"""Docker-based isolated sandbox for the Right Brain agent.

Lifecycle:
  sb = await DockerSandbox.create(image="python:3.12-slim")
  async with sb:
      obs = await sb.run("pip install pandas")
      state = await sb.observe()
"""

from __future__ import annotations

import asyncio
import docker
from dataclasses import dataclass, field
from typing import Any


_DEFAULT_IMAGE = "python:3.12-slim"

# Resource limits for the container
_CONTAINER_CONFIG: dict[str, Any] = {
    "mem_limit": "512m",
    "nano_cpus": int(1e9),        # 1 vCPU
    "network_disabled": False,    # network enabled but monitored by Lyra
    "read_only": False,
    "auto_remove": False,         # we remove manually on __aexit__
}


@dataclass
class SandboxObservation:
    """Snapshot of measurable state variables inside the container."""
    cpu_usage: float       # 0.0 – 1.0
    mem_used_bytes: int
    net_bytes_out: int
    file_deletions: int    # tracked by the loop externally; sandbox reports 0 by default
    net_connections: int


@dataclass
class DockerSandbox:
    image: str = _DEFAULT_IMAGE
    _container: Any = field(default=None, init=False, repr=False)
    _client: Any = field(default=None, init=False, repr=False)
    _net_bytes_out_baseline: int = field(default=0, init=False, repr=False)
    _workspace_file_count_high_water: int = field(default=0, init=False, repr=False)
    _cumulative_file_deletions: int = field(default=0, init=False, repr=False)
    _prev_cpu_total: int | None = field(default=None, init=False, repr=False)
    _prev_sys_total: int | None = field(default=None, init=False, repr=False)

    @classmethod
    async def create(cls, image: str = _DEFAULT_IMAGE) -> "DockerSandbox":
        sb = cls(image=image)
        await sb._start()
        return sb

    async def _start(self) -> None:
        loop = asyncio.get_running_loop()
        self._client = await loop.run_in_executor(None, docker.from_env)
        self._container = await loop.run_in_executor(
            None,
            lambda: self._client.containers.run(
                self.image,
                # /workspace must exist before the container starts: run()
                # always execs with workdir="/workspace", and most base
                # images (e.g. python:3.12-slim) don't ship that directory —
                # creating it here, not via a post-start run(), avoids a
                # chicken-and-egg chdir failure on the very first command.
                command=["sh", "-c", "mkdir -p /workspace && sleep infinity"],
                detach=True,
                **_CONTAINER_CONFIG,
            ),
        )
        # mkdir above is near-instant but detach=True returns as soon as the
        # container process starts, not once mkdir has run — a small margin
        # avoids a rare race against the first exec_run's workdir="/workspace".
        await asyncio.sleep(0.2)

    async def run(self, command: str, timeout: int = 30) -> str:
        """Execute a shell command inside the container and return stdout+stderr."""
        loop = asyncio.get_running_loop()

        def _exec():
            exit_code, output = self._container.exec_run(
                cmd=["bash", "-c", command],
                stdout=True,
                stderr=True,
                demux=False,
                workdir="/workspace",
            )
            return exit_code, output

        exit_code, raw = await asyncio.wait_for(
            loop.run_in_executor(None, _exec),
            timeout=timeout,
        )
        text = raw.decode("utf-8", errors="replace") if raw else ""
        return f"[exit:{exit_code}]\n{text}"

    async def observe(self) -> SandboxObservation:
        """Sample container resource metrics."""
        loop = asyncio.get_running_loop()

        def _stats():
            self._container.reload()
            return self._container.stats(stream=False)

        stats = await loop.run_in_executor(None, _stats)

        # Docker's one-shot `stats(stream=False)` duplicates cpu_usage.total_usage
        # into precpu_stats (only system_cpu_usage, a host-wide clock, actually
        # differs) — diffing cpu_stats against precpu_stats here always yields
        # zero, regardless of real container load. Confirmed empirically: a
        # CPU-bound busy loop still reported cpu_usage=0.0 with the old code.
        # Tracking our own previous reading across successive observe() calls
        # (the intended before/after-an-action usage pattern) fixes this.
        cpu_total = stats["cpu_stats"]["cpu_usage"]["total_usage"]
        sys_total = stats["cpu_stats"]["system_cpu_usage"]
        num_cpus = stats["cpu_stats"].get("online_cpus", 1)

        if self._prev_cpu_total is not None:
            cpu_delta = cpu_total - self._prev_cpu_total
            sys_delta = sys_total - self._prev_sys_total
            cpu_pct = (cpu_delta / sys_delta) * num_cpus if sys_delta > 0 else 0.0
        else:
            cpu_pct = 0.0  # no prior sample yet to diff against

        self._prev_cpu_total = cpu_total
        self._prev_sys_total = sys_total

        mem_used = stats["memory_stats"].get("usage", 0)

        net_out = sum(
            iface["tx_bytes"]
            for iface in stats.get("networks", {}).values()
        )
        net_connections = await self._count_connections()

        return SandboxObservation(
            cpu_usage=min(1.0, cpu_pct),
            mem_used_bytes=mem_used,
            net_bytes_out=max(0, net_out - self._net_bytes_out_baseline),
            file_deletions=await self._track_file_deletions(),
            net_connections=net_connections,
        )

    async def _count_connections(self) -> int:
        """Count ESTABLISHED outbound TCP connections in the container's network namespace.

        `ss`/`netstat` (iproute2/net-tools) aren't installed in minimal base
        images like `python:3.12-slim` — confirmed empirically (`ss: command
        not found`), which is why this always silently read 0 regardless of
        real connections made. Parsing `/proc/net/tcp{,6}` directly needs no
        extra tools, only `python3`, which every image this project targets
        already has (it's a Python agent sandbox). State `01` = ESTABLISHED;
        see `man 5 proc` / `Documentation/networking/proc_net_tcp.txt`.
        """
        script = (
            "import re\n"
            "n = 0\n"
            "for path in ('/proc/net/tcp', '/proc/net/tcp6'):\n"
            "    try:\n"
            "        with open(path) as f:\n"
            "            lines = f.readlines()[1:]\n"
            "    except FileNotFoundError:\n"
            "        continue\n"
            "    for line in lines:\n"
            "        parts = line.split()\n"
            "        if len(parts) > 3 and parts[3] == '01':\n"
            "            n += 1\n"
            "print(n)\n"
        )
        try:
            result = await self.run(f"python3 -c \"{script}\"")
            for line in result.split("\n"):
                line = line.strip()
                if line.isdigit():
                    return int(line)
        except Exception:
            pass
        return 0

    async def _track_file_deletions(self) -> int:
        """Cumulative count of files deleted from /workspace since sandbox start.

        Tracks a high-water mark of the file count seen so far; any drop below
        that mark means files were removed since the last observation, which
        gets folded into a running total. Matches how `file_deletions` is
        used as a state var elsewhere (an absolute cumulative count compared
        against `max_file_deletions`, not a per-step delta) — see
        no_file_deletion in config/security_research.yaml.
        """
        try:
            result = await self.run("find /workspace -type f 2>/dev/null | wc -l")
            file_count = None
            for line in result.split("\n"):
                line = line.strip()
                if line.isdigit():
                    file_count = int(line)
                    break
        except Exception:
            file_count = None

        if file_count is None:
            return self._cumulative_file_deletions

        if file_count > self._workspace_file_count_high_water:
            self._workspace_file_count_high_water = file_count
        elif file_count < self._workspace_file_count_high_water:
            self._cumulative_file_deletions += self._workspace_file_count_high_water - file_count
            self._workspace_file_count_high_water = file_count

        return self._cumulative_file_deletions

    async def destroy(self) -> None:
        if self._container:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: self._container.remove(force=True))
            self._container = None
        if self._client:
            self._client.close()
            self._client = None

    async def __aenter__(self) -> "DockerSandbox":
        return self

    async def __aexit__(self, *_) -> None:
        await self.destroy()
