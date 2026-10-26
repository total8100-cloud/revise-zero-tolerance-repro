"""Tests for DockerSandbox — no real Docker daemon needed (all mocked).

`_track_file_deletions` only depends on `self.run(...)`'s return value, so
it's tested here by monkeypatching `run` directly rather than spinning up a
container. TestCreateAsyncContextManager guards against a regression found
while running the first real end-to-end sandbox session: `create()` is an
async classmethod, so `async with DockerSandbox.create(...) as sb:` fails
with `TypeError: 'coroutine' object does not support the asynchronous
context manager protocol` — it must be `sb = await DockerSandbox.create(...)`
then `async with sb:`. loop.py had this exact bug, undiscovered until Docker
was actually available to run it.
"""

from unittest.mock import MagicMock

import pytest

from omni_lyra.sandbox import docker_env
from omni_lyra.sandbox.docker_env import DockerSandbox


def _sandbox_with_file_counts(counts: list[str]):
    """A DockerSandbox whose .run() yields each of `counts` in sequence, one per call."""
    sb = DockerSandbox()
    it = iter(counts)

    async def fake_run(command, timeout=30):
        return f"[exit:0]\n{next(it)}\n"

    sb.run = fake_run
    return sb


class TestCountConnections:
    """Host-side parsing of _count_connections' output.

    The actual counting happens via a python3 script run *inside* the
    container (parsing /proc/net/tcp{,6} — ss/netstat aren't installed in
    minimal images like python:3.12-slim, confirmed empirically), verified
    separately against a real container. This only tests that the host side
    correctly extracts the printed digit from self.run()'s output.
    """

    @pytest.mark.asyncio
    async def test_extracts_count_from_output(self):
        sb = _sandbox_with_file_counts(["3"])
        assert await sb._count_connections() == 3

    @pytest.mark.asyncio
    async def test_zero_connections(self):
        sb = _sandbox_with_file_counts(["0"])
        assert await sb._count_connections() == 0

    @pytest.mark.asyncio
    async def test_run_failure_returns_zero(self):
        sb = DockerSandbox()

        async def failing_run(command, timeout=30):
            raise RuntimeError("exec failed")

        sb.run = failing_run
        assert await sb._count_connections() == 0


class TestTrackFileDeletions:
    @pytest.mark.asyncio
    async def test_no_change_reports_zero(self):
        sb = _sandbox_with_file_counts(["5", "5", "5"])
        for _ in range(3):
            assert await sb._track_file_deletions() == 0

    @pytest.mark.asyncio
    async def test_growth_reports_zero_deletions(self):
        sb = _sandbox_with_file_counts(["5", "8", "12"])
        assert await sb._track_file_deletions() == 0
        assert await sb._track_file_deletions() == 0
        assert await sb._track_file_deletions() == 0

    @pytest.mark.asyncio
    async def test_drop_is_recorded_cumulatively(self):
        sb = _sandbox_with_file_counts(["10", "3", "3"])
        assert await sb._track_file_deletions() == 0
        assert await sb._track_file_deletions() == 7   # 10 -> 3, dropped 7
        assert await sb._track_file_deletions() == 7   # no further change

    @pytest.mark.asyncio
    async def test_repeated_drops_accumulate(self):
        sb = _sandbox_with_file_counts(["10", "6", "9", "2"])
        assert await sb._track_file_deletions() == 0
        assert await sb._track_file_deletions() == 4    # 10 -> 6
        assert await sb._track_file_deletions() == 4    # 6 -> 9, growth doesn't add
        assert await sb._track_file_deletions() == 4 + 7  # new high water 9 -> 2, +7

    @pytest.mark.asyncio
    async def test_unparseable_output_holds_previous_total(self):
        sb = _sandbox_with_file_counts(["10", "3", "not a number"])
        await sb._track_file_deletions()
        after_drop = await sb._track_file_deletions()
        held = await sb._track_file_deletions()
        assert held == after_drop


def _fake_stats(cpu_total: int, sys_total: int, num_cpus: int = 1) -> dict:
    return {
        "cpu_stats": {
            "cpu_usage": {"total_usage": cpu_total},
            "system_cpu_usage": sys_total,
            "online_cpus": num_cpus,
        },
        # precpu_stats deliberately mirrors cpu_stats' cpu_usage — this is
        # what real Docker one-shot stats() actually returns (confirmed
        # empirically), and observe() must not rely on it for the delta.
        "precpu_stats": {
            "cpu_usage": {"total_usage": cpu_total},
            "system_cpu_usage": sys_total - 1,
        },
        "memory_stats": {"usage": 0},
        "networks": {},
    }


def _sandbox_with_stats(stats_sequence: list[dict]):
    sb = DockerSandbox()
    sb._container = MagicMock()
    it = iter(stats_sequence)
    sb._container.stats.side_effect = lambda stream=False: next(it)
    sb._container.reload = MagicMock()

    async def fake_run(command, timeout=30):
        return "[exit:0]\n0\n"  # 0 connections, 0 files found

    sb.run = fake_run
    return sb


class TestObserveCpuTracking:
    """observe()'s CPU%% must diff its OWN previous reading, not Docker's precpu_stats."""

    @pytest.mark.asyncio
    async def test_first_observe_reports_zero_no_prior_sample(self):
        sb = _sandbox_with_stats([_fake_stats(cpu_total=1000, sys_total=2000)])
        obs = await sb.observe()
        assert obs.cpu_usage == 0.0

    @pytest.mark.asyncio
    async def test_second_observe_computes_real_delta(self):
        sb = _sandbox_with_stats(
            [
                _fake_stats(cpu_total=1000, sys_total=100_000, num_cpus=1),
                _fake_stats(cpu_total=6000, sys_total=105_000, num_cpus=1),
            ]
        )
        await sb.observe()
        obs2 = await sb.observe()
        # cpu_delta=5000, sys_delta=5000, num_cpus=1 -> 100% busy in that window
        assert obs2.cpu_usage == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_idle_between_two_observes_reports_near_zero(self):
        sb = _sandbox_with_stats(
            [
                _fake_stats(cpu_total=1000, sys_total=100_000, num_cpus=4),
                _fake_stats(cpu_total=1010, sys_total=200_000, num_cpus=4),
            ]
        )
        await sb.observe()
        obs2 = await sb.observe()
        assert obs2.cpu_usage < 0.01

    @pytest.mark.asyncio
    async def test_cpu_usage_capped_at_one(self):
        sb = _sandbox_with_stats(
            [
                _fake_stats(cpu_total=0, sys_total=1000, num_cpus=8),
                _fake_stats(cpu_total=900, sys_total=1010, num_cpus=8),
            ]
        )
        await sb.observe()
        obs2 = await sb.observe()
        assert obs2.cpu_usage == 1.0


class TestCreateAsyncContextManager:
    @pytest.mark.asyncio
    async def test_create_then_async_with(self, monkeypatch):
        fake_container = MagicMock()
        fake_client = MagicMock()
        fake_client.containers.run.return_value = fake_container
        monkeypatch.setattr(docker_env.docker, "from_env", lambda: fake_client)

        sandbox = await DockerSandbox.create("dummy:image")
        async with sandbox as sb:
            assert sb is sandbox

        fake_container.remove.assert_called_once_with(force=True)
