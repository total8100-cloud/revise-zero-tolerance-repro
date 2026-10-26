"""Self-Research Loop — orchestrates Right Brain → Lyra → Sandbox.

Flow per step:
  1. Right Brain proposes a CLI action
  2. Lyra estimates the post-action state and verifies [V_t, ΔV_t]
  3a. SAFE  → execute in sandbox, commit real state
  3b. WARN  → log warning, allow with revision note, execute, commit
  3c. BLOCK → skip execution, feed rejection back to Right Brain history

Two Right Brain backends are supported (`ResearchSession.use_guided_agent`):
  - Ollama (default, unchanged): ExplorationAgent talks to a local Ollama
    model over HTTP. No PPLM steering, no hidden-state access — see
    docs/research/lyapunov-guided-decoding.md, Decision 3, for why.
  - Guided (opt-in): GuidedExplorationAgent generates with a local
    `transformers` model, continuously PPLM-steered against constraints'
    danger_tokens, with a clause-boundary hard check reusing the same
    ModelStateEstimator + LyraVerifier pipeline as the Ollama path. Every
    clause checkpoint's hidden state gets paired with the REAL post-action
    state (once the full action executes) and logged via
    state_predictor.log_training_example — this is how real StatePredictor
    training data accumulates instead of the hand-curated examples in
    scripts/collect_real_training_data.py.
"""

from __future__ import annotations

import asyncio
import functools
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from .left_brain.compiler import LyraCompiler, CompiledPolicy
from .left_brain.verifier import LyraVerifier, Verdict, VerificationResult
from .right_brain.agent import ExplorationAgent
from .right_brain.state_estimator import ModelStateEstimator, StateEstimationError
from .right_brain.state_predictor import TrainingExample, log_training_example
from .sandbox.docker_env import DockerSandbox, SandboxObservation

# This module prints em dashes, arrows, and Greek letters (V, dV) via rich.
# On a non-UTF-8 Windows console (e.g. the Korean cp949 codepage), rich's
# legacy Windows renderer crashes the whole process with UnicodeEncodeError
# the first time one of those characters is printed — found for real when
# ModelStateEstimator's failure-fallback message ("... unavailable (...) -
# falling back...") hit this for the first time. Forcing UTF-8 (with
# errors='replace' as a last-resort safety net) avoids the crash regardless
# of the host codepage. Guarded because captured streams (e.g. pytest's
# capsys) may not support .reconfigure().
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

console = Console()


@dataclass
class StepRecord:
    step: int
    proposed_action: str
    verdict: Verdict
    v_before: float
    v_after: float
    delta_v: float
    observation: str
    latency_ms: float


@dataclass
class ResearchSession:
    domain_context: str
    policy: CompiledPolicy
    max_steps: int = 50
    sandbox_image: str = "python:3.12-slim"
    ollama_model: str = "nemotron-mini"
    ollama_url: str = "http://localhost:11434"

    # Right Brain backend selection — see module docstring.
    use_guided_agent: bool = False
    guided_model_name: str = "distilgpt2"
    guided_max_tokens: int = 48
    training_log_dir: str | None = None   # defaults to data/state_predictor_training/<domain>

    history: list[dict] = field(default_factory=list, init=False)
    records: list[StepRecord] = field(default_factory=list, init=False)

    async def run(self) -> list[StepRecord]:
        verifier = LyraVerifier(policy=self.policy)

        sandbox = await DockerSandbox.create(self.sandbox_image)
        async with sandbox:
            async with ModelStateEstimator(
                model=self.ollama_model,
                ollama_base_url=self.ollama_url,
            ) as estimator:
                # Ensure workspace exists
                await sandbox.run("mkdir -p /workspace")

                if self.use_guided_agent:
                    await self._run_guided(sandbox, verifier, estimator)
                else:
                    await self._run_ollama(sandbox, verifier, estimator)

        self._print_summary()
        return self.records

    async def _run_ollama(
        self, sandbox: DockerSandbox, verifier: LyraVerifier, estimator: ModelStateEstimator
    ) -> None:
        async with ExplorationAgent(
            model=self.ollama_model,
            ollama_base_url=self.ollama_url,
        ) as agent:
            for step in range(1, self.max_steps + 1):
                obs_before = await sandbox.observe()
                current_state = _obs_to_state(obs_before)

                console.rule(f"[bold cyan]Step {step}")

                action = await agent.propose_action(
                    domain_context=self.domain_context,
                    sandbox_state=current_state,
                    history=self.history,
                )

                if action.strip() == "DONE":
                    console.print("[green]Agent signalled DONE.")
                    break

                console.print(f"[yellow]PROPOSED:[/yellow] {action}")

                observation, _actual_state, result = await self._verify_and_execute(
                    sandbox, verifier, estimator, current_state, action
                )
                self._record_step(step, action, result, observation)

    async def _run_guided(
        self, sandbox: DockerSandbox, verifier: LyraVerifier, estimator: ModelStateEstimator
    ) -> None:
        # Lazy import: only the guided path needs torch/transformers (the
        # 'pplm' optional dependency group) — the default Ollama path must
        # stay usable without them installed.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .right_brain.guided_agent import GuidedExplorationAgent

        console.print(f"[cyan]Loading guided agent model ({self.guided_model_name}) ...[/cyan]")
        tokenizer = AutoTokenizer.from_pretrained(self.guided_model_name)
        model = AutoModelForCausalLM.from_pretrained(self.guided_model_name)
        model.eval()

        guided_agent = GuidedExplorationAgent(
            model=model,
            tokenizer=tokenizer,
            constraints=self.policy.spec.constraints,
            max_tokens=self.guided_max_tokens,
        )
        log_dir = self.training_log_dir or f"data/state_predictor_training/{self.policy.domain}"
        loop = asyncio.get_running_loop()

        for step in range(1, self.max_steps + 1):
            obs_before = await sandbox.observe()
            current_state = _obs_to_state(obs_before)

            console.rule(f"[bold cyan]Step {step}")

            prompt = _build_text_prompt(self.domain_context, current_state, self.history)

            def clause_check(partial_text: str, _state=current_state) -> bool:
                # Runs inside the worker thread propose_action() executes in
                # (see run_in_executor below) — bridges back to this
                # coroutine's event loop to reuse the same async estimator +
                # verifier the Ollama path uses, rather than a second
                # implementation of the hard-check logic.
                future = asyncio.run_coroutine_threadsafe(
                    self._clause_is_safe(partial_text, _state, estimator, verifier), loop
                )
                return future.result()

            action = await loop.run_in_executor(
                None, functools.partial(guided_agent.propose_action, prompt, clause_check)
            )

            if not action or action.strip() == "DONE":
                console.print("[green]Guided agent produced no further action, stopping.")
                break

            console.print(f"[yellow]PROPOSED:[/yellow] {action}")

            observation, actual_state, result = await self._verify_and_execute(
                sandbox, verifier, estimator, current_state, action
            )
            self._record_step(step, action, result, observation)

            # Real training data: pair every clause checkpoint's hidden state
            # from this step's generation with the REAL final state the full
            # action produced. Approximate — a mid-generation clause's hidden
            # state technically corresponds to that clause's own eventual
            # sub-state, not the full action's final state — but it's the
            # only ground truth available without separately executing every
            # partial clause. See docs/research/lyapunov-guided-decoding.md,
            # Decision 6.
            for record in guided_agent.clause_records:
                log_training_example(
                    log_dir, TrainingExample(hidden_state=record.hidden_state, actual_state=actual_state)
                )

    async def _clause_is_safe(
        self,
        partial_text: str,
        current_state: dict[str, float],
        estimator: ModelStateEstimator,
        verifier: LyraVerifier,
    ) -> bool:
        """True unless the partial command so far would BLOCK if run as-is.

        WARN still returns True (matches _verify_and_execute's behavior:
        only BLOCK stops execution) — a clause reaching a risky-but-allowed
        state shouldn't halt generation early.
        """
        try:
            predicted_state = await estimator.estimate(
                self.policy.spec.state_vars, current_state, partial_text
            )
        except StateEstimationError:
            predicted_state = _heuristic_estimate(current_state, partial_text)
        result = verifier.verify(current_state, predicted_state)
        return result.verdict != Verdict.BLOCK

    async def _verify_and_execute(
        self,
        sandbox: DockerSandbox,
        verifier: LyraVerifier,
        estimator: ModelStateEstimator,
        current_state: dict[str, float],
        action: str,
    ) -> tuple[str, dict[str, float], VerificationResult]:
        """Shared Left-Brain gate + Sandbox execution, used by both agent backends."""
        try:
            predicted_state = await estimator.estimate(
                self.policy.spec.state_vars, current_state, action
            )
        except StateEstimationError as exc:
            console.print(
                f"[red]Model state estimator unavailable ({exc}) — "
                "falling back to keyword heuristic[/red]"
            )
            predicted_state = _heuristic_estimate(current_state, action)

        result = verifier.verify(current_state, predicted_state)
        _log_verdict(result, action)

        if result.verdict == Verdict.BLOCK:
            observation = f"[BLOCKED by Lyra] {result.reason}"
            verifier.commit(current_state)   # state unchanged
            actual_state = current_state
        else:
            if result.verdict == Verdict.WARN:
                console.print(f"[orange3]WARN:[/orange3] {result.reason}")

            observation = await sandbox.run(action)
            obs_after = await sandbox.observe()
            actual_state = _obs_to_state(obs_after)
            verifier.commit(actual_state)
            console.print(f"[dim]{observation[:300]}[/dim]")

        return observation, actual_state, result

    def _record_step(self, step: int, action: str, result: VerificationResult, observation: str) -> None:
        self.history.append({"action": action, "observation": observation})
        self.records.append(
            StepRecord(
                step=step,
                proposed_action=action,
                verdict=result.verdict,
                v_before=result.v_current,
                v_after=result.v_predicted,
                delta_v=result.delta_v,
                observation=observation,
                latency_ms=result.latency_ms,
            )
        )

    def _print_summary(self) -> None:
        table = Table(title="OMNI-Lyra Session Summary")
        table.add_column("Step", style="cyan")
        table.add_column("Action", style="white", max_width=50)
        table.add_column("Verdict", style="bold")
        table.add_column("V_before", justify="right")
        table.add_column("V_after", justify="right")
        table.add_column("ΔV", justify="right")
        table.add_column("Lyra ms", justify="right")

        for r in self.records:
            color = {"SAFE": "green", "WARN": "yellow", "BLOCK": "red"}[r.verdict]
            table.add_row(
                str(r.step),
                r.proposed_action[:50],
                f"[{color}]{r.verdict}[/{color}]",
                f"{r.v_before:.3f}",
                f"{r.v_after:.3f}",
                f"{r.delta_v:+.3f}",
                f"{r.latency_ms:.2f}",
            )
        console.print(table)


# ── Helpers ─────────────────────────────────────────────────────────────────

_FEW_SHOT_PREAMBLE = (
    "$ pwd\n"
    "/workspace\n"
    "$ ls -la\n"
    "total 8\ndrwxr-xr-x 2 root root 4096 .\n"
)


def _build_text_prompt(domain_context: str, sandbox_state: dict, history: list[dict]) -> str:
    """Flatten domain context + recent history into a plain-text completion prompt.

    GuidedExplorationAgent generates via raw next-token completion, not
    chat-formatted messages (see agent.py's _build_messages for that style) —
    a small non-instruction-tuned local model (e.g. distilgpt2) has nothing
    to follow otherwise. A couple of static few-shot exemplars nudge it
    toward continuing in shell-transcript style; without them it tends to
    degenerate into echoing the state dict verbatim instead of proposing a
    command — found empirically running this for real. The raw state dict is
    kept as a comment, not placed directly before the "$" prompt, to avoid
    anchoring generation onto continuing that dict-like text. Only the last
    few history entries are kept: small local models have short context
    windows and long prompts slow down every one of PPLM's extra per-token
    forward passes.
    """
    lines = [f"# task: {domain_context}", f"# current state: {sandbox_state}", "", _FEW_SHOT_PREAMBLE.rstrip("\n")]
    for entry in history[-3:]:
        lines.append(f"$ {entry['action']}")
        lines.append(str(entry["observation"])[:200])
    lines.append("$")
    return "\n".join(lines)


def _obs_to_state(obs: SandboxObservation) -> dict[str, float]:
    return {
        "bytes_out": float(obs.net_bytes_out),
        "cpu_usage": obs.cpu_usage,
        "file_deletions": float(obs.file_deletions),
        "net_connections": float(obs.net_connections),
    }


def _heuristic_estimate(
    current: dict[str, float],
    action: str,
) -> dict[str, float]:
    """Keyword-matching fallback used only when ModelStateEstimator fails.

    Hardcoded to this domain's variable names, unlike the model-based
    estimator — it exists purely as a conservative safety net (never used
    when Ollama is reachable) so a dead model endpoint can't silently
    degrade into "no risk predicted."
    """
    est = dict(current)
    action_lower = action.lower()

    if any(kw in action_lower for kw in ["curl", "wget", "requests", "http", "ftp"]):
        est["bytes_out"] = current["bytes_out"] + 1_048_576    # +1 MB estimate
        est["net_connections"] = current["net_connections"] + 1

    if any(kw in action_lower for kw in ["rm ", "rm -", "rmdir", "shutil.rmtree"]):
        est["file_deletions"] = current["file_deletions"] + 1

    if any(kw in action_lower for kw in ["stress", "fork bomb", ":(){ :|:& };:", "dd if="]):
        est["cpu_usage"] = min(1.0, current["cpu_usage"] + 0.5)

    return est


def _log_verdict(result, action: str) -> None:
    color = {"SAFE": "green", "WARN": "yellow", "BLOCK": "red"}[result.verdict.value]
    console.print(
        f"[{color}]Lyra [{result.verdict}][/{color}] "
        f"V={result.v_current:.4f}→{result.v_predicted:.4f} "
        f"ΔV={result.delta_v:+.4f} "
        f"({result.latency_ms:.2f}ms)"
    )


# ── Entry point ─────────────────────────────────────────────────────────────

async def run_session(
    domain_context: str,
    constraint_yaml: str | Path,
    max_steps: int = 50,
    sandbox_image: str = "python:3.12-slim",
    ollama_model: str = "nemotron-mini",
    ollama_url: str = "http://localhost:11434",
    use_guided_agent: bool = False,
    guided_model_name: str = "distilgpt2",
    guided_max_tokens: int = 48,
    training_log_dir: str | None = None,
) -> list[StepRecord]:
    policy = LyraCompiler.from_yaml(constraint_yaml)
    session = ResearchSession(
        domain_context=domain_context,
        policy=policy,
        max_steps=max_steps,
        sandbox_image=sandbox_image,
        ollama_model=ollama_model,
        ollama_url=ollama_url,
        use_guided_agent=use_guided_agent,
        guided_model_name=guided_model_name,
        guided_max_tokens=guided_max_tokens,
        training_log_dir=training_log_dir,
    )
    return await session.run()
