"""Lyra Runtime Verifier — [V_t, ΔV_t] real-time safety gate.

Decision logic:
  V_t  = V(s_t)           current energy
  V_t' = V(s_t')          predicted energy after proposed action
  ΔV   = V_t' - V_t

  SAFE  : V_t' == 0  (strictly inside safe set)
  WARN  : 0 < V_t' ≤ block_threshold  OR  ΔV > 0
  BLOCK : V_t' > block_threshold
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .compiler import CompiledPolicy


class Verdict(str, Enum):
    SAFE = "SAFE"
    WARN = "WARN"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class VerificationResult:
    verdict: Verdict
    v_current: float
    v_predicted: float
    delta_v: float
    components_predicted: dict[str, float]
    latency_ms: float
    reason: str


@dataclass
class LyraVerifier:
    """Stateful verifier that tracks V_t across the agent trajectory."""

    policy: CompiledPolicy
    _v_prev: float = field(default=0.0, init=False, repr=False)

    def verify(
        self,
        current_state: dict[str, float],
        predicted_state: dict[str, float],
        v_margin: float = 0.0,
    ) -> VerificationResult:
        """
        Verify whether transitioning from current_state → predicted_state is safe.

        Args:
            current_state:   Observable state before the action.
            predicted_state: Estimated state if the proposed action executes.
            v_margin:        Optional scalar split-conformal margin (right_brain/
                              conformal.py's calibrate_scalar) added directly to
                              V_predicted before thresholding — an alternative to
                              perturbing individual state variables. Defaults to
                              0.0, reproducing today's behavior exactly.

        Returns:
            VerificationResult with SAFE / WARN / BLOCK verdict.
        """
        t0 = time.perf_counter()

        v_current = self.policy.evaluate(current_state)
        v_predicted = self.policy.evaluate(predicted_state) + v_margin
        delta_v = v_predicted - v_current
        components = self.policy.evaluate_components(predicted_state)

        verdict, reason = self._decide(v_current, v_predicted, delta_v)

        latency_ms = (time.perf_counter() - t0) * 1000

        return VerificationResult(
            verdict=verdict,
            v_current=v_current,
            v_predicted=v_predicted,
            delta_v=delta_v,
            components_predicted=components,
            latency_ms=latency_ms,
            reason=reason,
        )

    def _decide(
        self, v_current: float, v_predicted: float, delta_v: float
    ) -> tuple[Verdict, str]:
        block_thresh = self.policy.block_threshold
        warn_thresh = self.policy.warn_threshold

        if v_predicted > block_thresh:
            return Verdict.BLOCK, (
                f"V_predicted={v_predicted:.4f} exceeds block threshold {block_thresh}"
            )

        if v_predicted > warn_thresh:
            return Verdict.WARN, (
                f"V_predicted={v_predicted:.4f} > warn_threshold={warn_thresh:.4f}"
            )

        if delta_v > 0 and v_current > warn_thresh:
            # Already in warn zone and moving further away from safe set
            return Verdict.WARN, (
                f"ΔV={delta_v:.4f} > 0 while in warn zone (V_current={v_current:.4f})"
            )

        return Verdict.SAFE, "All constraints satisfied"

    def select_best(
        self,
        current_state: dict[str, float],
        candidates: list[tuple[str, dict[str, float]]],
        v_margin: float = 0.0,
    ) -> tuple[int, VerificationResult]:
        """REVISE(x, a) = argmin_j V(x, j) — the first real implementation of it in this codebase.

        Paper 1's Proposition 4.1 proves this selection rule never increases
        V beyond what the best available option achieves, for a small
        enumerable action set. `candidates` (action_text, predicted_state)
        pairs *are* that enumerable set here — e.g. K shell commands sampled
        from the Right Brain for one decision point — so this directly
        inherits the proven guarantee rather than asserting a new one for
        the continuous/free-form case (see docs/research/lyapunov-guided-
        decoding.md Decisions 1-10 for why that continuous extension was
        tried via PPLM steering and didn't hold up under scrutiny).

        Verifies every candidate via the existing `verify()` — no new
        energy math, just argmin over predicted V. Returns the index of the
        lowest-V candidate and its VerificationResult. Ties resolve to the
        first (lowest-index) candidate, for determinism.
        """
        if not candidates:
            raise ValueError("select_best requires at least one candidate")

        best_index = 0
        best_result = self.verify(current_state, candidates[0][1], v_margin=v_margin)

        for i, (_action, predicted_state) in enumerate(candidates[1:], start=1):
            result = self.verify(current_state, predicted_state, v_margin=v_margin)
            if result.v_predicted < best_result.v_predicted:
                best_index = i
                best_result = result

        return best_index, best_result

    def commit(self, actual_state: dict[str, float]) -> float:
        """Update V_prev after an action has been executed.

        Call this after the sandbox runs the action and returns the real new state.
        Returns the new V_t.
        """
        v_new = self.policy.evaluate(actual_state)
        self._v_prev = v_new
        return v_new

    @property
    def v_prev(self) -> float:
        return self._v_prev
