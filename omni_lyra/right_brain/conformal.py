"""Split-conformal calibration for `ModelStateEstimator` predictions —
Phase 3 of the pentest-ROE track (docs/research/pentest-roe-validation.md).

Phase 2 (`confidence_fusion.py`) tried *online* self-consistency (N repeated
samples of the same model, fused at decision time) and found the result
empirically unstable run-to-run for weak models, at N x the runtime
estimator-call cost of a point estimate.

This module instead calibrates a fixed, additive per-state-variable safety
margin *offline*, once, from a labeled calibration set of (predicted, actual)
pairs. Applying it at decision time costs nothing beyond the existing single
point-estimate call — margins are precomputed constants — which is a much
closer fit to this project's local/"micro-cost" positioning than paying a
repeated-sampling tax on every decision. In exchange it gives a formal,
textbook split-conformal finite-sample guarantee (under the standard
exchangeability assumption — see the module-level caveat below) instead of
an empirically-observed-but-unstable heuristic.

One-sided by design: the nonconformity score only penalizes
*under*-prediction (`actual - predicted`, clipped at 0). Missing a real
violation (estimator too optimistic) is the safety-relevant failure mode;
over-predicting (false alarms) is a cost/usability problem, not a safety
one, and inflating margins to also guard against it would only make the
already-documented over-blocking problem worse for no safety benefit.

Caveat (report honestly, do not hide): split-conformal's marginal coverage
guarantee formally requires calibration and test points to be exchangeable.
The calibration corpus here is hand-authored to cover known violation
categories, not drawn i.i.d. from a "natural" action distribution — so the
guarantee is a working hypothesis to be checked empirically (see
`scripts/conformal_calibration_study.py`'s repeated-split coverage
validation), not something to assert without evidence.
"""

from __future__ import annotations

import math


def nonconformity_scores(
    pairs: list[tuple[dict[str, float], dict[str, float]]],
) -> dict[str, list[float]]:
    """One-sided residuals `max(0, actual_i - predicted_i)` per state var,
    across a calibration set of (predicted_state, actual_state) pairs.
    """
    if not pairs:
        raise ValueError("nonconformity_scores requires at least one (predicted, actual) pair")

    all_vars: set[str] = set()
    for predicted, actual in pairs:
        all_vars.update(predicted.keys())
        all_vars.update(actual.keys())

    scores: dict[str, list[float]] = {var: [] for var in all_vars}
    for predicted, actual in pairs:
        for var in all_vars:
            p = predicted.get(var, 0.0)
            a = actual.get(var, 0.0)
            scores[var].append(max(0.0, a - p))
    return scores


def min_calibration_size(alpha: float) -> int:
    """Minimum calibration-set size for a valid split-conformal quantile at
    the given alpha (standard result: need ceil(1/alpha) - 1 points so the
    ceil((n+1)(1-alpha))-th order statistic exists within the sample).
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return math.ceil(1 / alpha) - 1


def calibrate(scores_by_var: dict[str, list[float]], alpha: float) -> dict[str, float]:
    """Per-var split-conformal margin q_hat at confidence level (1 - alpha).

    Uses the standard finite-sample-corrected order statistic — the
    ceil((n+1)(1-alpha))-th smallest score, clipped to the largest available
    index — NOT a naive quantile, which under-covers for finite n. Raises
    ValueError if any var's calibration set is smaller than
    `min_calibration_size(alpha)` rather than silently returning an
    under-supported margin.
    """
    required_n = min_calibration_size(alpha)
    margins: dict[str, float] = {}
    for var, scores in scores_by_var.items():
        n = len(scores)
        if n < required_n:
            raise ValueError(
                f"var {var!r}: calibration set has {n} points, need >= {required_n} "
                f"for alpha={alpha} (min_calibration_size(alpha) = ceil(1/alpha) - 1)"
            )
        sorted_scores = sorted(scores)
        rank = min(math.ceil((n + 1) * (1 - alpha)), n)  # 1-indexed order statistic
        margins[var] = sorted_scores[rank - 1]
    return margins


def apply_margin(predicted_state: dict[str, float], margins: dict[str, float]) -> dict[str, float]:
    """Return predicted_state with each var's conformal margin added.

    Pure, non-mutating — same integration pattern as
    `confidence_fusion.fuse_max`/`fuse_majority`: applied to a candidate's
    predicted state before it's handed to `LyraVerifier.verify`/
    `select_best`, no changes to the verifier's own math required.
    """
    adjusted = dict(predicted_state)
    for var, margin in margins.items():
        if var in adjusted:
            adjusted[var] = adjusted[var] + margin
    return adjusted


# --- Scalar-V conformalization (Phase 4) -----------------------------------
#
# The per-variable margins above were found (docs/research/pentest-roe-
# validation.md, nemotron-mini cross-check) to collapse to universal BLOCK
# in this zero-tolerance domain: since block_threshold=1.0 and the smallest
# constraint weight is 5.0, ANY single variable needing margin=1.0 alone
# produces V >= 5.0 > block_threshold for every candidate, regardless of how
# well-calibrated the other variables are. That collapse is not proportional
# to how many variables are miscalibrated — it's a binary switch.
#
# The functions below instead conformalize the scalar energy V(s) directly:
# calibrate a single q_hat from the *scalar* residual V(actual) - V(predicted)
# across the corpus, and add it once to V_predicted at decision time (see
# LyraVerifier.verify's v_margin parameter) rather than perturbing each state
# variable before V is computed.
#
# This is reported as an open empirical question, not a fix assumed to work:
# a scalar margin is still a single global additive constant applied to
# every candidate regardless of that candidate's own predicted value, which
# is the same structural property that caused the per-variable collapse. If
# the corpus contains even one severely under-predicted case (e.g. a missed
# privilege-escalation with V residual >> block_threshold), q_hat will
# inherit that magnitude and collapse every decision to BLOCK just the same
# — see scripts/scalar_conformal_calibration_study.py for the actual
# measurement, not an assumption either way.
def scalar_nonconformity_scores(
    pairs: list[tuple[dict[str, float], dict[str, float]]],
    policy,
) -> list[float]:
    """One-sided residuals `max(0, V(actual) - V(predicted))` computed via
    the compiled policy's own energy function, across a calibration set of
    (predicted_state, actual_state) pairs. `policy` is any object exposing
    `.evaluate(state: dict[str, float]) -> float` (duck-typed — normally a
    `CompiledPolicy` from `left_brain.compiler`; not imported here to keep
    this module dependency-light).
    """
    if not pairs:
        raise ValueError("scalar_nonconformity_scores requires at least one (predicted, actual) pair")
    return [max(0.0, policy.evaluate(actual) - policy.evaluate(predicted)) for predicted, actual in pairs]


def calibrate_scalar(scores: list[float], alpha: float) -> float:
    """Split-conformal margin q_hat for a scalar residual list, at
    confidence level (1 - alpha). Same finite-sample-corrected order
    statistic as `calibrate` (the ceil((n+1)(1-alpha))-th smallest score),
    just for a single list instead of a per-var dict.
    """
    required_n = min_calibration_size(alpha)
    n = len(scores)
    if n < required_n:
        raise ValueError(
            f"calibration set has {n} points, need >= {required_n} for alpha={alpha} "
            f"(min_calibration_size(alpha) = ceil(1/alpha) - 1)"
        )
    sorted_scores = sorted(scores)
    rank = min(math.ceil((n + 1) * (1 - alpha)), n)  # 1-indexed order statistic
    return sorted_scores[rank - 1]


def apply_scalar_margin(v_predicted: float, margin: float) -> float:
    """Add the scalar conformal margin to a single predicted V value."""
    return v_predicted + margin


# --- Hybrid (Mondrian-style) conformalization (Phase 5) --------------------
#
# Phase 4 found that nemotron-mini's scalar margin=0.0 wasn't genuine
# calibration -- it was a coincidence where over-predicted noise on
# unrelated variables (weight-sum 23-75) happened to exceed the missed
# privilege_escalation_level's own weight (15) in every tested case,
# clipping the one-sided residual to 0. A follow-up on llama3.2:1b showed
# this isn't tied to "how well-calibrated a model looks overall" -- it's a
# narrow, per-instance coincidence (does the false-positive weight sum
# happen to exceed the missed danger's own weight?) that can't be predicted
# from aggregate model quality.
#
# The fix: never let a safety-critical variable's own under-prediction be
# cancellable by *other* variables' over-prediction. Split state vars into
# `critical_vars` (the highest-weight/highest-consequence ones -- e.g.
# privilege_escalation_level, credentials_exfiltrated) and everything else.
# Calibrate ordinary per-var margins (`calibrate`/`apply_margin`) for the
# critical vars only -- each one's correction is isolated to its own
# variable, computed before any summation happens, so it can never be offset
# by noise elsewhere. Calibrate a single scalar margin for the remaining
# ("aggregate") vars via `restricted_energy`/`aggregate_scalar_nonconformity_
# scores` below, so the lower-stakes categories still get the more
# space-efficient scalar treatment instead of each contributing their own
# all-or-nothing per-var collapse risk.
#
# This does NOT eliminate collapse when a critical var genuinely needs
# correction (weight >> block_threshold means any real fix there still
# blocks everything) -- that's intentional: for the highest-consequence
# categories, "safe but blunt" is the correct default when the estimator
# can't be trusted. What it fixes is Phase 4's false all-clear: a critical
# miss can no longer be silently absorbed by unrelated noise.
def restricted_energy(policy, state: dict[str, float], zeroed_vars: set[str]) -> float:
    """V(state) with `zeroed_vars` forced to 0 first -- isolates the energy
    contribution of the *other* (non-zeroed) variables. `policy` is any
    object exposing `.evaluate(state) -> float` (see scalar_nonconformity_
    scores' docstring for why this isn't type-hinted more strictly)."""
    adjusted = dict(state)
    for var in zeroed_vars:
        if var in adjusted:
            adjusted[var] = 0.0
    return policy.evaluate(adjusted)


def aggregate_scalar_nonconformity_scores(
    pairs: list[tuple[dict[str, float], dict[str, float]]],
    policy,
    critical_vars: set[str],
) -> list[float]:
    """One-sided scalar residuals computed on the energy contributed by the
    non-critical ("aggregate") variables only -- `critical_vars` are zeroed
    out of both predicted and actual state before evaluating, so this
    residual can't be inflated or masked by critical-variable errors (those
    are calibrated independently via ordinary per-var `calibrate`).
    """
    if not pairs:
        raise ValueError("aggregate_scalar_nonconformity_scores requires at least one pair")
    return [
        max(0.0, restricted_energy(policy, actual, critical_vars) - restricted_energy(policy, predicted, critical_vars))
        for predicted, actual in pairs
    ]
