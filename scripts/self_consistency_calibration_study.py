"""Self-consistency calibration study: does repeated sampling of a *single*
model (temperature > 0, fused via max/majority vote) improve recall/false-
alarm tradeoff over a single deterministic (temperature=0) point estimate —
and does it do better than the 4-model cross-model max-ensemble that was
already shown to recover recall at the cost of nearly tripling false alarms
(`estimator_calibration_study.py`: 5/5 recall, 20 false alarms, vs. best
individual model's 6-8)?

Unlike that cross-model ensemble, this only ever needs the ONE model an
operator already runs — deployable in practice, and its cost is an honest,
quantifiable tradeoff (N x estimator calls per decision) worth reporting
given this project's "micro-cost, local model" positioning.

Not a pytest test. Requires Ollama serving all models in MODELS. Run manually:

    python scripts/self_consistency_calibration_study.py
"""

from __future__ import annotations

import asyncio

from omni_lyra.left_brain.compiler import LyraCompiler
from omni_lyra.right_brain.confidence_fusion import fuse_majority, fuse_max
from omni_lyra.right_brain.state_estimator import ModelStateEstimator, StateEstimationError
from estimator_calibration_study import MODELS, print_details, score
from pentest_test_cases import CONFIG_PATH, CURRENT_STATE, TEST_CASES

N_SAMPLES = 5
SAMPLING_TEMPERATURE = 0.7
MAJORITY_MIN_VOTE_FRACTION = 0.6  # 3-of-5

# Recorded from estimator_calibration_study.py's own summary (not re-derived
# here — this script's point is the self-consistency comparison, not
# re-measuring point-estimate/cross-model-ensemble numbers already on file).
PRIOR_POINT_RECALL = {
    "nemotron-mini": (4, 10),
    "qwen2.5-coder:0.5b": (1, 9),
    "llama3.2:1b": (1, 6),
    "phi3:mini": (5, 8),
}
PRIOR_ENSEMBLE_RECALL = (5, 20)  # (tp, fp) — 4-model max-ensemble


async def get_variant_predictions(model: str, policy) -> dict[str, dict[str, dict[str, float]]]:
    """Returns {variant: {case_name: predicted_state}} for one model."""
    variants: dict[str, dict[str, dict[str, float]]] = {"point": {}, "max": {}, "majority": {}}
    # Estimator failures use the fail-closed worst case (same rule as
    # PentestSession), not current_state. Counted and printed so that any
    # failure is visible in the run log rather than silently absorbed.
    worst_case = {sv.name: 1.0 for sv in policy.spec.state_vars}
    failures = {"point": 0, "samples": 0}
    async with ModelStateEstimator(model=model) as est:
        for name, action, _expected in TEST_CASES:
            try:
                point = await est.estimate(
                    policy.spec.state_vars, CURRENT_STATE, action, temperature=0.0
                )
            except StateEstimationError:
                point = dict(worst_case)
                failures["point"] += 1
            variants["point"][name] = point

            # Production policy (2026-10-07): failed draws are worst-case votes. Under max
            # fusion this blocks the same candidates as all-or-nothing, so one call serves both.
            samples = await est.estimate_samples(
                policy.spec.state_vars,
                CURRENT_STATE,
                action,
                n_samples=N_SAMPLES,
                temperature=SAMPLING_TEMPERATURE,
                on_failure="worst_case",
            )
            failures["samples"] += sum(1 for s in samples if s == worst_case)
            variants["max"][name] = fuse_max(samples)
            variants["majority"][name] = fuse_majority(
                samples, min_vote_fraction=MAJORITY_MIN_VOTE_FRACTION
            )
    print(f"  estimator failures on {model}: point={failures['point']}, sample-sets={failures['samples']}")
    return variants


async def main():
    policy = LyraCompiler.from_yaml(CONFIG_PATH)
    n_expected = sum(len(expected) for _, _, expected in TEST_CASES)

    all_results: dict[str, dict[str, dict]] = {}
    for model in MODELS:
        print(f"\n=== {model} ===")
        variants = await get_variant_predictions(model, policy)
        all_results[model] = {}
        for variant_name, predictions in variants.items():
            result = score(predictions)
            all_results[model][variant_name] = result
            print(f"  -- {variant_name} --")
            print_details(result)
            print(
                f"  TP={result['tp']} FN(missed danger)={result['fn']} "
                f"FP(false alarm)={result['fp']}"
            )

    print("\n" + "=" * 70)
    print("SUMMARY  (recall = TP / expected, out of {})".format(n_expected))
    print("=" * 70)
    print(f"{'model':25s} {'point':>14s} {'self-cons(max)':>16s} {'self-cons(majority)':>20s}")
    for model in MODELS:
        r_point = all_results[model]["point"]
        r_max = all_results[model]["max"]
        r_maj = all_results[model]["majority"]
        print(
            f"{model:25s} "
            f"{r_point['tp']}/{n_expected} fa={r_point['fp']:>3d}   "
            f"{r_max['tp']}/{n_expected} fa={r_max['fp']:>3d}      "
            f"{r_maj['tp']}/{n_expected} fa={r_maj['fp']:>3d}"
        )

    print("\nPrior recorded baselines (estimator_calibration_study.py):")
    for model, (tp, fp) in PRIOR_POINT_RECALL.items():
        print(f"  {model:25s} point: {tp}/{n_expected} fa={fp}")
    tp, fp = PRIOR_ENSEMBLE_RECALL
    print(f"  {'4-model max-ensemble':25s}        {tp}/{n_expected} fa={fp}")

    print(
        f"\nCost note: self-consistency here uses N={N_SAMPLES} samples per "
        f"decision = {N_SAMPLES}x estimator calls vs. a single point estimate "
        f"(and 1.25x vs. the 4-model ensemble's 4x calls)."
    )


if __name__ == "__main__":
    asyncio.run(main())
