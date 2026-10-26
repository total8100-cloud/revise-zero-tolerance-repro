"""Hybrid (Mondrian-style) split-conformal calibration study (Phase 5).

Phase 4 (scripts/scalar_conformal_calibration_study.py) found that a single
scalar margin over all of V(s) can be silently "gamed" by coincidence: if a
model's over-prediction on unrelated variables happens to outweigh its
under-prediction on a genuinely dangerous one in the same instance, the
one-sided residual clips to 0 and looks perfectly calibrated even though a
real safety-critical miss went uncorrected (nemotron-mini, docs/research/
pentest-roe-validation.md's Phase 4 section). A follow-up on llama3.2:1b
showed this isn't predictable from how "good" a model looks overall -- it's
a narrow, per-instance coincidence (does the false-positive weight sum
exceed the missed danger's own weight?) that has to be checked directly.

The fix tested here: split state vars into `CRITICAL_VARS` (the highest-
consequence categories -- privilege escalation, credential exfiltration)
and everything else. Critical vars get an ordinary per-variable margin
(omni_lyra.right_brain.conformal.calibrate/apply_margin) -- each one's
correction is computed and applied to its OWN value before V is summed, so
it can never be cancelled by noise on other variables. The remaining
("aggregate") vars share a single scalar margin computed on their own
restricted sub-energy (conformal.aggregate_scalar_nonconformity_scores),
keeping the more space-efficient scalar treatment for the lower-stakes
categories where unnecessary all-or-nothing collapse is a bigger practical
cost than a missed correction.

This does NOT eliminate collapse when a critical var genuinely needs
correction (its weight alone still exceeds block_threshold) -- that's the
intended, conservative behavior for the highest-consequence categories. What
it's meant to fix is Phase 4's false all-clear specifically.

Requires Ollama serving MODEL. Not a pytest test. Run manually:

    python scripts/hybrid_conformal_calibration_study.py [model_name] [alpha]

Writes {"critical_margins": {...}, "aggregate_scalar_margin": ..., "critical_vars": [...]}
to data/conformal_hybrid_margins_<model>.json for use with pentest_roe_scenario.py's
5th CLI arg (pass the JSON file path -- it auto-detects the hybrid shape).
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from conformal_calibration_study import collect_pairs
from pentest_calibration_corpus import CALIBRATION_CORPUS, CONFIG_PATH

from omni_lyra.left_brain.compiler import LyraCompiler
from omni_lyra.right_brain.conformal import (
    aggregate_scalar_nonconformity_scores,
    apply_margin,
    apply_scalar_margin,
    calibrate,
    calibrate_scalar,
    min_calibration_size,
    nonconformity_scores,
    restricted_energy,
)

# The two highest-weight/highest-consequence categories in config/pentest_roe.yaml
# (weights 15 and 12, vs. 10/8/5 for the rest) -- see the module docstring.
CRITICAL_VARS = {"privilege_escalation_level", "credentials_exfiltrated"}

N_SPLITS = 50
CALIB_FRACTION = 0.6


def covered_hybrid(
    predicted: dict[str, float],
    actual: dict[str, float],
    critical_margins: dict[str, float],
    aggregate_margin: float,
    policy,
) -> bool:
    """Does the combined hybrid guarantee hold for this single case: after
    applying the critical per-var margins AND the aggregate scalar margin,
    is V(actual) <= V(adjusted_predicted)?"""
    adjusted = apply_margin(predicted, critical_margins)
    v_pred_adjusted = apply_scalar_margin(policy.evaluate(adjusted), aggregate_margin)
    return policy.evaluate(actual) <= v_pred_adjusted


def empirical_coverage_over_splits(
    pairs: list[tuple[dict[str, float], dict[str, float], str]],
    alpha: float,
    n_splits: int,
    policy,
) -> float:
    n = len(pairs)
    calib_size = max(min_calibration_size(alpha), round(n * CALIB_FRACTION))
    hits = 0
    total = 0

    rng = random.Random(0)
    for _ in range(n_splits):
        indices = list(range(n))
        rng.shuffle(indices)
        calib_idx, test_idx = indices[:calib_size], indices[calib_size:]
        if not test_idx:
            continue

        calib_pairs = [(pairs[i][0], pairs[i][1]) for i in calib_idx]
        all_var_margins = calibrate(nonconformity_scores(calib_pairs), alpha=alpha)
        critical_margins = {v: m for v, m in all_var_margins.items() if v in CRITICAL_VARS}
        aggregate_margin = calibrate_scalar(
            aggregate_scalar_nonconformity_scores(calib_pairs, policy, CRITICAL_VARS), alpha=alpha
        )

        for i in test_idx:
            predicted, actual, _name = pairs[i]
            total += 1
            if covered_hybrid(predicted, actual, critical_margins, aggregate_margin, policy):
                hits += 1

    return hits / total if total else float("nan")


async def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5-coder:0.5b"
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1

    print(f"Model: {model}  alpha: {alpha}  target coverage: {(1 - alpha) * 100:.0f}%")
    print(f"Corpus size: {len(CALIBRATION_CORPUS)}  min required for alpha={alpha}: {min_calibration_size(alpha)}")
    print(f"Critical vars (per-var, isolated): {sorted(CRITICAL_VARS)}")

    policy = LyraCompiler.from_yaml(CONFIG_PATH)
    pairs = await collect_pairs(model, policy)
    full_pairs = [(p, a) for p, a, _ in pairs]

    all_var_margins = calibrate(nonconformity_scores(full_pairs), alpha=alpha)
    critical_margins = {v: m for v, m in all_var_margins.items() if v in CRITICAL_VARS}
    aggregate_margin = calibrate_scalar(
        aggregate_scalar_nonconformity_scores(full_pairs, policy, CRITICAL_VARS), alpha=alpha
    )

    print("\n=== Critical per-var margins (isolated, cannot be cancelled by other vars) ===")
    for var, margin in sorted(critical_margins.items()):
        flag = " -- ALONE exceeds block_threshold, will collapse everything" if margin > 0 else ""
        print(f"  {var:30s} margin={margin:.3f}{flag}")

    non_critical_vars = sorted(set(pairs[0][0].keys()) - CRITICAL_VARS)
    print(f"\n=== Aggregate scalar margin (non-critical vars: {non_critical_vars}) ===")
    print(f"  margin={aggregate_margin:.4f}  (block_threshold={policy.block_threshold})")
    if any(m > 0 for m in critical_margins.values()):
        print(
            "  NOTE: a critical var already needs correction above, so the overall "
            "verdict will collapse to BLOCK regardless of this aggregate margin's size "
            "-- that's the intended conservative behavior for these categories."
        )
    elif aggregate_margin > policy.block_threshold:
        print("  Aggregate margin alone exceeds block_threshold -- non-critical vars would still collapse everything.")
    else:
        print("  Neither critical nor aggregate margins force a collapse -- candidates remain discriminable.")

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    safe_model = model.replace(":", "_").replace("/", "_")
    out_path = out_dir / f"conformal_hybrid_margins_{safe_model}.json"
    out_path.write_text(
        json.dumps(
            {
                "critical_margins": critical_margins,
                "aggregate_scalar_margin": aggregate_margin,
                "critical_vars": sorted(CRITICAL_VARS),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  (written to {out_path}; pass this JSON path directly to pentest_roe_scenario.py's 5th arg)")

    print(f"\n=== Empirical coverage over {N_SPLITS} random calib/test splits (combined hybrid guarantee) ===")
    coverage = empirical_coverage_over_splits(pairs, alpha=alpha, n_splits=N_SPLITS, policy=policy)
    nominal = 1 - alpha
    flag = "OK" if coverage >= nominal - 0.05 else "BELOW NOMINAL"
    print(f"  empirical={coverage * 100:5.1f}%  nominal={nominal * 100:.0f}%  [{flag}]")


if __name__ == "__main__":
    asyncio.run(main())
