"""Split-conformal calibration study (Phase 4): does conformalizing the
scalar V(s) directly -- instead of each binary state variable independently
(Phase 3, scripts/conformal_calibration_study.py) -- avoid the all-or-
nothing collapse documented in docs/research/pentest-roe-validation.md?

Per-variable calibration was found to collapse to universal BLOCK in this
zero-tolerance domain: block_threshold=1.0 while the smallest constraint
weight is 5.0, so ANY single variable needing margin=1.0 alone produces
V >= 5.0 > block_threshold for every candidate, regardless of how
well-calibrated the other variables are -- and this held for both
qwen2.5-coder:0.5b (4/5 variables miscalibrated) and nemotron-mini (only
1/5), i.e. it wasn't proportional to estimator quality.

This script tests whether replacing "add margin_i to state var i, for all
i, then evaluate V" with "evaluate V, then add one q_hat" changes that
outcome. Reported honestly either way -- see conformal.py's module-level
note on scalar_nonconformity_scores: a scalar margin is still a single
global additive constant applied to every candidate, which is the same
structural property that caused the per-variable collapse, so there's no
a priori reason to expect this to behave differently.

Requires Ollama serving MODEL. Not a pytest test. Run manually:

    python scripts/scalar_conformal_calibration_study.py [model_name] [alpha]

Writes the full-corpus scalar margin to
data/conformal_scalar_margin_<model>.json for use with
scripts/pentest_roe_scenario.py's 5th CLI arg (pass the printed float value
directly, not the JSON file).
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
from omni_lyra.right_brain.conformal import calibrate_scalar, min_calibration_size, scalar_nonconformity_scores

N_SPLITS = 50
CALIB_FRACTION = 0.6


def covered_scalar(predicted: dict[str, float], actual: dict[str, float], margin: float, policy) -> bool:
    """Does the scalar guarantee actually hold: V(actual) <= V(predicted) + margin?"""
    return policy.evaluate(actual) <= policy.evaluate(predicted) + margin


def empirical_coverage_over_splits(
    pairs: list[tuple[dict[str, float], dict[str, float], str]],
    alpha: float,
    n_splits: int,
    policy,
) -> float:
    """Same repeated-random-split methodology as the per-variable study,
    reusing the same already-collected predictions (no extra Ollama calls)."""
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
        margin = calibrate_scalar(scalar_nonconformity_scores(calib_pairs, policy), alpha=alpha)

        for i in test_idx:
            predicted, actual, _name = pairs[i]
            total += 1
            if covered_scalar(predicted, actual, margin, policy):
                hits += 1

    return hits / total if total else float("nan")


async def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5-coder:0.5b"
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1

    print(f"Model: {model}  alpha: {alpha}  target coverage: {(1 - alpha) * 100:.0f}%")
    print(f"Corpus size: {len(CALIBRATION_CORPUS)}  min required for alpha={alpha}: {min_calibration_size(alpha)}")

    policy = LyraCompiler.from_yaml(CONFIG_PATH)
    pairs = await collect_pairs(model, policy)

    full_margin = calibrate_scalar(
        scalar_nonconformity_scores([(p, a) for p, a, _ in pairs], policy), alpha=alpha
    )
    print(f"\n=== Full-corpus scalar q_hat: margin={full_margin:.4f} (block_threshold={policy.block_threshold}) ===")
    if full_margin > policy.block_threshold:
        print(
            f"  COLLAPSE: margin ({full_margin:.4f}) alone exceeds block_threshold "
            f"({policy.block_threshold}) -- every candidate's V will be pushed past "
            f"BLOCK regardless of its own predicted value. Same failure mode as "
            f"per-variable conformal, just computed on the scalar directly."
        )
    else:
        print(
            f"  NOT collapsed at this alpha: margin stays within block_threshold, "
            f"so candidates can still be discriminated after the margin is applied."
        )

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    safe_model = model.replace(":", "_").replace("/", "_")
    out_path = out_dir / f"conformal_scalar_margin_{safe_model}.json"
    out_path.write_text(json.dumps({"margin": full_margin, "alpha": alpha}, indent=2), encoding="utf-8")
    print(f"  (written to {out_path}; pass the margin value {full_margin:.4f} directly as pentest_roe_scenario.py's 5th arg)")

    print(f"\n=== Empirical coverage over {N_SPLITS} random calib/test splits ===")
    coverage = empirical_coverage_over_splits(pairs, alpha=alpha, n_splits=N_SPLITS, policy=policy)
    nominal = 1 - alpha
    flag = "OK" if coverage >= nominal - 0.05 else "BELOW NOMINAL"
    print(f"  empirical={coverage * 100:5.1f}%  nominal={nominal * 100:.0f}%  [{flag}]")


if __name__ == "__main__":
    asyncio.run(main())
