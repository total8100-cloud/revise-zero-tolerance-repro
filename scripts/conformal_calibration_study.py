"""Split-conformal calibration study (Phase 3): does a one-sided, offline
per-variable margin recover the recall Phase 2's self-consistency needed
N=5x runtime estimator calls to (unstably) approach — while costing zero
extra estimator calls at decision time?

Two things this script reports, both real / not fabricated:

1. Full-corpus q_hat per state var for the target model (qwen2.5-coder:0.5b
   by default — the model with the worst point-estimate recall and the most
   unstable self-consistency results in Phase 2), for use in the live
   re-validation (pentest_roe_scenario.py).
2. Empirical coverage under repeated random calibration/test splits: does
   the nominal (1 - alpha) coverage target actually hold on held-out cases,
   or does the hand-authored/non-i.i.d. corpus break the exchangeability
   assumption split-conformal relies on? Reported honestly either way — see
   conformal.py's module docstring for why this isn't assumed for free.

Not a pytest test. Requires Ollama serving MODEL. Run manually:

    python scripts/conformal_calibration_study.py [model_name] [alpha]
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path

# Same fix as loop.py/pentest_session.py: rich/plain stdout on Windows can
# crash on the first non-cp949 character (e.g. an em dash) printed to the
# console — force UTF-8 rather than avoid such characters everywhere.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from omni_lyra.left_brain.compiler import LyraCompiler
from omni_lyra.right_brain.conformal import apply_margin, calibrate, min_calibration_size, nonconformity_scores
from omni_lyra.right_brain.state_estimator import ModelStateEstimator, StateEstimationError
from pentest_calibration_corpus import CALIBRATION_CORPUS, CONFIG_PATH, CURRENT_STATE

N_SPLITS = 50
CALIB_FRACTION = 0.6


def _actual_state(expected_flagged: set[str]) -> dict[str, float]:
    return {var: (1.0 if var in expected_flagged else 0.0) for var in CURRENT_STATE}


async def collect_pairs(model: str, policy) -> list[tuple[dict[str, float], dict[str, float], str]]:
    """Returns (predicted_state, actual_state, case_name) for every corpus case."""
    pairs = []
    async with ModelStateEstimator(model=model) as est:
        for name, action, expected in CALIBRATION_CORPUS:
            try:
                predicted = await est.estimate(
                    policy.spec.state_vars, CURRENT_STATE, action, temperature=0.0
                )
            except StateEstimationError:
                predicted = dict(CURRENT_STATE)
            pairs.append((predicted, _actual_state(expected), name))
    return pairs


def covered(predicted: dict[str, float], actual: dict[str, float], margins: dict[str, float]) -> dict[str, bool]:
    """Per-var: does actual_i <= predicted_i + margin_i (the guarantee this
    margin is supposed to provide)?"""
    adjusted = apply_margin(predicted, margins)
    return {var: actual.get(var, 0.0) <= adjusted.get(var, 0.0) for var in actual}


def empirical_coverage_over_splits(
    pairs: list[tuple[dict[str, float], dict[str, float], str]],
    alpha: float,
    n_splits: int,
) -> dict[str, float]:
    """Repeated random calib/test splits (reusing the SAME already-collected
    predictions — no extra Ollama calls) so a single expensive collection
    pass supports many splits. Returns, per var, the average fraction of
    held-out test cases where the coverage check holds."""
    n = len(pairs)
    calib_size = max(min_calibration_size(alpha), round(n * CALIB_FRACTION))
    all_vars = list(CURRENT_STATE.keys())
    hit_counts = {var: 0 for var in all_vars}
    total_counts = {var: 0 for var in all_vars}

    rng = random.Random(0)
    for split_i in range(n_splits):
        indices = list(range(n))
        rng.shuffle(indices)
        calib_idx, test_idx = indices[:calib_size], indices[calib_size:]
        if not test_idx:
            continue

        calib_pairs = [(pairs[i][0], pairs[i][1]) for i in calib_idx]
        margins = calibrate(nonconformity_scores(calib_pairs), alpha=alpha)

        for i in test_idx:
            predicted, actual, _name = pairs[i]
            result = covered(predicted, actual, margins)
            for var, ok in result.items():
                total_counts[var] += 1
                if ok:
                    hit_counts[var] += 1

    return {
        var: (hit_counts[var] / total_counts[var] if total_counts[var] else float("nan"))
        for var in all_vars
    }


async def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5-coder:0.5b"
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1

    print(f"Model: {model}  alpha: {alpha}  target coverage: {(1 - alpha) * 100:.0f}%")
    print(f"Corpus size: {len(CALIBRATION_CORPUS)}  min required for alpha={alpha}: {min_calibration_size(alpha)}")

    policy = LyraCompiler.from_yaml(CONFIG_PATH)
    pairs = await collect_pairs(model, policy)

    print("\n=== Full-corpus q_hat per state var (for live re-validation) ===")
    full_margins = calibrate(nonconformity_scores([(p, a) for p, a, _ in pairs]), alpha=alpha)
    for var, margin in sorted(full_margins.items()):
        print(f"  {var:30s} margin={margin:.3f}")

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    safe_model = model.replace(":", "_").replace("/", "_")
    out_path = out_dir / f"conformal_margins_{safe_model}.json"
    out_path.write_text(json.dumps(full_margins, indent=2), encoding="utf-8")
    print(f"  (written to {out_path} for use with pentest_roe_scenario.py)")

    print(f"\n=== Empirical coverage over {N_SPLITS} random calib/test splits ===")
    coverage = empirical_coverage_over_splits(pairs, alpha=alpha, n_splits=N_SPLITS)
    nominal = 1 - alpha
    for var, cov in sorted(coverage.items()):
        flag = "OK" if cov >= nominal - 0.05 else "BELOW NOMINAL"
        print(f"  {var:30s} empirical={cov * 100:5.1f}%  nominal={nominal * 100:.0f}%  [{flag}]")

    print(
        "\nInterpretation: 'OK' means held-out coverage roughly tracks the "
        "target (1-alpha) despite the corpus being hand-authored, not "
        "i.i.d.-sampled — evidence (not proof) the exchangeability "
        "assumption is a reasonable working approximation here. "
        "'BELOW NOMINAL' would mean the guarantee doesn't actually hold for "
        "this corpus and should be reported as a limitation, not hidden."
    )


if __name__ == "__main__":
    asyncio.run(main())
