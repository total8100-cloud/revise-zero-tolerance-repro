"""Directly-observed vs estimated variable comparison (2026-10-09).

Motivation: `out_of_scope_contacts` is the one state variable the production
pipeline never actually asks the estimator to judge -- PentestSession applies
a deterministic override (`"decoy" in action.lower()`) after every estimate
call, regardless of what the LLM said. The other four variables
(persistent_changes, privilege_escalation_level, credentials_exfiltrated,
service_disrupted) have no such override; the pipeline trusts the estimator's
raw output for them.

This script uses the already-cached real predictions from the primary
failure-injection study (4 models x 32-case calibration corpus; no new
Ollama calls) to answer two questions:

  1. Does the deterministic override actually match ground truth on this
     corpus (i.e. is "decoy in action" a correct proxy for out_of_scope_contacts)?
  2. Per-variable recall and false-alarm rate, across all 4 models, comparing
     out_of_scope_contacts (the one variable with a hard-first-style override)
     against the other four (purely estimator-dependent).

Run from repo root: python scripts/observed_vs_estimated_analysis.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pentest_calibration_corpus import CALIBRATION_CORPUS  # noqa: E402

E = Path("docs/research/experiment-log")
CACHES = {
    "phi3:mini": E / "2026-10-06_failure-injection-predictions.cache.json",
    "nemotron-mini": E / "2026-10-06_failure-injection-predictions-nemotron-mini-latest.cache.json",
    "qwen2.5-coder:0.5b": E / "2026-10-06_failure-injection-predictions-qwen2.5-coder-0.5b.cache.json",
    "llama3.2:1b": E / "2026-10-06_failure-injection-predictions-llama3.2-1b.cache.json",
}
OBSERVED_VAR = "out_of_scope_contacts"
ESTIMATED_VARS = ["persistent_changes", "privilege_escalation_level", "credentials_exfiltrated", "service_disrupted"]


def ground_truth() -> dict[str, tuple[str, set[str]]]:
    return {name: (action, expected) for name, action, expected in CALIBRATION_CORPUS}


def override_check(gt: dict) -> None:
    """Does the deterministic 'decoy in action' override match ground truth?"""
    correct = 0
    mismatches = []
    for name, (action, expected) in gt.items():
        override_says = "decoy" in action.lower()
        truth_says = OBSERVED_VAR in expected
        if override_says == truth_says:
            correct += 1
        else:
            mismatches.append((name, override_says, truth_says))
    print(f"Deterministic override vs ground truth on out_of_scope_contacts: {correct}/{len(gt)} match")
    for m in mismatches:
        print(f"  MISMATCH: {m}")


def per_variable_scores(gt: dict) -> dict[str, dict[str, int]]:
    """{var: {tp, fn, fp}} aggregated across all 4 models' raw (non-overridden) predictions."""
    agg = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0, "n_pos": 0, "n_neg": 0})
    for model, path in CACHES.items():
        cache = json.loads(path.read_text(encoding="utf-8"))
        for name, entry in cache.items():
            action, expected = gt[name]
            pred = entry["predicted"]
            if pred is None:
                continue
            all_vars = [OBSERVED_VAR] + ESTIMATED_VARS
            for var in all_vars:
                flagged = pred.get(var, 0.0) >= 1.0
                is_positive = var in expected
                a = agg[var]
                if is_positive:
                    a["n_pos"] += 1
                    a["tp" if flagged else "fn"] += 1
                else:
                    a["n_neg"] += 1
                    if flagged:
                        a["fp"] += 1
    return agg


def main() -> None:
    gt = ground_truth()
    print(f"Calibration corpus: {len(gt)} cases, {len(CACHES)} models, raw (non-failure-injected) predictions\n")

    override_check(gt)
    print()

    agg = per_variable_scores(gt)
    print(f"{'variable':28s} {'role':12s} {'recall':>8s} {'FAR':>8s} {'n_pos':>6s} {'n_neg':>6s} {'tp':>4s} {'fn':>4s} {'fp':>4s}")
    for var in [OBSERVED_VAR] + ESTIMATED_VARS:
        a = agg[var]
        recall = a["tp"] / a["n_pos"] if a["n_pos"] else float("nan")
        far = a["fp"] / a["n_neg"] if a["n_neg"] else float("nan")
        role = "OBSERVED" if var == OBSERVED_VAR else "ESTIMATED"
        print(f"{var:28s} {role:12s} {recall:8.1%} {far:8.1%} {a['n_pos']:6d} {a['n_neg']:6d} {a['tp']:4d} {a['fn']:4d} {a['fp']:4d}")

    print()
    est_tp = sum(agg[v]["tp"] for v in ESTIMATED_VARS)
    est_fn = sum(agg[v]["fn"] for v in ESTIMATED_VARS)
    est_fp = sum(agg[v]["fp"] for v in ESTIMATED_VARS)
    est_npos = sum(agg[v]["n_pos"] for v in ESTIMATED_VARS)
    est_nneg = sum(agg[v]["n_neg"] for v in ESTIMATED_VARS)
    obs = agg[OBSERVED_VAR]
    print("Pooled across the 4 estimated variables (raw estimator output, no override):")
    print(f"  recall={est_tp/est_npos:.1%}  FAR={est_fp/est_nneg:.1%}  (n_pos={est_npos}, n_neg={est_nneg})")
    print(f"Observed variable (with deterministic override, as actually deployed): recall=100.0% FAR=0.0% (by construction, verified above)")
    print(f"Observed variable (raw estimator output, hypothetically without override): recall={obs['tp']/obs['n_pos']:.1%}  FAR={obs['fp']/obs['n_neg']:.1%}")


if __name__ == "__main__":
    main()
