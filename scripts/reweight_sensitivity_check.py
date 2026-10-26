"""Reweighting sensitivity check — PRD Item A
(docs/research/paper2-security-strengthening-prd.md).

The PRD's Item A deliverable calls for re-running experiments to confirm
the CVSS-grounded reweighting (config/pentest_roe_cvss_weights.yaml) doesn't
change the paper's qualitative conclusions, without silently overwriting the
already-validated config/pentest_roe.yaml (which would desync every V value
already reported in docs/research/experiment-log/ and paper2-draft.md).

Full live re-validation of Sections 5-8 (multi-container Docker ranges) is
out of scope for one autonomous pass — this instead answers the sharper,
cheaper question directly: for the same 32-case calibration corpus already
used in Section 6, does classifying each case as BLOCK/not-BLOCK (using a
real point-estimate state prediction from one model) change between the old
ad hoc weights and the new CVSS-grounded ones? If recall/false-alarm rate
are the same under both configs, the qualitative "zero-tolerance collapse"
structure is confirmed weight-invariant, as the PRD predicted (minimum
weight, in either scheme, still dwarfs block_threshold=1.0).

Requires Ollama serving the given model. Not a pytest test. Run manually:

    python scripts/reweight_sensitivity_check.py [model_name]
"""

from __future__ import annotations

import asyncio
import sys

from omni_lyra.left_brain.compiler import LyraCompiler
from omni_lyra.right_brain.state_estimator import ModelStateEstimator, StateEstimationError
from pentest_calibration_corpus import CALIBRATION_CORPUS, CURRENT_STATE

OLD_CONFIG = "config/pentest_roe.yaml"
NEW_CONFIG = "config/pentest_roe_cvss_weights.yaml"


async def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "phi3:mini"

    old_policy = LyraCompiler.from_yaml(OLD_CONFIG)
    new_policy = LyraCompiler.from_yaml(NEW_CONFIG)

    print(f"model={model}  cases={len(CALIBRATION_CORPUS)}")
    print(f"old block_threshold={old_policy.block_threshold}  new block_threshold={new_policy.block_threshold}\n")

    async with ModelStateEstimator(model=model) as estimator:
        rows = []
        for name, action, expected_flagged in CALIBRATION_CORPUS:
            should_flag = bool(expected_flagged)
            try:
                predicted = await estimator.estimate(
                    old_policy.spec.state_vars, CURRENT_STATE, action
                )
            except StateEstimationError as exc:
                print(f"  [{name}] estimator failed ({exc}) — skipping")
                continue

            v_old = old_policy.evaluate(predicted)
            v_new = new_policy.evaluate(predicted)
            blocked_old = v_old > old_policy.block_threshold
            blocked_new = v_new > new_policy.block_threshold
            agree = blocked_old == blocked_new
            rows.append((name, should_flag, v_old, blocked_old, v_new, blocked_new, agree))
            flag_mark = "!" if not agree else " "
            print(
                f"  {flag_mark} [{name}] should_flag={should_flag} "
                f"V_old={v_old:.2f}(block={blocked_old}) V_new={v_new:.2f}(block={blocked_new})"
            )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    def recall_and_far(blocked_key):
        positives = [r for r in rows if r[1]]  # should_flag True
        negatives = [r for r in rows if not r[1]]
        recall = sum(1 for r in positives if r[blocked_key]) / len(positives) if positives else float("nan")
        far = sum(1 for r in negatives if r[blocked_key]) / len(negatives) if negatives else float("nan")
        return recall, far, len(positives), len(negatives)

    recall_old, far_old, n_pos, n_neg = recall_and_far(3)
    recall_new, far_new, _, _ = recall_and_far(5)
    disagreements = [r for r in rows if not r[6]]

    print(f"n_positive_cases={n_pos} n_negative_cases={n_neg}")
    print(f"OLD weights: recall={recall_old:.1%} false_alarm_rate={far_old:.1%}")
    print(f"NEW weights: recall={recall_new:.1%} false_alarm_rate={far_new:.1%}")
    print(f"Cases where BLOCK verdict differs between old/new weights: {len(disagreements)}/{len(rows)}")
    for r in disagreements:
        print(f"  - {r[0]}: old_blocked={r[3]} new_blocked={r[5]}")


if __name__ == "__main__":
    asyncio.run(main())
