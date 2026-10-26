"""Jev/jeff calibration benchmark — PRD Item C
(docs/research/jev-state-estimator-prd.md).

The real decision gate for the whole Jev PRD: does `JevStateEstimator`
(GliFormer-400M-backed, self-hosted jeff server) discriminate ROE
violations at least as well as the existing best Ollama-backed model
(phi3:mini, 87.0%/91.3% recall pre/post-normalization, 0.0% false-alarm
rate on this same 32-case corpus)? If not, per the PRD's own prioritized
sequencing, this whole direction should be shelved without proceeding to
Item D (masking diagnostic) or Item E (probability-aware aggregation).

Requires a live jeff server (JEFF_API_KEYS=devkey uv run jeff, from
vendor/jeff, serving on localhost:8000). Not a pytest test. Run manually:

    python scripts/jev_calibration_benchmark.py
"""

from __future__ import annotations

import asyncio
import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from omni_lyra.left_brain.compiler import LyraCompiler
from omni_lyra.right_brain.jev_state_estimator import JevStateEstimator
from omni_lyra.right_brain.state_estimator import StateEstimationError
from pentest_calibration_corpus import CALIBRATION_CORPUS
from pentest_test_cases import CONFIG_PATH, CURRENT_STATE

POLICY = LyraCompiler.from_yaml(CONFIG_PATH)
STATE_VARS = POLICY.spec.state_vars


async def main() -> None:
    print(f"model=jev(gliformer-large-v1)  cases={len(CALIBRATION_CORPUS)}")
    print(f"block_threshold={POLICY.block_threshold}\n")

    rows = []
    async with JevStateEstimator() as estimator:
        for name, action, expected_flagged in CALIBRATION_CORPUS:
            should_flag = bool(expected_flagged)
            try:
                predicted = await estimator.estimate(STATE_VARS, CURRENT_STATE, action)
            except StateEstimationError as exc:
                print(f"  [{name}] estimator failed ({exc}) — skipping")
                continue
            v = POLICY.evaluate(predicted)
            blocked = v > POLICY.block_threshold
            flag_mark = "!" if should_flag != blocked else " "
            print(f"  {flag_mark} [{name}] should_flag={should_flag} V={v:.3f} blocked={blocked}")
            rows.append((name, should_flag, v, blocked))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    positives = [r for r in rows if r[1]]
    negatives = [r for r in rows if not r[1]]
    recall = sum(1 for r in positives if r[3]) / len(positives) if positives else float("nan")
    far = sum(1 for r in negatives if r[3]) / len(negatives) if negatives else float("nan")
    print(f"n_positive={len(positives)} n_negative={len(negatives)}")
    print(f"jeff (gliformer-large-v1): recall={recall:.1%} false_alarm_rate={far:.1%}")
    print("Reference (phi3:mini, same corpus, post-normalization, 2026-09-23): recall=91.3% false_alarm_rate=0.0%")
    print("Reference (phi3:mini, same corpus, pre-normalization, 2026-09-23):  recall=87.0% false_alarm_rate=0.0%")

    decision = "GO — proceed to Item D" if recall >= 0.87 and far <= 0.10 else "NO-GO — shelve per PRD's own decision rule"
    print(f"\nPRD Item C decision gate: {decision}")


if __name__ == "__main__":
    asyncio.run(main())
