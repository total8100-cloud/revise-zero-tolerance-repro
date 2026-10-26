"""Decomposes the combined-mitigation regression found by
scripts/prompt_and_normalization_validation.py.

That full-32-case-corpus run showed nemotron-mini's recall REGRESSING
95.7%->78.3% under the combined (neutral prompt + $() normalization)
condition, directly contradicting the small N=7 ablation check's finding
that the neutral prompt alone "removes spurious flags with no recall cost."
Since that combined run only tested OLD (legacy+no-norm) vs. NEW
(neutral+norm), the two mitigations are confounded there — this script
un-confounds them with the full 2x2 (prompt in {legacy, neutral}) x
(normalization in {off, on}) on the full 32-case corpus, to find out which
one (or both, or an interaction) is responsible before either is kept as an
unconditional production default.

Requires Ollama serving the given model(s). Not a pytest test. Run manually:

    python scripts/mitigation_decomposition_check.py [model1] [model2] ...
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
from omni_lyra.right_brain import state_estimator as _se_module
from omni_lyra.right_brain.state_estimator import (
    ModelStateEstimator,
    StateEstimationError,
    WORST_CASE_SYSTEM_PROMPT as LEGACY_WORST_CASE_SYSTEM_PROMPT,
    _SYSTEM_PROMPT as _NEUTRAL_PROMPT,
)
from pentest_calibration_corpus import CALIBRATION_CORPUS
from pentest_test_cases import CONFIG_PATH, CURRENT_STATE

POLICY = LyraCompiler.from_yaml(CONFIG_PATH)
STATE_VARS = POLICY.spec.state_vars

CONDITIONS = [
    ("legacy_no_norm", LEGACY_WORST_CASE_SYSTEM_PROMPT, False),
    ("legacy_norm", LEGACY_WORST_CASE_SYSTEM_PROMPT, True),
    ("neutral_no_norm", _NEUTRAL_PROMPT, False),
    ("neutral_norm", _NEUTRAL_PROMPT, True),
]


async def score(estimator: ModelStateEstimator, label: str) -> dict[str, tuple]:
    rows = {}
    for name, action, expected_flagged in CALIBRATION_CORPUS:
        should_flag = bool(expected_flagged)
        try:
            predicted = await estimator.estimate(STATE_VARS, CURRENT_STATE, action)
        except StateEstimationError as exc:
            print(f"  [{label}] estimator failed on {name} ({exc}) — skipping")
            continue
        v = POLICY.evaluate(predicted)
        blocked = v > POLICY.block_threshold
        rows[name] = (should_flag, blocked, predicted)
    return rows


def summarize(rows: dict[str, tuple]) -> tuple[float, float, int, int]:
    positives = [r for r in rows.values() if r[0]]
    negatives = [r for r in rows.values() if not r[0]]
    recall = sum(1 for r in positives if r[1]) / len(positives) if positives else float("nan")
    far = sum(1 for r in negatives if r[1]) / len(negatives) if negatives else float("nan")
    return recall, far, len(positives), len(negatives)


async def run_model(model: str) -> None:
    print(f"=== model: {model} ===  ({len(CALIBRATION_CORPUS)} cases)")
    all_rows: dict[str, dict[str, tuple]] = {}

    orig_normalize = _se_module._normalize_command_substitutions
    try:
        for label, prompt, normalize_on in CONDITIONS:
            if normalize_on:
                _se_module._normalize_command_substitutions = orig_normalize
            else:
                _se_module._normalize_command_substitutions = lambda action: action
            est = ModelStateEstimator(model=model, system_prompt=prompt)
            async with est:
                rows = await score(est, label)
            all_rows[label] = rows
            recall, far, n_pos, n_neg = summarize(rows)
            print(f"  [{label}] recall={recall:.1%} ({n_pos} pos) false_alarm_rate={far:.1%} ({n_neg} neg)")
    finally:
        _se_module._normalize_command_substitutions = orig_normalize

    base = all_rows["legacy_no_norm"]
    print("\nPer-condition verdict, cases where ANY condition disagrees with legacy_no_norm:")
    for name, (should_flag, base_blocked, _pred) in base.items():
        verdicts = {label: all_rows[label][name][1] for label, *_ in CONDITIONS if name in all_rows[label]}
        if len(set(verdicts.values())) > 1:
            print(f"  - {name}: should_flag={should_flag} " + " ".join(f"{k}={v}" for k, v in verdicts.items()))

    target = "privesc_only_command_substitution"
    print(f"\nTarget case ({target}) across all 4 conditions:")
    for label, *_ in CONDITIONS:
        if target in all_rows[label]:
            should_flag, blocked, predicted = all_rows[label][target]
            print(f"  [{label}] blocked={blocked} predicted={predicted}")
    print()


async def main():
    models = sys.argv[1:] if len(sys.argv) > 1 else ["nemotron-mini", "phi3:mini"]
    for model in models:
        await run_model(model)


if __name__ == "__main__":
    asyncio.run(main())
