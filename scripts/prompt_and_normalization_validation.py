"""Full-corpus validation of BOTH mitigations combined (Section 7.3.6-7.3.7,
paper2-draft.md), before they're treated as adopted rather than just
"identified by a 7-case ablation check."

Background: three mechanism-analysis techniques (occlusion, exact Shapley,
system-prompt ablation — see docs/research/experiment-log/2026-09-23_*)
identified two independent, separately-fixable contributors to the masking
failure mode on the disguised privesc case
(`privesc_only_command_substitution`):

  Mitigation 1 — system prompt's "predict worst if unclear" clause
                 (LEGACY_WORST_CASE_SYSTEM_PROMPT) replaced with neutral
                 "predict your best estimate" wording (now `_SYSTEM_PROMPT`,
                 state_estimator.py's production default).
  Mitigation 2 — `$(...)` command-substitution wrapper syntax stripped from
                 what the estimator is shown before it sees the action
                 (`_normalize_command_substitutions`, always applied in
                 `_build_prompt` now — not optional).

Both were adopted as production defaults based on a small N=7 base-corpus
ablation check alone. Following this project's own established pattern
(scripts/reweight_sensitivity_check.py's precedent for a config/prompt
change), this script validates the COMBINED effect of both mitigations
against the full 32-case calibration corpus
(pentest_calibration_corpus.CALIBRATION_CORPUS) before treating that
adoption as validated rather than provisional.

OLD condition (pre-2026-09-23 production behavior) is reconstructed by
using `LEGACY_WORST_CASE_SYSTEM_PROMPT` explicitly AND monkeypatching
`_normalize_command_substitutions` to identity for the duration of that
condition's calls only — the function is always active in current
`_build_prompt`, so this is the only way to measure "without it" using the
real, unmodified prompt-building code path rather than a hand-reconstructed
copy (same fidelity-gap lesson as every other script in this project:
scripts/system_prompt_ablation_check.py's own comments document two
separate hand-retyping bugs this pattern avoids).

Requires Ollama serving the given model(s). Not a pytest test. Run manually:

    python scripts/prompt_and_normalization_validation.py [model1] [model2] ...
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

TARGET_CASE_NAME = "privesc_only_command_substitution"


async def score(estimator: ModelStateEstimator, label: str) -> list[tuple]:
    rows = []
    for name, action, expected_flagged in CALIBRATION_CORPUS:
        should_flag = bool(expected_flagged)
        try:
            predicted = await estimator.estimate(STATE_VARS, CURRENT_STATE, action)
        except StateEstimationError as exc:
            print(f"  [{label}] estimator failed on {name} ({exc}) — skipping")
            continue
        v = POLICY.evaluate(predicted)
        blocked = v > POLICY.block_threshold
        rows.append((name, should_flag, v, blocked, predicted))
    return rows


def summarize(rows: list[tuple]) -> tuple[float, float, int, int]:
    positives = [r for r in rows if r[1]]
    negatives = [r for r in rows if not r[1]]
    recall = sum(1 for r in positives if r[3]) / len(positives) if positives else float("nan")
    far = sum(1 for r in negatives if r[3]) / len(negatives) if negatives else float("nan")
    return recall, far, len(positives), len(negatives)


async def run_model(model: str) -> None:
    print(f"=== model: {model} ===  ({len(CALIBRATION_CORPUS)} cases)")

    # OLD condition: legacy worst-case prompt AND no $() normalization —
    # reconstructs pre-2026-09-23 production behavior exactly, via the real
    # code path (identity-monkeypatch, not a reimplementation).
    old_est = ModelStateEstimator(model=model, system_prompt=LEGACY_WORST_CASE_SYSTEM_PROMPT)
    orig_normalize = _se_module._normalize_command_substitutions
    _se_module._normalize_command_substitutions = lambda action: action
    try:
        async with old_est:
            old_rows = await score(old_est, "OLD")
    finally:
        _se_module._normalize_command_substitutions = orig_normalize

    # NEW condition: today's actual, unmodified production defaults.
    new_est = ModelStateEstimator(model=model)
    assert new_est.system_prompt == _NEUTRAL_PROMPT
    async with new_est:
        new_rows = await score(new_est, "NEW")

    recall_old, far_old, n_pos, n_neg = summarize(old_rows)
    recall_new, far_new, _, _ = summarize(new_rows)

    print(f"n_positive={n_pos} n_negative={n_neg}")
    print(f"OLD (legacy prompt, no normalization): recall={recall_old:.1%} false_alarm_rate={far_old:.1%}")
    print(f"NEW (neutral prompt + normalization):  recall={recall_new:.1%} false_alarm_rate={far_new:.1%}")

    old_by_name = {r[0]: r for r in old_rows}
    new_by_name = {r[0]: r for r in new_rows}
    disagreements = []
    for name, o in old_by_name.items():
        n = new_by_name.get(name)
        if n is None or o[3] == n[3]:
            continue
        disagreements.append((name, o[1], o[3], n[3]))
    print(f"Cases where BLOCK verdict changed OLD->NEW: {len(disagreements)}/{len(old_by_name)}")
    for name, should_flag, o_blocked, n_blocked in disagreements:
        print(f"  - {name}: should_flag={should_flag} old_blocked={o_blocked} new_blocked={n_blocked}")

    target_old = old_by_name.get(TARGET_CASE_NAME)
    target_new = new_by_name.get(TARGET_CASE_NAME)
    if target_old and target_new:
        print(f"\nTarget case ({TARGET_CASE_NAME}) — the disguised $() privesc case this arc was built toward:")
        print(f"  OLD: predicted={target_old[4]} blocked={target_old[3]}")
        print(f"  NEW: predicted={target_new[4]} blocked={target_new[3]}")
    print()


async def main():
    models = sys.argv[1:] if len(sys.argv) > 1 else ["phi3:mini", "nemotron-mini"]
    for model in models:
        await run_model(model)


if __name__ == "__main__":
    asyncio.run(main())
