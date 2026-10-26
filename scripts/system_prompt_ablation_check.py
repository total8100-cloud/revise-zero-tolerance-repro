"""System-prompt ablation — the mechanism hypothesis proposed before the
Shapley work (docs/research/experiment-log/2026-09-23_occlusion-attribution
-check.md, 2026-09-23_exact-shapley-attribution.md): is the masking
over-prediction pattern (Section 7.3) a direct side effect of
`_SYSTEM_PROMPT`'s own "if unclear, predict the worst plausible value"
instruction -- added early in this project specifically to prevent
optimistic false negatives on ambiguous commands, but potentially a direct
contributor to THIS failure mode (spurious over-prediction on unrelated
variables masking a genuine miss)?

Three system-prompt variants, differing ONLY in the uncertainty-handling
clause, tested via `ModelStateEstimator(system_prompt=...)` (added this
session specifically for this check; default unchanged, see the
class docstring):

  A (worst-case) -- "if unclear, predict the worst plausible value"
  B (neutral)    -- no uncertainty guidance; just predict your best estimate
  C (opposite)   -- "if unclear, predict 0 (no change)"

(Which one is the actual current production default is checked at runtime
below, not assumed by this labeling — see the 2026-09-23 note.)

Measured on two things:
  1. The corpus's 7 base cases (pentest_test_cases.TEST_CASES) -- recall
     (of the positive/violation cases) and false-alarm rate (of the benign
     cases), to see whether the instruction trades recall for false alarms
     as the hypothesis predicts.
  2. The specific known-masked case (privesc_only_command_substitution) --
     does the SPURIOUS over-prediction on unrelated variables change under
     B/C relative to A, and does privilege_escalation_level itself change?

Requires Ollama serving the given model(s). Not a pytest test; no Docker
needed. Run manually:

    python scripts/system_prompt_ablation_check.py [model1] [model2] ...
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
from omni_lyra.right_brain.state_estimator import (
    ModelStateEstimator,
    StateEstimationError,
    WORST_CASE_SYSTEM_PROMPT,
    NEUTRAL_SYSTEM_PROMPT,
    _SYSTEM_PROMPT as _CURRENT_PRODUCTION_PROMPT,
)
from pentest_calibration_corpus import CALIBRATION_CORPUS
from pentest_test_cases import CONFIG_PATH, CURRENT_STATE, TEST_CASES

STATE_VARS = LyraCompiler.from_yaml(CONFIG_PATH).spec.state_vars
STATE_VAR_NAMES = [sv.name for sv in STATE_VARS]

# NOTE (2026-09-23, after this check's own finding was chased further):
# this small N=7 ablation's B_neutral_best_estimate result was NOT
# confirmed at full-corpus scale -- see
# scripts/prompt_and_normalization_validation.py and
# scripts/mitigation_decomposition_check.py, which found the neutral
# wording costs nemotron-mini real recall (95.7%->73.9%) with ZERO
# false-alarm benefit for either tested model. The production default
# (state_estimator._SYSTEM_PROMPT) therefore stayed WORST_CASE_SYSTEM_PROMPT
# -- this script no longer assumes which variant is "current"; it checks at
# runtime and prints whichever one actually matches, so it can't silently
# go stale again if the default changes a third time.

# Variants A and B are imported directly (not retyped) -- a hand-retyped
# copy of this exact wording silently diverged from the real one on this
# script's first attempt (missing/extra whitespace from the prompt's
# multi-line string literal), which would have compared two
# already-different prompts instead of isolating the one clause intended.
# Only C (no production constant exists for it) still needs the
# anchor-derived clause-substitution approach.
_ANCHOR = "- Every variable name given must appear as a key.\n"
assert _ANCHOR in WORST_CASE_SYSTEM_PROMPT
_UNCERTAINTY_CLAUSE_WORST_CASE = WORST_CASE_SYSTEM_PROMPT.split(_ANCHOR, 1)[1]
_UNCERTAINTY_CLAUSE_OPPOSITE = (
    "- If a command's effect on a variable is unclear, predict 0 (no change) for "
    "that variable rather than guessing a violation.\n"
)

PROMPTS = {
    "A_worst_case": WORST_CASE_SYSTEM_PROMPT,
    "B_neutral_best_estimate": NEUTRAL_SYSTEM_PROMPT,
    "C_opposite_assume_no_change": WORST_CASE_SYSTEM_PROMPT.replace(
        _UNCERTAINTY_CLAUSE_WORST_CASE, _UNCERTAINTY_CLAUSE_OPPOSITE
    ),
}

assert PROMPTS["A_worst_case"] == WORST_CASE_SYSTEM_PROMPT
assert PROMPTS["B_neutral_best_estimate"] == NEUTRAL_SYSTEM_PROMPT
assert PROMPTS["A_worst_case"] != PROMPTS["B_neutral_best_estimate"]
assert PROMPTS["C_opposite_assume_no_change"] != PROMPTS["A_worst_case"]

_production_label = next(
    (label for label, prompt in PROMPTS.items() if prompt == _CURRENT_PRODUCTION_PROMPT),
    "<none of A/B/C — production prompt has diverged from all three>",
)
print(f"NOTE: current production default (_SYSTEM_PROMPT) == variant {_production_label!r}\n")

TARGET_CASE_NAME = "privesc_only_command_substitution"
TARGET_CASE = next(c for c in CALIBRATION_CORPUS if c[0] == TARGET_CASE_NAME)


_POLICY = LyraCompiler.from_yaml(CONFIG_PATH)


async def score_corpus(estimator: ModelStateEstimator) -> dict:
    positives = [c for c in TEST_CASES if c[2]]
    negatives = [c for c in TEST_CASES if not c[2]]

    async def flagged(action: str) -> bool:
        try:
            predicted = await estimator.estimate(STATE_VARS, CURRENT_STATE, action)
        except StateEstimationError:
            return False
        return _POLICY.evaluate(predicted) > _POLICY.block_threshold

    pos_flagged = 0
    for _n, a, _e in positives:
        if await flagged(a):
            pos_flagged += 1
    neg_flagged = 0
    for _n, a, _e in negatives:
        if await flagged(a):
            neg_flagged += 1
    return {
        "recall": pos_flagged / len(positives) if positives else float("nan"),
        "false_alarm_rate": neg_flagged / len(negatives) if negatives else float("nan"),
        "n_positive": len(positives),
        "n_negative": len(negatives),
    }


async def run_model(model: str) -> None:
    print(f"=== model: {model} ===")
    for variant, prompt in PROMPTS.items():
        est = ModelStateEstimator(model=model, system_prompt=prompt)
        async with est:
            corpus_result = await score_corpus(est)
            _name, action, _expected = TARGET_CASE
            try:
                target_predicted = await est.estimate(STATE_VARS, CURRENT_STATE, action)
            except StateEstimationError as exc:
                target_predicted = {"ERROR": str(exc)}

        print(
            f"  [{variant}] recall={corpus_result['recall']:.1%} "
            f"({corpus_result['n_positive']} pos) "
            f"false_alarm_rate={corpus_result['false_alarm_rate']:.1%} "
            f"({corpus_result['n_negative']} neg)"
        )
        print(f"    target case ({TARGET_CASE_NAME}) predicted: {target_predicted}")
    print()


async def main():
    models = sys.argv[1:] if len(sys.argv) > 1 else ["phi3:mini", "nemotron-mini"]
    for model in models:
        await run_model(model)


if __name__ == "__main__":
    asyncio.run(main())
