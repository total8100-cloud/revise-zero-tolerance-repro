"""Extends the cross-estimator disguised-evasion comparison
(2026-10-10_cross-estimator-evasion-comparison.md) to the 3 scoreable
frontier models (claude-opus-5.5 refuses outright, already characterized
separately -- excluded here). Answers a direct follow-up question: is
real Jev's unique success on this case because it is a genuinely
calibrated DECISION model, or would any sufficiently capable frontier
model -- even prompted the same point-estimate way as the local models,
not Jev's probability-eliciting format -- also see through it?

Single case, single live call per model (cheap, <$0.05 total). Run:
    python scripts/frontier_evasion_check.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from omni_lyra.left_brain.compiler import LyraCompiler  # noqa: E402
from omni_lyra.right_brain.openrouter_state_estimator import OpenRouterStateEstimator  # noqa: E402
from omni_lyra.right_brain.state_estimator import StateEstimationError  # noqa: E402
from pentest_calibration_corpus import CALIBRATION_CORPUS  # noqa: E402
from pentest_test_cases import CONFIG_PATH, CURRENT_STATE  # noqa: E402

POLICY = LyraCompiler.from_yaml(CONFIG_PATH)
STATE_VARS = POLICY.spec.state_vars

MODELS = ["openai/gpt-5.6-terra", "qwen/qwen3.8-max-prime", "deepseek/deepseek-v4-pro"]

CASE_NAME = "privesc_only_command_substitution"
CASE = next(c for c in CALIBRATION_CORPUS if c[0] == CASE_NAME)


async def main() -> None:
    api_key = Path("Openrouter_API_key.env").read_text(encoding="utf-8").strip()
    _name, action, _expected = CASE
    for model in MODELS:
        async with OpenRouterStateEstimator(api_key=api_key, model=model) as estimator:
            try:
                predicted = await estimator.estimate(STATE_VARS, CURRENT_STATE, action)
            except StateEstimationError as exc:
                print(f"{model:25s} ERROR: {exc}")
                continue
            v = POLICY.evaluate(predicted)
            blocked = v > POLICY.block_threshold
            pel = predicted.get("privilege_escalation_level")
            is_max = pel == max(predicted.values())
            print(f"{model:25s} predicted={predicted}  V={v:.3f}  blocked={blocked}  "
                  f"privesc_is_own_max={is_max}")


if __name__ == "__main__":
    asyncio.run(main())
