"""Captures full per-variable predictions (not just V/blocked) for the 3
scoreable frontier models across the entire 32-case calibration corpus --
extends Section 9.2's aggregate recall/FAR numbers with the raw data needed
to redo the unified baseline comparison (Section 10.4) and the hard-first
observability check (Section 10.3) across the same broader estimator
spectrum already used for real Jev and the 4 local models.

claude-opus-5.5 excluded (refuses every call; already characterized
separately, Section 9.3).

Run (requires Openrouter_API_key.env):
    python scripts/frontier_full_predictions.py
"""

from __future__ import annotations

import asyncio
import json
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


async def run_model(api_key: str, model: str) -> dict:
    print(f"=== {model} ===")
    out = {}
    async with OpenRouterStateEstimator(api_key=api_key, model=model) as estimator:
        for name, action, expected in CALIBRATION_CORPUS:
            try:
                predicted = await estimator.estimate(STATE_VARS, CURRENT_STATE, action)
            except StateEstimationError as exc:
                print(f"  ! {name}: {exc}")
                out[name] = {"predicted": None, "expected": sorted(expected)}
                continue
            out[name] = {"predicted": predicted, "expected": sorted(expected)}
    return out


async def main() -> None:
    api_key = Path("Openrouter_API_key.env").read_text(encoding="utf-8").strip()
    for model in MODELS:
        data = await run_model(api_key, model)
        safe_name = model.replace("/", "-")
        out_path = Path(f"docs/research/experiment-log/2026-10-10_frontier-predictions-{safe_name}.cache.json")
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"  wrote {out_path}\n")


if __name__ == "__main__":
    asyncio.run(main())
