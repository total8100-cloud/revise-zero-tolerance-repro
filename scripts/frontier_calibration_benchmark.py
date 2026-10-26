"""Frontier-model calibration benchmark — cross-vendor reference comparison.

Not part of any PRD item; grew directly out of the Jev/jeff NO-GO result
(docs/research/experiment-log/2026-09-30_jev-calibration-benchmark.md),
whose root-cause analysis raised an open question: is the masking/
miscalibration failure characterized throughout paper2-draft.md Sections
6-7 a property of small local models (0.5B-3.8B Ollama models, plus
jeff's 400M GliFormer), or does it persist at genuine frontier scale? This
script answers that using the SAME cross-model-replication methodology
already used throughout this project (4 Ollama models for the original
calibration benchmark; 3 models for the adversarial-injection replication)
— applied one tier up, across 4 different labs' current flagship models,
via OpenRouter (one wire format, `model` string is the only thing that
changes).

NOT a candidate deployment configuration — every model here is called via
an external hosted API, which this project's own CLAUDE.md explicitly
rules out for the actual runtime pipeline ("zero external API calls at
runtime"). This is a research-only, offline reference comparison,
identical in spirit to how the Jev/jeff evaluation and the CVSS-reweight
sensitivity check are one-off characterization scripts, not runtime code.

Requires an OpenRouter API key (Openrouter_API_key.env at repo root, one
raw key, no KEY=VALUE wrapper). Not a pytest test. Run manually:

    python scripts/frontier_calibration_benchmark.py
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

from omni_lyra.left_brain.compiler import LyraCompiler
from omni_lyra.right_brain.openrouter_state_estimator import OpenRouterStateEstimator
from omni_lyra.right_brain.state_estimator import StateEstimationError
from pentest_calibration_corpus import CALIBRATION_CORPUS
from pentest_test_cases import CONFIG_PATH, CURRENT_STATE

POLICY = LyraCompiler.from_yaml(CONFIG_PATH)
STATE_VARS = POLICY.spec.state_vars

MODELS = [
    "openai/gpt-5.6-terra",
    "anthropic/claude-opus-5.5",
    "qwen/qwen3.8-max-prime",
    "deepseek/deepseek-v4-pro",
]


async def run_model(api_key: str, model: str) -> None:
    print(f"=== model: {model} ===  ({len(CALIBRATION_CORPUS)} cases)")
    rows = []
    refusals = []
    errors = []
    async with OpenRouterStateEstimator(api_key=api_key, model=model) as estimator:
        for name, action, expected_flagged in CALIBRATION_CORPUS:
            should_flag = bool(expected_flagged)
            try:
                predicted = await estimator.estimate(STATE_VARS, CURRENT_STATE, action)
            except StateEstimationError as exc:
                msg = str(exc)
                if "refused" in msg:
                    refusals.append((name, should_flag))
                    print(f"  [REFUSED] [{name}] should_flag={should_flag}: {msg}")
                else:
                    errors.append((name, should_flag, msg))
                    print(f"  [ERROR]   [{name}] should_flag={should_flag}: {msg}")
                continue
            v = POLICY.evaluate(predicted)
            blocked = v > POLICY.block_threshold
            flag_mark = "!" if should_flag != blocked else " "
            print(f"  {flag_mark} [{name}] should_flag={should_flag} V={v:.3f} blocked={blocked}")
            rows.append((name, should_flag, v, blocked))

    print("\n" + "-" * 70)
    positives = [r for r in rows if r[1]]
    negatives = [r for r in rows if not r[1]]
    recall = sum(1 for r in positives if r[3]) / len(positives) if positives else float("nan")
    far = sum(1 for r in negatives if r[3]) / len(negatives) if negatives else float("nan")
    n_pos_refused = sum(1 for _n, sf in refusals if sf)
    n_neg_refused = sum(1 for _n, sf in refusals if not sf)
    print(f"scored: n_positive={len(positives)} n_negative={len(negatives)}")
    print(f"recall={recall:.1%} false_alarm_rate={far:.1%}  (computed only over cases that got a real answer)")
    print(f"refused: {len(refusals)}/{len(CALIBRATION_CORPUS)} total "
          f"({n_pos_refused} of the {len(positives) + n_pos_refused} positive/dangerous cases, "
          f"{n_neg_refused} of the {len(negatives) + n_neg_refused} negative/benign cases)")
    if errors:
        print(f"other errors (not refusals): {len(errors)}")
    print()


async def main() -> None:
    key_path = Path("Openrouter_API_key.env")
    api_key = key_path.read_text(encoding="utf-8").strip()
    for model in MODELS:
        await run_model(api_key, model)


if __name__ == "__main__":
    asyncio.run(main())
