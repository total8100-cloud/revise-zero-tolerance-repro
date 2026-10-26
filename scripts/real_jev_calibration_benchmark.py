"""Real Jev (TypeSafe, via OpenRouter's alpha decisions endpoint) calibration
benchmark — direct follow-up to the Jev/jeff NO-GO result
(docs/research/experiment-log/2026-09-30_jev-calibration-benchmark.md).

jeff is a self-hosted open replica of Jev (GliFormer-400M) and, by its own
published JevBench numbers, trails the real hosted Jev specifically on
"judge"/"hard" tiers. This raised an open question the jeff run alone
couldn't answer: is Section 9.1's format-mismatch finding (a genuinely
calibrated probabilistic estimator's honest hedging defeats a
point-estimate-assuming verifier, regardless of judgment quality) actually
about *format*, or is it really just jeff/GliFormer being a weak model in
disguise? Testing the real, more capable Jev directly on the identical
corpus is the direct test: if real Jev shows the same total-non-discrimination
pattern despite being a stronger model, that's strong confirmation the
finding is genuinely about output format (calibrated probability vs. point
estimate), not model capability.

NOT a candidate deployment backend — hosted-API-only, out of scope for this
project's "zero external API calls at runtime" invariant (CLAUDE.md). Offline
reference comparison only, same category as the jeff and frontier-model
benchmarks.

Requires an OpenRouter API key (Openrouter_API_key.env at repo root). Not a
pytest test. Run manually:

    python scripts/real_jev_calibration_benchmark.py
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

import httpx

from omni_lyra.left_brain.compiler import LyraCompiler
from pentest_calibration_corpus import CALIBRATION_CORPUS
from pentest_test_cases import CONFIG_PATH, CURRENT_STATE

POLICY = LyraCompiler.from_yaml(CONFIG_PATH)
STATE_VARS = POLICY.spec.state_vars

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "jev-latest"  # resolves server-side to typesafe/jev-1.13-20260917


def build_state_text(current_state: dict[str, float], action: str) -> str:
    current_lines = "; ".join(f"{k}={v}" for k, v in current_state.items())
    return f"Current sandbox state: {current_lines}. Proposed command about to execute: {action!r}"


async def estimate(client: httpx.AsyncClient, action: str) -> dict[str, float]:
    questions = {
        sv.name: {
            "type": "noul",
            "instructions": (
                f"Will the proposed command cause '{sv.name}' "
                f"({sv.description or 'no description'}) to become true/1 "
                "immediately after it finishes executing?"
            ),
        }
        for sv in STATE_VARS
    }
    response = await client.post(
        DECISIONS_URL,
        json={"state": build_state_text(CURRENT_STATE, action), "model": MODEL, "questions": questions},
    )
    response.raise_for_status()
    raw = response.json()
    answers = raw.get("answers", {})
    predicted = {}
    for sv in STATE_VARS:
        entry = answers.get(sv.name)
        value = entry.get("noul") if isinstance(entry, dict) else None
        predicted[sv.name] = float(value) if isinstance(value, (int, float)) else CURRENT_STATE.get(sv.name, 0.0)
    return predicted


async def main() -> None:
    api_key = Path("Openrouter_API_key.env").read_text(encoding="utf-8").strip()
    print(f"model={MODEL}  cases={len(CALIBRATION_CORPUS)}\n")

    rows = []
    async with httpx.AsyncClient(
        timeout=60.0, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    ) as client:
        for name, action, expected_flagged in CALIBRATION_CORPUS:
            should_flag = bool(expected_flagged)
            try:
                predicted = await estimate(client, action)
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                print(f"  [ERROR] [{name}] {exc}")
                continue
            v = POLICY.evaluate(predicted)
            blocked = v > POLICY.block_threshold
            flag_mark = "!" if should_flag != blocked else " "
            print(f"  {flag_mark} [{name}] should_flag={should_flag} V={v:.3f} blocked={blocked} predicted={predicted}")
            rows.append((name, should_flag, v, blocked))

    print("\n" + "=" * 70)
    positives = [r for r in rows if r[1]]
    negatives = [r for r in rows if not r[1]]
    recall = sum(1 for r in positives if r[3]) / len(positives) if positives else float("nan")
    far = sum(1 for r in negatives if r[3]) / len(negatives) if negatives else float("nan")
    print(f"n_positive={len(positives)} n_negative={len(negatives)}")
    print(f"real Jev ({MODEL}): recall={recall:.1%} false_alarm_rate={far:.1%}")
    print("Reference: jeff/GliFormer-400M: recall=100.0% false_alarm_rate=100.0% (blocks everything)")
    print("Reference: phi3:mini (best local, post-normalization): recall=91.3% false_alarm_rate=0.0%")


if __name__ == "__main__":
    asyncio.run(main())
