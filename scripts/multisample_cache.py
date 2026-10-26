"""Cache real multi-sample estimator outputs for the multi-sample failure study.

For each calibration-corpus case, draw N=5 predictions from phi3:mini at
temperature 0.7 (the same settings as the self-consistency study). Real
StateEstimationError draws are stored as None. The study script
(multisample_failure_study.py) injects additional failures on top of these.

Run from repo root: python scripts/multisample_cache.py
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

sys.path.insert(0, str(Path(__file__).parent))
from omni_lyra.left_brain.compiler import LyraCompiler  # noqa: E402
from omni_lyra.right_brain.state_estimator import ModelStateEstimator, StateEstimationError  # noqa: E402
from pentest_calibration_corpus import CALIBRATION_CORPUS  # noqa: E402
from pentest_test_cases import CONFIG_PATH, CURRENT_STATE  # noqa: E402

CACHE_PATH = Path("docs/research/experiment-log/2026-10-06_multisample-predictions.cache.json")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "phi3:mini"
if MODEL != "phi3:mini":
    CACHE_PATH = Path(f"docs/research/experiment-log/2026-10-06_multisample-predictions-{MODEL.replace(':', '-')}.cache.json")
N_DRAWS = 5
TEMPERATURE = 0.7


async def main() -> None:
    policy = LyraCompiler.from_yaml(CONFIG_PATH)
    state_vars = policy.spec.state_vars
    cache = {}
    real_failures = 0
    async with ModelStateEstimator(model=MODEL) as est:
        for name, action, expected in CALIBRATION_CORPUS:
            draws = []
            for _ in range(N_DRAWS):
                try:
                    draws.append(await est.estimate(state_vars, CURRENT_STATE, action, temperature=TEMPERATURE))
                except StateEstimationError:
                    draws.append(None)
                    real_failures += 1
            cache[name] = {"action": action, "dangerous": bool(expected), "draws": draws}
            print(f"  cached {name}: {sum(d is not None for d in draws)}/{N_DRAWS} draws ok")
    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {CACHE_PATH} ({len(cache)} cases, {real_failures} real draw failures)")


if __name__ == "__main__":
    asyncio.run(main())
