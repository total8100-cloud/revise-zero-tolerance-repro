"""Failure-model validation on the live estimator (5th-priority analysis, 2026-10-06).

The failure-injection studies assume per-call failures are independent (Bernoulli f).
This script measures whether real estimator failures on the local Ollama server are
independent, using load- and latency-induced timeouts as the failure source:

  1. Calibrate: median latency M of phi3:mini over sequential calls (generous timeout).
  2. Sequential: S calls at timeout = M, so roughly half fail. Measures the failure
     rate p and the lag-1 autocorrelation of outcomes. Consecutive triples are grouped
     into pseudo decision steps (K=3) and compared with 1-(1-p)^3 (independence).
  3. Burst: B bursts of 6 concurrent calls at timeout = M. Queueing on the single
     local server makes later requests in a burst time out more often. The
     dispersion index (variance / mean of failures per burst) is compared with 1
     (binomial, independent).

Limitation: this measures timeout-induced failures from load, not real server
outages (crash, OOM, model unload). It tests the within-run dependence structure,
which is the assumption the injection studies rely on. Results are not a claim about
production outage rates.

Run from repo root: python scripts/failure_model_validation.py
"""

from __future__ import annotations

import asyncio
import json
import statistics
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

MODEL = "phi3:mini"
N_CAL = 12
N_SEQ = 240
N_BURST = 40
BURST = 6
OUT = Path("docs/research/experiment-log/2026-10-06_failure-model-validation.json")


async def one(est: ModelStateEstimator, state_vars, action: str) -> bool:
    try:
        await est.estimate(state_vars, CURRENT_STATE, action)
        return True
    except StateEstimationError:
        return False


def lag1_autocorr(x: list[int]) -> float:
    m = statistics.fmean(x)
    num = sum((x[i] - m) * (x[i + 1] - m) for i in range(len(x) - 1))
    den = sum((v - m) ** 2 for v in x)
    return num / den if den else 0.0


async def main() -> None:
    policy = LyraCompiler.from_yaml(CONFIG_PATH)
    sv = policy.spec.state_vars
    actions = [a for _, a, _ in CALIBRATION_CORPUS]

    # 1. calibrate latency with a generous timeout
    lat = []
    async with ModelStateEstimator(model=MODEL, timeout=120.0) as est:
        import time
        for i in range(N_CAL):
            t0 = time.perf_counter()
            await est.estimate(sv, CURRENT_STATE, actions[i % len(actions)])
            lat.append(time.perf_counter() - t0)
    M = statistics.median(lat)
    print(f"latency over {N_CAL} calls: median={M:.2f}s min={min(lat):.2f}s max={max(lat):.2f}s")

    # 2. sequential at timeout = M
    seq = []
    async with ModelStateEstimator(model=MODEL, timeout=M) as est:
        for i in range(N_SEQ):
            ok = await one(est, sv, actions[i % len(actions)])
            seq.append(0 if ok else 1)  # 1 = failure
    p = statistics.fmean(seq)
    rho = lag1_autocorr(seq)
    triples = [seq[i:i + 3] for i in range(0, len(seq) - len(seq) % 3, 3)]
    step_fail = sum(1 for t in triples if any(t)) / len(triples)
    indep_step = 1 - (1 - p) ** 3
    print(f"sequential: n={N_SEQ} failure rate p={p:.3f} lag-1 autocorr={rho:+.3f}")
    print(f"pseudo-steps (K=3): observed P(>=1 failure)={step_fail:.3f} vs independence {indep_step:.3f} ({len(triples)} steps)")

    # 3. bursts of concurrent calls at several timeouts. At timeout = M the failure rate
    # saturates under concurrency (near-total), which makes the dispersion test uninformative;
    # looser timeouts (2M, 4M) probe intermediate failure rates.
    bursts = {}
    for mult in (1.0, 2.0, 4.0):
        t_out = M * mult
        counts = []
        for b in range(N_BURST):
            async with ModelStateEstimator(model=MODEL, timeout=t_out) as est:
                res = await asyncio.gather(*[
                    one(est, sv, actions[(b * BURST + j) % len(actions)]) for j in range(BURST)
                ])
            counts.append(sum(1 for ok in res if not ok))
        mean_b = statistics.fmean(counts)
        var_b = statistics.pvariance(counts)
        p_b = mean_b / BURST
        binom_var = BURST * p_b * (1 - p_b)
        disp = var_b / binom_var if binom_var else float("nan")
        bursts[str(mult)] = {"timeout_s": t_out, "failures_per_burst": counts, "mean_failures": mean_b,
                             "rate": p_b, "dispersion_index": disp}
        print(f"bursts at timeout={mult}M ({t_out:.2f}s): {N_BURST} x {BURST} concurrent; failures/burst mean={mean_b:.2f} (p={p_b:.3f}), dispersion index={disp:.2f} (1 = independent)")

    result = {
        "model": MODEL, "timeout_s": M, "latency_median_s": M, "latency_samples": lat,
        "sequential": {"n": N_SEQ, "failure_rate": p, "lag1_autocorr": rho,
                       "pseudo_step_failure_observed": step_fail, "pseudo_step_failure_independent": indep_step,
                       "outcomes": seq},
        "burst": {"bursts": N_BURST, "size": BURST, "by_timeout_multiple": bursts},
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
