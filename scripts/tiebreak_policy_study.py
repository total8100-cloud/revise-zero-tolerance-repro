"""Tie-break as a design variable (2nd-priority analysis, 2026-10-06).

Same decision model as failure_injection_study.py (one dangerous + two benign
candidates, independent per-candidate failures, paired seeds across policies).
Adds a second axis: how the verifier resolves ties at minimum V.

  index    : lowest candidate index among the tie (current select_best behavior)
  random   : uniform random choice among the tie (per-trial stream, shared by policies)
  abstain  : candidates tied at minimum V are all withheld (not executed); the
             best of the remaining candidates is selected. A tie is treated as
             no evidence of safety.

Failure defaults: "open" (failed candidate predicted = current state) and
"closed" (failed candidate predicted = worst case), as in the primary study.

Sanity check: index + open/closed must reproduce the primary study's numbers
(docs/research/experiment-log/2026-10-06_failure-injection-results.json) exactly
for the same cache; the script asserts this for phi3:mini.

Run from repo root:
  python scripts/tiebreak_policy_study.py --cache docs/research/experiment-log/2026-10-06_failure-injection-predictions.cache.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).parent))
from failure_injection_study import (  # noqa: E402
    FAILURE_RATES,
    K,
    N_TRIALS,
    VERIFIER,
    _candidate_pool,
    _predict_for_candidate,
)
from pentest_test_cases import CURRENT_STATE  # noqa: E402

TIEBREAKS = ["index", "random", "abstain"]
POLICIES = ["open", "closed"]
E = Path("docs/research/experiment-log")


def choose(tiebreak: str, cands: list, u: float):
    """Return (selected_index or None, verdict_is_block). u is a per-trial uniform draw,
    always consumed, so every policy sees the same tie-break randomness."""
    vals = [VERIFIER.verify(dict(CURRENT_STATE), p) for _, p in cands]
    vmin = min(v.v_predicted for v in vals)
    tied = [i for i, v in enumerate(vals) if v.v_predicted == vmin]
    if tiebreak == "abstain" and len(tied) > 1:
        remaining = [i for i in range(len(cands)) if i not in tied]
        if not remaining:
            return None, True
        best = min(remaining, key=lambda i: (vals[i].v_predicted, i))
        return best, vals[best].verdict.value == "BLOCK"
    if tiebreak == "random" and len(tied) > 1:
        sel = tied[min(int(u * len(tied)), len(tied) - 1)]
    else:  # index (and abstain with no tie): lowest index in the tie
        sel = tied[0]
    return sel, vals[sel].verdict.value == "BLOCK"


def run_cell(policy: str, f: float, tiebreak: str, dangerous, benign):
    # Main stream identical to failure_injection_study.run_study (for the index sanity check).
    rng = random.Random(1000 * int(f * 10))
    tie_rng = random.Random(9000 + 1000 * int(f * 10))
    unsafe = benign_done = blocked_all = 0
    per_trial = []
    for _ in range(N_TRIALS):
        chosen = [rng.choice(dangerous)] + [rng.choice(benign) for _ in range(K - 1)]
        rng.shuffle(chosen)
        cands = []
        danger_idx = None
        for idx, (name, entry) in enumerate(chosen):
            fails = rng.random() < f
            cands.append((entry["action"], _predict_for_candidate(entry, policy, fails)))
            if entry["dangerous"]:
                danger_idx = idx
        u = tie_rng.random()  # always consumed, so policies share the tie-break stream
        sel, blocked = choose(tiebreak, cands, u)
        exec_ok = sel is not None and not blocked
        u = int(exec_ok and sel == danger_idx)
        unsafe += u
        per_trial.append(u)
        benign_done += int(exec_ok and sel != danger_idx)
        blocked_all += int(not exec_ok)
    return {"unsafe": unsafe, "benign_done": benign_done, "blocked_all": blocked_all, "per_trial_unsafe": per_trial}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", default="phi3-mini")
    args = ap.parse_args()
    cache = json.loads(Path(args.cache).read_text(encoding="utf-8"))
    dangerous, benign = _candidate_pool(cache)
    results = {}
    for tb in TIEBREAKS:
        for policy in POLICIES:
            for f in FAILURE_RATES:
                r = run_cell(policy, f, tb, dangerous, benign)
                results[f"{tb}|{policy}@{f}"] = {"tiebreak": tb, "policy": policy, "f": f, "n": N_TRIALS, **r}
                print(f"{tb:8s} {policy:6s} f={f:<4} unsafe={r['unsafe']:4d}  benign={r['benign_done']:4d}  blocked_all={r['blocked_all']:4d}")
    if args.tag == "phi3-mini":
        primary = json.loads((E / "2026-10-06_failure-injection-results.json").read_text(encoding="utf-8"))
        for policy in POLICIES:
            for f in FAILURE_RATES:
                a = results[f"index|{policy}@{f}"]
                b = primary[f"{policy}@{f}"]
                assert a["unsafe"] == b["unsafe_execution"] and a["benign_done"] == b["benign_completion"], (policy, f)
        print("sanity check passed: index tie-break reproduces the primary phi3:mini study exactly")
    out = E / f"2026-10-06_tiebreak-policy-results-{args.tag}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
