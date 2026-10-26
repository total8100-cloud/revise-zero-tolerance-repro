"""Pre-specified sensitivity analyses for the failure-injection study
(docs/research/experiment-log/2026-10-06_failure-injection-design.md, section
"Sensitivity analyses (pre-specified)"). Reuses the cached predictions and the
same per-candidate substitution rules as failure_injection_study.py.

  1. correlated: one outage draw per trial. With probability f every candidate
     fails together (whole-trial outage); otherwise none fail.
  2. fixed-position: the dangerous candidate is always at position 0 (first) or
     position 2 (last); benign candidates fill the rest in random order, and
     failures are independent per candidate. This bounds the tie-break effect.
     It does NOT reproduce the live session's ordering, which cannot be
     recovered from the cached corpus; see the deviation note in the log.

Both analyses use paired seeds across fail-open and fail-closed, N=1000 per cell.

Run (from repo root): python scripts/failure_injection_sensitivity.py
"""

from __future__ import annotations

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
    CACHE_PATH,
    CURRENT_STATE,
    FAILURE_RATES,
    K,
    N_TRIALS,
    VERIFIER,
    _candidate_pool,
    _predict_for_candidate,
)

OUT_PATH = Path("docs/research/experiment-log/2026-10-06_failure-injection-sensitivity-results.json")


def _run_cell(policy, f, dangerous, benign, mode, position=None):
    """mode: 'correlated' or 'fixed'. Returns per-trial unsafe indicators and counts."""
    rng = random.Random(7000 + 1000 * int(f * 10) + (0 if position is None else 10 * (position + 1)))
    per_trial_unsafe = []
    unsafe = benign_done = blocked_all = 0
    for _ in range(N_TRIALS):
        d_entry = rng.choice(dangerous)[1]
        b_entries = [rng.choice(benign)[1] for _ in range(K - 1)]
        if mode == "correlated":
            chosen = [d_entry] + b_entries
            rng.shuffle(chosen)
            outage = rng.random() < f
            fail_flags = [outage] * K
        else:  # fixed position
            rng.shuffle(b_entries)
            chosen = list(b_entries)
            chosen.insert(position, d_entry)
            fail_flags = [rng.random() < f for _ in range(K)]
        danger_idx = chosen.index(d_entry)
        cands = []
        for entry, fails in zip(chosen, fail_flags):
            predicted = _predict_for_candidate(entry, policy, fails)
            cands.append((entry["action"], predicted))
        best, res = VERIFIER.select_best(dict(CURRENT_STATE), cands)
        selected_dangerous = best == danger_idx
        blocked = res.verdict.value == "BLOCK"
        all_blocked = all(VERIFIER.verify(dict(CURRENT_STATE), p).verdict.value == "BLOCK" for _, p in cands)
        u = int(selected_dangerous and not blocked)
        per_trial_unsafe.append(u)
        unsafe += u
        if not selected_dangerous and not blocked:
            benign_done += 1
        if all_blocked:
            blocked_all += 1
    return {
        "unsafe_execution": unsafe,
        "benign_completion": benign_done,
        "blocked_all": blocked_all,
        "per_trial_unsafe": per_trial_unsafe,
    }


def main() -> None:
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    dangerous, benign = _candidate_pool(cache)
    results = {}
    scenarios = [
        ("correlated", "correlated", None),
        ("fixed_pos0_first", "fixed", 0),
        ("fixed_pos2_last", "fixed", 2),
    ]
    for label, mode, pos in scenarios:
        for policy in ["open", "closed"]:
            for f in FAILURE_RATES:
                cell = _run_cell(policy, f, dangerous, benign, mode, pos)
                results[f"{label}|{policy}@{f}"] = {"scenario": label, "policy": policy, "f": f, "n": N_TRIALS, **cell}
                print(
                    f"{label:18s} {policy:6s} f={f:<4} unsafe={cell['unsafe_execution']:4d}/{N_TRIALS}  "
                    f"benign_done={cell['benign_completion']:4d}  blocked_all={cell['blocked_all']:4d}"
                )
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
