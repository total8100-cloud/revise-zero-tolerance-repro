"""Self-consistency fusion — combine N repeated `ModelStateEstimator` samples
of the *same* model into a single predicted state, as a calibration-aware
alternative to a single deterministic (temperature=0) point estimate.

Motivated by `scripts/estimator_calibration_study.py`: a 4-model cross-model
max-ensemble recovered full recall on ROE violations but nearly tripled false
alarms (20 vs. best individual model's 6-8) — not a usable mitigation. Rather
than run 4 heterogeneous models per decision (impractical, and contrary to
this project's local-single-model "micro-cost" positioning), these functions
fuse repeated *samples of one model* (temperature > 0) instead, so the
calibration question becomes "is this one model's own uncertainty a useful
signal?" rather than "do other models disagree with it?".

Pure functions, no I/O — testable without Ollama, same pattern as
`sandbox.pentest_range.parse_connection_ips`.
"""

from __future__ import annotations


def fuse_max(samples: list[dict[str, float]]) -> dict[str, float]:
    """Per state var, take the max value across all samples.

    The most conservative fusion: if any single sample flags a violation,
    the fused state reflects it — a single confident "unsafe" draw can never
    be diluted by other draws that missed it.
    """
    if not samples:
        raise ValueError("fuse_max requires at least one sample")

    fused: dict[str, float] = dict(samples[0])
    for sample in samples[1:]:
        for var, val in sample.items():
            fused[var] = max(fused.get(var, val), val)
    return fused


def fuse_majority(
    samples: list[dict[str, float]],
    flag_threshold: float = 1.0,
    min_vote_fraction: float = 0.6,
) -> dict[str, float]:
    """Per state var, flag only if a majority of samples agree it's flagged.

    A sample "votes to flag" a var if its value >= flag_threshold (the same
    >=1.0 convention `estimator_calibration_study.score()` uses to decide
    what counts as a predicted violation). If the fraction of flagging votes
    meets `min_vote_fraction`, the fused value is the max among the flagging
    samples (preserving severity); otherwise the fused value is the max
    among the *non*-flagging samples (i.e., treated as not flagged), so a
    single confident-but-wrong sample can't dictate the fused output.

    Trades recall for precision relative to `fuse_max`: an isolated
    hallucinated "unsafe" draw gets outvoted instead of always winning.
    """
    if not samples:
        raise ValueError("fuse_majority requires at least one sample")

    all_vars = set()
    for sample in samples:
        all_vars.update(sample.keys())

    fused: dict[str, float] = {}
    for var in all_vars:
        values = [sample[var] for sample in samples if var in sample]
        flagging = [v for v in values if v >= flag_threshold]
        non_flagging = [v for v in values if v < flag_threshold]

        vote_fraction = len(flagging) / len(values) if values else 0.0
        if vote_fraction >= min_vote_fraction and flagging:
            fused[var] = max(flagging)
        elif non_flagging:
            fused[var] = max(non_flagging)
        else:
            # Every sample flagged, but not enough to clear the majority
            # threshold (only possible if min_vote_fraction > the fraction
            # implied by "all of them") — fall back to the lowest flagging
            # value rather than crediting full severity.
            fused[var] = min(flagging)
    return fused


def disagreement(samples: list[dict[str, float]], flag_threshold: float = 1.0) -> dict[str, float]:
    """Per state var, fraction of samples that disagree with the majority
    binary flagged/not-flagged classification.

    0.0 = every sample agreed (fully consistent); approaches 0.5 = maximally
    split. Diagnostic only — not used by `fuse_max`/`fuse_majority` — kept
    separate so the actual fusion decision logic stays simple to test and
    reason about independently of how disagreement is reported/logged.
    """
    if not samples:
        raise ValueError("disagreement requires at least one sample")

    all_vars = set()
    for sample in samples:
        all_vars.update(sample.keys())

    result: dict[str, float] = {}
    for var in all_vars:
        values = [sample[var] for sample in samples if var in sample]
        flagged_count = sum(1 for v in values if v >= flag_threshold)
        n = len(values)
        majority_count = max(flagged_count, n - flagged_count)
        result[var] = (n - majority_count) / n if n else 0.0
    return result
