"""Tests for the Lyra Runtime Verifier [V_t, ΔV_t]."""

import pytest
from pathlib import Path
from omni_lyra.left_brain.compiler import LyraCompiler
from omni_lyra.left_brain.verifier import LyraVerifier, Verdict

FIXTURE = Path(__file__).parent / "fixtures" / "sample_constraint.yaml"


@pytest.fixture
def policy():
    return LyraCompiler.from_yaml(FIXTURE)


@pytest.fixture
def verifier(policy):
    return LyraVerifier(policy=policy)


def safe() -> dict:
    return {"bytes_out": 0, "cpu_usage": 0.5, "file_deletions": 0, "net_connections": 2}


def mild_violation() -> dict:
    # V slightly > 0 but well below block_threshold=50
    return {"bytes_out": 0, "cpu_usage": 0.90, "file_deletions": 0, "net_connections": 2}


def heavy_violation() -> dict:
    # V >> block_threshold=50
    return {"bytes_out": 50_000_000, "cpu_usage": 0.99, "file_deletions": 10, "net_connections": 20}


class TestVerdicts:
    def test_safe_to_safe(self, verifier):
        result = verifier.verify(safe(), safe())
        assert result.verdict == Verdict.SAFE

    def test_safe_to_mild_warn(self, verifier):
        result = verifier.verify(safe(), mild_violation())
        assert result.verdict == Verdict.WARN

    def test_safe_to_heavy_block(self, verifier):
        result = verifier.verify(safe(), heavy_violation())
        assert result.verdict == Verdict.BLOCK

    def test_delta_v_positive_causes_warn(self, verifier):
        # Move from mild to slightly worse — ΔV > 0 while in warn zone
        worse = {**mild_violation(), "cpu_usage": 0.92}
        result = verifier.verify(mild_violation(), worse)
        assert result.verdict in (Verdict.WARN, Verdict.BLOCK)
        assert result.delta_v > 0


class TestLatency:
    def test_verification_under_1ms(self, verifier):
        result = verifier.verify(safe(), mild_violation())
        assert result.latency_ms < 1.0, (
            f"Verifier exceeded 1ms budget: {result.latency_ms:.3f}ms"
        )


class TestCommit:
    def test_commit_updates_v_prev(self, verifier):
        assert verifier.v_prev == 0.0
        new_v = verifier.commit(mild_violation())
        assert new_v > 0.0
        assert verifier.v_prev == new_v

    def test_commit_safe_state_resets_to_zero(self, verifier):
        verifier.commit(mild_violation())
        verifier.commit(safe())
        assert verifier.v_prev == pytest.approx(0.0, abs=1e-9)


class TestSelectBest:
    """REVISE(x,a) = argmin_j V(x,j) — see verifier.py's select_best."""

    def test_picks_lowest_v_candidate(self, verifier):
        candidates = [
            ("heavy", heavy_violation()),
            ("safe", safe()),
            ("mild", mild_violation()),
        ]
        index, result = verifier.select_best(safe(), candidates)
        assert index == 1
        assert result.verdict == Verdict.SAFE

    def test_single_candidate_returns_it(self, verifier):
        index, result = verifier.select_best(safe(), [("only", mild_violation())])
        assert index == 0
        assert result.verdict == Verdict.WARN

    def test_ties_resolve_to_first_index(self, verifier):
        candidates = [("a", mild_violation()), ("b", mild_violation())]
        index, _ = verifier.select_best(safe(), candidates)
        assert index == 0

    def test_empty_candidates_raises(self, verifier):
        with pytest.raises(ValueError):
            verifier.select_best(safe(), [])

    def test_all_candidates_bad_still_picks_least_bad(self, verifier):
        worse = {**heavy_violation(), "cpu_usage": 1.0, "bytes_out": 100_000_000}
        candidates = [("worse", worse), ("heavy", heavy_violation())]
        index, result = verifier.select_best(safe(), candidates)
        assert index == 1  # heavy_violation is less bad than worse
        assert result.verdict == Verdict.BLOCK  # still blocked, just the better of two bad options

    def test_does_not_mutate_verifier_state(self, verifier):
        """select_best should not call commit() implicitly — the caller decides
        what actually executes and commits, this only picks a candidate."""
        verifier.select_best(safe(), [("mild", mild_violation())])
        assert verifier.v_prev == 0.0


class TestVMargin:
    """Scalar split-conformal margin (right_brain/conformal.py's
    calibrate_scalar), added directly to V_predicted."""

    def test_default_margin_is_zero_no_behavior_change(self, verifier):
        result = verifier.verify(safe(), mild_violation())
        result_explicit = verifier.verify(safe(), mild_violation(), v_margin=0.0)
        assert result.v_predicted == pytest.approx(result_explicit.v_predicted)
        assert result.verdict == result_explicit.verdict

    def test_margin_added_to_v_predicted(self, verifier):
        base = verifier.verify(safe(), safe())
        margined = verifier.verify(safe(), safe(), v_margin=2.5)
        assert margined.v_predicted == pytest.approx(base.v_predicted + 2.5)

    def test_margin_can_push_safe_into_block(self, verifier):
        result = verifier.verify(safe(), safe(), v_margin=1000.0)
        assert result.verdict == Verdict.BLOCK

    def test_select_best_applies_margin_to_every_candidate(self, verifier):
        candidates = [("heavy", heavy_violation()), ("safe", safe())]
        index, result = verifier.select_best(safe(), candidates, v_margin=1000.0)
        # "safe" still wins the argmin (lowest V), but the huge margin pushes
        # even the best candidate's V past block_threshold — the same
        # all-or-nothing collapse mode documented for per-variable margins,
        # now reproducible directly against the scalar verifier.
        assert index == 1
        assert result.verdict == Verdict.BLOCK


class TestResultFields:
    def test_result_has_all_fields(self, verifier):
        result = verifier.verify(safe(), mild_violation())
        assert isinstance(result.v_current, float)
        assert isinstance(result.v_predicted, float)
        assert isinstance(result.delta_v, float)
        assert isinstance(result.components_predicted, dict)
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0
