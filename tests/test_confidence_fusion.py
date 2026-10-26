"""Tests for self-consistency fusion (omni_lyra.right_brain.confidence_fusion)."""

import pytest

from omni_lyra.right_brain.confidence_fusion import disagreement, fuse_majority, fuse_max


class TestFuseMax:
    def test_picks_per_var_max_across_samples(self):
        samples = [
            {"a": 0.0, "b": 1.0},
            {"a": 1.0, "b": 0.0},
            {"a": 0.5, "b": 0.5},
        ]
        assert fuse_max(samples) == {"a": 1.0, "b": 1.0}

    def test_single_sample_returned_verbatim(self):
        samples = [{"a": 3.0, "b": 0.0}]
        assert fuse_max(samples) == {"a": 3.0, "b": 0.0}

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            fuse_max([])


class TestFuseMajority:
    def test_majority_flag_uses_max_of_flagging_samples(self):
        # 3 of 5 flag (>=1.0) -> vote_fraction=0.6 clears default 0.6 threshold
        samples = [
            {"risky": 1.0},
            {"risky": 2.0},
            {"risky": 1.5},
            {"risky": 0.0},
            {"risky": 0.0},
        ]
        assert fuse_majority(samples) == {"risky": 2.0}

    def test_minority_flag_falls_back_to_non_flagging(self):
        # only 2 of 5 flag -> vote_fraction=0.4 < default 0.6 threshold
        samples = [
            {"risky": 1.0},
            {"risky": 1.0},
            {"risky": 0.0},
            {"risky": 0.0},
            {"risky": 0.2},
        ]
        assert fuse_majority(samples) == {"risky": 0.2}

    def test_unanimous_agreement_not_flagged(self):
        samples = [{"x": 0.0}, {"x": 0.0}, {"x": 0.1}]
        assert fuse_majority(samples) == {"x": 0.1}

    def test_unanimous_flagged(self):
        samples = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
        assert fuse_majority(samples) == {"x": 3.0}

    def test_custom_threshold(self):
        # 2 of 5 = 0.4 vote fraction; with min_vote_fraction=0.4 this now clears
        samples = [
            {"risky": 1.0},
            {"risky": 1.0},
            {"risky": 0.0},
            {"risky": 0.0},
            {"risky": 0.0},
        ]
        assert fuse_majority(samples, min_vote_fraction=0.4) == {"risky": 1.0}

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            fuse_majority([])


class TestDisagreement:
    def test_full_agreement_is_zero(self):
        samples = [{"x": 0.0}, {"x": 0.0}, {"x": 0.0}]
        assert disagreement(samples) == {"x": 0.0}

    def test_one_dissenter_out_of_five(self):
        samples = [{"x": 0.0}, {"x": 0.0}, {"x": 0.0}, {"x": 0.0}, {"x": 1.0}]
        assert disagreement(samples) == {"x": 0.2}

    def test_maximally_split(self):
        samples = [{"x": 0.0}, {"x": 0.0}, {"x": 1.0}, {"x": 1.0}]
        assert disagreement(samples) == {"x": 0.5}

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            disagreement([])
