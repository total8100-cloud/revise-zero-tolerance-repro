"""Tests for split-conformal calibration (omni_lyra.right_brain.conformal)."""

import pytest

from omni_lyra.right_brain.conformal import (
    aggregate_scalar_nonconformity_scores,
    apply_margin,
    apply_scalar_margin,
    calibrate,
    calibrate_scalar,
    min_calibration_size,
    nonconformity_scores,
    restricted_energy,
    scalar_nonconformity_scores,
)


class _FakePolicy:
    """Minimal duck-typed stand-in for CompiledPolicy: linear weighted sum
    instead of the real weighted-squared-ReLU V(s), sufficient for testing
    that scalar_nonconformity_scores/calibrate_scalar call .evaluate()
    correctly — the real energy math is compiler.py's responsibility, not
    this module's."""

    def __init__(self, weights: dict[str, float]):
        self.weights = weights

    def evaluate(self, state: dict[str, float]) -> float:
        return sum(self.weights.get(k, 0.0) * v for k, v in state.items())


class TestNonconformityScores:
    def test_one_sided_underprediction_only(self):
        pairs = [
            ({"x": 0.0}, {"x": 1.0}),  # underpredicted by 1.0
            ({"x": 1.0}, {"x": 0.0}),  # overpredicted -> clipped to 0.0
            ({"x": 0.5}, {"x": 0.5}),  # exact -> 0.0
        ]
        scores = nonconformity_scores(pairs)
        assert scores["x"] == [1.0, 0.0, 0.0]

    def test_missing_var_defaults_to_zero(self):
        pairs = [({"x": 0.0}, {"x": 0.0, "y": 1.0})]
        scores = nonconformity_scores(pairs)
        assert scores["y"] == [1.0]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            nonconformity_scores([])


class TestMinCalibrationSize:
    def test_alpha_0_1_needs_9(self):
        assert min_calibration_size(0.1) == 9

    def test_alpha_0_2_needs_4(self):
        assert min_calibration_size(0.2) == 4

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            min_calibration_size(0.0)
        with pytest.raises(ValueError):
            min_calibration_size(1.0)


class TestCalibrate:
    def test_picks_correct_order_statistic(self):
        # n=9, alpha=0.1 -> rank = ceil(10*0.9) = 9th smallest (i.e. the max)
        scores = {"x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
        margins = calibrate(scores, alpha=0.1)
        assert margins["x"] == pytest.approx(0.9)

    def test_larger_n_picks_interior_order_statistic(self):
        # n=10, alpha=0.5 -> rank = ceil(11*0.5) = ceil(5.5) = 6th smallest
        scores = {"x": [float(i) for i in range(1, 11)]}  # 1..10
        margins = calibrate(scores, alpha=0.5)
        assert margins["x"] == pytest.approx(6.0)

    def test_rank_clipped_to_n_when_alpha_very_small(self):
        scores = {"x": [1.0, 2.0, 3.0, 4.0]}
        margins = calibrate(scores, alpha=0.2)  # min n for alpha=0.2 is 4
        assert margins["x"] == pytest.approx(4.0)

    def test_too_few_points_raises(self):
        scores = {"x": [1.0, 2.0, 3.0]}  # n=3, alpha=0.1 needs n>=9
        with pytest.raises(ValueError):
            calibrate(scores, alpha=0.1)

    def test_multiple_vars_calibrated_independently(self):
        scores = {
            "x": [1.0, 1.0, 1.0, 1.0],
            "y": [0.0, 0.0, 0.0, 5.0],
        }
        margins = calibrate(scores, alpha=0.2)
        assert margins["x"] == pytest.approx(1.0)
        assert margins["y"] == pytest.approx(5.0)


class TestApplyMargin:
    def test_adds_margin_per_var(self):
        predicted = {"a": 0.0, "b": 1.0}
        margins = {"a": 0.5}
        result = apply_margin(predicted, margins)
        assert result == {"a": 0.5, "b": 1.0}

    def test_does_not_mutate_input(self):
        predicted = {"a": 0.0}
        apply_margin(predicted, {"a": 1.0})
        assert predicted == {"a": 0.0}

    def test_margin_for_absent_var_is_ignored(self):
        predicted = {"a": 0.0}
        result = apply_margin(predicted, {"unknown_var": 5.0})
        assert result == {"a": 0.0}


class TestScalarNonconformityScores:
    def test_one_sided_underprediction_only(self):
        policy = _FakePolicy({"x": 2.0})
        pairs = [
            ({"x": 0.0}, {"x": 1.0}),  # V: predicted=0, actual=2 -> residual 2.0
            ({"x": 1.0}, {"x": 0.0}),  # V: predicted=2, actual=0 -> overpredicted, clipped to 0
            ({"x": 0.5}, {"x": 0.5}),  # exact -> 0.0
        ]
        scores = scalar_nonconformity_scores(pairs, policy)
        assert scores == [2.0, 0.0, 0.0]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            scalar_nonconformity_scores([], _FakePolicy({}))


class TestCalibrateScalar:
    def test_picks_correct_order_statistic(self):
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        margin = calibrate_scalar(scores, alpha=0.1)
        assert margin == pytest.approx(0.9)

    def test_too_few_points_raises(self):
        scores = [1.0, 2.0, 3.0]  # n=3, alpha=0.1 needs n>=9
        with pytest.raises(ValueError):
            calibrate_scalar(scores, alpha=0.1)


class TestApplyScalarMargin:
    def test_adds_margin(self):
        assert apply_scalar_margin(5.0, 1.5) == pytest.approx(6.5)

    def test_zero_margin_is_identity(self):
        assert apply_scalar_margin(5.0, 0.0) == pytest.approx(5.0)


class TestRestrictedEnergy:
    def test_zeroes_out_given_vars(self):
        policy = _FakePolicy({"a": 2.0, "b": 3.0})
        result = restricted_energy(policy, {"a": 5.0, "b": 1.0}, {"a"})
        assert result == pytest.approx(3.0)  # a forced to 0, only b's 3.0 contributes

    def test_missing_var_in_zeroed_set_is_ignored(self):
        policy = _FakePolicy({"a": 2.0})
        result = restricted_energy(policy, {"a": 5.0}, {"unknown"})
        assert result == pytest.approx(10.0)

    def test_does_not_mutate_input(self):
        policy = _FakePolicy({"a": 1.0})
        state = {"a": 5.0}
        restricted_energy(policy, state, {"a"})
        assert state == {"a": 5.0}


class TestAggregateScalarNonconformityScores:
    def test_critical_var_errors_do_not_leak_into_aggregate_residual(self):
        # policy: critical var 'crit' weight 15, aggregate var 'agg' weight 5.
        policy = _FakePolicy({"crit": 15.0, "agg": 5.0})
        pairs = [
            # predicted misses 'crit' entirely (0 vs actual 1) but 'agg' is
            # predicted correctly -- with critical_vars zeroed out, this
            # should contribute residual 0 to the aggregate score, NOT be
            # masked/inflated by the critical miss.
            ({"crit": 0.0, "agg": 1.0}, {"crit": 1.0, "agg": 1.0}),
        ]
        scores = aggregate_scalar_nonconformity_scores(pairs, policy, critical_vars={"crit"})
        assert scores == [0.0]

    def test_genuine_aggregate_underprediction_is_captured(self):
        policy = _FakePolicy({"crit": 15.0, "agg": 5.0})
        pairs = [({"crit": 1.0, "agg": 0.0}, {"crit": 1.0, "agg": 1.0})]
        scores = aggregate_scalar_nonconformity_scores(pairs, policy, critical_vars={"crit"})
        assert scores == [5.0]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aggregate_scalar_nonconformity_scores([], _FakePolicy({}), critical_vars=set())
