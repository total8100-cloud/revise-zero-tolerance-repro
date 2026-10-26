"""Tests for the differentiable hidden-state -> state-var predictor.

Pure synthetic data — no HF model download required.
"""

import pytest

torch = pytest.importorskip("torch")

from omni_lyra.left_brain.schema import StateVar
from omni_lyra.right_brain.state_predictor import (
    StatePredictor,
    TrainingExample,
    load_training_examples,
    log_training_example,
    train_state_predictor,
)

STATE_VARS = [
    StateVar(name="bytes_out", unit="bytes"),
    StateVar(name="cpu_usage", unit="ratio"),
]


class TestStatePredictorShape:
    def test_output_dim_matches_state_vars(self):
        predictor = StatePredictor(hidden_size=16, state_vars=STATE_VARS)
        hidden = torch.randn(16)
        out = predictor(hidden)
        assert out.shape == (2,)

    def test_predict_dict_keys_match_state_var_names(self):
        predictor = StatePredictor(hidden_size=16, state_vars=STATE_VARS)
        hidden = torch.randn(16)
        result = predictor.predict_dict(hidden)
        assert set(result.keys()) == {"bytes_out", "cpu_usage"}

    def test_batched_input(self):
        predictor = StatePredictor(hidden_size=16, state_vars=STATE_VARS)
        hidden = torch.randn(4, 16)
        out = predictor(hidden)
        assert out.shape == (4, 2)


class TestTraining:
    def test_loss_decreases_over_training(self):
        torch.manual_seed(0)
        predictor = StatePredictor(hidden_size=8, state_vars=STATE_VARS)
        examples = [
            TrainingExample(
                hidden_state=torch.randn(8),
                actual_state={"bytes_out": float(i), "cpu_usage": 0.1 * i},
            )
            for i in range(20)
        ]
        history = train_state_predictor(predictor, examples, epochs=200, lr=0.01)
        assert history[-1] < history[0]

    def test_converges_on_toy_linear_mapping(self):
        """A tiny linear ground-truth relationship should be learnable to low error."""
        torch.manual_seed(0)
        true_weight = torch.tensor([2.0, -1.0, 0.5, 3.0])
        examples = []
        for _ in range(50):
            h = torch.randn(4)
            target_val = float((h @ true_weight).item())
            examples.append(
                TrainingExample(hidden_state=h, actual_state={"bytes_out": target_val, "cpu_usage": 0.0})
            )
        predictor = StatePredictor(hidden_size=4, state_vars=STATE_VARS, hidden_dim=32)
        history = train_state_predictor(predictor, examples, epochs=500, lr=0.01)
        assert history[-1] < 0.5

    def test_empty_examples_raises(self):
        predictor = StatePredictor(hidden_size=8, state_vars=STATE_VARS)
        with pytest.raises(ValueError):
            train_state_predictor(predictor, [])


class TestTrainingDataPersistence:
    def test_log_and_load_roundtrip(self, tmp_path):
        example = TrainingExample(
            hidden_state=torch.randn(8), actual_state={"bytes_out": 123.0, "cpu_usage": 0.4}
        )
        log_training_example(tmp_path, example)
        loaded = load_training_examples(tmp_path)
        assert len(loaded) == 1
        assert torch.allclose(loaded[0].hidden_state, example.hidden_state)
        assert loaded[0].actual_state == example.actual_state

    def test_multiple_examples_each_get_own_file(self, tmp_path):
        for i in range(5):
            log_training_example(
                tmp_path,
                TrainingExample(hidden_state=torch.randn(4), actual_state={"bytes_out": float(i), "cpu_usage": 0.0}),
            )
        assert len(list(tmp_path.glob("*.pt"))) == 5
        assert len(load_training_examples(tmp_path)) == 5

    def test_load_from_nonexistent_dir_returns_empty(self, tmp_path):
        missing = tmp_path / "does-not-exist"
        assert load_training_examples(missing) == []

    def test_log_creates_dir_if_missing(self, tmp_path):
        target = tmp_path / "nested" / "log_dir"
        log_training_example(
            target, TrainingExample(hidden_state=torch.randn(4), actual_state={"bytes_out": 0.0, "cpu_usage": 0.0})
        )
        assert target.exists()
        assert len(list(target.glob("*.pt"))) == 1
