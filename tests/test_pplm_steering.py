"""Pure-logic tests for PPLM math/config — no model download required."""

import torch

from omni_lyra.left_brain.schema import StateVar
from omni_lyra.right_brain.pplm_steering import (
    PPLMConfig,
    PPLMSteering,
    bow_attribute_loss,
    fuse_distributions,
    mask_danger_tokens,
    pick_next_token,
    resolve_danger_token_ids,
)
from omni_lyra.right_brain.state_predictor import StatePredictor


class FakeTokenizer:
    """Minimal stand-in for a HF tokenizer's `.encode`, no model needed."""

    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab

    def encode(self, text, add_special_tokens=False):
        if text in self.vocab:
            return [self.vocab[text]]
        return [0, 1]  # stand-in for "tokenizes into more than one piece"


class TestResolveDangerTokenIds:
    def test_single_token_words_resolved(self):
        tok = FakeTokenizer({"curl": 10, " curl": 11, "rm": 20})
        ids = resolve_danger_token_ids(tok, ["curl", "rm"])
        assert ids == [10, 11, 20]

    def test_multi_token_words_skipped(self):
        tok = FakeTokenizer({"rm": 20})
        ids = resolve_danger_token_ids(tok, ["rm", "not-a-single-token"])
        assert ids == [20]

    def test_empty_input_returns_empty(self):
        tok = FakeTokenizer({"curl": 10})
        assert resolve_danger_token_ids(tok, []) == []

    def test_dedupes_ids(self):
        tok = FakeTokenizer({"curl": 10, " curl": 10})
        ids = resolve_danger_token_ids(tok, ["curl"])
        assert ids == [10]


class TestBowAttributeLoss:
    def test_empty_danger_ids_returns_zero(self):
        logits = torch.randn(1, 50)
        loss = bow_attribute_loss(logits, [])
        assert loss.item() == 0.0

    def test_loss_increases_with_danger_mass(self):
        vocab_size = 10
        low_logits = torch.zeros(1, vocab_size)
        low_logits[0, 0] = -10.0  # token 0 is the "danger" token, suppressed

        high_logits = torch.zeros(1, vocab_size)
        high_logits[0, 0] = 10.0  # token 0 dominates the distribution

        low_loss = bow_attribute_loss(low_logits, [0])
        high_loss = bow_attribute_loss(high_logits, [0])
        assert high_loss.item() > low_loss.item()

    def test_gradient_flows_to_logits(self):
        logits = torch.randn(1, 10, requires_grad=True)
        loss = bow_attribute_loss(logits, [3, 4])
        loss.backward()
        assert logits.grad is not None
        assert torch.any(logits.grad != 0)


class TestFuseDistributions:
    def test_gamma_one_matches_perturbed_softmax(self):
        perturbed = torch.tensor([[1.0, 2.0, 3.0]])
        original = torch.tensor([[5.0, -1.0, 0.0]])
        fused = fuse_distributions(perturbed, original, gamma=1.0)
        expected = torch.softmax(perturbed, dim=-1)
        assert torch.allclose(fused, expected, atol=1e-5)

    def test_gamma_zero_matches_original_softmax(self):
        perturbed = torch.tensor([[1.0, 2.0, 3.0]])
        original = torch.tensor([[5.0, -1.0, 0.0]])
        fused = fuse_distributions(perturbed, original, gamma=0.0)
        expected = torch.softmax(original, dim=-1)
        assert torch.allclose(fused, expected, atol=1e-5)

    def test_output_is_valid_probability_distribution(self):
        perturbed = torch.randn(1, 20)
        original = torch.randn(1, 20)
        fused = fuse_distributions(perturbed, original, gamma=0.7)
        assert torch.allclose(fused.sum(dim=-1), torch.tensor([1.0]), atol=1e-5)
        assert torch.all(fused >= 0)


class TestPPLMConfig:
    def test_defaults_are_sane(self):
        cfg = PPLMConfig()
        assert cfg.step_size > 0
        assert cfg.num_iterations >= 0
        assert 0.0 <= cfg.gamma <= 1.0
        assert cfg.temperature == 0.0  # greedy by default — backward compatible


class TestPickNextToken:
    def test_zero_temperature_matches_argmax_on_logits(self):
        logits = torch.tensor([[1.0, 5.0, 2.0]])
        assert pick_next_token(logits, temperature=0.0).item() == 1

    def test_zero_temperature_matches_argmax_on_probs(self):
        probs = torch.tensor([[0.1, 0.7, 0.2]])
        assert pick_next_token(probs, temperature=0.0, is_probs=True).item() == 1

    def test_positive_temperature_samples_from_support(self):
        torch.manual_seed(0)
        logits = torch.tensor([[0.0, 0.0, 10.0]])  # index 2 overwhelmingly likely
        picks = {pick_next_token(logits, temperature=1.0).item() for _ in range(20)}
        assert picks <= {0, 1, 2}
        assert 2 in picks

    def test_never_samples_zero_probability_token(self):
        torch.manual_seed(0)
        logits = torch.full((1, 5), -1e9)
        logits[0, 3] = 10.0  # only index 3 has any real mass
        for _ in range(20):
            assert pick_next_token(logits, temperature=1.0).item() == 3

    def test_sampling_from_probs_respects_distribution_support(self):
        torch.manual_seed(0)
        probs = torch.tensor([[0.0, 0.5, 0.5]])
        for _ in range(20):
            assert pick_next_token(probs, temperature=1.0, is_probs=True).item() in {1, 2}

    def test_output_shape_matches_argmax(self):
        logits = torch.randn(1, 10)
        greedy = pick_next_token(logits, temperature=0.0)
        sampled = pick_next_token(logits, temperature=1.0)
        assert greedy.shape == sampled.shape == (1, 1)


class _FakeCacheLayer:
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values


class _FakeCache:
    def __init__(self, layers):
        self.layers = layers


class _FakeOutput:
    def __init__(self, logits, hidden_states, past_key_values):
        self.logits = logits
        self.hidden_states = hidden_states
        self.past_key_values = past_key_values


class _FakeModel:
    """Toy HF-model stand-in: hidden state = sum of the cached key/value tensors.

    Additive dependence on the cache is enough for gradients to flow from a
    loss computed off `hidden_states` back to a perturbation added to the
    cache, which is all `PPLMSteering.perturb_past` needs to be exercised
    without downloading a real model.
    """

    def __init__(self, hidden_size=4, vocab_size=6):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def __call__(self, input_ids, past_key_values=None, use_cache=True, output_hidden_states=False):
        batch = input_ids.shape[0]
        if past_key_values is None:
            keys = torch.zeros(batch, self.hidden_size)
            values = torch.zeros(batch, self.hidden_size)
        else:
            layer = past_key_values.layers[0]
            keys, values = layer.keys, layer.values

        hidden_vec = keys + values
        hidden = hidden_vec.unsqueeze(1)  # (batch, 1, hidden_size)
        logits = torch.zeros(batch, 1, self.vocab_size)
        new_cache = _FakeCache([_FakeCacheLayer(keys, values)])
        hidden_states = (hidden,) if output_hidden_states else None
        return _FakeOutput(logits=logits, hidden_states=hidden_states, past_key_values=new_cache)


class TestVGradientSteering:
    """PPLMSteering steering against a real (predicted) V(s) instead of BoW danger_tokens."""

    def _make_steerer(self, **config_kwargs):
        state_vars = [StateVar(name="x")]
        predictor = StatePredictor(hidden_size=4, state_vars=state_vars)
        v_fn = lambda state: state["x"] ** 2  # noqa: E731 — simple, always >=0, min at 0
        model = _FakeModel(hidden_size=4)
        config = PPLMConfig(step_size=0.1, num_iterations=5, **config_kwargs)
        steerer = PPLMSteering(
            model, tokenizer=None, config=config, state_predictor=predictor, differentiable_v_fn=v_fn
        )
        return steerer, predictor, v_fn

    def test_uses_v_gradient_true_when_both_supplied(self):
        steerer, _, _ = self._make_steerer()
        assert steerer.uses_v_gradient is True

    def test_uses_v_gradient_false_by_default(self):
        steerer = PPLMSteering(_FakeModel(), tokenizer=None)
        assert steerer.uses_v_gradient is False

    def test_perturb_past_reduces_predicted_v(self):
        steerer, predictor, v_fn = self._make_steerer()
        past = _FakeCache([_FakeCacheLayer(torch.ones(1, 4), torch.ones(1, 4))])
        last_token = torch.tensor([[0]])

        hidden_before = (past.layers[0].keys + past.layers[0].values).unsqueeze(1)[:, -1, :]
        v_before = v_fn(predictor.predict_dict(hidden_before)).item()

        new_past = steerer.perturb_past(past, last_token, danger_token_ids=[])
        hidden_after = (new_past.layers[0].keys + new_past.layers[0].values).unsqueeze(1)[:, -1, :]
        v_after = v_fn(predictor.predict_dict(hidden_after)).item()

        assert v_after < v_before

    def test_perturb_past_unchanged_when_no_signal_configured(self):
        model = _FakeModel(hidden_size=4)
        steerer = PPLMSteering(model, tokenizer=None, config=PPLMConfig(num_iterations=5))
        past = _FakeCache([_FakeCacheLayer(torch.ones(1, 4), torch.ones(1, 4))])
        last_token = torch.tensor([[0]])

        result = steerer.perturb_past(past, last_token, danger_token_ids=[])
        assert result is past


class TestMaskDangerTokens:
    def test_sets_negative_infinity_at_danger_ids(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        masked = mask_danger_tokens(logits, [1])
        assert masked[0, 1] == float("-inf")
        assert masked[0, 0] == 1.0
        assert masked[0, 2] == 3.0

    def test_empty_danger_ids_returns_logits_unchanged(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        masked = mask_danger_tokens(logits, [])
        assert torch.equal(masked, logits)

    def test_multiple_danger_ids_all_masked(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        masked = mask_danger_tokens(logits, [0, 2])
        assert masked[0, 0] == float("-inf")
        assert masked[0, 2] == float("-inf")
        assert masked[0, 1] == 2.0
        assert masked[0, 3] == 4.0


class TestHardMaskGenerateStep:
    """PPLMConfig(hard_mask=True) — the baseline scripts/baseline_comparison_study.py
    (Decision 10) compares gradient-based steering against: no gradient
    computation, just make the danger tokens impossible to select."""

    def test_never_selects_masked_token_on_first_call(self):
        model = _FakeModel(hidden_size=4, vocab_size=6)  # always returns all-zero logits
        steerer = PPLMSteering(model, tokenizer=None, config=PPLMConfig(hard_mask=True))
        input_ids = torch.tensor([[0]])
        next_id, _ = steerer.generate_step(input_ids, None, danger_token_ids=[0])
        assert next_id.item() != 0

    def test_never_selects_masked_token_on_subsequent_call(self):
        model = _FakeModel(hidden_size=4, vocab_size=6)
        steerer = PPLMSteering(model, tokenizer=None, config=PPLMConfig(hard_mask=True))
        input_ids = torch.tensor([[0]])
        next_id, past = steerer.generate_step(input_ids, None, danger_token_ids=[])
        next_id2, _ = steerer.generate_step(next_id, past, danger_token_ids=[0])
        assert next_id2.item() != 0

    def test_no_danger_ids_behaves_like_unmasked_greedy(self):
        model = _FakeModel(hidden_size=4, vocab_size=6)
        steerer = PPLMSteering(model, tokenizer=None, config=PPLMConfig(hard_mask=True))
        input_ids = torch.tensor([[0]])
        next_id, _ = steerer.generate_step(input_ids, None, danger_token_ids=[])
        assert next_id.item() == 0  # all-zero logits, argmax ties to index 0


class TestLastHiddenStateSideChannel:
    """generate_step should always populate last_hidden_state, regardless of steering signal."""

    def test_none_before_any_call(self):
        steerer = PPLMSteering(_FakeModel(hidden_size=4), tokenizer=None)
        assert steerer.last_hidden_state is None

    def test_set_on_first_call(self):
        steerer = PPLMSteering(_FakeModel(hidden_size=4), tokenizer=None)
        input_ids = torch.tensor([[0]])
        steerer.generate_step(input_ids, None, danger_token_ids=[])
        assert steerer.last_hidden_state is not None
        assert steerer.last_hidden_state.shape == (1, 4)

    def test_set_on_subsequent_call(self):
        steerer = PPLMSteering(_FakeModel(hidden_size=4), tokenizer=None)
        input_ids = torch.tensor([[0]])
        next_id, past = steerer.generate_step(input_ids, None, danger_token_ids=[])
        steerer.generate_step(next_id, past, danger_token_ids=[])
        assert steerer.last_hidden_state.shape == (1, 4)
