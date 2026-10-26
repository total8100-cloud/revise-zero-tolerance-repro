"""Tests for clause-boundary splitting and GuidedExplorationAgent."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from omni_lyra.right_brain.guided_agent import GuidedExplorationAgent, split_clauses


class TestSplitClauses:
    def test_no_boundary_returns_single_clause(self):
        assert split_clauses("ls -la") == ["ls -la"]

    def test_pipe_boundary(self):
        clauses = split_clauses("cat file.txt | grep foo")
        assert clauses == ["cat file.txt |", " grep foo"]

    def test_semicolon_boundary(self):
        clauses = split_clauses("cd /tmp; rm -rf x")
        assert clauses == ["cd /tmp;", " rm -rf x"]

    def test_double_ampersand_boundary(self):
        clauses = split_clauses("mkdir foo && cd foo")
        assert clauses == ["mkdir foo &&", " cd foo"]

    def test_double_pipe_boundary(self):
        clauses = split_clauses("test -f x || touch x")
        assert clauses == ["test -f x ||", " touch x"]

    def test_newline_boundary(self):
        clauses = split_clauses("echo a\necho b")
        assert clauses == ["echo a\n", "echo b"]

    def test_multiple_boundaries(self):
        clauses = split_clauses("a; b | c && d")
        assert clauses == ["a;", " b |", " c &&", " d"]

    def test_roundtrip_preserves_original_text(self):
        for text in [
            "ls -la",
            "cat file.txt | grep foo",
            "cd /tmp; rm -rf x",
            "mkdir foo && cd foo || echo fail",
            "echo a\necho b\necho c",
            "",
        ]:
            assert "".join(split_clauses(text)) == text

    def test_empty_string_returns_empty_list(self):
        assert split_clauses("") == []


class _ScriptedFakeModel:
    """Ignores its actual input and emits a scripted token id at each call.

    Enough to drive GuidedExplorationAgent's generation loop deterministically
    without a real HF model — no danger_tokens/state_predictor are configured
    in these tests, so PPLMSteering's no-signal shortcut means each decoding
    step is exactly one forward call.
    """

    def __init__(self, script_ids, hidden_size=4, vocab_size=100):
        self.script_ids = script_ids
        self.call_count = 0
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def __call__(self, input_ids, past_key_values=None, use_cache=True, output_hidden_states=False):
        target_id = self.script_ids[min(self.call_count, len(self.script_ids) - 1)]
        self.call_count += 1
        batch = input_ids.shape[0]
        logits = torch.full((batch, 1, self.vocab_size), -10.0)
        logits[:, :, target_id] = 10.0
        hidden_states = (torch.randn(batch, 1, self.hidden_size),) if output_hidden_states else None
        return SimpleNamespace(logits=logits, hidden_states=hidden_states, past_key_values=(self.call_count,))


class _ScriptedFakeTokenizer:
    id_to_text = {0: "a", 1: "|", 2: "b", 99: ""}
    eos_token_id = 99

    def encode(self, prompt, return_tensors="pt"):
        return torch.tensor([[0]])

    def decode(self, ids, skip_special_tokens=True):
        return self.id_to_text[int(ids[0])]


class TestGuidedExplorationAgent:
    def test_generates_scripted_sequence(self):
        agent = GuidedExplorationAgent(
            model=_ScriptedFakeModel([0, 1, 2, 99]),
            tokenizer=_ScriptedFakeTokenizer(),
            constraints=[],
        )
        result = agent.propose_action("ignored prompt", clause_check=lambda text: True)
        assert result == "a|b"

    def test_clause_record_captured_at_boundary(self):
        agent = GuidedExplorationAgent(
            model=_ScriptedFakeModel([0, 1, 2, 99]),
            tokenizer=_ScriptedFakeTokenizer(),
            constraints=[],
        )
        agent.propose_action("ignored prompt", clause_check=lambda text: True)
        assert len(agent.clause_records) == 1
        assert agent.clause_records[0].text == "a|"
        assert agent.clause_records[0].hidden_state.shape == (1, 4)

    def test_clause_check_false_aborts_generation(self):
        agent = GuidedExplorationAgent(
            model=_ScriptedFakeModel([0, 1, 2, 99]),
            tokenizer=_ScriptedFakeTokenizer(),
            constraints=[],
        )
        result = agent.propose_action("ignored prompt", clause_check=lambda text: False)
        assert result == "a|"  # generation stopped right after the first clause boundary

    def test_clause_records_reset_between_calls(self):
        agent = GuidedExplorationAgent(
            model=_ScriptedFakeModel([0, 1, 2, 99]),
            tokenizer=_ScriptedFakeTokenizer(),
            constraints=[],
        )
        agent.propose_action("ignored prompt", clause_check=lambda text: True)
        assert len(agent.clause_records) == 1
        agent._steerer = agent._steerer.__class__(agent.model, agent.tokenizer, agent.pplm_config)
        agent.model.call_count = 0
        agent.propose_action("ignored prompt", clause_check=lambda text: True)
        assert len(agent.clause_records) == 1

    def test_newline_ends_generation_even_though_more_tokens_remain(self):
        """A model that keeps generating past a newline (e.g. echoing fake
        transcript output) must be cut off there, not run to max_tokens."""
        tokenizer = _ScriptedFakeTokenizer()
        tokenizer.id_to_text = {0: "a", 3: "\n", 2: "b", 99: ""}
        agent = GuidedExplorationAgent(
            model=_ScriptedFakeModel([0, 3, 2, 2, 2, 99]),  # "a", "\n", then it would keep going with "b"s
            tokenizer=tokenizer,
            constraints=[],
            max_tokens=10,
        )
        result = agent.propose_action("ignored prompt", clause_check=lambda text: True)
        assert result == "a"
        assert agent.model.call_count == 2  # stopped right after the newline token, no further steps
