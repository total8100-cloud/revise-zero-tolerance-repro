"""Tests for ResearchSession's guided-agent backend (loop.py).

Everything is mocked/faked — no real Docker, Ollama, or HF model download.
The main thing being exercised is the sync-callback-in-a-worker-thread <->
async-event-loop bridge (clause_check / run_coroutine_threadsafe) that
_run_guided relies on, since GuidedExplorationAgent.propose_action is
synchronous CPU-bound torch code but needs to call back into the async
ModelStateEstimator + LyraVerifier pipeline mid-generation.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from omni_lyra.left_brain.compiler import LyraCompiler
from omni_lyra.left_brain.verifier import LyraVerifier
from omni_lyra.loop import ResearchSession, _build_text_prompt
from omni_lyra.right_brain.state_estimator import ModelStateEstimator
from omni_lyra.right_brain.state_predictor import load_training_examples
from omni_lyra.sandbox.docker_env import SandboxObservation

FIXTURE = Path(__file__).parent / "fixtures" / "sample_constraint.yaml"


class TestBuildTextPrompt:
    def test_includes_domain_and_state(self):
        prompt = _build_text_prompt("test domain", {"x": 1}, [])
        assert "test domain" in prompt
        assert prompt.endswith("$")

    def test_includes_recent_history(self):
        history = [{"action": "ls", "observation": "file.txt"}]
        prompt = _build_text_prompt("d", {}, history)
        assert "$ ls" in prompt
        assert "file.txt" in prompt

    def test_only_keeps_last_three_history_entries(self):
        history = [{"action": f"cmd{i}", "observation": "o"} for i in range(6)]
        prompt = _build_text_prompt("d", {}, history)
        assert "cmd0" not in prompt
        assert "cmd5" in prompt


class TestClauseIsSafe:
    @pytest.fixture
    def session(self):
        policy = LyraCompiler.from_yaml(FIXTURE)
        return ResearchSession(domain_context="d", policy=policy)

    @pytest.mark.asyncio
    async def test_safe_prediction_returns_true(self, session):
        verifier = LyraVerifier(policy=session.policy)
        estimator = AsyncMock()
        current_state = {"bytes_out": 0, "cpu_usage": 0.1, "file_deletions": 0, "net_connections": 1}
        estimator.estimate.return_value = dict(current_state)

        assert await session._clause_is_safe("ls", current_state, estimator, verifier) is True

    @pytest.mark.asyncio
    async def test_block_prediction_returns_false(self, session):
        verifier = LyraVerifier(policy=session.policy)
        estimator = AsyncMock()
        current_state = {"bytes_out": 0, "cpu_usage": 0.1, "file_deletions": 0, "net_connections": 1}
        estimator.estimate.return_value = {
            "bytes_out": 0, "cpu_usage": 0.1, "file_deletions": 999, "net_connections": 1,
        }

        assert await session._clause_is_safe("rm -rf /", current_state, estimator, verifier) is False

    @pytest.mark.asyncio
    async def test_warn_prediction_still_returns_true(self, session):
        """Matches _verify_and_execute: only BLOCK stops the flow, WARN doesn't."""
        verifier = LyraVerifier(policy=session.policy)
        estimator = AsyncMock()
        current_state = {"bytes_out": 0, "cpu_usage": 0.1, "file_deletions": 0, "net_connections": 1}
        estimator.estimate.return_value = {
            "bytes_out": 0, "cpu_usage": 0.9, "file_deletions": 0, "net_connections": 1,  # mild WARN, not BLOCK
        }

        assert await session._clause_is_safe("stress --cpu 1", current_state, estimator, verifier) is True


class _ScriptedFakeModel:
    """Ignores its actual input, emits a scripted token id at each call.

    No danger_tokens/state_predictor are configured in these tests, so
    PPLMSteering's no-signal shortcut means each decoding step is exactly
    one forward call and past_key_values never needs to look Cache-like.
    """

    def __init__(self, script_ids, hidden_size=4, vocab_size=100):
        self.script_ids = script_ids
        self.call_count = 0
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.config = SimpleNamespace(n_embd=hidden_size)

    def __call__(self, input_ids, past_key_values=None, use_cache=True, output_hidden_states=False):
        target_id = self.script_ids[min(self.call_count, len(self.script_ids) - 1)]
        self.call_count += 1
        batch = input_ids.shape[0]
        logits = torch.full((batch, 1, self.vocab_size), -10.0)
        logits[:, :, target_id] = 10.0
        hidden_states = (torch.randn(batch, 1, self.hidden_size),) if output_hidden_states else None
        return SimpleNamespace(logits=logits, hidden_states=hidden_states, past_key_values=(self.call_count,))

    def eval(self):
        return self


class _ScriptedFakeTokenizer:
    id_to_text = {0: "a", 1: "|", 2: "b", 99: ""}
    eos_token_id = 99

    def encode(self, prompt, return_tensors="pt"):
        return torch.tensor([[0]])

    def decode(self, ids, skip_special_tokens=True):
        return self.id_to_text[int(ids[0])]


class _FakeSandbox:
    def __init__(self):
        self.commands_run: list[str] = []

    async def run(self, command, timeout=30):
        self.commands_run.append(command)
        return "[exit:0]\nok"

    async def observe(self):
        return SandboxObservation(cpu_usage=0.01, mem_used_bytes=0, net_bytes_out=0, file_deletions=0, net_connections=0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


class TestRunGuidedEndToEnd:
    @pytest.mark.asyncio
    async def test_generates_executes_and_logs_training_data(self, monkeypatch, tmp_path):
        policy = LyraCompiler.from_yaml(FIXTURE)

        fake_sandbox = _FakeSandbox()

        async def fake_create(image):
            return fake_sandbox

        monkeypatch.setattr("omni_lyra.loop.DockerSandbox.create", fake_create)

        fake_model = _ScriptedFakeModel([0, 1, 2, 99])  # "a", "|", "b", eos -> action "a|b"
        fake_tokenizer = _ScriptedFakeTokenizer()
        monkeypatch.setattr(
            transformers.AutoModelForCausalLM, "from_pretrained", classmethod(lambda cls, *a, **kw: fake_model)
        )
        monkeypatch.setattr(
            transformers.AutoTokenizer, "from_pretrained", classmethod(lambda cls, *a, **kw: fake_tokenizer)
        )

        async def fake_estimate(self, state_vars, current_state, action):
            return dict(current_state)  # always predicts "no change" -> SAFE

        monkeypatch.setattr(ModelStateEstimator, "estimate", fake_estimate)

        session = ResearchSession(
            domain_context="test domain",
            policy=policy,
            max_steps=1,
            use_guided_agent=True,
            training_log_dir=str(tmp_path),
        )
        records = await session.run()

        assert len(records) == 1
        assert records[0].proposed_action == "a|b"
        assert records[0].verdict == "SAFE"
        assert fake_sandbox.commands_run == ["mkdir -p /workspace", "a|b"]

        logged = load_training_examples(tmp_path)
        assert len(logged) == 1  # one clause boundary hit, at "a|"
        assert logged[0].hidden_state.shape == (1, 4)
