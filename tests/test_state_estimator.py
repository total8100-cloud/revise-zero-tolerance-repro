"""Tests for the model-based post-action state estimator (Right Brain)."""

import json

import httpx
import pytest

from omni_lyra.left_brain.schema import StateVar
from omni_lyra.right_brain.state_estimator import ModelStateEstimator, StateEstimationError

STATE_VARS = [
    StateVar(name="bytes_out", description="bytes sent out", unit="bytes"),
    StateVar(name="cpu_usage", description="cpu ratio", unit="ratio"),
]


def _estimator(handler) -> ModelStateEstimator:
    est = ModelStateEstimator()
    est._client = httpx.AsyncClient(
        base_url=est.ollama_base_url,
        transport=httpx.MockTransport(handler),
    )
    return est


def _ollama_response(content_obj) -> httpx.Response:
    return httpx.Response(
        200,
        json={"message": {"content": json.dumps(content_obj)}},
    )


class TestSuccessfulPrediction:
    @pytest.mark.asyncio
    async def test_valid_json_used_verbatim(self):
        def handler(request):
            return _ollama_response({"bytes_out": 2048.0, "cpu_usage": 0.7})

        est = _estimator(handler)
        result = await est.estimate(STATE_VARS, {"bytes_out": 0.0, "cpu_usage": 0.1}, "curl foo")
        assert result == {"bytes_out": 2048.0, "cpu_usage": 0.7}

    @pytest.mark.asyncio
    async def test_missing_key_holds_current_value(self):
        def handler(request):
            return _ollama_response({"bytes_out": 2048.0})   # cpu_usage omitted

        est = _estimator(handler)
        current = {"bytes_out": 0.0, "cpu_usage": 0.42}
        result = await est.estimate(STATE_VARS, current, "curl foo")
        assert result["bytes_out"] == 2048.0
        assert result["cpu_usage"] == 0.42

    @pytest.mark.asyncio
    async def test_non_numeric_value_holds_current_value(self):
        def handler(request):
            return _ollama_response({"bytes_out": "a lot", "cpu_usage": 0.9})

        est = _estimator(handler)
        current = {"bytes_out": 5.0, "cpu_usage": 0.1}
        result = await est.estimate(STATE_VARS, current, "curl foo")
        assert result["bytes_out"] == 5.0
        assert result["cpu_usage"] == 0.9


class TestEstimateSamples:
    @pytest.mark.asyncio
    async def test_issues_n_requests_and_returns_n_results(self):
        calls = []

        def handler(request):
            calls.append(request)
            return _ollama_response({"bytes_out": 1.0, "cpu_usage": 0.2})

        est = _estimator(handler)
        results = await est.estimate_samples(
            STATE_VARS, {"bytes_out": 0.0, "cpu_usage": 0.1}, "curl foo", n_samples=3
        )
        assert len(calls) == 3
        assert len(results) == 3
        assert all(r == {"bytes_out": 1.0, "cpu_usage": 0.2} for r in results)

    @pytest.mark.asyncio
    async def test_temperature_override_reaches_request_body(self):
        seen_temperatures = []

        def handler(request):
            body = json.loads(request.content)
            seen_temperatures.append(body["options"]["temperature"])
            return _ollama_response({"bytes_out": 1.0, "cpu_usage": 0.2})

        est = _estimator(handler)
        await est.estimate_samples(
            STATE_VARS,
            {"bytes_out": 0.0, "cpu_usage": 0.1},
            "curl foo",
            n_samples=2,
            temperature=0.9,
        )
        assert seen_temperatures == [0.9, 0.9]

    @pytest.mark.asyncio
    async def test_single_failed_draw_aborts_whole_candidate(self):
        # All-or-nothing: a failed draw must not be replaced by current_state
        # (fail-open) nor kept in the sample set (majority fusion would let
        # it outvote a real detection). The whole candidate is aborted.
        call_count = {"n": 0}

        def handler(request):
            call_count["n"] += 1
            if call_count["n"] == 2:
                return httpx.Response(500, text="boom")
            return _ollama_response({"bytes_out": 1.0, "cpu_usage": 0.2})

        est = _estimator(handler)
        current = {"bytes_out": 9.0, "cpu_usage": 0.5}
        with pytest.raises(StateEstimationError, match="draw 2/3 failed"):
            await est.estimate_samples(STATE_VARS, current, "curl foo", n_samples=3)

    @pytest.mark.asyncio
    async def test_failed_draw_cannot_outvote_real_detection_in_majority(self):
        # Regression for the fail-open bug in multi-sample mode: failed draws
        # used to be substituted with current_state and then outvoted a real
        # detection under majority fusion. With all-or-nothing, a candidate
        # with any failed draw never reaches fusion at all.
        call_count = {"n": 0}

        def handler(request):
            call_count["n"] += 1
            if call_count["n"] > 1:
                return httpx.Response(500, text="boom")
            return _ollama_response({"bytes_out": 0.0, "cpu_usage": 1.0})

        est = _estimator(handler)
        with pytest.raises(StateEstimationError):
            await est.estimate_samples(STATE_VARS, {"bytes_out": 0.0, "cpu_usage": 0.0}, "x", n_samples=3)

    @pytest.mark.asyncio
    async def test_worst_case_policy_replaces_failed_draw_with_worst_case_vote(self):
        call_count = {"n": 0}

        def handler(request):
            call_count["n"] += 1
            if call_count["n"] == 2:
                return httpx.Response(500, text="boom")
            return _ollama_response({"bytes_out": 1.0, "cpu_usage": 0.2})

        est = _estimator(handler)
        results = await est.estimate_samples(
            STATE_VARS, {"bytes_out": 9.0, "cpu_usage": 0.5}, "curl foo", n_samples=3, on_failure="worst_case"
        )
        assert len(results) == 3
        assert results[1] == {"bytes_out": 1.0, "cpu_usage": 1.0}  # worst case: every var at 1.0
        assert results[0] == {"bytes_out": 1.0, "cpu_usage": 0.2}

    @pytest.mark.asyncio
    async def test_worst_case_policy_failures_cannot_outvote_real_detection(self):
        # One real flag + two failed draws: failures are flagging votes, so the flag stands.
        from omni_lyra.right_brain.confidence_fusion import fuse_majority

        call_count = {"n": 0}

        def handler(request):
            call_count["n"] += 1
            if call_count["n"] > 1:
                return httpx.Response(500, text="boom")
            return _ollama_response({"bytes_out": 0.0, "cpu_usage": 1.0})

        est = _estimator(handler)
        samples = await est.estimate_samples(
            STATE_VARS, {"bytes_out": 0.0, "cpu_usage": 0.0}, "x", n_samples=3, on_failure="worst_case"
        )
        fused = fuse_majority(samples)
        assert fused["cpu_usage"] >= 1.0

    @pytest.mark.asyncio
    async def test_unknown_on_failure_policy_is_rejected(self):
        est = _estimator(lambda request: _ollama_response({"bytes_out": 1.0, "cpu_usage": 0.2}))
        with pytest.raises(ValueError, match="unknown on_failure"):
            await est.estimate_samples(STATE_VARS, {"bytes_out": 0.0, "cpu_usage": 0.0}, "x", n_samples=1, on_failure="bogus")


class TestHistoryContext:
    """History is optional and, by default, absent — these confirm the
    opt-in path works and the default path is unaffected (see
    ModelStateEstimator.estimate's docstring for why this stays opt-in)."""

    @pytest.mark.asyncio
    async def test_no_history_prompt_unchanged(self):
        seen_prompt = {}

        def handler(request):
            body = json.loads(request.content)
            seen_prompt["text"] = body["messages"][1]["content"]
            return _ollama_response({"bytes_out": 1.0, "cpu_usage": 0.1})

        est = _estimator(handler)
        await est.estimate(STATE_VARS, {"bytes_out": 0.0, "cpu_usage": 0.1}, "curl foo")
        assert "Execution history" not in seen_prompt["text"]

    @pytest.mark.asyncio
    async def test_history_included_when_provided(self):
        seen_prompt = {}

        def handler(request):
            body = json.loads(request.content)
            seen_prompt["text"] = body["messages"][1]["content"]
            return _ollama_response({"bytes_out": 1.0, "cpu_usage": 0.1})

        est = _estimator(handler)
        history = [{"action": "cat /etc/passwd", "observation": "root:x:0:0..."}]
        await est.estimate(
            STATE_VARS, {"bytes_out": 0.0, "cpu_usage": 0.1}, "curl foo", history=history
        )
        assert "Execution history" in seen_prompt["text"]
        assert "cat /etc/passwd" in seen_prompt["text"]
        assert "root:x:0:0" in seen_prompt["text"]

    @pytest.mark.asyncio
    async def test_empty_history_list_omits_block(self):
        seen_prompt = {}

        def handler(request):
            body = json.loads(request.content)
            seen_prompt["text"] = body["messages"][1]["content"]
            return _ollama_response({"bytes_out": 1.0, "cpu_usage": 0.1})

        est = _estimator(handler)
        await est.estimate(
            STATE_VARS, {"bytes_out": 0.0, "cpu_usage": 0.1}, "curl foo", history=[]
        )
        assert "Execution history" not in seen_prompt["text"]


class TestSystemPromptOverride:
    """Overridable per-instance, defaulting to the production prompt — added
    for scripts/system_prompt_ablation_check.py."""

    @pytest.mark.asyncio
    async def test_default_uses_production_prompt(self):
        # A neutral-wording alternative (NEUTRAL_SYSTEM_PROMPT) was tested
        # as a default candidate on 2026-09-23 (paper2-draft.md Section
        # 7.3.6/7.3.7) but REJECTED after full-corpus validation showed it
        # cost nemotron-mini real recall with zero false-alarm benefit for
        # either model — the "worst case if unclear" wording remains the
        # default. This checks the CURRENT default dynamically, not a fixed
        # string, so it can't silently drift out of sync with
        # omni_lyra.right_brain.state_estimator._SYSTEM_PROMPT.
        seen_system = {}

        def handler(request):
            body = json.loads(request.content)
            seen_system["text"] = body["messages"][0]["content"]
            return _ollama_response({"bytes_out": 1.0, "cpu_usage": 0.1})

        est = _estimator(handler)
        await est.estimate(STATE_VARS, {"bytes_out": 0.0, "cpu_usage": 0.1}, "curl foo")
        from omni_lyra.right_brain.state_estimator import _SYSTEM_PROMPT as _current_default
        assert seen_system["text"] == _current_default
        assert "predict the worst plausible" in seen_system["text"]
        assert "assuming no change" in seen_system["text"]

    @pytest.mark.asyncio
    async def test_override_reaches_request(self):
        seen_system = {}

        def handler(request):
            body = json.loads(request.content)
            seen_system["text"] = body["messages"][0]["content"]
            return _ollama_response({"bytes_out": 1.0, "cpu_usage": 0.1})

        est = ModelStateEstimator(system_prompt="A custom test prompt.")
        est._client = httpx.AsyncClient(
            base_url=est.ollama_base_url, transport=httpx.MockTransport(handler)
        )
        await est.estimate(STATE_VARS, {"bytes_out": 0.0, "cpu_usage": 0.1}, "curl foo")
        assert seen_system["text"] == "A custom test prompt."


class TestCommandSubstitutionNormalization:
    """Mitigation 2 (Section 7.3.4-7.3.5, paper2-draft.md) — strips $(...)
    wrappers from what the estimator SEES, without touching the real action
    string used anywhere else."""

    def test_no_substitution_unchanged(self):
        from omni_lyra.right_brain.state_estimator import _normalize_command_substitutions
        action = 'curl -s "http://target:8080/health"'
        assert _normalize_command_substitutions(action) == action

    def test_strips_wrapper_keeps_content(self):
        from omni_lyra.right_brain.state_estimator import _normalize_command_substitutions
        action = 'curl -s "http://target:8080/diag?host=$(sudo%20backup.sh)"'
        result = _normalize_command_substitutions(action)
        assert result == 'curl -s "http://target:8080/diag?host=sudo%20backup.sh"'

    def test_handles_nested_parens(self):
        from omni_lyra.right_brain.state_estimator import _normalize_command_substitutions
        action = 'curl -s "http://target:8080/diag?host=$(echo $(whoami))"'
        result = _normalize_command_substitutions(action)
        assert "$(" not in result
        assert "whoami" in result

    def test_unbalanced_input_does_not_crash(self):
        from omni_lyra.right_brain.state_estimator import _normalize_command_substitutions
        action = 'curl -s "http://target:8080/diag?host=$(sudo backup.sh'  # missing close paren
        result = _normalize_command_substitutions(action)  # should not raise
        assert "sudo backup.sh" in result

    @pytest.mark.asyncio
    async def test_prompt_shows_unwrapped_action(self):
        seen_prompt = {}

        def handler(request):
            body = json.loads(request.content)
            seen_prompt["text"] = body["messages"][1]["content"]
            return _ollama_response({"bytes_out": 1.0, "cpu_usage": 0.1})

        est = _estimator(handler)
        action = 'curl -s "http://target:8080/diag?host=$(sudo%20backup.sh)"'
        await est.estimate(STATE_VARS, {"bytes_out": 0.0, "cpu_usage": 0.1}, action)
        assert "$(" not in seen_prompt["text"]
        assert "sudo%20backup.sh" in seen_prompt["text"]


class TestFailureModes:
    @pytest.mark.asyncio
    async def test_http_error_raises(self):
        def handler(request):
            return httpx.Response(500, text="boom")

        est = _estimator(handler)
        with pytest.raises(StateEstimationError):
            await est.estimate(STATE_VARS, {"bytes_out": 0.0, "cpu_usage": 0.1}, "curl foo")

    @pytest.mark.asyncio
    async def test_invalid_json_raises(self):
        def handler(request):
            return httpx.Response(200, json={"message": {"content": "not json"}})

        est = _estimator(handler)
        with pytest.raises(StateEstimationError):
            await est.estimate(STATE_VARS, {"bytes_out": 0.0, "cpu_usage": 0.1}, "curl foo")

    @pytest.mark.asyncio
    async def test_non_object_json_raises(self):
        def handler(request):
            return _ollama_response([1, 2, 3])

        est = _estimator(handler)
        with pytest.raises(StateEstimationError):
            await est.estimate(STATE_VARS, {"bytes_out": 0.0, "cpu_usage": 0.1}, "curl foo")
