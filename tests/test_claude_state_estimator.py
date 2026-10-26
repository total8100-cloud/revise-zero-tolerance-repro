"""Tests for the Claude (Anthropic API) state estimator — frontier-model
reference comparison only (see module docstring in
omni_lyra/right_brain/claude_state_estimator.py for why this exists and
why it is never wired into any live/runtime pipeline).

Mocked at the httpx transport level (no real API calls, no real key
needed) — these verify request construction and response parsing against
the real Anthropic Messages API wire shape.
"""

import json

import httpx
import pytest

from omni_lyra.left_brain.schema import StateVar
from omni_lyra.right_brain.claude_state_estimator import ClaudeStateEstimator
from omni_lyra.right_brain.state_estimator import StateEstimationError

STATE_VARS = [
    StateVar(name="out_of_scope_contacts", description="contact with an unauthorized host", unit="count"),
    StateVar(name="privilege_escalation_level", description="root access obtained", unit="count"),
]


def _estimator(handler, api_key: str = "testkey") -> ClaudeStateEstimator:
    est = ClaudeStateEstimator(api_key=api_key)
    est._client = httpx.AsyncClient(
        base_url=est.base_url,
        transport=httpx.MockTransport(handler),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    return est


def _messages_response(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "model": "claude-sonnet-5",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    )


class TestRequestConstruction:
    @pytest.mark.asyncio
    async def test_sends_api_key_header(self):
        seen_headers = {}

        def handler(request):
            seen_headers.update(request.headers)
            return _messages_response('{"out_of_scope_contacts": 0.0, "privilege_escalation_level": 0.0}')

        est = _estimator(handler, api_key="sk-ant-test123")
        await est.estimate(
            STATE_VARS, {"out_of_scope_contacts": 0.0, "privilege_escalation_level": 0.0}, "ls"
        )
        assert seen_headers.get("x-api-key") == "sk-ant-test123"

    @pytest.mark.asyncio
    async def test_uses_production_system_prompt(self):
        seen_body = {}

        def handler(request):
            seen_body.update(json.loads(request.content))
            return _messages_response('{"out_of_scope_contacts": 0.0, "privilege_escalation_level": 0.0}')

        est = _estimator(handler)
        await est.estimate(
            STATE_VARS, {"out_of_scope_contacts": 0.0, "privilege_escalation_level": 0.0}, "curl foo"
        )
        from omni_lyra.right_brain.state_estimator import WORST_CASE_SYSTEM_PROMPT
        assert seen_body["system"] == WORST_CASE_SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_action_reaches_prompt(self):
        seen_body = {}

        def handler(request):
            seen_body.update(json.loads(request.content))
            return _messages_response('{"out_of_scope_contacts": 0.0, "privilege_escalation_level": 0.0}')

        est = _estimator(handler)
        await est.estimate(
            STATE_VARS,
            {"out_of_scope_contacts": 0.0, "privilege_escalation_level": 0.0},
            "sudo /opt/app/backup_tool.sh x",
        )
        assert "sudo /opt/app/backup_tool.sh x" in seen_body["messages"][0]["content"]


class TestSuccessfulPrediction:
    @pytest.mark.asyncio
    async def test_valid_json_used_verbatim(self):
        def handler(request):
            return _messages_response('{"out_of_scope_contacts": 0.05, "privilege_escalation_level": 0.87}')

        est = _estimator(handler)
        result = await est.estimate(
            STATE_VARS, {"out_of_scope_contacts": 0.0, "privilege_escalation_level": 0.0}, "curl foo"
        )
        assert result == {"out_of_scope_contacts": 0.05, "privilege_escalation_level": 0.87}

    @pytest.mark.asyncio
    async def test_strips_markdown_code_fence(self):
        """Claude sometimes wraps JSON in ```json ... ``` despite being
        told not to — a real behavior Ollama's models essentially never
        show, worth handling rather than crashing on."""

        def handler(request):
            return _messages_response(
                '```json\n{"out_of_scope_contacts": 0.1, "privilege_escalation_level": 0.9}\n```'
            )

        est = _estimator(handler)
        result = await est.estimate(
            STATE_VARS, {"out_of_scope_contacts": 0.0, "privilege_escalation_level": 0.0}, "curl foo"
        )
        assert result == {"out_of_scope_contacts": 0.1, "privilege_escalation_level": 0.9}

    @pytest.mark.asyncio
    async def test_missing_key_holds_current_value(self):
        def handler(request):
            return _messages_response('{"out_of_scope_contacts": 0.2}')

        est = _estimator(handler)
        current = {"out_of_scope_contacts": 0.0, "privilege_escalation_level": 0.33}
        result = await est.estimate(STATE_VARS, current, "curl foo")
        assert result["out_of_scope_contacts"] == 0.2
        assert result["privilege_escalation_level"] == 0.33


class TestFailureModes:
    @pytest.mark.asyncio
    async def test_http_error_raises(self):
        def handler(request):
            return httpx.Response(500, text="boom")

        est = _estimator(handler)
        with pytest.raises(StateEstimationError):
            await est.estimate(
                STATE_VARS, {"out_of_scope_contacts": 0.0, "privilege_escalation_level": 0.0}, "curl foo"
            )

    @pytest.mark.asyncio
    async def test_non_json_text_raises(self):
        def handler(request):
            return _messages_response("I think this is probably safe.")

        est = _estimator(handler)
        with pytest.raises(StateEstimationError):
            await est.estimate(
                STATE_VARS, {"out_of_scope_contacts": 0.0, "privilege_escalation_level": 0.0}, "curl foo"
            )

    @pytest.mark.asyncio
    async def test_unexpected_body_shape_raises(self):
        def handler(request):
            return httpx.Response(200, json={"unexpected": "shape"})

        est = _estimator(handler)
        with pytest.raises(StateEstimationError):
            await est.estimate(
                STATE_VARS, {"out_of_scope_contacts": 0.0, "privilege_escalation_level": 0.0}, "curl foo"
            )
