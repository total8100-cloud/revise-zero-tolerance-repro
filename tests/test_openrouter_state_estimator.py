"""Tests for the OpenRouter-backed state estimator — frontier-model
reference comparison only (see module docstring in
omni_lyra/right_brain/openrouter_state_estimator.py for why this exists
and why it is never wired into any live/runtime pipeline).

Mocked at the httpx transport level (no real API calls, no real key
needed) — these verify request construction and response parsing against
the real OpenAI-compatible chat-completions wire shape OpenRouter uses
for every provider it routes to.
"""

import json

import httpx
import pytest

from omni_lyra.left_brain.schema import StateVar
from omni_lyra.right_brain.openrouter_state_estimator import OpenRouterStateEstimator
from omni_lyra.right_brain.state_estimator import StateEstimationError

STATE_VARS = [
    StateVar(name="out_of_scope_contacts", description="contact with an unauthorized host", unit="count"),
    StateVar(name="privilege_escalation_level", description="root access obtained", unit="count"),
]


def _estimator(handler, model: str = "openai/gpt-5.6-terra", api_key: str = "testkey") -> OpenRouterStateEstimator:
    est = OpenRouterStateEstimator(api_key=api_key, model=model)
    est._client = httpx.AsyncClient(
        base_url=est.base_url,
        transport=httpx.MockTransport(handler),
        headers={"Authorization": f"Bearer {api_key}", "content-type": "application/json"},
    )
    return est


def _chat_response(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "gen-test",
            "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


class TestRequestConstruction:
    @pytest.mark.asyncio
    async def test_sends_bearer_auth_and_model(self):
        seen_headers = {}
        seen_body = {}

        def handler(request):
            seen_headers.update(request.headers)
            seen_body.update(json.loads(request.content))
            return _chat_response('{"out_of_scope_contacts": 0.0, "privilege_escalation_level": 0.0}')

        est = _estimator(handler, model="deepseek/deepseek-v4-pro", api_key="sk-or-test")
        await est.estimate(
            STATE_VARS, {"out_of_scope_contacts": 0.0, "privilege_escalation_level": 0.0}, "ls"
        )
        assert seen_headers.get("authorization") == "Bearer sk-or-test"
        assert seen_body["model"] == "deepseek/deepseek-v4-pro"

    @pytest.mark.asyncio
    async def test_system_and_user_messages_both_present(self):
        seen_body = {}

        def handler(request):
            seen_body.update(json.loads(request.content))
            return _chat_response('{"out_of_scope_contacts": 0.0, "privilege_escalation_level": 0.0}')

        est = _estimator(handler)
        await est.estimate(
            STATE_VARS,
            {"out_of_scope_contacts": 0.0, "privilege_escalation_level": 0.0},
            "sudo /opt/app/backup_tool.sh x",
        )
        assert seen_body["messages"][0]["role"] == "system"
        from omni_lyra.right_brain.state_estimator import WORST_CASE_SYSTEM_PROMPT
        assert seen_body["messages"][0]["content"] == WORST_CASE_SYSTEM_PROMPT
        assert seen_body["messages"][1]["role"] == "user"
        assert "sudo /opt/app/backup_tool.sh x" in seen_body["messages"][1]["content"]


class TestSuccessfulPrediction:
    @pytest.mark.asyncio
    async def test_valid_json_used_verbatim(self):
        def handler(request):
            return _chat_response('{"out_of_scope_contacts": 0.05, "privilege_escalation_level": 0.87}')

        est = _estimator(handler)
        result = await est.estimate(
            STATE_VARS, {"out_of_scope_contacts": 0.0, "privilege_escalation_level": 0.0}, "curl foo"
        )
        assert result == {"out_of_scope_contacts": 0.05, "privilege_escalation_level": 0.87}

    @pytest.mark.asyncio
    async def test_strips_markdown_code_fence(self):
        def handler(request):
            return _chat_response(
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
            return _chat_response('{"out_of_scope_contacts": 0.2}')

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
            return _chat_response("I think this is probably safe.")

        est = _estimator(handler)
        with pytest.raises(StateEstimationError):
            await est.estimate(
                STATE_VARS, {"out_of_scope_contacts": 0.0, "privilege_escalation_level": 0.0}, "curl foo"
            )

    @pytest.mark.asyncio
    async def test_provider_refusal_raises_distinct_from_parse_error(self):
        """Observed live 2026-09-30: anthropic/claude-opus-5.5 via
        OpenRouter refused to answer on a dangerous-phrasing case
        (native_finish_reason="refusal", content=null) — this must not
        crash with an unhandled AttributeError on None.strip()."""

        def handler(request):
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "refusal": "This request triggered restrictions on violative cyber content.",
                            },
                            "finish_reason": "content_filter",
                        }
                    ]
                },
            )

        est = _estimator(handler, model="anthropic/claude-opus-5.5")
        with pytest.raises(StateEstimationError, match="refused"):
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
