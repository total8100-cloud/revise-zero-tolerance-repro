"""Tests for the Jev/jeff-backed state estimator (Right Brain, PRD Item B).

Mocked at the httpx transport level (no real jeff server involved) — these
verify request construction and response parsing against the real wire
shape, confirmed live 2026-09-30 (see JevStateEstimator's module
docstring): top-level key is "answers", not "nouls" as first assumed from
the README's SDK example.
"""

import json

import httpx
import pytest

from omni_lyra.left_brain.schema import StateVar
from omni_lyra.right_brain.jev_state_estimator import JevStateEstimator
from omni_lyra.right_brain.state_estimator import StateEstimationError

STATE_VARS = [
    StateVar(name="out_of_scope_contacts", description="contact with an unauthorized host", unit="count"),
    StateVar(name="privilege_escalation_level", description="root access obtained", unit="count"),
]


def _estimator(handler, api_key: str = "devkey") -> JevStateEstimator:
    est = JevStateEstimator(api_key=api_key)
    est._client = httpx.AsyncClient(
        base_url=est.base_url,
        transport=httpx.MockTransport(handler),
        headers={"Authorization": f"Bearer {api_key}"},
    )
    return est


def _jeff_response(nouls: dict[str, float]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "gliformer-large-v1",
            "answers": {name: {"type": "noul", "noul": value} for name, value in nouls.items()},
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )


class TestRequestConstruction:
    @pytest.mark.asyncio
    async def test_posts_one_noul_question_per_state_var(self):
        seen_body = {}

        def handler(request):
            seen_body.update(json.loads(request.content))
            return _jeff_response({"out_of_scope_contacts": 0.1, "privilege_escalation_level": 0.9})

        est = _estimator(handler)
        await est.estimate(
            STATE_VARS,
            {"out_of_scope_contacts": 0.0, "privilege_escalation_level": 0.0},
            "sudo /opt/app/backup_tool.sh x",
        )
        assert set(seen_body["questions"].keys()) == {"out_of_scope_contacts", "privilege_escalation_level"}
        assert all(q["type"] == "noul" for q in seen_body["questions"].values())
        assert "sudo /opt/app/backup_tool.sh x" in seen_body["state"]

    @pytest.mark.asyncio
    async def test_sends_bearer_auth_header(self):
        seen_headers = {}

        def handler(request):
            seen_headers.update(request.headers)
            return _jeff_response({"out_of_scope_contacts": 0.0, "privilege_escalation_level": 0.0})

        est = _estimator(handler, api_key="testkey")
        await est.estimate(
            STATE_VARS, {"out_of_scope_contacts": 0.0, "privilege_escalation_level": 0.0}, "ls"
        )
        assert seen_headers.get("authorization") == "Bearer testkey"


class TestSuccessfulPrediction:
    @pytest.mark.asyncio
    async def test_valid_response_used_verbatim(self):
        def handler(request):
            return _jeff_response({"out_of_scope_contacts": 0.05, "privilege_escalation_level": 0.87})

        est = _estimator(handler)
        result = await est.estimate(
            STATE_VARS, {"out_of_scope_contacts": 0.0, "privilege_escalation_level": 0.0}, "curl foo"
        )
        assert result == {"out_of_scope_contacts": 0.05, "privilege_escalation_level": 0.87}

    @pytest.mark.asyncio
    async def test_missing_noul_holds_current_value(self):
        def handler(request):
            # response shape has "privilege_escalation_level" missing entirely
            return _jeff_response({"out_of_scope_contacts": 0.2})

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
    async def test_unexpected_response_shape_falls_back_for_every_var(self):
        """If a future jeff response shape differs from the verified one,
        every variable should fall back to holding current_state rather
        than crash or silently zero out."""

        def handler(request):
            return httpx.Response(200, json={"unexpected": "shape"})

        est = _estimator(handler)
        current = {"out_of_scope_contacts": 0.4, "privilege_escalation_level": 0.6}
        result = await est.estimate(STATE_VARS, current, "curl foo")
        assert result == current

    @pytest.mark.asyncio
    async def test_non_json_response_raises(self):
        def handler(request):
            return httpx.Response(200, text="not json")

        est = _estimator(handler)
        with pytest.raises(StateEstimationError):
            await est.estimate(
                STATE_VARS, {"out_of_scope_contacts": 0.0, "privilege_escalation_level": 0.0}, "curl foo"
            )
