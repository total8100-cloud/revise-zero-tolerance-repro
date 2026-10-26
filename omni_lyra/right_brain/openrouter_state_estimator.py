"""Right Brain — OpenRouter-backed state estimator, frontier-model reference.

NOT a candidate deployment backend. This deliberately calls an external
hosted API (OpenRouter, routing to whichever provider the `model` string
names), which violates this project's own "zero external API calls at
runtime — local inference via Ollama or vLLM only" invariant (CLAUDE.md)
by design. It exists solely as a controlled, offline reference/upper-bound
comparison point for docs/research/paper2-draft.md Sections 6-7: does the
masking/miscalibration failure mode characterized there for local models
(4 Ollama models, 0.5B-3.8B, plus jeff's 400M GliFormer — see
docs/research/jev-state-estimator-prd.md) persist across genuinely
frontier-scale models from different labs, or is it a
capability/scale-specific artifact? This is never wired into
PentestSession or any live/runtime pipeline.

One adapter, not one per vendor: OpenRouter exposes every provider
(OpenAI, Anthropic, Qwen, DeepSeek, ...) through the same OpenAI-compatible
chat-completions wire format, so a single `model` field (e.g.
"openai/gpt-5.6-terra", "anthropic/claude-opus-5.5", "qwen/qwen3.8-max-prime",
"deepseek/deepseek-v4-pro") is the only thing that changes between runs —
matching this project's existing pattern (ModelStateEstimator(model=...)
swapping among Ollama models via one class).

Matches ModelStateEstimator's estimate() interface exactly so it drops
into the same calibration-benchmark scripts
(pentest_calibration_corpus.py, LyraCompiler.evaluate) with zero changes
to evaluation methodology. Reuses the identical prompt-construction logic
(`ModelStateEstimator._build_prompt`, `WORST_CASE_SYSTEM_PROMPT`) so the
comparison isolates the model backend, not a different prompt.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import httpx

from ..left_brain.schema import StateVar
from .claude_state_estimator import _strip_code_fence
from .state_estimator import ModelStateEstimator, StateEstimationError, WORST_CASE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class OpenRouterStateEstimator:
    """Predicts post-action state via any model OpenRouter routes to.

    Reference/upper-bound comparison only — see module docstring. Same
    public shape as ModelStateEstimator: estimate() takes and returns
    exactly what that class's does.
    """

    api_key: str
    model: str
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.0
    timeout: float = 60.0
    # Same production system prompt every Ollama-backed benchmark in this
    # project used — isolates the model backend as the only variable.
    system_prompt: str = WORST_CASE_SYSTEM_PROMPT

    _client: httpx.AsyncClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "content-type": "application/json",
            },
        )

    async def estimate(
        self,
        state_vars: list[StateVar],
        current_state: dict[str, float],
        action: str,
        temperature: float | None = None,
        history: list[dict] | None = None,
    ) -> dict[str, float]:
        """Same contract as ModelStateEstimator.estimate() — raises
        StateEstimationError on any failure rather than returning a
        silently optimistic guess."""
        prompt = ModelStateEstimator._build_prompt(state_vars, current_state, action, history)
        effective_temperature = self.temperature if temperature is None else temperature
        try:
            response = await self._client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "temperature": effective_temperature,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            response.raise_for_status()
            body = response.json()
            message = body["choices"][0]["message"]
            content_text = message["content"]
            if content_text is None:
                # A real, distinct outcome (not a parsing bug): the
                # provider's own content-moderation layer refused to
                # answer at all (native_finish_reason == "refusal"),
                # observed live for anthropic/claude-opus-5.5 on this
                # domain's own danger-adjacent phrasing. Surfaced as its
                # own StateEstimationError so callers/benchmarks can
                # count refusals separately from ordinary malformed
                # output — collapsing the two would hide a real, distinct
                # estimator-availability failure mode.
                refusal_reason = message.get("refusal") or "no content, no refusal reason given"
                raise StateEstimationError(
                    f"OpenRouter model {self.model} refused to answer: {refusal_reason}"
                )
            predicted_raw = json.loads(_strip_code_fence(content_text))
        except (httpx.HTTPError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise StateEstimationError(f"OpenRouter call failed ({self.model}): {exc}") from exc

        if not isinstance(predicted_raw, dict):
            raise StateEstimationError(f"{self.model} returned non-object JSON: {predicted_raw!r}")

        predicted: dict[str, float] = {}
        for sv in state_vars:
            value = predicted_raw.get(sv.name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                predicted[sv.name] = float(value)
            else:
                logger.warning(
                    "OpenRouterStateEstimator(%s) omitted/mispredicted %r (got %r) — holding current value",
                    self.model, sv.name, value,
                )
                predicted[sv.name] = current_state.get(sv.name, 0.0)
        return predicted

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OpenRouterStateEstimator":
        return self

    async def __aexit__(self, *_) -> None:
        await self.aclose()
