"""Right Brain — Claude (Anthropic API) state estimator, frontier-model reference.

NOT a candidate deployment backend. This deliberately calls an external
hosted API, which violates this project's own "zero external API calls at
runtime — local inference via Ollama or vLLM only" invariant (CLAUDE.md)
by design. It exists solely as a controlled, offline reference/upper-bound
comparison point for docs/research/paper2-draft.md Sections 6-7: does the
masking/miscalibration failure mode characterized there for local models
(4 Ollama models, 0.5B-3.8B, plus jeff's 400M GliFormer — see
docs/research/jev-state-estimator-prd.md) persist even in a frontier-scale
model, or is it a capability/scale-specific artifact? Either answer is
informative for the paper (see the experiment-log entry this produces) —
this is never wired into PentestSession or any live/runtime pipeline.

Matches ModelStateEstimator's estimate() interface exactly so it drops
into the same calibration-benchmark scripts
(pentest_calibration_corpus.py, LyraCompiler.evaluate) with zero changes
to evaluation methodology. Reuses the identical prompt-construction logic
(`ModelStateEstimator._build_prompt`, `WORST_CASE_SYSTEM_PROMPT`) from
state_estimator.py so the comparison isolates the model backend, not a
different prompt.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import httpx

from ..left_brain.schema import StateVar
from .state_estimator import ModelStateEstimator, StateEstimationError, WORST_CASE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_code_fence(text: str) -> str:
    """Claude sometimes wraps JSON in a ```json ... ``` fence despite being
    told not to (Ollama's models, prompted identically, essentially never
    do this — a real behavioral difference, not a bug to hide). Stripping
    the fence is pure text cleanup, not a judgment-quality change: it does
    not alter what values the model predicted, only how they're unwrapped
    for json.loads."""
    return _CODE_FENCE_PATTERN.sub("", text.strip()).strip()


@dataclass
class ClaudeStateEstimator:
    """Predicts post-action state via the Anthropic Messages API.

    Reference/upper-bound comparison only — see module docstring. Same
    public shape as ModelStateEstimator: estimate() takes and returns
    exactly what that class's does, so calibration-benchmark scripts don't
    need to know which backend they're scoring.
    """

    api_key: str
    model: str = "claude-sonnet-5"
    base_url: str = "https://api.anthropic.com"
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
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
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
                "/v1/messages",
                json={
                    "model": self.model,
                    "max_tokens": 512,
                    "temperature": effective_temperature,
                    "system": self.system_prompt,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
            body = response.json()
            content_text = body["content"][0]["text"]
            predicted_raw = json.loads(_strip_code_fence(content_text))
        except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise StateEstimationError(f"Claude API call failed: {exc}") from exc

        if not isinstance(predicted_raw, dict):
            raise StateEstimationError(f"Claude returned non-object JSON: {predicted_raw!r}")

        predicted: dict[str, float] = {}
        for sv in state_vars:
            value = predicted_raw.get(sv.name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                predicted[sv.name] = float(value)
            else:
                logger.warning(
                    "ClaudeStateEstimator omitted/mispredicted %r (got %r) — holding current value",
                    sv.name, value,
                )
                predicted[sv.name] = current_state.get(sv.name, 0.0)
        return predicted

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "ClaudeStateEstimator":
        return self

    async def __aexit__(self, *_) -> None:
        await self.aclose()
