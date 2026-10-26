"""Right Brain — Jev-style "System One" decision-model state estimator.

PRD Item B (docs/research/jev-state-estimator-prd.md): an alternative
backend for the same role `ModelStateEstimator` (state_estimator.py) plays —
predict the post-action value of each ROE state variable — but via a
schema-constrained typed-decision model instead of a generic chat LLM
prompted to emit JSON. Matches `ModelStateEstimator`'s public interface
exactly (`estimate()` signature, `StateEstimationError`) so it drops into
`PentestSession`/`pentest_roe_scenario.py` as a straight substitute.

Talks to a jeff server (github.com/logan-markewich/jeff), a self-hosted
open replica of TypeSafe's Jev API — NOT the official hosted Jev, which is
closed-weight and API-only and therefore out of scope for this project's
"zero external API calls at runtime" invariant (CLAUDE.md); see the PRD for
why jeff specifically was chosen. Raw httpx is used (matching
`ModelStateEstimator`'s existing style) rather than the `typesafe-sdk`
package, to avoid adding a new dependency to this project's own environment
for what is, at the wire level, a plain JSON POST.

All 5 of this project's ROE state variables are already treated as sticky
binary flags (see sandbox/pentest_range.py's module docstring), which maps
directly onto jeff/Jev's `noul` question type (a calibrated yes/no
probability) — one noul question per state variable, no domain-specific
redesign needed.

VERIFICATION STATUS (updated 2026-09-30 against a live server): the
request shape (POST /v1/systemone, {state, model, questions}) was correct.
The response shape, however, was NOT: the README's Python-SDK example
(`result.nouls["billing"].noul`) implied a raw-JSON key of "nouls", but a
real live call returns {"model": ..., "answers": {"<name>": {"type":
"noul", "noul": <float>}}, "usage": {...}} — the top-level key is
"answers", not "nouls" (the SDK apparently renames this field when
wrapping the raw response into its own object; the raw wire format does
not match the SDK's own attribute name). Confirmed via a direct curl
smoke test against a live jeff server (GliFormer backend) before trusting
any benchmark result — this is exactly the class of fidelity-gap bug this
project has hit before (assumed vs. real shape), and running the full
32-case benchmark against the wrong key would have silently produced an
all-fallback, uninformatively bad "jeff can't do this" result rather than
a real measurement. `_parse_response` still falls back defensively
(holding `current_state` with a logged warning) if a future jeff version
changes the shape again, but the key it now reads is verified correct.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

from ..left_brain.schema import StateVar
from .state_estimator import StateEstimationError

logger = logging.getLogger(__name__)


@dataclass
class JevStateEstimator:
    """Predicts post-action state via a jeff (self-hosted Jev-replica) server.

    Same public shape as `ModelStateEstimator` — `estimate()` takes and
    returns exactly what that class's does — so the two are interchangeable
    at every call site (`PentestSession`, `pentest_roe_scenario.py`).
    """

    base_url: str = "http://localhost:8000"
    api_key: str = "devkey"
    # jeff's README example uses "jev-latest" as the model alias jeff itself
    # answers to (JEFF_MODEL_ALIASES defaults to "jev-latest,jev") — this is
    # NOT a request to the real hosted Jev; it's the alias jeff's own server
    # accepts for its local GliFormer backend.
    model: str = "jev-latest"
    timeout: float = 30.0

    _client: httpx.AsyncClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    async def estimate(
        self,
        state_vars: list[StateVar],
        current_state: dict[str, float],
        action: str,
        temperature: float | None = None,
    ) -> dict[str, float]:
        """Return the predicted post-action state, keyed by state variable name.

        `temperature` is accepted for interface parity with
        `ModelStateEstimator.estimate()` but ignored — jeff's calibration
        temperature (JEFF_TEMPERATURE, default 3.2) is a server-side
        deployment setting, not a per-request sampling parameter; there is
        no per-call analogue to a chat model's generation temperature here.

        Raises StateEstimationError if the server is unreachable or returns
        nothing usable — same contract as ModelStateEstimator.estimate(), so
        callers (including estimate_samples()-style fallback logic) behave
        identically regardless of which estimator backend is in use.
        """
        state_text = self._build_state_text(current_state, action)
        questions = {
            sv.name: {
                "type": "noul",
                "instructions": (
                    f"Will the proposed command cause '{sv.name}' "
                    f"({sv.description or 'no description'}) to become true/1 "
                    "immediately after it finishes executing?"
                ),
            }
            for sv in state_vars
        }

        try:
            response = await self._client.post(
                "/v1/systemone",
                json={"state": state_text, "model": self.model, "questions": questions},
            )
            response.raise_for_status()
            raw = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise StateEstimationError(f"jeff call failed: {exc}") from exc

        return self._parse_response(raw, state_vars, current_state)

    @staticmethod
    def _build_state_text(current_state: dict[str, float], action: str) -> str:
        """jeff's `state` field is free text (unlike ModelStateEstimator's
        structured prompt) — it's the classification input, not a chat
        message, so this stays a plain description rather than a system/user
        message pair."""
        current_lines = "; ".join(f"{k}={v}" for k, v in current_state.items())
        return (
            f"Current sandbox state: {current_lines}. "
            f"Proposed command about to execute: {action!r}"
        )

    @staticmethod
    def _parse_response(
        raw: dict,
        state_vars: list[StateVar],
        current_state: dict[str, float],
    ) -> dict[str, float]:
        """See module docstring's VERIFICATION STATUS note — the response's
        top-level key is "answers" (confirmed live), not "nouls" (the SDK's
        own attribute name, which does not match the raw wire format)."""
        answers = raw.get("answers", {}) if isinstance(raw, dict) else {}
        predicted: dict[str, float] = {}
        for sv in state_vars:
            entry = answers.get(sv.name) if isinstance(answers, dict) else None
            value = entry.get("noul") if isinstance(entry, dict) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                predicted[sv.name] = float(value)
            else:
                logger.warning(
                    "JevStateEstimator: no usable noul for %r in response %r — "
                    "holding current value",
                    sv.name, raw,
                )
                predicted[sv.name] = current_state.get(sv.name, 0.0)
        return predicted

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "JevStateEstimator":
        return self

    async def __aexit__(self, *_) -> None:
        await self.aclose()
