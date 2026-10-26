"""Right Brain — Nemotron-Terminal exploration agent via Ollama REST API.

The agent receives:
  - domain_context : high-level research goal
  - sandbox_state  : current observable state from the Docker sandbox
  - history        : prior (action, observation) pairs

It returns a single proposed CLI action string.
"""

from __future__ import annotations

import httpx
import json
from dataclasses import dataclass, field
from typing import AsyncIterator


_SYSTEM_PROMPT = """\
You are an autonomous terminal research agent running inside a sandboxed Linux environment.
Your goal is to explore the given research domain by issuing terminal commands.

Rules:
- Output ONLY the next single shell command to execute. No explanation, no markdown.
- One command per response.
- If you believe the research goal is complete, output exactly: DONE
"""


@dataclass
class ExplorationAgent:
    """Async agent that streams a proposed CLI action from a local Ollama model."""

    model: str = "nemotron-mini"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 256
    # 60s was too tight in practice: a pentest session's Ollama calls can
    # race 3 containers concurrently apt-get installing (docker/pentest_
    # fixtures*'s setup scripts), and this project has already documented
    # >90s real-world contention under that exact load (see pentest_range.py's
    # own 180s readiness timeout for the same root cause) — hit a real
    # httpx.ReadTimeout here at 60s during adversarial_injection_study.py's
    # first live run.
    timeout: float = 120.0

    _client: httpx.AsyncClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.ollama_base_url,
            timeout=self.timeout,
        )

    async def propose_action(
        self,
        domain_context: str,
        sandbox_state: dict,
        history: list[dict],
    ) -> str:
        """Ask the model for the next terminal command.

        Returns the raw command string (e.g. "pip install pandas") or "DONE".
        """
        messages = self._build_messages(domain_context, sandbox_state, history)
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        response = await self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"].strip()

    async def propose_action_stream(
        self,
        domain_context: str,
        sandbox_state: dict,
        history: list[dict],
    ) -> AsyncIterator[str]:
        """Streaming variant — yields token chunks as they arrive."""
        messages = self._build_messages(domain_context, sandbox_state, history)
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        async with self._client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if token := chunk.get("message", {}).get("content"):
                    yield token
                if chunk.get("done"):
                    break

    @staticmethod
    def _build_messages(
        domain_context: str,
        sandbox_state: dict,
        history: list[dict],
    ) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

        user_init = (
            f"Research domain:\n{domain_context}\n\n"
            f"Current sandbox state:\n{json.dumps(sandbox_state, indent=2)}"
        )
        messages.append({"role": "user", "content": user_init})

        for entry in history:
            messages.append({"role": "assistant", "content": entry["action"]})
            messages.append({
                "role": "user",
                "content": f"Observation:\n{entry['observation']}",
            })

        return messages

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "ExplorationAgent":
        return self

    async def __aexit__(self, *_) -> None:
        await self.aclose()
