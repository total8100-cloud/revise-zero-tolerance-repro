"""Right Brain — clause-boundary guided exploration agent (PPLM steering + hard checks).

Unlike `ExplorationAgent` (agent.py), which talks to Ollama over HTTP and has
no access to model internals, this agent loads a HF causal LM directly
in-process so that `PPLMSteering` can perturb `past_key_values` between
tokens — Ollama's serving API exposes no gradient path, so true PPLM-style
steering is fundamentally incompatible with using it as the backend (see
docs/research/lyapunov-guided-decoding.md, Decision 3).

Generation runs token-by-token, steering continuously against the union of
all constraints' danger_tokens. Whenever a clause boundary is produced (a
shell metacharacter: |, ;, &&, &, or newline), generation pauses and the
completed text so far is handed to an injected `clause_check` callback,
which is expected to run the existing ModelStateEstimator + LyraVerifier
hard-gate pipeline — a periodic heavyweight check layered on top of the
continuous, cheap per-token steering signal. loop.py's SAFE/WARN/BLOCK gate
itself is not modified; this composes with it.

Each clause checkpoint also snapshots the LM's hidden state (via
`PPLMSteering.last_hidden_state`) into `self.clause_records`. This agent has
no way to know the *actual* post-execution state on its own — only whatever
executes the real command (loop.py's DockerSandbox) does — so it only
captures the (text, hidden_state) half of a StatePredictor training pair.
Once a caller learns the real outcome, pair it up and persist it with
`state_predictor.log_training_example` for later training. See
docs/research/lyapunov-guided-decoding.md, Decision 5.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import torch

from ..left_brain.schema import Constraint
from .pplm_steering import PPLMConfig, PPLMSteering, resolve_danger_token_ids

CLAUSE_BOUNDARY_RE = re.compile(r"(\|\||\||;|&&|&|\n)")

MAX_CLAUSE_TOKENS_DEFAULT = 64


def split_clauses(text: str) -> list[str]:
    """Split shell text into clauses on |, ;, &&, ||, &, and newline.

    Boundary characters stay attached to the end of the preceding clause, so
    `"".join(split_clauses(text)) == text` always holds.
    """
    parts = CLAUSE_BOUNDARY_RE.split(text)
    clauses: list[str] = []
    buf = ""
    for part in parts:
        buf += part
        if CLAUSE_BOUNDARY_RE.fullmatch(part):
            clauses.append(buf)
            buf = ""
    if buf:
        clauses.append(buf)
    return clauses


@dataclass
class ClauseRecord:
    """Half of a StatePredictor training pair, captured at one clause checkpoint.

    Missing the other half (the real post-execution state) on purpose — this
    agent has no access to sandbox execution outcomes, only whoever calls
    `propose_action` does. Pair `hidden_state` with the real observed state
    once known and pass both to `state_predictor.log_training_example`.
    """

    text: str
    hidden_state: torch.Tensor


@dataclass
class GuidedExplorationAgent:
    """Token-by-token PPLM-steered command generation with clause-level hard gates."""

    model: object
    tokenizer: object
    constraints: list[Constraint]
    pplm_config: PPLMConfig = field(default_factory=PPLMConfig)
    max_tokens: int = 256
    max_clause_tokens: int = MAX_CLAUSE_TOKENS_DEFAULT

    _steerer: PPLMSteering = field(init=False, repr=False)
    _danger_token_ids: list[int] = field(init=False, repr=False)
    clause_records: list[ClauseRecord] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._steerer = PPLMSteering(self.model, self.tokenizer, self.pplm_config)
        all_danger_words = [w for c in self.constraints for w in c.danger_tokens]
        self._danger_token_ids = resolve_danger_token_ids(self.tokenizer, all_danger_words)

    def propose_action(self, prompt: str, clause_check: Callable[[str], bool]) -> str:
        """Generate a full command, pausing at clause boundaries for a hard check.

        `clause_check(generated_so_far) -> bool` should return False to abort
        generation (treated the same as hitting BLOCK). Kept synchronous and
        as an injected callback rather than calling the async
        ModelStateEstimator directly, since this generation loop is
        synchronous CPU-bound torch code — callers running under asyncio
        should wrap the estimator call (e.g. via a loop-safe adapter) before
        passing it in here.

        Resets and repopulates `self.clause_records` — one entry per clause
        checkpoint reached (whether or not `clause_check` passed it) — so a
        caller that goes on to execute the accepted command can log training
        pairs for the StatePredictor afterward.

        A newline ends generation outright rather than just triggering a
        checkpoint like the other boundary characters do — this project's
        action convention is one shell command per action (see agent.py's
        "One command per response" system prompt), and without this a
        non-instruction-tuned model prompted with a shell-transcript style
        (see loop.py's _build_text_prompt) will keep generating past the
        first line into hallucinated fake command output, found running this
        for real against distilgpt2.
        """
        self.clause_records = []
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        past = None
        generated = ""
        clause_tokens = 0

        for _ in range(self.max_tokens):
            next_id, past = self._steerer.generate_step(input_ids, past, self._danger_token_ids)

            if next_id.item() == self.tokenizer.eos_token_id:
                break

            token_text = self.tokenizer.decode(next_id[0], skip_special_tokens=True)
            generated += token_text
            clause_tokens += 1
            input_ids = next_id

            boundary_match = CLAUSE_BOUNDARY_RE.search(token_text)
            if boundary_match or clause_tokens >= self.max_clause_tokens:
                if self._steerer.last_hidden_state is not None:
                    self.clause_records.append(
                        ClauseRecord(text=generated, hidden_state=self._steerer.last_hidden_state)
                    )
                if not clause_check(generated):
                    break
                if boundary_match and boundary_match.group() == "\n":
                    break
                clause_tokens = 0

        return generated.strip()
