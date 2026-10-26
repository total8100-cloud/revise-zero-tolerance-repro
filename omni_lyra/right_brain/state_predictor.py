"""Right Brain — differentiable state predictor (hidden state -> state vars).

Bridges the LM's hidden state at a generation position to Lyra's actual state
variables, so PPLM steering can gradient-descend the *real* V(s) (via
`CompiledPolicy.to_torch()`, see left_brain/compiler.py) instead of a
hand-picked bag-of-words proxy over danger_tokens. This is the differentiable
surrogate discussed in docs/research/lyapunov-guided-decoding.md — closing
the gap where the old bow_attribute_loss had no formal connection to the
actual Lyapunov energy.

Domain-agnostic by construction: driven by `ConstraintSpec.state_vars`
(names/order), no hardcoded variable names, matching the rest of the project.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from ..left_brain.schema import StateVar


class StatePredictor(nn.Module):
    """Small MLP: LM hidden state -> predicted value of each state var."""

    def __init__(self, hidden_size: int, state_vars: list[StateVar], hidden_dim: int = 64):
        super().__init__()
        self.state_var_names: list[str] = [sv.name for sv in state_vars]
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(state_vars)),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """hidden_state: (..., hidden_size) -> (..., num_state_vars)."""
        return self.net(hidden_state)

    def predict_dict(self, hidden_state: torch.Tensor) -> dict[str, torch.Tensor]:
        """Map the raw output vector back to {state_var_name: value}, in output order."""
        out = self.forward(hidden_state)
        return {name: out[..., i] for i, name in enumerate(self.state_var_names)}


@dataclass
class TrainingExample:
    """One (hidden_state, ground-truth post-action state) supervised pair.

    In production, `hidden_state` comes from a real GuidedExplorationAgent
    generation step and `actual_state` from the real sandbox observation
    after that action executed (see guided_agent.py's clause-check logging
    hook) — this dataclass has no opinion on where the data came from.
    """

    hidden_state: torch.Tensor   # shape: (hidden_size,)
    actual_state: dict[str, float]


def log_training_example(log_dir: str | Path, example: TrainingExample) -> Path:
    """Persist one (hidden_state, actual_state) pair to `log_dir` as its own file.

    One file per example (rather than one growing file) so concurrent
    ResearchSession runs can log independently without a shared-file lock.
    Returns the path written, keyed by a random id since example order
    doesn't matter for later training.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{uuid.uuid4().hex}.pt"
    torch.save(example, path)
    return path


def load_training_examples(log_dir: str | Path) -> list[TrainingExample]:
    """Load every example previously written by `log_training_example`."""
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return []
    return [torch.load(p, weights_only=False) for p in sorted(log_dir.glob("*.pt"))]


def train_state_predictor(
    predictor: StatePredictor,
    examples: list[TrainingExample],
    epochs: int = 100,
    lr: float = 1e-3,
) -> list[float]:
    """Supervised MSE regression: hidden_state -> actual_state.

    Returns the per-epoch loss history so callers can check convergence.
    Ordinary full-batch training loop — the only non-obvious part is that
    each example's target vector must be assembled in
    `predictor.state_var_names` order to line up with the network's output.
    """
    if not examples:
        raise ValueError("train_state_predictor requires at least one example")

    inputs = torch.stack([ex.hidden_state for ex in examples])
    targets = torch.stack(
        [
            torch.tensor(
                [ex.actual_state[name] for name in predictor.state_var_names],
                dtype=torch.float32,
            )
            for ex in examples
        ]
    )

    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history: list[float] = []
    for _ in range(epochs):
        optimizer.zero_grad()
        preds = predictor(inputs)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    return history
