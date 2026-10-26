from __future__ import annotations
from pydantic import BaseModel, Field


class StateVar(BaseModel):
    name: str
    description: str = ""
    unit: str = ""


class Constraint(BaseModel):
    name: str
    expression: str          # SymPy-parseable string using state_var names + constant names
    weight: float = 1.0
    description: str = ""
    danger_tokens: list[str] = Field(default_factory=list)
    # Bag-of-words vocabulary associated with this constraint's violation, used
    # by the PPLM-style guided decoder (right_brain/pplm_steering.py) as a
    # differentiable proxy: suppressing these words' probability mass at the
    # LM head is the attribute-model signal, since the constraint expression
    # itself (a SymPy inequality over state vars) has no gradient path back
    # into token logits.


class ConstraintSpec(BaseModel):
    domain: str
    description: str = ""
    state_vars: list[StateVar]
    constants: dict[str, float] = Field(default_factory=dict)
    constraints: list[Constraint]
    warn_threshold: float = 0.0
    block_threshold: float = float("inf")
