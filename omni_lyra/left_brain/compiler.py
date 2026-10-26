"""YAML → SymPy → V(s) Lyapunov energy function compiler.

V(s) = Σ w_i * max(0, g_i(s))²
where g_i(s) is the i-th constraint expression evaluated at state s.

V(s) ≥ 0 always; V(s) = 0 iff all constraints are satisfied.
"""

from __future__ import annotations

import yaml
import sympy as sp
from pathlib import Path
from typing import Callable

from .schema import ConstraintSpec, Constraint


class CompiledPolicy:
    """Holds the compiled V(s) function and metadata for a domain."""

    def __init__(
        self,
        spec: ConstraintSpec,
        state_symbols: dict[str, sp.Symbol],
        v_expr: sp.Expr,
        constraint_exprs: list[tuple[str, sp.Expr, float]],
    ) -> None:
        self.spec = spec
        self.state_symbols = state_symbols
        self.v_expr = v_expr                    # symbolic V(s)
        self.constraint_exprs = constraint_exprs  # [(name, g_i_expr, weight), ...]

        # Lambdify for fast numeric evaluation
        sym_list = list(state_symbols.values())
        self._v_fn: Callable[..., float] = sp.lambdify(sym_list, v_expr, modules="math")
        self._sym_order: list[str] = list(state_symbols.keys())

    def evaluate(self, state: dict[str, float]) -> float:
        """Compute V(state) numerically."""
        args = [state[k] for k in self._sym_order]
        result = self._v_fn(*args)
        return float(max(0.0, result))

    def evaluate_components(self, state: dict[str, float]) -> dict[str, float]:
        """Return per-constraint energy contributions for diagnostics."""
        sym_subs = {self.state_symbols[k]: v for k, v in state.items()}
        components: dict[str, float] = {}
        for name, g_expr, weight in self.constraint_exprs:
            val = float(g_expr.subs(sym_subs))
            components[name] = weight * max(0.0, val) ** 2
        return components

    def to_torch(self):
        """Return a torch-differentiable V(s): fn(state: dict[str, Tensor]) -> Tensor.

        Lambdifies the same `v_expr` used by `evaluate()` with the 'torch'
        printer instead of 'math', so `sp.Piecewise` (the hard ReLU) compiles
        to `torch.where` and gradients flow through it. Requires the `pplm`
        optional dependency group (`torch`) — imported lazily here so the
        base package (`evaluate()`, the SymPy-only runtime verifier) never
        needs torch installed.
        """
        import torch  # noqa: F401  (imported for sp.lambdify's 'torch' module resolution)

        sym_list = list(self.state_symbols.values())
        torch_fn = sp.lambdify(sym_list, self.v_expr, modules="torch")
        order = self._sym_order

        def v_fn(state: dict[str, "torch.Tensor"]):
            args = [state[k] for k in order]
            return torch_fn(*args)

        return v_fn

    @property
    def domain(self) -> str:
        return self.spec.domain

    @property
    def warn_threshold(self) -> float:
        return self.spec.warn_threshold

    @property
    def block_threshold(self) -> float:
        return self.spec.block_threshold


class LyraCompiler:
    """Compiles a YAML constraint file into a CompiledPolicy."""

    @classmethod
    def from_yaml(cls, path: str | Path) -> CompiledPolicy:
        raw = Path(path).read_text(encoding="utf-8")
        return cls.from_string(raw)

    @classmethod
    def from_string(cls, yaml_text: str) -> CompiledPolicy:
        data = yaml.safe_load(yaml_text)
        # Flatten numeric underscores (Python allows 10_485_760 in code but YAML parses as str)
        if "constants" in data:
            data["constants"] = {
                k: float(str(v).replace("_", "")) for k, v in data["constants"].items()
            }
        spec = ConstraintSpec(**data)
        return cls._compile(spec)

    @classmethod
    def _compile(cls, spec: ConstraintSpec) -> CompiledPolicy:
        # Create SymPy symbols for each state variable
        state_symbols: dict[str, sp.Symbol] = {
            sv.name: sp.Symbol(sv.name, real=True) for sv in spec.state_vars
        }

        # Constant substitution map
        const_subs: dict[str, float] = dict(spec.constants)

        constraint_exprs: list[tuple[str, sp.Expr, float]] = []
        v_expr: sp.Expr = sp.Integer(0)

        for c in spec.constraints:
            g_expr = cls._parse_expression(c.expression, state_symbols, const_subs)
            # V_i = w_i * ReLU(g_i)^2  — smooth, always ≥ 0
            relu_g = sp.Piecewise((g_expr, g_expr > 0), (sp.Integer(0), True))
            v_i = c.weight * relu_g**2
            constraint_exprs.append((c.name, g_expr, c.weight))
            v_expr = v_expr + v_i

        return CompiledPolicy(
            spec=spec,
            state_symbols=state_symbols,
            v_expr=v_expr,
            constraint_exprs=constraint_exprs,
        )

    @staticmethod
    def _parse_expression(
        expr_str: str,
        state_symbols: dict[str, sp.Symbol],
        constants: dict[str, float],
    ) -> sp.Expr:
        local_dict: dict = {**state_symbols}
        for k, v in constants.items():
            local_dict[k] = sp.Float(v)
        return sp.sympify(expr_str, locals=local_dict)
