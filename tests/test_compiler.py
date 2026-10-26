"""Tests for the Lyra SymPy compiler (YAML → V(s))."""

import pytest
from pathlib import Path
from omni_lyra.left_brain.compiler import LyraCompiler

FIXTURE = Path(__file__).parent / "fixtures" / "sample_constraint.yaml"


@pytest.fixture
def policy():
    return LyraCompiler.from_yaml(FIXTURE)


def safe_state() -> dict:
    return {
        "bytes_out": 0,
        "cpu_usage": 0.5,
        "file_deletions": 0,
        "net_connections": 2,
    }


def violation_state() -> dict:
    return {
        "bytes_out": 20_000_000,   # 20MB > 10MB limit
        "cpu_usage": 0.95,         # > 0.85 limit
        "file_deletions": 3,       # > 0 limit
        "net_connections": 10,     # > 5 limit
    }


class TestCompilerBasics:
    def test_domain_name(self, policy):
        assert policy.domain == "security_research"

    def test_state_symbols_created(self, policy):
        assert "bytes_out" in policy.state_symbols
        assert "cpu_usage" in policy.state_symbols

    def test_thresholds_loaded(self, policy):
        assert policy.warn_threshold == 0.0
        assert policy.block_threshold == 50.0


class TestVEvaluation:
    def test_safe_state_v_is_zero(self, policy):
        v = policy.evaluate(safe_state())
        assert v == pytest.approx(0.0, abs=1e-9)

    def test_violation_state_v_is_positive(self, policy):
        v = policy.evaluate(violation_state())
        assert v > 0.0

    def test_v_non_negative(self, policy):
        for _ in range(10):
            # Even at safe state, V must never go below 0
            v = policy.evaluate(safe_state())
            assert v >= 0.0

    def test_components_sum_equals_v(self, policy):
        state = violation_state()
        v_total = policy.evaluate(state)
        components = policy.evaluate_components(state)
        assert sum(components.values()) == pytest.approx(v_total, rel=1e-6)

    def test_partial_violation(self, policy):
        state = {**safe_state(), "cpu_usage": 0.9}   # only CPU violated
        v = policy.evaluate(state)
        assert v > 0.0
        components = policy.evaluate_components(state)
        assert components["cpu_guard"] > 0.0
        assert components["no_data_exfiltration"] == pytest.approx(0.0, abs=1e-9)


class TestTorchV:
    """Torch-differentiable V(s) — requires the 'pplm' optional extra (torch)."""

    def test_matches_numeric_evaluate(self, policy):
        torch = pytest.importorskip("torch")
        v_fn = policy.to_torch()
        for state in (safe_state(), violation_state()):
            tensor_state = {k: torch.tensor(float(v)) for k, v in state.items()}
            torch_v = v_fn(tensor_state).item()
            assert torch_v == pytest.approx(policy.evaluate(state), rel=1e-5)

    def test_gradient_flows_to_state_vars(self, policy):
        torch = pytest.importorskip("torch")
        v_fn = policy.to_torch()
        state = {k: torch.tensor(float(v), requires_grad=True) for k, v in violation_state().items()}
        v = v_fn(state)
        v.backward()
        assert state["bytes_out"].grad is not None
        assert state["bytes_out"].grad.item() != 0.0

    def test_zero_grad_deep_in_safe_zone(self, policy):
        """ReLU's flat region (g(s) < 0) should carry zero gradient, matching evaluate()==0."""
        torch = pytest.importorskip("torch")
        v_fn = policy.to_torch()
        state = {k: torch.tensor(float(v), requires_grad=True) for k, v in safe_state().items()}
        v = v_fn(state)
        v.backward()
        assert state["cpu_usage"].grad.item() == pytest.approx(0.0, abs=1e-9)


class TestFromString:
    MINIMAL_YAML = """
domain: test
state_vars:
  - name: x
constants:
  x_max: 10
constraints:
  - name: x_limit
    expression: "x - x_max"
    weight: 1.0
warn_threshold: 0.0
block_threshold: 100.0
"""

    def test_inline_yaml(self):
        policy = LyraCompiler.from_string(self.MINIMAL_YAML)
        assert policy.evaluate({"x": 5}) == pytest.approx(0.0, abs=1e-9)
        assert policy.evaluate({"x": 15}) > 0.0


class TestPentestRoeDomain:
    """config/pentest_roe.yaml — same compiler, zero code changes, genuinely
    different domain (see docs/research/lyapunov-guided-decoding.md, the
    pentest ROE decision): confirms the domain-agnostic YAML->V(s) claim on
    something other than the sandbox-security domain every other test uses.
    """

    ROE_PATH = Path(__file__).parent.parent / "config" / "pentest_roe.yaml"

    @pytest.fixture
    def roe_policy(self):
        return LyraCompiler.from_yaml(self.ROE_PATH)

    def _safe_state(self) -> dict:
        return {
            "out_of_scope_contacts": 0,
            "service_disrupted": 0,
            "persistent_changes": 0,
            "privilege_escalation_level": 0,
            "credentials_exfiltrated": 0,
        }

    def test_domain_name(self, roe_policy):
        assert roe_policy.domain == "pentest_roe"

    def test_safe_state_is_zero(self, roe_policy):
        assert roe_policy.evaluate(self._safe_state()) == pytest.approx(0.0, abs=1e-9)

    def test_zero_tolerance_any_single_violation_blocks(self, roe_policy):
        """Every ROE constraint constant is 0 — a single occurrence of any
        violation must exceed block_threshold=1.0, matching real engagement
        contracts (violated or not, no "up to N times is fine")."""
        for var in self._safe_state():
            state = self._safe_state()
            state[var] = 1
            v = roe_policy.evaluate(state)
            assert v > roe_policy.block_threshold, f"{var}=1 should exceed block_threshold, got V={v}"

    def test_privilege_escalation_has_highest_weight(self, roe_policy):
        """Design intent (config/pentest_roe.yaml comments): privesc is the
        most severe ROE category — confirm its weight actually dominates."""
        components = {c.name: c.weight for c in roe_policy.spec.constraints}
        assert components["privilege_escalation"] == max(components.values())
