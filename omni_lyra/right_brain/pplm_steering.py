"""Right Brain — PPLM-style gradient steering for Lyapunov-guided decoding.

Paper 1's WARN-zone revision, REVISE(x,a) = argmin_{j!=a} V(x,j), assumes a
small enumerable discrete action space. OMNI-Lyra's Right Brain proposes
free-form shell commands, so that revision rule has no direct target to
minimize over. This module extends the same "descend an energy" spirit into
token-by-token generation instead: gradient-descend the model's own hidden
state cache to suppress probability mass on each constraint's danger_tokens,
following Dathathri et al.'s PPLM (Plug-and-Play Language Models).

A purely symbolic constraint (SymPy inequality over sandbox state vars, see
left_brain/schema.py) has no gradient path into token logits. The bag-of-words
attribute model below sidesteps that by using the LM's own softmax over a
curated danger_tokens vocabulary as the differentiable proxy.

That bag-of-words proxy has no formal connection to the actual Lyapunov V(s)
— it is a stand-in, not V itself. `PPLMSteering` optionally accepts a
`state_predictor` (right_brain/state_predictor.py) + `differentiable_v_fn`
(from `CompiledPolicy.to_torch()`, see left_brain/compiler.py) pair; when
both are supplied, the attribute loss becomes the model's own predicted V(s)
— true gradient descent on the same energy function the runtime verifier
uses — instead of the danger_tokens stand-in. See
docs/research/lyapunov-guided-decoding.md for the full design rationale and
the current limitation (state_predictor needs real sandbox-execution data to
train properly; untrained/synthetically-trained predictors only demonstrate
the mechanism, not calibrated predictions).

Recent `transformers` versions (confirmed on 5.14.1) wrap `past_key_values`
in a `DynamicCache` object — a list of per-layer objects exposing `.keys`/
`.values` tensors — rather than the plain tuple-of-tuples the original PPLM
implementation (and GPT-2's older API) used. `_extract_layer_pairs`/
`_rebuild_cache` below normalize between that Cache-object form and the
legacy tuple form so the gradient math is agnostic to which one a given
model/transformers version hands back. Architectures whose cache doesn't
expose a `.layers` list of `.keys`/`.values` tensors at all (e.g. some
StaticCache variants) are not handled and would need a further adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache


@dataclass
class PPLMConfig:
    step_size: float = 0.02
    num_iterations: int = 3
    gamma: float = 0.9          # fusion weight toward the perturbed distribution
    grad_norm_eps: float = 1e-8
    temperature: float = 0.0    # 0.0 = greedy argmax (default, deterministic); >0 = sample
    hard_mask: bool = False     # baseline mode: -inf out danger_token_ids instead of gradient steering


def resolve_danger_token_ids(tokenizer, danger_tokens: Sequence[str]) -> list[int]:
    """Map each danger word/phrase to a HF tokenizer's single-token id, if any.

    Tries both the bare surface form and a leading-space variant, since BPE
    vocabularies (GPT-2 style) tokenize "curl" and " curl" differently and
    shell commands almost always appear after a space in generated text.
    Multi-token words are skipped — the bag-of-words loss below only operates
    on single next-token logits, matching PPLM's original formulation.
    """
    ids: set[int] = set()
    for word in danger_tokens:
        for variant in (word, f" {word}"):
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            if len(encoded) == 1:
                ids.add(encoded[0])
    return sorted(ids)


def bow_attribute_loss(logits: torch.Tensor, danger_token_ids: Sequence[int]) -> torch.Tensor:
    """log P(next token is in the danger bag), from raw logits at one position.

    Minimizing this via gradient descent on `past_key_values` is PPLM's
    mechanism, retargeted at Lyra's per-constraint danger_tokens instead of a
    fixed sentiment/topic bag.
    """
    if not danger_token_ids:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    probs = F.softmax(logits, dim=-1)
    danger_mass = probs[..., danger_token_ids].sum(dim=-1)
    return torch.log(danger_mass.clamp_min(1e-10))


def fuse_distributions(
    perturbed_logits: torch.Tensor,
    original_logits: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Geometric-mean fusion of perturbed and unperturbed next-token distributions.

    p_fused ∝ p_perturbed^gamma * p_original^(1-gamma), renormalized. Keeps
    generation close to the base model's fluent distribution while still
    pulling mass away from the attribute-loss direction, rather than fully
    replacing the distribution.
    """
    log_p_pert = F.log_softmax(perturbed_logits, dim=-1)
    log_p_orig = F.log_softmax(original_logits, dim=-1)
    fused_log = gamma * log_p_pert + (1.0 - gamma) * log_p_orig
    return F.softmax(fused_log, dim=-1)


def pick_next_token(distribution: torch.Tensor, temperature: float, is_probs: bool = False) -> torch.Tensor:
    """Argmax if temperature<=0 (deterministic, matches the original default
    behavior everywhere in this module); otherwise multinomial-sample.

    `distribution` is raw logits unless `is_probs=True` (fuse_distributions
    already returns a normalized probability tensor). Argmax-only decoding
    is why intervention_study.py's v1 saw 0/10 real-execution differences
    despite steering measurably shifting probability mass (Decision 8) —
    sampling is what actually lets a shifted-but-not-first-place probability
    occasionally get realized as a different generated token.
    """
    if temperature <= 0:
        return torch.argmax(distribution, dim=-1, keepdim=True)
    if is_probs:
        probs = distribution
        if temperature != 1.0:
            probs = F.softmax(torch.log(probs.clamp_min(1e-12)) / temperature, dim=-1)
    else:
        probs = F.softmax(distribution / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def mask_danger_tokens(logits: torch.Tensor, danger_token_ids: Sequence[int]) -> torch.Tensor:
    """Set danger_token_ids' logits to -inf, making them impossible to select.

    The simplest possible safety intervention: a hard vocabulary constraint,
    no gradient computation needed. This is `PPLMConfig.hard_mask` mode — the
    baseline that `scripts/baseline_comparison_study.py` (Decision 10)
    compares PPLM's gradient-based steering against, to answer "is the
    gradient machinery worth it over just banning the words outright."
    """
    if not danger_token_ids:
        return logits
    masked = logits.clone()
    masked[..., danger_token_ids] = float("-inf")
    return masked


def _extract_layer_pairs(past_key_values) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return per-layer (key, value) tensors regardless of Cache-object vs legacy-tuple form."""
    if hasattr(past_key_values, "layers"):
        return [(layer.keys, layer.values) for layer in past_key_values.layers]
    return list(past_key_values)


def _rebuild_cache(pairs: list[tuple[torch.Tensor, torch.Tensor]], like):
    """Rebuild a past_key_values object of the same kind as `like` from (key, value) pairs."""
    if hasattr(like, "layers"):
        cache = DynamicCache()
        for i, (k, v) in enumerate(pairs):
            cache.update(k, v, i)
        return cache
    return tuple(pairs)


class PPLMSteering:
    """Wraps a HF causal LM to steer generation away from danger, per step.

    Defaults to the danger_tokens bag-of-words proxy. Pass `state_predictor`
    and `differentiable_v_fn` to steer against the real (predicted) V(s)
    instead — see module docstring.
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: PPLMConfig | None = None,
        state_predictor=None,
        differentiable_v_fn=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or PPLMConfig()
        self.state_predictor = state_predictor
        self.differentiable_v_fn = differentiable_v_fn
        self.last_hidden_state: torch.Tensor | None = None
        # Populated by every generate_step() call (last decoder layer's hidden
        # state at the newly generated position, detached). Exists so callers
        # like GuidedExplorationAgent can log (hidden_state, ...) pairs for
        # later StatePredictor training without an extra forward pass.

    @property
    def uses_v_gradient(self) -> bool:
        return self.state_predictor is not None and self.differentiable_v_fn is not None

    def _attribute_loss(self, outputs, danger_token_ids: Sequence[int]) -> torch.Tensor:
        """Compute the scalar loss to gradient-descend, from whichever signal is configured."""
        if self.uses_v_gradient:
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            predicted_state = self.state_predictor.predict_dict(last_hidden)
            v = self.differentiable_v_fn(predicted_state)
            return v.mean() if v.dim() > 0 else v
        logits = outputs.logits[:, -1, :]
        return bow_attribute_loss(logits, danger_token_ids)

    def perturb_past(
        self,
        past_key_values,
        last_token_id: torch.Tensor,
        danger_token_ids: Sequence[int],
    ):
        """Gradient-descend `past_key_values` to lower the attribute loss.

        Returns a new past_key_values object, of the same kind (Cache object
        or legacy tuple) as the input, nudged toward lower attribute loss at
        the next step (see `_attribute_loss`). Returns the input unchanged if
        there is nothing to steer against (no danger vocabulary and no
        V-gradient configured) or iteration is disabled.
        """
        if self.config.num_iterations <= 0:
            return past_key_values
        if not self.uses_v_gradient and not danger_token_ids:
            return past_key_values

        layer_pairs = _extract_layer_pairs(past_key_values)
        perturbations = [
            [torch.zeros_like(t, requires_grad=True) for t in pair]
            for pair in layer_pairs
        ]

        for _ in range(self.config.num_iterations):
            perturbed_pairs = [
                tuple(t + p for t, p in zip(pair, layer_pert))
                for pair, layer_pert in zip(layer_pairs, perturbations)
            ]
            perturbed_cache = _rebuild_cache(perturbed_pairs, past_key_values)
            outputs = self.model(
                input_ids=last_token_id,
                past_key_values=perturbed_cache,
                use_cache=True,
                output_hidden_states=self.uses_v_gradient,
            )
            loss = self._attribute_loss(outputs, danger_token_ids)

            flat_params = [p for layer in perturbations for p in layer]
            grads = torch.autograd.grad(loss, flat_params)

            grad_iter = iter(grads)
            with torch.no_grad():
                for layer_pert in perturbations:
                    for i, p in enumerate(layer_pert):
                        g = next(grad_iter)
                        norm = g.norm() + self.config.grad_norm_eps
                        layer_pert[i] = (p - self.config.step_size * g / norm).requires_grad_(True)

        with torch.no_grad():
            new_pairs = [
                tuple((t + p).detach() for t, p in zip(pair, layer_pert))
                for pair, layer_pert in zip(layer_pairs, perturbations)
            ]
        return _rebuild_cache(new_pairs, past_key_values)

    def generate_step(
        self,
        input_ids: torch.Tensor,
        past_key_values,
        danger_token_ids: Sequence[int],
    ):
        """Run one steered decoding step.

        On the first call (`past_key_values is None`) there is nothing yet to
        perturb, so the prompt is forwarded once, unsteered, purely to build
        the initial cache. Every step after that steers the *previous* step's
        cache, then fuses the resulting logits with an unperturbed forward
        pass's logits before picking the next token — perturbation is not
        carried forward compounding across steps (PPLM found that unstable
        over longer generations).
        """
        if past_key_values is None:
            with torch.no_grad():
                out = self.model(input_ids=input_ids, use_cache=True, output_hidden_states=True)
            self.last_hidden_state = out.hidden_states[-1][:, -1, :].detach()
            logits = out.logits[:, -1, :]
            if self.config.hard_mask:
                logits = mask_danger_tokens(logits, danger_token_ids)
            next_token_id = pick_next_token(logits, self.config.temperature)
            return next_token_id, out.past_key_values

        with torch.no_grad():
            unperturbed = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
        self.last_hidden_state = unperturbed.hidden_states[-1][:, -1, :].detach()
        original_logits = unperturbed.logits[:, -1, :]
        new_past = unperturbed.past_key_values

        if self.config.hard_mask:
            masked_logits = mask_danger_tokens(original_logits, danger_token_ids)
            next_token_id = pick_next_token(masked_logits, self.config.temperature)
            return next_token_id, new_past

        if not danger_token_ids and not self.uses_v_gradient:
            next_token_id = pick_next_token(original_logits, self.config.temperature)
            return next_token_id, new_past

        last_token_id = input_ids[:, -1:]
        perturbed_past = self.perturb_past(past_key_values, last_token_id, danger_token_ids)

        with torch.no_grad():
            perturbed_out = self.model(
                input_ids=last_token_id, past_key_values=perturbed_past, use_cache=True
            )
        perturbed_logits = perturbed_out.logits[:, -1, :]

        fused = fuse_distributions(perturbed_logits, original_logits, self.config.gamma)
        next_token_id = pick_next_token(fused, self.config.temperature, is_probs=True)
        return next_token_id, new_past
