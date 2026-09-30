import torch
import torch.nn as nn
from typing import Dict, List, Tuple


def _naive_post_projection_hook(head_idx, num_heads):
    """Reproduces this repo's original `HeadAblator` intervention: zeroes a
    head_dim-wide channel slice of the attention block's OUTPUT, i.e. AFTER
    `c_proj` has linearly mixed all heads together. Kept only so it can be
    directly compared against the correct pre-projection method (RQ0)."""

    def hook(module, inputs, output):
        attn_output = output[0].clone()
        head_dim = attn_output.shape[-1] // num_heads
        start = head_idx * head_dim
        end = start + head_dim
        attn_output[..., start:end] = 0
        return (attn_output, *output[1:])

    return hook


def _head_ablation_pre_hook(head_idx, num_heads):
    """Correct head ablation: zeroes the head_dim-wide slice of `c_proj`'s
    INPUT (the concatenated, not-yet-mixed per-head outputs), via a
    forward pre-hook on `c_proj`. Verified against the installed
    `transformers` GPT2Attention.forward source: `attn_output` is
    reshaped/merged across heads and THEN passed to `c_proj`, so
    `c_proj`'s input has exactly one contiguous head_dim-wide chunk per
    head, unmixed."""

    def hook(module, inputs):
        x = inputs[0]
        head_dim = x.shape[-1] // num_heads
        start = head_idx * head_dim
        end = start + head_dim
        x = x.clone()
        x[..., start:end] = 0
        return (x,) + tuple(inputs[1:])

    return hook


class ValidatedHeadAblator:
    """
    Head ablation with two selectable intervention points, used to test
    RQ0 (does the ablation implementation change the answer) and to
    compute full per-head causal-effect vectors for RQ1/RQ2.

    method="naive_post_projection": the original (invalid) intervention.
    method="head_ablation": the correct, pre-c_proj intervention.

    Only "head_ablation" is used for any causal claim in the paper;
    "naive_post_projection" is retained solely for the RQ0 comparison.
    """

    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_layers = model.config.num_hidden_layers
        self.num_heads = model.config.num_attention_heads

    def _register(self, layer_idx, head_idx, method):
        attn_layer = self.model.transformer.h[layer_idx].attn
        if method == "head_ablation":
            hook_fn = _head_ablation_pre_hook(head_idx, self.num_heads)
            return attn_layer.c_proj.register_forward_pre_hook(hook_fn)
        elif method == "naive_post_projection":
            hook_fn = _naive_post_projection_hook(head_idx, self.num_heads)
            return attn_layer.register_forward_hook(hook_fn)
        else:
            raise ValueError(f"Unknown ablation method: {method}")

    def compute_all_head_effects(
        self, questions: List[str], answers: List[str], method: str = "head_ablation"
    ) -> Tuple[List[float], Dict[Tuple[int, int], List[float]]]:
        """
        Computes, for EVERY (layer, head), the per-example teacher-forced
        log-probability of the gold continuation with that head ablated
        (one batched forward pass per head -- not per example, so this is
        cheap: num_layers * num_heads batched calls total).

        Returns (baseline_per_example, effects) where
        effects[(layer, head)] is a list of per-example ablated log-probs
        (same order as `questions`/`answers`). Callers compute
        delta = baseline - ablated as needed.
        """
        from src.metrics import logprob_of_continuation_batch

        baseline = logprob_of_continuation_batch(
            self.model, self.tokenizer, questions, answers, self.device
        )

        effects: Dict[Tuple[int, int], List[float]] = {}
        for layer_idx in range(self.num_layers):
            for head_idx in range(self.num_heads):
                handle = self._register(layer_idx, head_idx, method)
                try:
                    ablated = logprob_of_continuation_batch(
                        self.model, self.tokenizer, questions, answers, self.device
                    )
                finally:
                    handle.remove()
                effects[(layer_idx, head_idx)] = ablated

        return baseline, effects
