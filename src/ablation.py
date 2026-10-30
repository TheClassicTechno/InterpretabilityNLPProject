import torch
import torch.nn as nn
from typing import Dict, List, Tuple


def attention_modules(model, layer_idx):
    """(attention block, output projection) for one layer. The output
    projection's input is the concatenation of the per-head outputs in
    every supported architecture, so pre-projection hooks isolate heads.
    GPT-2 family: `transformer.h[l].attn` / `.c_proj`.
    Llama/Qwen2 family: `model.layers[l].self_attn` / `.o_proj`."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        attn = model.transformer.h[layer_idx].attn
        return attn, attn.c_proj
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        attn = model.model.layers[layer_idx].self_attn
        return attn, attn.o_proj
    raise ValueError(f"Unsupported architecture: {type(model).__name__}")


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


def _head_mean_ablation_pre_hook(head_idx, num_heads, mean_vec):
    """Mean ablation at the same pre-`c_proj` location as
    `_head_ablation_pre_hook`, but the head's slice is overwritten with a
    fixed reference mean (shape [head_dim]) instead of zero, so the
    replacement stays on the model's activation distribution."""

    def hook(module, inputs):
        x = inputs[0]
        head_dim = x.shape[-1] // num_heads
        start = head_idx * head_dim
        end = start + head_dim
        x = x.clone()
        x[..., start:end] = mean_vec.to(dtype=x.dtype, device=x.device)
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

    @torch.no_grad()
    def compute_head_means(self, questions: List[str], answers: List[str]) -> Dict[int, torch.Tensor]:
        """Per-layer mean of `c_proj`'s input over all non-padding token
        positions of the given (question, answer) set (one batched forward
        pass). Returns {layer_idx: tensor[hidden]}; head h's reference mean
        is the h-th head_dim-wide slice."""
        from src.metrics import _build_batch

        input_ids, attention_mask, _ = _build_batch(self.tokenizer, questions, answers, self.device)
        mask = attention_mask.bool()
        means: Dict[int, torch.Tensor] = {}
        handles = []

        def make_capture(layer_idx):
            def capture(module, inputs):
                means[layer_idx] = inputs[0][mask].mean(dim=0).detach().clone()
            return capture

        for layer_idx in range(self.num_layers):
            _, out_proj = attention_modules(self.model, layer_idx)
            handles.append(out_proj.register_forward_pre_hook(make_capture(layer_idx)))
        try:
            self.model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            for handle in handles:
                handle.remove()
        return means

    def _register(self, layer_idx, head_idx, method, means=None):
        attn_layer, out_proj = attention_modules(self.model, layer_idx)
        if method == "head_ablation":
            hook_fn = _head_ablation_pre_hook(head_idx, self.num_heads)
            return out_proj.register_forward_pre_hook(hook_fn)
        elif method == "mean_ablation":
            if means is None:
                raise ValueError("mean_ablation requires reference means")
            layer_mean = means[layer_idx]
            head_dim = layer_mean.shape[-1] // self.num_heads
            mean_vec = layer_mean[head_idx * head_dim:(head_idx + 1) * head_dim]
            hook_fn = _head_mean_ablation_pre_hook(head_idx, self.num_heads, mean_vec)
            return out_proj.register_forward_pre_hook(hook_fn)
        elif method == "naive_post_projection":
            hook_fn = _naive_post_projection_hook(head_idx, self.num_heads)
            return attn_layer.register_forward_hook(hook_fn)
        else:
            raise ValueError(f"Unknown ablation method: {method}")

    def compute_all_head_effects(
        self, questions: List[str], answers: List[str], method: str = "head_ablation",
        means: Dict[int, torch.Tensor] = None,
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
                handle = self._register(layer_idx, head_idx, method, means)
                try:
                    ablated = logprob_of_continuation_batch(
                        self.model, self.tokenizer, questions, answers, self.device
                    )
                finally:
                    handle.remove()
                effects[(layer_idx, head_idx)] = ablated

        return baseline, effects


def _head_resample_ablation_pre_hook(head_idx, num_heads, source_inputs):
    """Resample ablation at the same pre-`c_proj` location: the head's
    slice is overwritten, per example and per position, with the slice
    recorded on a *different* (corrupted/resampled) input of identical
    token length. `source_inputs` has the same shape as `c_proj`'s input
    ([batch, seq, hidden]). Unlike zero or mean ablation, the replacement
    is an on-distribution activation that carries no information about
    the clean example."""

    def hook(module, inputs):
        x = inputs[0]
        if x.shape != source_inputs.shape:
            raise ValueError(
                f"resample source shape {tuple(source_inputs.shape)} != input shape {tuple(x.shape)}; "
                "clean and source batches must be token-aligned"
            )
        head_dim = x.shape[-1] // num_heads
        start = head_idx * head_dim
        end = start + head_dim
        x = x.clone()
        x[..., start:end] = source_inputs[..., start:end].to(dtype=x.dtype, device=x.device)
        return (x,) + tuple(inputs[1:])

    return hook


@torch.no_grad()
def capture_out_proj_inputs(model, input_ids, attention_mask) -> Dict[int, torch.Tensor]:
    """{layer_idx: c_proj/o_proj input tensor [batch, seq, hidden]} for one batch."""
    captured: Dict[int, torch.Tensor] = {}
    handles = []

    def make_capture(layer_idx):
        def capture(module, inputs):
            captured[layer_idx] = inputs[0].detach().clone()
        return capture

    for layer_idx in range(model.config.num_hidden_layers):
        _, out_proj = attention_modules(model, layer_idx)
        handles.append(out_proj.register_forward_pre_hook(make_capture(layer_idx)))
    try:
        model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def register_head_intervention(model, layer_idx, head_idx, method, means=None, source=None):
    """Register one pre-projection intervention on (layer, head) and return
    the hook handle. method in {"zero", "mean", "resample"}; `means` is the
    {layer: mean vector} dict for "mean", `source` the {layer: tensor}
    dict from `capture_out_proj_inputs` for "resample"."""
    num_heads = model.config.num_attention_heads
    _, out_proj = attention_modules(model, layer_idx)
    if method == "zero":
        fn = _head_ablation_pre_hook(head_idx, num_heads)
    elif method == "mean":
        layer_mean = means[layer_idx]
        head_dim = layer_mean.shape[-1] // num_heads
        fn = _head_mean_ablation_pre_hook(
            head_idx, num_heads, layer_mean[head_idx * head_dim:(head_idx + 1) * head_dim])
    elif method == "resample":
        fn = _head_resample_ablation_pre_hook(head_idx, num_heads, source[layer_idx])
    else:
        raise ValueError(f"Unknown method: {method}")
    return out_proj.register_forward_pre_hook(fn)
