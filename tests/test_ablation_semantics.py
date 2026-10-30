"""Regression tests for the intervention semantics that the paper's RQ0
claim depends on.

These run on a tiny randomly-initialised GPT-2 so they need no download
and no cached weights: the properties under test are structural (which
tensor the hook edits, and which channels it leaves alone), not learned.
"""

import unittest
from unittest import mock

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from src.ablation import (
    ValidatedHeadAblator,
    _head_ablation_pre_hook,
    _head_mean_ablation_pre_hook,
    _naive_post_projection_hook,
)

LAYER, HEAD = 1, 2


def tiny_model():
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=64, n_positions=16, n_embd=32, n_layer=2, n_head=4
    )
    return GPT2LMHeadModel(config).eval()


def logits(model, ids):
    with torch.no_grad():
        return model(ids).logits.clone()


def capture_c_proj_input(model, layer, ids, extra_hook=None):
    """Runs a forward pass and returns what `c_proj` actually received.

    The capture hook is registered *after* `extra_hook` so it observes the
    input as modified by the intervention under test.
    """
    seen = {}
    c_proj = model.transformer.h[layer].attn.c_proj
    handles = []
    if extra_hook is not None:
        handles.append(c_proj.register_forward_pre_hook(extra_hook))

    def capture(module, inputs):
        seen["x"] = inputs[0].detach().clone()

    handles.append(c_proj.register_forward_pre_hook(capture))
    try:
        with torch.no_grad():
            model(ids)
    finally:
        for handle in handles:
            handle.remove()
    return seen["x"]


class PreProjectionInterventionTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.num_heads = self.model.config.num_attention_heads
        self.ids = torch.tensor([[1, 2, 3, 4, 5]])
        self.head_dim = self.model.config.n_embd // self.num_heads

    def test_zeroes_only_the_requested_head_slice_of_the_projection_input(self):
        """The core RQ0 claim: the valid hook removes exactly one head's
        contribution and perturbs nothing else."""
        clean = capture_c_proj_input(self.model, LAYER, self.ids)
        ablated = capture_c_proj_input(
            self.model,
            LAYER,
            self.ids,
            extra_hook=_head_ablation_pre_hook(HEAD, self.num_heads),
        )

        start, end = HEAD * self.head_dim, (HEAD + 1) * self.head_dim
        self.assertTrue(torch.all(ablated[..., start:end] == 0))
        self.assertTrue(torch.equal(ablated[..., :start], clean[..., :start]))
        self.assertTrue(torch.equal(ablated[..., end:], clean[..., end:]))
        # Guard against a vacuous pass if the slice were already zero.
        self.assertFalse(torch.all(clean[..., start:end] == 0))

    def test_ablation_changes_logits(self):
        baseline = logits(self.model, self.ids)
        handle = self.model.transformer.h[LAYER].attn.c_proj.register_forward_pre_hook(
            _head_ablation_pre_hook(HEAD, self.num_heads)
        )
        try:
            ablated = logits(self.model, self.ids)
        finally:
            handle.remove()

        self.assertFalse(torch.equal(baseline, ablated))

    def test_hook_removal_restores_the_exact_baseline(self):
        baseline = logits(self.model, self.ids)
        handle = self.model.transformer.h[LAYER].attn.c_proj.register_forward_pre_hook(
            _head_ablation_pre_hook(HEAD, self.num_heads)
        )
        handle.remove()

        self.assertTrue(torch.equal(baseline, logits(self.model, self.ids)))

    def test_naive_hook_leaves_the_per_head_projection_input_untouched(self):
        """Why the naive method cannot isolate a head: it edits the block's
        returned output, so `c_proj` still receives every head intact."""
        clean = capture_c_proj_input(self.model, LAYER, self.ids)

        attn = self.model.transformer.h[LAYER].attn
        handle = attn.register_forward_hook(
            _naive_post_projection_hook(HEAD, self.num_heads)
        )
        try:
            during_naive = capture_c_proj_input(self.model, LAYER, self.ids)
        finally:
            handle.remove()

        self.assertTrue(torch.equal(clean, during_naive))

    def test_naive_and_preprojection_interventions_disagree(self):
        attn = self.model.transformer.h[LAYER].attn

        handle = attn.c_proj.register_forward_pre_hook(
            _head_ablation_pre_hook(HEAD, self.num_heads)
        )
        try:
            valid = logits(self.model, self.ids)
        finally:
            handle.remove()

        handle = attn.register_forward_hook(
            _naive_post_projection_hook(HEAD, self.num_heads)
        )
        try:
            naive = logits(self.model, self.ids)
        finally:
            handle.remove()

        self.assertFalse(torch.equal(valid, naive))

    def test_ablator_registers_each_method_at_its_documented_module(self):
        ablator = ValidatedHeadAblator(self.model, tokenizer=None, device="cpu")
        attn = self.model.transformer.h[LAYER].attn

        handle = ablator._register(LAYER, HEAD, "head_ablation")
        try:
            self.assertIn(handle.id, attn.c_proj._forward_pre_hooks)
        finally:
            handle.remove()

        handle = ablator._register(LAYER, HEAD, "naive_post_projection")
        try:
            self.assertIn(handle.id, attn._forward_hooks)
        finally:
            handle.remove()

        with self.assertRaises(ValueError):
            ablator._register(LAYER, HEAD, "not_a_method")


class MeanAblationTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.num_heads = self.model.config.num_attention_heads
        self.ids = torch.tensor([[1, 2, 3, 4, 5]])
        self.head_dim = self.model.config.n_embd // self.num_heads

    def test_writes_exactly_the_reference_mean_into_the_head_slice(self):
        mean_vec = torch.arange(self.head_dim, dtype=torch.float32) + 0.5
        clean = capture_c_proj_input(self.model, LAYER, self.ids)
        ablated = capture_c_proj_input(
            self.model,
            LAYER,
            self.ids,
            extra_hook=_head_mean_ablation_pre_hook(HEAD, self.num_heads, mean_vec),
        )

        start, end = HEAD * self.head_dim, (HEAD + 1) * self.head_dim
        self.assertTrue(torch.equal(ablated[..., start:end], mean_vec.expand_as(ablated[..., start:end])))
        self.assertTrue(torch.equal(ablated[..., :start], clean[..., :start]))
        self.assertTrue(torch.equal(ablated[..., end:], clean[..., end:]))

    def test_hook_removal_restores_the_exact_baseline(self):
        baseline = logits(self.model, self.ids)
        handle = self.model.transformer.h[LAYER].attn.c_proj.register_forward_pre_hook(
            _head_mean_ablation_pre_hook(HEAD, self.num_heads, torch.ones(self.head_dim))
        )
        try:
            self.assertFalse(torch.equal(baseline, logits(self.model, self.ids)))
        finally:
            handle.remove()
        self.assertTrue(torch.equal(baseline, logits(self.model, self.ids)))

    def test_head_means_exclude_padding_positions(self):
        """Means are averaged over attended (non-padding) positions only."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 0, 0, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]])
        ablator = ValidatedHeadAblator(self.model, tokenizer=None, device="cpu")
        with mock.patch(
            "src.metrics._build_batch", return_value=(input_ids, attention_mask, None)
        ):
            means = ablator.compute_head_means(["q"], ["a"])

        seen = {}
        c_proj = self.model.transformer.h[LAYER].attn.c_proj
        handle = c_proj.register_forward_pre_hook(
            lambda m, inputs: seen.setdefault("x", inputs[0].detach().clone())
        )
        try:
            with torch.no_grad():
                self.model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            handle.remove()
        expected = seen["x"][attention_mask.bool()].mean(dim=0)
        unmasked = seen["x"].reshape(-1, seen["x"].shape[-1]).mean(dim=0)

        self.assertEqual(set(means), set(range(self.model.config.n_layer)))
        self.assertTrue(torch.allclose(means[LAYER], expected))
        self.assertFalse(torch.allclose(means[LAYER], unmasked))

    def test_mean_ablation_requires_means_and_registers_on_c_proj(self):
        ablator = ValidatedHeadAblator(self.model, tokenizer=None, device="cpu")
        with self.assertRaises(ValueError):
            ablator._register(LAYER, HEAD, "mean_ablation")
        means = {l: torch.zeros(self.model.config.n_embd) for l in range(2)}
        handle = ablator._register(LAYER, HEAD, "mean_ablation", means)
        try:
            self.assertIn(handle.id, self.model.transformer.h[LAYER].attn.c_proj._forward_pre_hooks)
        finally:
            handle.remove()


class Qwen2ArchitectureTests(unittest.TestCase):
    """The same intervention semantics on a Llama/Qwen2-style block, where the
    output projection is `self_attn.o_proj` (rotary embeddings, grouped-query
    attention, no projection bias)."""

    def setUp(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM

        torch.manual_seed(0)
        config = Qwen2Config(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=16,
        )
        self.model = Qwen2ForCausalLM(config).eval()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.ids = torch.tensor([[1, 2, 3, 4, 5]])
        self.o_proj = self.model.model.layers[LAYER].self_attn.o_proj

    def capture(self, extra_hook=None, module_hook=None):
        seen, handles = {}, []
        if extra_hook is not None:
            handles.append(self.o_proj.register_forward_pre_hook(extra_hook))
        if module_hook is not None:
            attn = self.model.model.layers[LAYER].self_attn
            handles.append(attn.register_forward_hook(module_hook))
        handles.append(self.o_proj.register_forward_pre_hook(
            lambda m, inputs: seen.setdefault("x", inputs[0].detach().clone())))
        try:
            with torch.no_grad():
                out = self.model(self.ids).logits.clone()
        finally:
            for h in handles:
                h.remove()
        return seen["x"], out

    def test_preprojection_ablation_zeroes_only_the_head_slice(self):
        clean, _ = self.capture()
        ablated, _ = self.capture(extra_hook=_head_ablation_pre_hook(HEAD, self.num_heads))
        start, end = HEAD * self.head_dim, (HEAD + 1) * self.head_dim
        self.assertTrue(torch.all(ablated[..., start:end] == 0))
        self.assertTrue(torch.equal(ablated[..., :start], clean[..., :start]))
        self.assertTrue(torch.equal(ablated[..., end:], clean[..., end:]))
        self.assertFalse(torch.all(clean[..., start:end] == 0))

    def test_naive_hook_changes_logits_but_not_the_projection_input(self):
        clean, clean_logits = self.capture()
        during, naive_logits = self.capture(
            module_hook=_naive_post_projection_hook(HEAD, self.num_heads))
        self.assertTrue(torch.equal(clean, during))
        self.assertFalse(torch.equal(clean_logits, naive_logits))

    def test_ablator_registers_at_o_proj(self):
        ablator = ValidatedHeadAblator(self.model, tokenizer=None, device="cpu")
        handle = ablator._register(LAYER, HEAD, "head_ablation")
        try:
            self.assertIn(handle.id, self.o_proj._forward_pre_hooks)
        finally:
            handle.remove()
        handle = ablator._register(LAYER, HEAD, "naive_post_projection")
        try:
            self.assertIn(handle.id, self.model.model.layers[LAYER].self_attn._forward_hooks)
        finally:
            handle.remove()


if __name__ == "__main__":
    unittest.main()
