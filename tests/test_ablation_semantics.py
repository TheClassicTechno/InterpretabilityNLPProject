"""Regression tests for the intervention semantics that the paper's RQ0
claim depends on.

These run on a tiny randomly-initialised GPT-2 so they need no download
and no cached weights: the properties under test are structural (which
tensor the hook edits, and which channels it leaves alone), not learned.
"""

import unittest

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from src.ablation import (
    ValidatedHeadAblator,
    _head_ablation_pre_hook,
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


if __name__ == "__main__":
    unittest.main()
