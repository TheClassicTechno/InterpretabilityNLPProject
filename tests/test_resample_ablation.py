"""Tests for the pre-projection resample-ablation hook and the IOI dataset."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ablation import _head_resample_ablation_pre_hook  # noqa: E402


class TestResampleHook(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.num_heads, self.head_dim = 4, 3
        self.x = torch.randn(2, 5, self.num_heads * self.head_dim)
        self.src = torch.randn(2, 5, self.num_heads * self.head_dim)

    def test_only_target_head_slice_changes(self):
        hook = _head_resample_ablation_pre_hook(2, self.num_heads, self.src)
        (out,) = hook(None, (self.x,))
        s, e = 2 * self.head_dim, 3 * self.head_dim
        self.assertTrue(torch.equal(out[..., s:e], self.src[..., s:e]))
        self.assertTrue(torch.equal(out[..., :s], self.x[..., :s]))
        self.assertTrue(torch.equal(out[..., e:], self.x[..., e:]))

    def test_resampling_from_itself_is_identity(self):
        hook = _head_resample_ablation_pre_hook(1, self.num_heads, self.x)
        (out,) = hook(None, (self.x,))
        self.assertTrue(torch.equal(out, self.x))

    def test_input_not_mutated_in_place(self):
        before = self.x.clone()
        hook = _head_resample_ablation_pre_hook(0, self.num_heads, self.src)
        hook(None, (self.x,))
        self.assertTrue(torch.equal(self.x, before))

    def test_shape_mismatch_fails_loudly(self):
        hook = _head_resample_ablation_pre_hook(0, self.num_heads, self.src[:, :4])
        with self.assertRaises(ValueError):
            hook(None, (self.x,))


class TestIOIDataset(unittest.TestCase):
    def test_alignment_and_answer_roles(self):
        from transformers import AutoTokenizer
        from src.dataset import get_ioi_tasks
        tok = AutoTokenizer.from_pretrained("gpt2")
        exs = get_ioi_tasks(tok, n=60, seed=0)
        for e in exs:
            self.assertEqual(len(tok.encode(e["prompt"])), len(tok.encode(e["abc_prompt"])))
            self.assertEqual(len(tok.encode(e["io"])), 1)
            self.assertEqual(len(tok.encode(e["s"])), 1)
            # IO appears once, S twice in the clean prompt; S appears once in the ABC prompt
            self.assertEqual(e["prompt"].count(e["io"].strip()), 1)
            self.assertEqual(e["prompt"].count(e["s"].strip()), 2)
            self.assertEqual(e["abc_prompt"].count(e["s"].strip()), 1)


if __name__ == "__main__":
    unittest.main()


class TestResampleSourceMatching(unittest.TestCase):
    """Source matching for resample ablation on the copying task
    (neurips/camera_ready_design.md, C3)."""

    def test_same_signature_same_split_never_self(self):
        import numpy as np
        from src.ablation import match_resample_sources
        sigs = [(1, 1), (2, 1), (1, 1), (2, 1), (1, 1), (2, 1), (1, 1), (2, 1)]
        split = [0, 0, 0, 0, 1, 1, 1, 1]
        src = match_resample_sources(sigs, split, np.random.default_rng(0))
        for i, j in enumerate(src):
            self.assertIsNotNone(j)
            self.assertNotEqual(i, j)
            self.assertEqual(sigs[i], sigs[j])
            self.assertEqual(split[i], split[j])

    def test_unmatched_example_gets_none(self):
        import numpy as np
        from src.ablation import match_resample_sources
        sigs = [(1,), (1,), (3,), (1,)]
        split = [0, 0, 0, 1]
        src = match_resample_sources(sigs, split, np.random.default_rng(0))
        self.assertIsNone(src[2])  # no other (3,) anywhere
        self.assertIsNone(src[3])  # only (1,) in its split is itself
        self.assertEqual(sorted([src[0], src[1]]), [0, 1])

    def test_deterministic_given_seed(self):
        import numpy as np
        from src.ablation import match_resample_sources
        sigs = [(1,)] * 10
        split = [0] * 10
        a = match_resample_sources(sigs, split, np.random.default_rng(2))
        b = match_resample_sources(sigs, split, np.random.default_rng(2))
        self.assertEqual(a, b)


class TestCopyingSignature(unittest.TestCase):
    def test_signature_sums_to_prefix_length(self):
        from transformers import AutoTokenizer
        from src.dataset import get_induction_tasks
        from src.metrics import tokenize_qa
        from src.ablation import copying_token_signature
        tok = AutoTokenizer.from_pretrained("gpt2")
        q, a = get_induction_tasks(tok, n=40, seed=0)
        for qq, aa in zip(q, a):
            prefix, full = tokenize_qa(tok, qq, aa)
            sig = copying_token_signature(tok, qq, aa)
            self.assertEqual(sum(sig[:-1]), len(prefix))
            self.assertEqual(sum(sig), len(full))
