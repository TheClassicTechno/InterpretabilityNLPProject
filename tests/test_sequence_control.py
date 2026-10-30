"""Tests for the format-matched ordered-sequence control task."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestSequenceContinuationTasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        from src.dataset import get_sequence_continuation_tasks
        cls.tok = AutoTokenizer.from_pretrained("gpt2")
        cls.qs, cls.ans, cls.families = get_sequence_continuation_tasks(cls.tok)

    def test_format_matches_copying_template(self):
        # "x1 x2 x3 x4 . x5 x6 x7" -> " x8": 7 items around one period
        for q, a in zip(self.qs, self.ans):
            left, right = q.split(" . ")
            self.assertEqual(len(left.split(" ")), 4)
            self.assertEqual(len(right.split(" ")), 3)
            self.assertTrue(a.startswith(" "))
            self.assertEqual(len(self.tok.encode(a)), 1)

    def test_answer_never_appears_in_context(self):
        # the rule cannot be solved by in-context copying
        for q, a in zip(self.qs, self.ans):
            self.assertNotIn(a.strip(), q.replace(".", " ").split())

    def test_consecutive_and_unique(self):
        self.assertEqual(len(set(self.qs)), len(self.qs))
        self.assertIn("1 2 3 4 . 5 6 7", self.qs)
        self.assertEqual(self.ans[self.qs.index("1 2 3 4 . 5 6 7")], " 8")

    def test_deterministic_and_all_families_present(self):
        from src.dataset import get_sequence_continuation_tasks
        again = get_sequence_continuation_tasks(self.tok)
        self.assertEqual(again[0], self.qs)
        self.assertEqual(set(self.families), {"numerals", "months", "letters", "number_words"})


if __name__ == "__main__":
    unittest.main()
