import unittest
from collections import Counter

from src.controls import empirical_percentile, sample_layer_matched_head_sets

# Three candidates share layer 0, which is what makes naive per-candidate
# sampling degenerate.
CANDIDATES = [(0, 0), (0, 7), (1, 3), (11, 0), (0, 10)]
NUM_HEADS = 12


class LayerMatchedControlTests(unittest.TestCase):
    """A layer-matched null must preserve set size and layer composition
    while replacing every candidate identity."""

    def draws(self, n=200, seed=43):
        return sample_layer_matched_head_sets(CANDIDATES, NUM_HEADS, n, seed=seed)

    def test_every_draw_has_the_candidate_set_size_with_no_repeats(self):
        for draw in self.draws():
            self.assertEqual(len(draw), len(CANDIDATES))
            self.assertEqual(len(set(draw)), len(CANDIDATES))

    def test_no_draw_contains_a_candidate_head(self):
        candidates = set(CANDIDATES)
        for draw in self.draws():
            self.assertEqual(candidates & set(draw), set())

    def test_layer_composition_is_preserved(self):
        expected = Counter(layer for layer, _ in CANDIDATES)
        for draw in self.draws():
            self.assertEqual(Counter(layer for layer, _ in draw), expected)

    def test_sampling_is_deterministic_given_the_seed(self):
        self.assertEqual(self.draws(seed=43), self.draws(seed=43))
        self.assertNotEqual(self.draws(seed=43), self.draws(seed=44))

    def test_rejects_a_layer_that_cannot_supply_enough_distinct_heads(self):
        crowded = [(0, h) for h in range(3)]
        with self.assertRaises(ValueError):
            sample_layer_matched_head_sets(crowded, num_heads_per_layer=4, n_draws=1)


class EmpiricalPercentileTests(unittest.TestCase):
    def test_uses_add_one_correction_when_no_control_matches_candidate(self):
        controls = [0.1, 0.2, 0.3]

        self.assertEqual(empirical_percentile(1.0, controls), 1 / 4)

    def test_counts_matches_and_exceedances_with_add_one_correction(self):
        controls = [0.2, 0.5, 0.8]

        self.assertEqual(empirical_percentile(0.5, controls), 3 / 4)


if __name__ == "__main__":
    unittest.main()
