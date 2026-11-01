"""
Statistical controls for head-ablation causal claims: matched random-head
and layer-matched-head control distributions, bootstrap confidence
intervals, rank-stability (discovery vs. held-out), and a task-specificity
statistic. No forward passes happen here -- everything resamples from
already-computed per-head effect vectors (see
`ValidatedHeadAblator.compute_all_head_effects`), so control distributions
are essentially free to build.
"""

import math
import random
from typing import Dict, List, Tuple

import numpy as np


def bootstrap_mean_ci(
    values: List[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of `values` (resampling
    examples with replacement). Returns (mean, lo, hi)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_means[i] = values[idx].mean()
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact (Clopper-Pearson) two-sided CI for a binomial proportion with
    k successes in n trials. Unlike a percentile bootstrap, it does not
    collapse to [0, 0] when k = 0 (or [1, 1] when k = n)."""
    if not 0 <= k <= n or n == 0:
        raise ValueError(f"need 0 <= k <= n and n > 0, got k={k}, n={n}")

    def solve(f, target):
        # f is monotone decreasing in p on [0, 1]; find f(p) = target.
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if f(mid) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(lambda q: -(1 - _binom_cdf(k - 1, n, q)), -alpha / 2)
    upper = 1.0 if k == n else solve(lambda q: _binom_cdf(k, n, q), alpha / 2)
    return lower, upper


def sample_random_head_sets(
    all_heads: List[Tuple[int, int]], k: int, n_draws: int, seed: int = 0
) -> List[List[Tuple[int, int]]]:
    """`n_draws` independent draws of `k` distinct heads sampled uniformly
    (without replacement within a draw) from `all_heads`."""
    rng = random.Random(seed)
    return [rng.sample(all_heads, k) for _ in range(n_draws)]


def sample_layer_matched_head_sets(
    candidate_heads: List[Tuple[int, int]], num_heads_per_layer: int, n_draws: int, seed: int = 0
) -> List[List[Tuple[int, int]]]:
    """`n_draws` draws that preserve the candidate set's size and layer
    composition while replacing every candidate identity -- controls for
    "this layer is generally sensitive" rather than "this specific head
    matters".

    Heads are drawn per layer WITHOUT replacement from that layer's
    non-candidate heads. Both exclusions matter when a layer supplies more
    than one candidate: sampling each candidate's replacement independently
    would let one head stand in for several candidates (reweighting it in
    the mean) and would let a candidate head itself enter its own control
    distribution, biasing the control toward the candidates.
    """
    heads_by_layer: Dict[int, List[int]] = {}
    for (layer, head) in candidate_heads:
        heads_by_layer.setdefault(layer, []).append(head)

    for layer, heads in heads_by_layer.items():
        available = num_heads_per_layer - len(heads)
        if available < len(heads):
            raise ValueError(
                f"layer {layer} supplies {len(heads)} candidates but only "
                f"{available} non-candidate heads; cannot build a "
                f"layer-matched control set without reuse"
            )

    rng = random.Random(seed)
    draws = []
    for _ in range(n_draws):
        draw = []
        for layer, heads in heads_by_layer.items():
            pool = [h for h in range(num_heads_per_layer) if h not in heads]
            draw.extend((layer, h) for h in rng.sample(pool, len(heads)))
        draws.append(draw)
    return draws


def mean_effect_for_head_set(
    head_effect_means: Dict[Tuple[int, int], float], head_set: List[Tuple[int, int]]
) -> float:
    return float(np.mean([head_effect_means[h] for h in head_set]))


def empirical_percentile(candidate_value: float, control_values: List[float]) -> float:
    """Add-one Monte Carlo p-value for ``control >= candidate``.

    The finite-sample correction prevents an estimated p-value of zero
    when no sampled control matches or exceeds the candidate.
    """
    control_values = np.asarray(control_values, dtype=float)
    exceedances = int(np.count_nonzero(control_values >= candidate_value))
    return float((exceedances + 1) / (len(control_values) + 1))


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average-tie ranks (0-indexed), no scipy dependency."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        avg_rank = (i + j) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman_rank_correlation(
    vec_a: Dict[Tuple[int, int], float], vec_b: Dict[Tuple[int, int], float]
) -> float:
    """Spearman rank correlation between two per-head effect vectors that
    share the same keys (e.g. discovery-split vs. held-out-split effects
    for all 144 heads) -- used as the head-ranking stability check."""
    keys = sorted(vec_a.keys())
    a = np.array([vec_a[k] for k in keys], dtype=float)
    b = np.array([vec_b[k] for k in keys], dtype=float)
    ra, rb = _rankdata(a), _rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def specificity_statistic(
    target_effect_means: Dict[Tuple[int, int], float],
    control_effect_means: Dict[Tuple[int, int], float],
    head_set: List[Tuple[int, int]],
) -> float:
    """Mean(Delta on target task) - Mean(Delta on off-target control task)
    for a given set of heads. Large positive values indicate the
    intervention selectively disrupts the target task; values near zero
    indicate generic damage that hits both tasks equally."""
    target = mean_effect_for_head_set(target_effect_means, head_set)
    control = mean_effect_for_head_set(control_effect_means, head_set)
    return target - control
