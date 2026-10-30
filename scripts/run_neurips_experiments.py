#!/usr/bin/env python3
"""
End-to-end pipeline for the NeurIPS 2026 "Interpretability as a Science"
short paper: "When Do Attention-Head Ablations Support Causal Claims?"

RQ0 (method validity): does the ablation *implementation* (naive
    post-projection slice vs. correct pre-projection head ablation)
    change the measured per-head effect? -- computed on the induction
    discovery split.
RQ1 (floor effects / metric validity): baseline capability (accuracy AND
    continuous log-prob) on an off-target floor task (arithmetic) vs. an
    on-target task with headroom (induction/copying).
RQ2 (causal specificity): do the top discovery-selected heads exceed
    matched random-head and layer-matched-head controls on a HELD-OUT
    split, replicate in rank across splits, and show effects specific to
    the target task rather than the off-target control task?

Every number that ends up in the paper is written to
neurips/experiment_results/<model>_results.json by this script -- no
number is hand-typed into the .tex.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ablation import ValidatedHeadAblator
from src.controls import (
    bootstrap_mean_ci,
    empirical_percentile,
    mean_effect_for_head_set,
    sample_layer_matched_head_sets,
    sample_random_head_sets,
    specificity_statistic,
    spearman_rank_correlation,
)
from src.dataset import get_arithmetic_control_tasks, get_induction_tasks
from src.metrics import logprob_of_continuation_batch, teacher_forced_exact_match_batch
from src.model_loader import ModelLoader

TOP_K = 5
N_INDUCTION = 200
N_ARITHMETIC = 200
N_BOOTSTRAP = 2000
N_CONTROL_DRAWS = 1000
INDUCTION_SEED = 0
ARITHMETIC_SEED = 1
CONTROL_SAMPLING_SEED = 42


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1]
        ).decode().strip()
    except Exception:
        return "unknown"


def per_head_delta_means(baseline, effects):
    """dict[(layer,head)] -> mean(baseline_i - ablated_i) over examples."""
    baseline = np.asarray(baseline)
    return {
        head: float(np.mean(baseline - np.asarray(vals))) for head, vals in effects.items()
    }


def per_head_delta_arrays(baseline, effects):
    """dict[(layer,head)] -> np.array of per-example (baseline_i - ablated_i)."""
    baseline = np.asarray(baseline)
    return {head: baseline - np.asarray(vals) for head, vals in effects.items()}


def jsonable_head_dict(d):
    return {f"{layer}_{head}": v for (layer, head), v in d.items()}


def rq2_statistics(all_heads, num_heads, delta_disc, delta_held, delta_ctrl, delta_held_arrays):
    """Candidate selection on discovery, held-out comparison against matched
    random / layer-matched controls, rank stability, and specificity.
    Shared by the zero- and mean-ablation pipelines."""
    candidate_heads = sorted(all_heads, key=lambda h: delta_disc[h], reverse=True)[:TOP_K]
    print(f"  Candidate heads (top-{TOP_K} by discovery delta): {candidate_heads}")

    # Rank stability: discovery vs held-out, over all 144 heads
    rank_stability_spearman = spearman_rank_correlation(delta_disc, delta_held)
    print(f"  Spearman(discovery_delta, heldout_delta) over all heads = {rank_stability_spearman:.3f}")

    # Candidate effect on held-out (bootstrap over held-out examples, averaged across candidate heads)
    candidate_per_example_held = np.mean(
        [delta_held_arrays[h] for h in candidate_heads], axis=0
    )
    cand_mean, cand_lo, cand_hi = bootstrap_mean_ci(list(candidate_per_example_held), N_BOOTSTRAP, seed=3)
    print(f"  Candidate held-out delta: {cand_mean:.4f} [{cand_lo:.4f},{cand_hi:.4f}]")

    # Matched random-head control distribution, evaluated on held-out
    random_draws = sample_random_head_sets(all_heads, TOP_K, N_CONTROL_DRAWS, seed=CONTROL_SAMPLING_SEED)
    random_control_held = [mean_effect_for_head_set(delta_held, draw) for draw in random_draws]
    random_p = empirical_percentile(cand_mean, random_control_held)
    print(f"  Random-control held-out delta: mean={np.mean(random_control_held):.4f} "
          f"[{np.percentile(random_control_held, 5):.4f},{np.percentile(random_control_held, 95):.4f}] "
          f"| P(random >= candidate) = {random_p:.4f}")

    # Layer-matched control distribution, evaluated on held-out
    layer_draws = sample_layer_matched_head_sets(candidate_heads, num_heads, N_CONTROL_DRAWS, seed=CONTROL_SAMPLING_SEED + 1)
    layer_control_held = [mean_effect_for_head_set(delta_held, draw) for draw in layer_draws]
    layer_p = empirical_percentile(cand_mean, layer_control_held)
    print(f"  Layer-matched control held-out delta: mean={np.mean(layer_control_held):.4f} "
          f"[{np.percentile(layer_control_held, 5):.4f},{np.percentile(layer_control_held, 95):.4f}] "
          f"| P(layer-matched >= candidate) = {layer_p:.4f}")

    # Specificity: candidate vs. random, target-task delta minus control-task delta
    candidate_specificity = specificity_statistic(delta_held, delta_ctrl, candidate_heads)
    random_specificity = [specificity_statistic(delta_held, delta_ctrl, draw) for draw in random_draws]
    specificity_p = empirical_percentile(candidate_specificity, random_specificity)
    print(f"  Candidate specificity (held-out delta - control delta) = {candidate_specificity:.4f}")
    print(f"  Random-control specificity: mean={np.mean(random_specificity):.4f} "
          f"| P(random >= candidate) = {specificity_p:.4f}")
    return {
        "candidate_heads": [list(h) for h in candidate_heads],
        "candidate_discovery_delta": {f"{l}_{h}": delta_disc[(l, h)] for (l, h) in candidate_heads},
        "candidate_heldout_delta_mean": cand_mean,
        "candidate_heldout_delta_ci95": [cand_lo, cand_hi],
        "random_control_heldout_delta": {
            "mean": float(np.mean(random_control_held)),
            "p5": float(np.percentile(random_control_held, 5)),
            "p95": float(np.percentile(random_control_held, 95)),
            "n_draws": N_CONTROL_DRAWS,
            "empirical_p_candidate_beats_random": random_p,
        },
        "layer_matched_control_heldout_delta": {
            "mean": float(np.mean(layer_control_held)),
            "p5": float(np.percentile(layer_control_held, 5)),
            "p95": float(np.percentile(layer_control_held, 95)),
            "n_draws": N_CONTROL_DRAWS,
            "empirical_p_candidate_beats_layer_matched": layer_p,
        },
        "rank_stability_spearman_discovery_vs_heldout": rank_stability_spearman,
        "specificity": {
            "candidate": candidate_specificity,
            "random_control_mean": float(np.mean(random_specificity)),
            "random_control_p5": float(np.percentile(random_specificity, 5)),
            "random_control_p95": float(np.percentile(random_specificity, 95)),
            "empirical_p_candidate_beats_random": specificity_p,
        },
    }


def run(model_name: str, output_path: Path, n_induction: int = N_INDUCTION, n_arithmetic: int = N_ARITHMETIC,
        arithmetic_max_operand: int = 20, ablation: str = "zero"):
    print(f"\n{'=' * 70}\nRunning pipeline for {model_name} (n_induction={n_induction}, n_arithmetic={n_arithmetic})\n{'=' * 70}")

    loader = ModelLoader(model_name)
    loader.load()
    model = loader.get_model()
    tokenizer = loader.get_tokenizer()
    device = loader.get_device()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    all_heads = [(l, h) for l in range(num_layers) for h in range(num_heads)]
    print(f"Model: {model_name} | layers={num_layers} heads/layer={num_heads} total_heads={len(all_heads)}")

    # ---- Datasets -----------------------------------------------------
    print("\n[data] Generating induction (target) and arithmetic (control) tasks...")
    induction_q, induction_a = get_induction_tasks(tokenizer, n=n_induction, seed=INDUCTION_SEED)
    arith_q, arith_a = get_arithmetic_control_tasks(
        tokenizer, n=n_arithmetic, seed=ARITHMETIC_SEED, max_operand=arithmetic_max_operand
    )

    half = len(induction_q) // 2
    disc_q, disc_a = induction_q[:half], induction_a[:half]
    held_q, held_a = induction_q[half:], induction_a[half:]
    print(f"Induction: {len(induction_q)} total -> discovery={len(disc_q)} held-out={len(held_q)}")
    print(f"Arithmetic control: {len(arith_q)} (specificity-only, never used for selection)")

    # ---- RQ1: baseline capability --------------------------------------
    print("\n[RQ1] Baseline capability (accuracy + log-prob) per task...")

    def summarize_baseline(qs, ans, label):
        acc = teacher_forced_exact_match_batch(model, tokenizer, qs, ans, device)
        lp = logprob_of_continuation_batch(model, tokenizer, qs, ans, device)
        acc_mean, acc_lo, acc_hi = bootstrap_mean_ci([float(x) for x in acc], N_BOOTSTRAP, seed=1)
        lp_mean, lp_lo, lp_hi = bootstrap_mean_ci(lp, N_BOOTSTRAP, seed=2)
        print(f"  {label}: n={len(qs)} acc={acc_mean:.3f} [{acc_lo:.3f},{acc_hi:.3f}] "
              f"logprob={lp_mean:.3f} [{lp_lo:.3f},{lp_hi:.3f}]")
        return {
            "n": len(qs), "accuracy_mean": acc_mean, "accuracy_ci95": [acc_lo, acc_hi],
            "logprob_mean": lp_mean, "logprob_ci95": [lp_lo, lp_hi],
            "logprob_per_example": lp,
        }

    baselines = {
        "induction_full": summarize_baseline(induction_q, induction_a, "induction (full)"),
        "arithmetic_control": summarize_baseline(arith_q, arith_a, "arithmetic (control)"),
    }

    # ---- RQ0: does the ablation implementation matter? -----------------
    print("\n[RQ0] Naive post-projection vs. correct pre-projection ablation "
          "(all heads, induction-discovery split)...")
    ablator = ValidatedHeadAblator(model, tokenizer, device)

    base_disc, effects_correct_disc = ablator.compute_all_head_effects(disc_q, disc_a, method="head_ablation")
    base_disc_naive, effects_naive_disc = ablator.compute_all_head_effects(disc_q, disc_a, method="naive_post_projection")
    assert np.allclose(base_disc, base_disc_naive), "baseline should not depend on ablation method"

    delta_correct_disc = per_head_delta_means(base_disc, effects_correct_disc)
    delta_naive_disc = per_head_delta_means(base_disc, effects_naive_disc)

    corr_vec = np.array([delta_correct_disc[h] for h in all_heads])
    naive_vec = np.array([delta_naive_disc[h] for h in all_heads])
    rq0_pearson = float(np.corrcoef(corr_vec, naive_vec)[0, 1])
    rq0_spearman = spearman_rank_correlation(delta_correct_disc, delta_naive_disc)
    rq0_mean_abs_diff = float(np.mean(np.abs(corr_vec - naive_vec)))

    top5_correct = set(sorted(all_heads, key=lambda h: delta_correct_disc[h], reverse=True)[:TOP_K])
    top5_naive = set(sorted(all_heads, key=lambda h: delta_naive_disc[h], reverse=True)[:TOP_K])
    rq0_top5_overlap = len(top5_correct & top5_naive)

    print(f"  Pearson r(naive, correct) = {rq0_pearson:.3f} | Spearman rho = {rq0_spearman:.3f}")
    print(f"  mean|delta_correct - delta_naive| = {rq0_mean_abs_diff:.4f}")
    print(f"  top-{TOP_K} head overlap (naive vs correct) = {rq0_top5_overlap}/{TOP_K}")

    # ---- RQ2 setup: full head-effect vectors on all 3 splits (correct method) ----
    print("\n[RQ2] Full per-head effect vectors (correct method) on held-out + control splits...")
    base_held, effects_held = ablator.compute_all_head_effects(held_q, held_a, method="head_ablation")
    base_ctrl, effects_ctrl = ablator.compute_all_head_effects(arith_q, arith_a, method="head_ablation")

    delta_disc = delta_correct_disc  # alias for clarity below
    delta_held = per_head_delta_means(base_held, effects_held)
    delta_ctrl = per_head_delta_means(base_ctrl, effects_ctrl)
    delta_held_arrays = per_head_delta_arrays(base_held, effects_held)

    rq2 = rq2_statistics(all_heads, num_heads, delta_disc, delta_held, delta_ctrl, delta_held_arrays)

    # ---- Robustness: mean ablation (only with --ablation mean) -----------
    mean_ablation = None
    if ablation == "mean":
        print("\n[mean ablation] Reference means from the discovery split (non-padding positions); "
              "the same means are applied on every split...")
        means = ablator.compute_head_means(disc_q, disc_a)
        base_disc_m, eff_disc_m = ablator.compute_all_head_effects(disc_q, disc_a, "mean_ablation", means)
        base_held_m, eff_held_m = ablator.compute_all_head_effects(held_q, held_a, "mean_ablation", means)
        base_ctrl_m, eff_ctrl_m = ablator.compute_all_head_effects(arith_q, arith_a, "mean_ablation", means)
        assert np.allclose(base_disc_m, base_disc), "baseline should not depend on ablation method"

        m_disc = per_head_delta_means(base_disc_m, eff_disc_m)
        m_held = per_head_delta_means(base_held_m, eff_held_m)
        m_ctrl = per_head_delta_means(base_ctrl_m, eff_ctrl_m)
        m_held_arrays = per_head_delta_arrays(base_held_m, eff_held_m)
        rq2_mean = rq2_statistics(all_heads, num_heads, m_disc, m_held, m_ctrl, m_held_arrays)

        zero_vec = np.array([delta_disc[h] for h in all_heads])
        mean_vec = np.array([m_disc[h] for h in all_heads])
        top_zero = set(tuple(h) for h in rq2["candidate_heads"])
        top_mean = set(tuple(h) for h in rq2_mean["candidate_heads"])
        mean_vs_zero = {
            "pearson_r_discovery": float(np.corrcoef(zero_vec, mean_vec)[0, 1]),
            "spearman_rho_discovery": spearman_rank_correlation(delta_disc, m_disc),
            "top5_overlap_with_zero_candidates": len(top_zero & top_mean),
        }
        print(f"  zero vs mean (discovery): Pearson r={mean_vs_zero['pearson_r_discovery']:.3f} "
              f"Spearman rho={mean_vs_zero['spearman_rho_discovery']:.3f} "
              f"top-{TOP_K} overlap={mean_vs_zero['top5_overlap_with_zero_candidates']}/{TOP_K}")
        mean_ablation = {
            "reference_split": "induction_discovery",
            "reference_positions": "all non-padding token positions",
            "mean_vs_zero": mean_vs_zero,
            "rq2_candidate_vs_controls": {
                **rq2_mean,
                "per_head_delta_discovery": jsonable_head_dict(m_disc),
                "per_head_delta_heldout": jsonable_head_dict(m_held),
                "per_head_delta_control": jsonable_head_dict(m_ctrl),
            },
        }

    # ---- Assemble results ----------------------------------------------
    results = {
        "config": {
            "model_name": model_name,
            "device": device,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "git_commit": git_commit(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "seeds": {
                "induction": INDUCTION_SEED,
                "arithmetic": ARITHMETIC_SEED,
                "control_sampling": CONTROL_SAMPLING_SEED,
            },
            "n_induction_total": len(induction_q),
            "n_induction_discovery": len(disc_q),
            "n_induction_heldout": len(held_q),
            "n_arithmetic_control": len(arith_q),
            "arithmetic_max_operand": arithmetic_max_operand,
            "top_k_candidates": TOP_K,
            "n_bootstrap": N_BOOTSTRAP,
            "n_control_draws": N_CONTROL_DRAWS,
        },
        "baselines": baselines,
        "rq0_naive_vs_correct": {
            "pearson_r": rq0_pearson,
            "spearman_rho": rq0_spearman,
            "mean_abs_delta_diff": rq0_mean_abs_diff,
            "top5_head_overlap": rq0_top5_overlap,
            "top5_correct": [list(h) for h in top5_correct],
            "top5_naive": [list(h) for h in top5_naive],
            "per_head_delta_correct": jsonable_head_dict(delta_correct_disc),
            "per_head_delta_naive": jsonable_head_dict(delta_naive_disc),
        },
        "rq2_candidate_vs_controls": {
            **rq2,
            "per_head_delta_discovery": jsonable_head_dict(delta_disc),
            "per_head_delta_heldout": jsonable_head_dict(delta_held),
            "per_head_delta_control": jsonable_head_dict(delta_ctrl),
        },
    }

    if mean_ablation is not None:
        results["config"]["ablation_baseline"] = "mean"
        results["mean_ablation"] = mean_ablation

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--output", default=None)
    parser.add_argument("--n-induction", type=int, default=N_INDUCTION)
    parser.add_argument("--n-arithmetic", type=int, default=N_ARITHMETIC)
    parser.add_argument("--arithmetic-max-operand", type=int, default=20)
    parser.add_argument("--ablation", choices=["zero", "mean"], default="zero",
                        help="'mean' additionally runs the mean-ablation robustness check")
    args = parser.parse_args()

    suffix = "_meanablation_results.json" if args.ablation == "mean" else "_results.json"
    out = Path(args.output) if args.output else Path(__file__).resolve().parents[1] / "neurips" / "experiment_results" / f"{args.model}{suffix}"
    if args.output is None and args.ablation == "mean" and out.exists():
        raise SystemExit(f"Refusing to overwrite existing {out}")
    run(args.model, out, n_induction=args.n_induction, n_arithmetic=args.n_arithmetic,
        arithmetic_max_operand=args.arithmetic_max_operand, ablation=args.ablation)
