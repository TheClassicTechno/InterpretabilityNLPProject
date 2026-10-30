#!/usr/bin/env python3
"""
Camera-ready robustness checks (design: neurips/camera_ready_design.md, Sec. 3),
specified before running and reported regardless of outcome.

C1  Partition sensitivity: 20 random 100/100 discovery/held-out partitions
    of the 200 copying examples; selection, matched controls, rank
    stability and arithmetic specificity are recomputed per partition.
C2  Format-matched control task: ordered-sequence continuation
    ("1 2 3 4 . 5 6 7" -> " 8"), same surface template as copying but the
    answer is never in context. Specificity of the primary zero-selected
    candidates against it (primary), and of the mean-selected candidates
    under mean-ablation (secondary).

Zero-ablation (pre-projection) throughout, except the C2 secondary.
Writes neurips/experiment_results/<model>_camera_ready_robustness.json and
refuses to overwrite it. The original result files are only read.
"""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_neurips_experiments import (  # noqa: E402
    ARITHMETIC_SEED,
    CONTROL_SAMPLING_SEED,
    INDUCTION_SEED,
    N_CONTROL_DRAWS,
    TOP_K,
    git_commit,
    jsonable_head_dict,
    per_head_delta_arrays,
    per_head_delta_means,
)
from src.ablation import ValidatedHeadAblator  # noqa: E402
from src.controls import (  # noqa: E402
    empirical_percentile,
    mean_effect_for_head_set,
    sample_layer_matched_head_sets,
    sample_random_head_sets,
    spearman_rank_correlation,
    specificity_statistic,
)
from src.dataset import (  # noqa: E402
    get_arithmetic_control_tasks,
    get_induction_tasks,
    get_sequence_continuation_tasks,
)
from src.metrics import logprob_of_continuation_batch, teacher_forced_exact_match_batch  # noqa: E402
from src.model_loader import ModelLoader  # noqa: E402

N_PARTITIONS = 20
PARTITION_SEED_BASE = 1000
SANITY_TOL = 1e-4
RESULTS_DIR = Path(__file__).resolve().parents[1] / "neurips" / "experiment_results"


def head_key(h):
    return f"{h[0]}_{h[1]}"


def contrast_vs_random(target, control, candidates, random_draws):
    """Specificity (target minus control Delta) of `candidates`, its random
    head-set distribution, and the candidates' control-task Delta alone."""
    cand = specificity_statistic(target, control, candidates)
    rand = [specificity_statistic(target, control, d) for d in random_draws]
    ctrl_cand = mean_effect_for_head_set(control, candidates)
    ctrl_rand = [mean_effect_for_head_set(control, d) for d in random_draws]
    return {
        "candidate_specificity": cand,
        "random_specificity_mean": float(np.mean(rand)),
        "random_specificity_p5": float(np.percentile(rand, 5)),
        "random_specificity_p95": float(np.percentile(rand, 95)),
        "empirical_p_candidate_beats_random": empirical_percentile(cand, rand),
        "candidate_control_task_delta": ctrl_cand,
        "random_control_task_delta_mean": float(np.mean(ctrl_rand)),
        "random_control_task_delta_p5": float(np.percentile(ctrl_rand, 5)),
        "random_control_task_delta_p95": float(np.percentile(ctrl_rand, 95)),
        "empirical_p_candidate_control_delta_ge_random": empirical_percentile(ctrl_cand, ctrl_rand),
    }


def run(model_name: str, output_path: Path):
    t0 = time.time()
    primary = json.load(open(RESULTS_DIR / f"{model_name}_results.json"))
    mean_file = RESULTS_DIR / f"{model_name}_meanablation_results.json"

    loader = ModelLoader(model_name)
    loader.load()
    model, tokenizer, device = loader.get_model(), loader.get_tokenizer(), loader.get_device()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    num_layers, num_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    all_heads = [(l, h) for l in range(num_layers) for h in range(num_heads)]

    copy_q, copy_a = get_induction_tasks(tokenizer, n=200, seed=INDUCTION_SEED)
    arith_q, arith_a = get_arithmetic_control_tasks(tokenizer, n=200, seed=ARITHMETIC_SEED, max_operand=20)
    seq_q, seq_a, seq_fam = get_sequence_continuation_tasks(tokenizer)
    n = len(copy_q)
    ablator = ValidatedHeadAblator(model, tokenizer, device)

    # ---- one zero-ablation sweep per task ---------------------------------
    print(f"[{model_name}] sweeps: copying (all {n}), arithmetic, sequence control")
    base_copy, eff_copy = ablator.compute_all_head_effects(copy_q, copy_a, "head_ablation")
    base_arith, eff_arith = ablator.compute_all_head_effects(arith_q, arith_a, "head_ablation")
    base_seq, eff_seq = ablator.compute_all_head_effects(seq_q, seq_a, "head_ablation")
    copy_arrays = per_head_delta_arrays(base_copy, eff_copy)  # head -> (n,)
    delta_arith = per_head_delta_means(base_arith, eff_arith)
    delta_seq = per_head_delta_means(base_seq, eff_seq)

    # ---- sanity: the original partition reproduces the saved per-head deltas
    half = n // 2
    rq2_saved = primary["rq2_candidate_vs_controls"]
    max_dev = 0.0
    for split, idx in (("per_head_delta_discovery", slice(0, half)), ("per_head_delta_heldout", slice(half, n))):
        for h in all_heads:
            max_dev = max(max_dev, abs(float(copy_arrays[h][idx].mean()) - rq2_saved[split][head_key(h)]))
    for h in all_heads:
        max_dev = max(max_dev, abs(delta_arith[h] - rq2_saved["per_head_delta_control"][head_key(h)]))
    print(f"  sanity: max |recomputed - saved| per-head delta = {max_dev:.2e}")
    if max_dev > SANITY_TOL:
        raise SystemExit(f"sanity check failed: {max_dev} > {SANITY_TOL}")

    random_draws = sample_random_head_sets(all_heads, TOP_K, N_CONTROL_DRAWS, seed=CONTROL_SAMPLING_SEED)

    # ---- C1: partition sensitivity ----------------------------------------
    primary_cands = [tuple(h) for h in rq2_saved["candidate_heads"]]
    partitions = []
    for i in range(N_PARTITIONS):
        perm = np.random.default_rng(PARTITION_SEED_BASE + i).permutation(n)
        disc_idx, held_idx = perm[:half], perm[half:]
        d_disc = {h: float(copy_arrays[h][disc_idx].mean()) for h in all_heads}
        d_held = {h: float(copy_arrays[h][held_idx].mean()) for h in all_heads}
        cands = sorted(all_heads, key=lambda h: d_disc[h], reverse=True)[:TOP_K]
        cand_held = mean_effect_for_head_set(d_held, cands)
        rand_held = [mean_effect_for_head_set(d_held, d) for d in random_draws]
        layer_draws = sample_layer_matched_head_sets(cands, num_heads, N_CONTROL_DRAWS, seed=CONTROL_SAMPLING_SEED + 1)
        layer_held = [mean_effect_for_head_set(d_held, d) for d in layer_draws]
        spec = specificity_statistic(d_held, delta_arith, cands)
        spec_rand = [specificity_statistic(d_held, delta_arith, d) for d in random_draws]
        partitions.append({
            "seed": PARTITION_SEED_BASE + i,
            "candidate_heads": [list(h) for h in cands],
            "overlap_with_primary_candidates": len(set(cands) & set(primary_cands)),
            "candidate_heldout_delta": cand_held,
            "random_control_mean": float(np.mean(rand_held)),
            "p_random": empirical_percentile(cand_held, rand_held),
            "layer_matched_control_mean": float(np.mean(layer_held)),
            "p_layer_matched": empirical_percentile(cand_held, layer_held),
            "rank_stability_spearman": spearman_rank_correlation(d_disc, d_held),
            "arith_specificity": spec,
            "p_arith_specificity": empirical_percentile(spec, spec_rand),
        })
        print(f"  C1 partition {i}: cands={cands} held={cand_held:.3f} "
              f"p_rand={partitions[-1]['p_random']:.3f} p_layer={partitions[-1]['p_layer_matched']:.3f} "
              f"p_spec={partitions[-1]['p_arith_specificity']:.3f}")

    def summary(key):
        v = np.array([p[key] for p in partitions], dtype=float)
        return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}

    selection_counts = Counter(head_key(h) for p in partitions for h in p["candidate_heads"])
    c1 = {
        "n_partitions": N_PARTITIONS,
        "partition_seeds": [PARTITION_SEED_BASE + i for i in range(N_PARTITIONS)],
        "split": "random permutation of the 200 copying examples; first 100 discovery, last 100 held-out",
        "primary_candidates": [list(h) for h in primary_cands],
        "candidate_heldout_delta": summary("candidate_heldout_delta"),
        "rank_stability_spearman": summary("rank_stability_spearman"),
        "arith_specificity": summary("arith_specificity"),
        "n_partitions_p_random_le_0.05": sum(p["p_random"] <= 0.05 for p in partitions),
        "n_partitions_p_layer_matched_le_0.05": sum(p["p_layer_matched"] <= 0.05 for p in partitions),
        "n_partitions_p_arith_specificity_le_0.05": sum(p["p_arith_specificity"] <= 0.05 for p in partitions),
        "p_arith_specificity": summary("p_arith_specificity"),
        "overlap_with_primary_candidates": summary("overlap_with_primary_candidates"),
        "overlap_with_primary_counts": dict(Counter(p["overlap_with_primary_candidates"] for p in partitions)),
        "head_selection_counts": dict(selection_counts.most_common()),
        "falsifier": "fail p<=0.05 vs either matched control in more than 2 of 20 partitions",
        "partitions": partitions,
    }

    # ---- C2: format-matched sequence-continuation control -----------------
    seq_acc = teacher_forced_exact_match_batch(model, tokenizer, seq_q, seq_a, device)
    seq_lp = logprob_of_continuation_batch(model, tokenizer, seq_q, seq_a, device)
    fam_counts = Counter(seq_fam)
    seq_baseline = {
        "n": len(seq_q),
        "family_counts": dict(fam_counts),
        "accuracy_mean": float(np.mean(seq_acc)),
        "logprob_mean": float(np.mean(seq_lp)),
        "by_family": {
            f: {"n": fam_counts[f],
                "accuracy_mean": float(np.mean([a for a, ff in zip(seq_acc, seq_fam) if ff == f])),
                "logprob_mean": float(np.mean([v for v, ff in zip(seq_lp, seq_fam) if ff == f]))}
            for f in fam_counts
        },
        "question_token_lengths": dict(Counter(len(tokenizer(q)["input_ids"]) for q in seq_q)),
        "copying_question_token_lengths": dict(Counter(len(tokenizer(q)["input_ids"]) for q in copy_q)),
    }
    print(f"  C2 sequence baseline: n={len(seq_q)} acc={seq_baseline['accuracy_mean']:.3f} "
          f"logprob={seq_baseline['logprob_mean']:.3f}")

    delta_held_primary = {h: rq2_saved["per_head_delta_heldout"][head_key(h)] for h in all_heads}
    c2_primary = contrast_vs_random(delta_held_primary, delta_seq, primary_cands, random_draws)
    c2_primary["candidates"] = [list(h) for h in primary_cands]
    c2_primary["arith_reference_specificity_p"] = rq2_saved["specificity"]["empirical_p_candidate_beats_random"]
    print(f"  C2 primary (zero): spec={c2_primary['candidate_specificity']:.3f} "
          f"p={c2_primary['empirical_p_candidate_beats_random']:.3f} "
          f"seq-delta cand={c2_primary['candidate_control_task_delta']:.3f}")

    c2_secondary = None
    if mean_file.exists():
        mean_saved = json.load(open(mean_file))["mean_ablation"]["rq2_candidate_vs_controls"]
        disc_q, disc_a = copy_q[:half], copy_a[:half]
        means = ablator.compute_head_means(disc_q, disc_a)
        base_seq_m, eff_seq_m = ablator.compute_all_head_effects(seq_q, seq_a, "mean_ablation", means)
        delta_seq_m = per_head_delta_means(base_seq_m, eff_seq_m)
        m_held = {h: mean_saved["per_head_delta_heldout"][head_key(h)] for h in all_heads}
        mean_cands = [tuple(h) for h in mean_saved["candidate_heads"]]
        c2_secondary = contrast_vs_random(m_held, delta_seq_m, mean_cands, random_draws)
        c2_secondary["candidates"] = [list(h) for h in mean_cands]
        c2_secondary["ablation"] = "mean (reference means: copying discovery split, all non-padding positions)"
        c2_secondary["per_head_delta_sequence_control"] = jsonable_head_dict(delta_seq_m)
        print(f"  C2 secondary (mean): spec={c2_secondary['candidate_specificity']:.3f} "
              f"p={c2_secondary['empirical_p_candidate_beats_random']:.3f}")

    results = {
        "config": {
            "model_name": model_name,
            "device": device,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "git_commit": git_commit(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "design": "neurips/camera_ready_design.md (Sec. 3, C1/C2), pre-specified",
            "seeds": {"induction": INDUCTION_SEED, "arithmetic": ARITHMETIC_SEED,
                      "control_sampling": CONTROL_SAMPLING_SEED, "partition_base": PARTITION_SEED_BASE},
            "n_copying": n, "n_arithmetic": len(arith_q), "n_sequence_control": len(seq_q),
            "top_k_candidates": TOP_K, "n_control_draws": N_CONTROL_DRAWS,
            "ablation": "zero (pre-projection) unless stated",
            "sanity_max_abs_dev_vs_saved": max_dev,
            "wall_clock_seconds": None,
        },
        "c1_partition_sensitivity": c1,
        "c2_sequence_control": {
            "baseline": seq_baseline,
            "primary_zero_candidates": c2_primary,
            "secondary_mean_candidates": c2_secondary,
            "per_head_delta_sequence_control_zero": jsonable_head_dict(delta_seq),
        },
    }
    results["config"]["wall_clock_seconds"] = round(time.time() - t0, 2)
    with open(output_path, "x") as f:  # "x": never overwrite
        json.dump(results, f, indent=2)
    print(f"Saved {output_path} ({results['config']['wall_clock_seconds']} s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", choices=["gpt2", "distilgpt2"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    out = Path(args.output) if args.output else RESULTS_DIR / f"{args.model}_camera_ready_robustness.json"
    run(args.model, out)
