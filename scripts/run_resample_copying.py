#!/usr/bin/env python3
"""
C3: resample ablation on the copying task (design: neurips/camera_ready_design.md,
Sec. 3, C3), specified before running and reported regardless of outcome.

Each head's slice of the pre-projection input is replaced, at every
position, by its value on a different copying example from the same split
with the same per-word token signature (so every position plays the same
role). Zero- and mean-ablation per-head effects are read from the saved
result files, not recomputed.

R1  held-out resample effect of the five zero-selected candidates vs
    random and layer-matched controls (primary; source seed 2)
R2  full selection under resample (top-5 on discovery, held-out vs controls)
R3  per-head agreement of resample with zero and with mean
R4  R1/R2 repeated with source seeds 3 and 4

Writes neurips/experiment_results/<model>_resample_copying.json and refuses
to overwrite it. The original result files are only read.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_neurips_experiments import (  # noqa: E402
    CONTROL_SAMPLING_SEED,
    INDUCTION_SEED,
    N_BOOTSTRAP,
    N_CONTROL_DRAWS,
    TOP_K,
    git_commit,
    jsonable_head_dict,
)
from src.ablation import (  # noqa: E402
    capture_out_proj_inputs,
    copying_token_signature,
    match_resample_sources,
    register_head_intervention,
)
from src.controls import (  # noqa: E402
    bootstrap_mean_ci,
    empirical_percentile,
    mean_effect_for_head_set,
    sample_layer_matched_head_sets,
    sample_random_head_sets,
    spearman_rank_correlation,
)
from src.dataset import get_induction_tasks  # noqa: E402
from src.metrics import _build_batch, logprob_of_continuation_batch, tokenize_qa  # noqa: E402
from src.model_loader import ModelLoader  # noqa: E402

PRIMARY_SOURCE_SEED = 2
SENSITIVITY_SOURCE_SEEDS = (3, 4)
R1_HALF_ZERO_EFFECT_FACTOR = 0.5  # reading (B) vs (C), fixed in the design
SELF_SOURCE_TOL = 1e-5
BASELINE_TOL = 1e-4
RESULTS_DIR = Path(__file__).resolve().parents[1] / "neurips" / "experiment_results"


def head_key(h):
    return f"{h[0]}_{h[1]}"


def heads_from_saved(d, all_heads):
    return {h: d[head_key(h)] for h in all_heads}


def pearson(a, b, heads):
    return float(np.corrcoef([a[h] for h in heads], [b[h] for h in heads])[0, 1])


def top_k(d, k=TOP_K):
    return sorted(d, key=lambda h: d[h], reverse=True)[:k]


def set_vs_controls(heads, num_heads, all_heads, d_held, held_arrays):
    """Held-out effect of a fixed head set vs random and layer-matched
    controls, with the same seeds and procedure as the RQ2 pipeline."""
    per_ex = np.mean([held_arrays[h] for h in heads], axis=0)
    m, lo, hi = bootstrap_mean_ci(list(per_ex), N_BOOTSTRAP, seed=3)
    rnd = [mean_effect_for_head_set(d_held, s)
           for s in sample_random_head_sets(all_heads, TOP_K, N_CONTROL_DRAWS, seed=CONTROL_SAMPLING_SEED)]
    lay = [mean_effect_for_head_set(d_held, s)
           for s in sample_layer_matched_head_sets(heads, num_heads, N_CONTROL_DRAWS, seed=CONTROL_SAMPLING_SEED + 1)]
    return {
        "heads": [list(h) for h in heads],
        "heldout_delta_mean": m,
        "heldout_delta_ci95": [lo, hi],
        "random_control": {"mean": float(np.mean(rnd)), "p5": float(np.percentile(rnd, 5)),
                           "p95": float(np.percentile(rnd, 95)), "empirical_p": empirical_percentile(m, rnd)},
        "layer_matched_control": {"mean": float(np.mean(lay)), "p5": float(np.percentile(lay, 5)),
                                  "p95": float(np.percentile(lay, 95)), "empirical_p": empirical_percentile(m, lay)},
    }


@torch.no_grad()
def resample_sweep(model, tokenizer, device, tq, ta, sq, sa, all_heads):
    """Per-head, per-example resample delta (baseline minus ablated log-prob)
    for targets (tq, ta) with token-aligned sources (sq, sa)."""
    t_ids, t_mask, t_spans = _build_batch(tokenizer, tq, ta, device)
    s_ids, s_mask, s_spans = _build_batch(tokenizer, sq, sa, device)
    if t_ids.shape != s_ids.shape or not torch.equal(t_mask, s_mask) or t_spans != s_spans:
        raise SystemExit("sanity (iii) failed: source and target batches are not token-aligned")
    source = capture_out_proj_inputs(model, s_ids, s_mask)
    base = np.array(logprob_of_continuation_batch(model, tokenizer, tq, ta, device))
    deltas = {}
    for (layer, head) in all_heads:
        handle = register_head_intervention(model, layer, head, "resample", source=source)
        try:
            abl = np.array(logprob_of_continuation_batch(model, tokenizer, tq, ta, device))
        finally:
            handle.remove()
        deltas[(layer, head)] = base - abl
    return base, deltas


@torch.no_grad()
def self_source_check(model, tokenizer, device, tq, ta, heads):
    """Sanity (i): resampling from the target itself must change nothing."""
    ids, mask, _ = _build_batch(tokenizer, tq, ta, device)
    source = capture_out_proj_inputs(model, ids, mask)
    base = np.array(logprob_of_continuation_batch(model, tokenizer, tq, ta, device))
    worst = 0.0
    for (layer, head) in heads:
        handle = register_head_intervention(model, layer, head, "resample", source=source)
        try:
            abl = np.array(logprob_of_continuation_batch(model, tokenizer, tq, ta, device))
        finally:
            handle.remove()
        worst = max(worst, float(np.abs(base - abl).max()))
    return worst


def run_one_seed(seed, model, tokenizer, device, copy_q, copy_a, sigs, split_ids, all_heads, num_heads,
                 zero_cands, mean_cands, zero_saved, mean_saved):
    sources = match_resample_sources(sigs, split_ids, np.random.default_rng(seed))
    out = {"source_seed": seed}
    deltas = {}
    for name, sid in (("discovery", 0), ("heldout", 1)):
        idx = [i for i in range(len(copy_q)) if split_ids[i] == sid and sources[i] is not None]
        excluded = [i for i in range(len(copy_q)) if split_ids[i] == sid and sources[i] is None]
        tq, ta = [copy_q[i] for i in idx], [copy_a[i] for i in idx]
        sq, sa = [copy_q[sources[i]] for i in idx], [copy_a[sources[i]] for i in idx]
        _, d = resample_sweep(model, tokenizer, device, tq, ta, sq, sa, all_heads)
        deltas[name] = d
        out[f"n_{name}"] = len(idx)
        out[f"excluded_{name}"] = excluded
        print(f"  seed {seed} {name}: n={len(idx)} excluded={excluded}")
    d_disc = {h: float(v.mean()) for h, v in deltas["discovery"].items()}
    d_held = {h: float(v.mean()) for h, v in deltas["heldout"].items()}

    # R1: the zero-selected candidates under resample
    r1 = set_vs_controls(zero_cands, num_heads, all_heads, d_held, deltas["heldout"])
    # R2: reselection under resample
    res_cands = top_k(d_disc)
    r2 = set_vs_controls(res_cands, num_heads, all_heads, d_held, deltas["heldout"])
    r2["rank_stability_spearman"] = spearman_rank_correlation(d_disc, d_held)
    r2["overlap_with_zero_selected"] = len(set(res_cands) & set(zero_cands))
    r2["overlap_with_mean_selected"] = len(set(res_cands) & set(mean_cands))
    # R3: agreement with the saved zero / mean per-head effects
    r3 = {}
    for split, d in (("heldout", d_held), ("discovery", d_disc)):
        for other, saved in (("zero", zero_saved), ("mean", mean_saved)):
            o = saved[split]
            r3[f"{split}_resample_vs_{other}"] = {
                "pearson": pearson(d, o, all_heads),
                "spearman": spearman_rank_correlation(d, o),
                "top5_overlap": len(set(top_k(d)) & set(top_k(o))),
            }
    print(f"  seed {seed}: R1 zero-selected held-out resample delta={r1['heldout_delta_mean']:.4f} "
          f"p_rand={r1['random_control']['empirical_p']:.3f} p_layer={r1['layer_matched_control']['empirical_p']:.3f}")
    print(f"  seed {seed}: R2 resample-selected={res_cands} held={r2['heldout_delta_mean']:.4f} "
          f"p_rand={r2['random_control']['empirical_p']:.3f} p_layer={r2['layer_matched_control']['empirical_p']:.3f}")
    out.update({
        "r1_zero_selected_under_resample": r1,
        "r2_resample_selected": r2,
        "r3_agreement": r3,
        "per_head_delta_discovery": jsonable_head_dict(d_disc),
        "per_head_delta_heldout": jsonable_head_dict(d_held),
    })
    return out


def reading(r1, zero_effect):
    p = r1["random_control"]["empirical_p"]
    if p > 0.05:
        return "A"
    return "B" if r1["heldout_delta_mean"] >= R1_HALF_ZERO_EFFECT_FACTOR * zero_effect else "C"


def run(model_name: str, output_path: Path):
    t0 = time.time()
    primary = json.load(open(RESULTS_DIR / f"{model_name}_results.json"))
    mean_file = json.load(open(RESULTS_DIR / f"{model_name}_meanablation_results.json"))

    loader = ModelLoader(model_name)
    loader.load()
    model, tokenizer, device = loader.get_model(), loader.get_tokenizer(), loader.get_device()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    num_layers, num_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    all_heads = [(l, h) for l in range(num_layers) for h in range(num_heads)]

    copy_q, copy_a = get_induction_tasks(tokenizer, n=200, seed=INDUCTION_SEED)
    n = len(copy_q)
    half = n // 2
    split_ids = [0] * half + [1] * (n - half)
    sigs = [copying_token_signature(tokenizer, q, a) for q, a in zip(copy_q, copy_a)]
    for q, a, s in zip(copy_q, copy_a, sigs):
        if sum(s) != len(tokenize_qa(tokenizer, q, a)[1]):
            raise SystemExit(f"signature does not match tokenization: {q!r}")

    # Sanity (ii): unablated per-example log-probs match the saved baseline
    base = logprob_of_continuation_batch(model, tokenizer, copy_q, copy_a, device)
    saved_base = primary["baselines"]["induction_full"]["logprob_per_example"]
    base_dev = float(np.max(np.abs(np.array(base) - np.array(saved_base))))
    print(f"[{model_name}] sanity (ii): max |baseline - saved| = {base_dev:.2e}")
    if base_dev > BASELINE_TOL:
        raise SystemExit(f"sanity (ii) failed: {base_dev}")

    rq2 = primary["rq2_candidate_vs_controls"]
    mrq2 = mean_file["mean_ablation"]["rq2_candidate_vs_controls"]
    zero_cands = [tuple(h) for h in rq2["candidate_heads"]]
    mean_cands = [tuple(h) for h in mrq2["candidate_heads"]]
    zero_saved = {"discovery": heads_from_saved(rq2["per_head_delta_discovery"], all_heads),
                  "heldout": heads_from_saved(rq2["per_head_delta_heldout"], all_heads)}
    mean_saved = {"discovery": heads_from_saved(mrq2["per_head_delta_discovery"], all_heads),
                  "heldout": heads_from_saved(mrq2["per_head_delta_heldout"], all_heads)}
    zero_effect = rq2["candidate_heldout_delta_mean"]

    # Sanity (i): self-source resampling is the identity
    self_dev = self_source_check(model, tokenizer, device, copy_q[half:], copy_a[half:], zero_cands)
    print(f"  sanity (i): max |delta| when resampling from the target itself = {self_dev:.2e}")
    if self_dev > SELF_SOURCE_TOL:
        raise SystemExit(f"sanity (i) failed: {self_dev}")

    runs = [run_one_seed(s, model, tokenizer, device, copy_q, copy_a, sigs, split_ids, all_heads, num_heads,
                         zero_cands, mean_cands, zero_saved, mean_saved)
            for s in (PRIMARY_SOURCE_SEED,) + SENSITIVITY_SOURCE_SEEDS]
    primary_run, sens = runs[0], runs[1:]
    r1_vals = [r["r1_zero_selected_under_resample"]["heldout_delta_mean"] for r in runs]

    results = {
        "config": {
            "model_name": model_name,
            "device": device,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "git_commit": git_commit(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "design": "neurips/camera_ready_design.md (Sec. 3, C3), pre-specified",
            "intervention": "pre-projection resample: head slice replaced at every position by its value on a "
                            "different same-split copying example with the same per-word token signature",
            "seeds": {"induction": INDUCTION_SEED, "control_sampling": CONTROL_SAMPLING_SEED,
                      "bootstrap": 3, "primary_source": PRIMARY_SOURCE_SEED,
                      "sensitivity_sources": list(SENSITIVITY_SOURCE_SEEDS)},
            "n_copying": n, "top_k_candidates": TOP_K, "n_control_draws": N_CONTROL_DRAWS,
            "n_bootstrap": N_BOOTSTRAP,
            "signature_counts": {str(k): v for k, v in
                                 sorted({s: sigs.count(s) for s in set(sigs)}.items(), key=lambda kv: -kv[1])},
            "sanity_baseline_max_abs_dev": base_dev,
            "sanity_self_source_max_abs_delta": self_dev,
            "wall_clock_seconds": None,
        },
        "reference": {
            "zero_selected_heads": [list(h) for h in zero_cands],
            "zero_selected_heldout_delta_under_zero": zero_effect,
            "mean_selected_heads": [list(h) for h in mean_cands],
            "reading_rule": "A: p_random > 0.05; B: p_random <= 0.05 and delta >= 0.5 * zero effect; C: otherwise",
        },
        "primary": primary_run,
        "r1_reading": reading(primary_run["r1_zero_selected_under_resample"], zero_effect),
        "r4_source_sensitivity": {
            "runs": sens,
            "r1_heldout_delta_by_seed": dict(zip([r["source_seed"] for r in runs], r1_vals)),
            "r1_heldout_delta_range": [min(r1_vals), max(r1_vals)],
            "r1_reading_by_seed": {r["source_seed"]: reading(r["r1_zero_selected_under_resample"], zero_effect)
                                   for r in runs},
            "r2_heads_by_seed": {r["source_seed"]: r["r2_resample_selected"]["heads"] for r in runs},
        },
    }
    results["config"]["wall_clock_seconds"] = round(time.time() - t0, 2)
    with open(output_path, "x") as f:  # "x": never overwrite
        json.dump(results, f, indent=2)
    print(f"R1 reading: {results['r1_reading']}  (by seed: {results['r4_source_sensitivity']['r1_reading_by_seed']})")
    print(f"Saved {output_path} ({results['config']['wall_clock_seconds']} s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", choices=["gpt2", "distilgpt2"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    out = Path(args.output) if args.output else RESULTS_DIR / f"{args.model}_resample_copying.json"
    run(args.model, out)
