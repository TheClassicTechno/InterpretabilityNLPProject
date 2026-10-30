#!/usr/bin/env python3
"""
POST-HOC analysis (not pre-committed): how large is the held-out effect of
the ZERO-ablation-selected candidate heads when those same heads are
mean-ablated instead? Reads only the saved <model>_meanablation_results.json
files (no forward passes) and writes mean_vs_zero_posthoc.json.
"""

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "neurips" / "experiment_results"


def summarize(model):
    d = json.loads((RESULTS / f"{model}_meanablation_results.json").read_text())
    zero_cands = [f"{l}_{h}" for l, h in d["rq2_candidate_vs_controls"]["candidate_heads"]]
    zero_held = d["rq2_candidate_vs_controls"]["per_head_delta_heldout"]
    mean_held = d["mean_ablation"]["rq2_candidate_vs_controls"]["per_head_delta_heldout"]
    return {
        "zero_selected_candidates": zero_cands,
        "heldout_delta_under_zero": {k: zero_held[k] for k in zero_cands},
        "heldout_delta_under_mean": {k: mean_held[k] for k in zero_cands},
        "heldout_delta_under_zero_mean_over_candidates": float(np.mean([zero_held[k] for k in zero_cands])),
        "heldout_delta_under_mean_mean_over_candidates": float(np.mean([mean_held[k] for k in zero_cands])),
    }


if __name__ == "__main__":
    out = {"note": "post hoc; computed from saved per-head held-out deltas",
           "gpt2": summarize("gpt2"), "distilgpt2": summarize("distilgpt2"),
           "qwen2.5-1.5b-instruct": summarize("qwen2.5-1.5b-instruct")}
    path = RESULTS / "mean_vs_zero_posthoc.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
