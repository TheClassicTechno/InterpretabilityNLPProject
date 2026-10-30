#!/usr/bin/env python3
"""
Generates the figures used in the paper, reading only from the saved
experiment JSON (neurips/experiment_results/<model>_results.json) --
no numbers are re-derived or guessed here.

Figure 1 (3-panel main figure):
  A. Baseline capability (accuracy vs. log-prob) per task.
  B. Candidate vs. matched-control (random, layer-matched) delta
     distributions on held-out copying-task examples.
  C. Specificity: target-task delta vs. control-task delta for candidates
     vs. the random-control null distribution.

Figure 2 (supplementary, RQ0):
  Per-head delta under the naive (post-projection) vs. correct
  (pre-projection) ablation method, discovery split.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "neurips" / "experiment_results" / "gpt2_results.json"
FIG_DIR = ROOT / "neurips" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})


def load():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def make_main_figure(d):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))

    # --- Panel A: baseline capability, accuracy vs log-prob ---
    ax = axes[0]
    tasks = ["copying\n(target)", "arithmetic\n(control)"]
    acc = [d["baselines"]["induction_full"]["accuracy_mean"], d["baselines"]["arithmetic_control"]["accuracy_mean"]]
    lp = [d["baselines"]["induction_full"]["logprob_mean"], d["baselines"]["arithmetic_control"]["logprob_mean"]]
    lp_lo = [d["baselines"]["induction_full"]["logprob_ci95"][0], d["baselines"]["arithmetic_control"]["logprob_ci95"][0]]
    lp_hi = [d["baselines"]["induction_full"]["logprob_ci95"][1], d["baselines"]["arithmetic_control"]["logprob_ci95"][1]]

    x = np.arange(len(tasks))
    width = 0.35
    ax2 = ax.twinx()
    b1 = ax.bar(x - width / 2, acc, width, color="#4C72B0", label="accuracy")
    b2 = ax2.bar(x + width / 2, lp, width, color="#DD8452", label="log-prob")
    ax2.errorbar(
        x + width / 2, lp,
        yerr=[np.array(lp) - np.array(lp_lo), np.array(lp_hi) - np.array(lp)],
        fmt="none", ecolor="black", capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("accuracy (teacher-forced exact match)", color="#4C72B0")
    ax2.set_ylabel("mean log-prob of gold continuation (nats)", color="#DD8452")
    ax.set_ylim(0, 1.05)
    ax.set_title("A. Accuracy is thresholded;\nlog-prob stays continuous")
    ax.tick_params(axis="y", labelcolor="#4C72B0")
    ax2.tick_params(axis="y", labelcolor="#DD8452")

    # --- Panel B: candidate vs matched controls, held-out ---
    ax = axes[1]
    rq2 = d["rq2_candidate_vs_controls"]
    cand_mean = rq2["candidate_heldout_delta_mean"]
    cand_ci = rq2["candidate_heldout_delta_ci95"]
    rand = rq2["random_control_heldout_delta"]
    layer = rq2["layer_matched_control_heldout_delta"]

    labels = ["candidate\n(top-5)", "random\ncontrol", "layer-matched\ncontrol"]
    means = [cand_mean, rand["mean"], layer["mean"]]
    lo = [cand_ci[0], rand["p5"], layer["p5"]]
    hi = [cand_ci[1], rand["p95"], layer["p95"]]
    colors = ["#4C72B0", "#999999", "#999999"]

    xpos = np.arange(3)
    ax.bar(xpos, means, color=colors)
    ax.errorbar(
        xpos, means,
        yerr=[np.array(means) - np.array(lo), np.array(hi) - np.array(means)],
        fmt="none", ecolor="black", capsize=3,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$\Delta$ log-prob (baseline $-$ ablated)")
    ax.set_title("B. Candidate heads vs. matched\ncontrols, held-out split")

    # --- Panel C: specificity ---
    ax = axes[2]
    spec = rq2["specificity"]
    labels = ["candidate", "random\ncontrol (mean)"]
    means = [spec["candidate"], spec["random_control_mean"]]
    err_lo = [0, spec["random_control_mean"] - spec["random_control_p5"]]
    err_hi = [0, spec["random_control_p95"] - spec["random_control_mean"]]
    xpos = np.arange(2)
    ax.bar(xpos, means, color=["#4C72B0", "#999999"])
    ax.errorbar(xpos, means, yerr=[err_lo, err_hi], fmt="none", ecolor="black", capsize=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"specificity: $\Delta_{copying} - \Delta_{arithmetic}$")
    ax.set_title(f"C. Specificity vs. random-control\nnull (Monte Carlo p={spec['empirical_p_candidate_beats_random']:.3f})")

    fig.tight_layout()
    out = FIG_DIR / "main_figure.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")  # vector copy for LaTeX
    plt.close(fig)
    print(f"Saved {out}")


def make_rq0_figure(d):
    rq0 = d["rq0_naive_vs_correct"]
    correct = rq0["per_head_delta_correct"]
    naive = rq0["per_head_delta_naive"]
    keys = sorted(correct.keys())
    x = np.array([correct[k] for k in keys])
    y = np.array([naive[k] for k in keys])

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.scatter(x, y, s=14, alpha=0.7, color="#4C72B0")
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, color="gray", linestyle="--", linewidth=1, label="y = x")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel(r"$\Delta$ log-prob, correct (pre-projection) ablation")
    ax.set_ylabel(r"$\Delta$ log-prob, naive (post-projection) ablation")
    ax.set_title(
        f"RQ0: per-head effect, correct vs. naive ablation\n"
        f"Pearson r={rq0['pearson_r']:.3f}, top-5 overlap={rq0['top5_head_overlap']}/5"
    )
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    out = FIG_DIR / "rq0_naive_vs_correct.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")  # vector copy for LaTeX
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    d = load()
    make_main_figure(d)
    make_rq0_figure(d)
