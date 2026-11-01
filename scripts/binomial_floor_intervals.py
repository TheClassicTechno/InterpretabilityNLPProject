#!/usr/bin/env python3
"""
Exact (Clopper-Pearson) 95% intervals for the arithmetic accuracy floors.

When every example fails, the percentile bootstrap reported in the result
files is degenerate ([0, 0]): every resample also has accuracy 0, so it says
nothing about accuracy on new examples. This script reads only the saved
baselines (no forward passes) and writes binomial_floor_intervals.json,
refusing to overwrite an existing file.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.controls import clopper_pearson_ci  # noqa: E402

RESULTS = ROOT / "neurips" / "experiment_results"
SOURCES = {
    "gpt2": "gpt2_results.json",
    "distilgpt2": "distilgpt2_results.json",
    "gpt2_n500": "gpt2_n500_results.json",
    # Its "baselines" block is the unablated model, shared by zero and mean.
    "qwen2.5-1.5b-instruct": "qwen2.5-1.5b-instruct_meanablation_results.json",
}


def floor_interval(filename):
    b = json.loads((RESULTS / filename).read_text())["baselines"]["arithmetic_control"]
    n = b["n"]
    k = round(b["accuracy_mean"] * n)
    assert abs(k / n - b["accuracy_mean"]) < 1e-12, "accuracy is not k/n"
    lo, hi = clopper_pearson_ci(k, n)
    return {
        "source": filename,
        "n": n,
        "k_correct": k,
        "accuracy": k / n,
        "bootstrap_ci95_saved": b["accuracy_ci95"],
        "clopper_pearson_ci95": [lo, hi],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(RESULTS / "binomial_floor_intervals.json"))
    args = parser.parse_args()
    out = {
        "note": "exact Clopper-Pearson 95% intervals for arithmetic exact match; "
                "computed from saved baselines, no forward passes",
        **{name: floor_interval(f) for name, f in SOURCES.items()},
    }
    with open(args.output, "x") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
