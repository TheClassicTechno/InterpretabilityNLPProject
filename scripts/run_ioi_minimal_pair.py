"""
IOI minimal-pair resample control (NAACL round 3, 2026-10-12).

QUESTION
    The E1 resample ablation replaces a head's pre-projection slice with the
    activation from the ABC prompt of a *different* same-template example, so
    names, places and objects all change at once. Is the resulting head
    ranking driven by the IOI cue (repeated subject name), or by the other
    content that changes simultaneously?

CONTROL
    Minimal pair: the source is the ABC prompt of the *same* example. It differs
    from the clean prompt only in the second subject occurrence (S2 -> a third
    name C). All other tokens are identical and positions align. This is
    Wang et al.'s ABC counterfactual applied per example.

PREDICTION (written before running; idss-backend/NAACL/cross_paper_plan.md, E6)
    Spearman (all 144 heads) between the minimal-pair and different-example
    rankings >= 0.8 and top-5 overlap >= 3.
KILL / WEAKENING
    rho < 0.6 or top-5 overlap <= 2: the different-example ranking depends on
    non-cue content, and the paper's resample results are scoped accordingly.

REGRESSION CHECK
    The different-example resample is recomputed with the E1 seeds and must
    reproduce the E1 per-head means (results/naacl_ioi/gpt2_ioi_ablation_methods.json).
OUTPUT
    results/naacl_ioi/<model>_ioi_minimal_pair.json (never overwrites).
"""

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_ioi_ablation_methods import (  # noqa: E402
    DATA_SEED, RESAMPLE_SEED, SPLIT_SEED, TOP_K, WANG_HEADS, build_batch, git_commit,
    means_of, per_head_deltas, rank_of, resample_sources, selection_stats, spearman_boot, topk,
)
from src.ablation import capture_out_proj_inputs  # noqa: E402
from src.controls import spearman_rank_correlation  # noqa: E402
from src.dataset import get_ioi_tasks  # noqa: E402


def minimal_pair_sources(model, tokenizer, exs, device):
    """Same-example ABC prompt: only the S2 token differs from the clean prompt."""
    for e in exs:
        a, b = tokenizer.encode(e["prompt"]), tokenizer.encode(e["abc_prompt"])
        assert len(a) == len(b), "minimal pair not token-aligned"
        n_diff = sum(x != y for x, y in zip(a, b))
        assert n_diff == 1, f"minimal pair differs in {n_diff} tokens"
    ids, mask, _ = build_batch(tokenizer, [e["abc_prompt"] for e in exs], device)
    return capture_out_proj_inputs(model, ids, mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = ROOT / "results" / "naacl_ioi"
    out_path = out_dir / f"{args.model.replace('/', '_')}_ioi_minimal_pair.json"
    e1_path = out_dir / f"{args.model.replace('/', '_')}_ioi_ablation_methods.json"
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite {out_path}")

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device).eval()
    L, H = model.config.num_hidden_layers, model.config.num_attention_heads
    all_heads = [(l, h) for l in range(L) for h in range(H)]

    exs = get_ioi_tasks(tok, n=args.n, seed=DATA_SEED)
    idx = list(range(len(exs)))
    random.Random(SPLIT_SEED).shuffle(idx)
    half = len(idx) // 2
    splits = {"discovery": [exs[i] for i in idx[:half]], "heldout": [exs[i] for i in idx[half:]]}

    rng = random.Random(RESAMPLE_SEED)  # same draw order as E1
    per_split = {}
    for name, se in splits.items():
        batch = build_batch(tok, [e["prompt"] for e in se], args.device)
        io = torch.tensor([tok.encode(e["io"])[0] for e in se])
        s = torch.tensor([tok.encode(e["s"])[0] for e in se])
        src_diff, _ = resample_sources(model, tok, se, args.device, rng)
        src_same = minimal_pair_sources(model, tok, se, args.device)
        per_split[name] = {"batch": batch, "io": io, "s": s, "diff": src_diff, "same": src_same}

    deltas = {}
    for variant in ("diff", "same"):
        deltas[variant] = {}
        for name, d in per_split.items():
            _, dl = per_head_deltas(model, d["batch"], d["io"], d["s"], "resample", source=d[variant])
            deltas[variant][name] = dl
        print(f"done resample_{variant}", flush=True)

    full = {v: {h: np.concatenate([deltas[v]["discovery"][h], deltas[v]["heldout"][h]]) for h in all_heads}
            for v in deltas}
    fm = {v: means_of(full[v]) for v in full}

    regression = None
    if e1_path.exists():
        e1 = json.loads(e1_path.read_text())["per_head_mean_effect"]["resample"]
        max_abs = max(abs(fm["diff"][(l, h)] - e1[f"{l}.{h}"]) for (l, h) in all_heads)
        regression = {"max_abs_diff_vs_E1_resample": max_abs, "reproduced": bool(max_abs < 1e-3)}
        print("regression vs E1:", regression, flush=True)

    res = {"methods": {}}
    for v, label in (("diff", "resample_different_example"), ("same", "resample_minimal_pair")):
        st = selection_stats(all_heads, H, deltas[v]["discovery"], deltas[v]["heldout"])
        st["top10_all_prompts"] = [list(h) for h in topk(fm[v], 10)]
        if args.model == "gpt2":
            st["wang_head_ranks"] = {g: {f"{l}.{h}": rank_of(fm[v], (l, h)) for (l, h) in hs}
                                     for g, hs in WANG_HEADS.items()}
        res["methods"][label] = st

    res["comparison"] = {
        "spearman_all_heads": spearman_rank_correlation(fm["diff"], fm["same"]),
        "spearman_ci95_bootstrap_prompts": spearman_boot(full["diff"], full["same"], 1000, seed=7),
        "top5_overlap": len(set(topk(fm["diff"], 5)) & set(topk(fm["same"], 5))),
        "top10_overlap": len(set(topk(fm["diff"], 10)) & set(topk(fm["same"], 10))),
        "prediction": "rho >= 0.8 and top5 overlap >= 3",
        "kill": "rho < 0.6 or top5 overlap <= 2",
    }
    if e1_path.exists():
        e1m = json.loads(e1_path.read_text())["per_head_mean_effect"]
        for m in ("zero", "mean"):
            other = {tuple(map(int, k.split("."))): v for k, v in e1m[m].items()}
            res["comparison"][f"spearman_minimal_pair_vs_{m}"] = spearman_rank_correlation(fm["same"], other)
            res["comparison"][f"top5_overlap_minimal_pair_vs_{m}"] = len(set(topk(fm["same"], 5)) & set(topk(other, 5)))
    res["regression_check"] = regression
    res["per_head_mean_effect"] = {v: {f"{l}.{h}": x for (l, h), x in fm[v].items()} for v in fm}
    res["config"] = {
        "model": args.model, "n_prompts": len(exs), "split": [half, len(exs) - half],
        "metric": "logit(IO) - logit(S) at final position; effect = clean - ablated",
        "location": "pre-output-projection (c_proj input)",
        "minimal_pair_source": "ABC prompt of the same example (only S2 token differs; asserted)",
        "different_example_source": "ABC prompt of a different same-template example, same split (E1)",
        "top_k": TOP_K, "seeds": {"data": DATA_SEED, "split": SPLIT_SEED, "resample": RESAMPLE_SEED},
        "device": args.device, "torch": torch.__version__, "transformers": transformers.__version__,
        "git_commit": git_commit(), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_path.write_text(json.dumps(res, indent=2))
    print(f"wrote {out_path}")
    print(json.dumps({"comparison": res["comparison"], "regression": regression,
                      "selection": {k: {kk: vv for kk, vv in v.items()} for k, v in res["methods"].items()}},
                     indent=1))


if __name__ == "__main__":
    main()
