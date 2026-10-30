"""
IOI replication of the ablation-baseline question (NAACL revision, 2026-10-12).

QUESTION
    On an established natural-language circuit task (indirect object
    identification; Wang et al., 2023), do zero-, mean-, and resample-ablation
    at the pre-projection location agree on which heads matter?

PREDICTION (written before running; see idss-backend/NAACL/cross_paper_plan.md, E1)
    The three methods disagree substantially on head rankings (Spearman well
    below 1, partial top-k overlap), and mean and resample agree more with
    each other than either does with zero.
KILL CRITERION
    If all three give Spearman > 0.9 and identical top-5 sets, the baseline
    choice does not change the conclusion on this natural task; that is
    reported as such.

PROTOCOL (mirrors scripts/run_neurips_experiments.py rq2_statistics)
    - N IOI prompts from the 15 templates of Wang et al. (Appendix E), single-
      token names/places/objects; split 50/50 into discovery and held-out.
    - Metric: logit difference LD = logit(IO) - logit(S) at the final prompt
      position. Per-head effect = LD(clean) - LD(ablated), per example.
    - zero:     head's pre-c_proj slice set to 0.
    - mean:     slice set to its mean over all non-padding positions of the
                DISCOVERY split clean prompts (no held-out leakage).
    - resample: slice replaced, per example and position, by the activation
                recorded on the ABC variant (third name in the S2 slot) of a
                *different* example with the same template (token-aligned),
                drawn within the same split.
    - Per method: top-5 heads by discovery effect; held-out effect with
      bootstrap CI; 1000 random and 1000 layer-matched control draws.
    - Cross-method: Spearman over all heads (all prompts) with a bootstrap CI
      over prompts; top-5 / top-10 overlap; ranks of the heads Wang et al.
      report (name movers 9.6, 9.9, 10.0; negative name movers 10.7, 11.10;
      S-inhibition 7.3, 7.9, 8.6, 8.10; GPT-2 small only).
OUTPUT
    results/naacl_ioi/<model>_ioi_ablation_methods.json (never overwrites).
"""

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ablation import capture_out_proj_inputs, register_head_intervention  # noqa: E402
from src.controls import (  # noqa: E402
    bootstrap_mean_ci, empirical_percentile, mean_effect_for_head_set,
    sample_layer_matched_head_sets, sample_random_head_sets, spearman_rank_correlation,
)
from src.dataset import get_ioi_tasks  # noqa: E402

TOP_K = 5
N_BOOT = 2000
N_SPEARMAN_BOOT = 1000
N_CONTROL_DRAWS = 1000
DATA_SEED = 0
SPLIT_SEED = 1
RESAMPLE_SEED = 2
CONTROL_SEED = 42
WANG_HEADS = {
    "name_mover": [(9, 6), (9, 9), (10, 0)],
    "negative_name_mover": [(10, 7), (11, 10)],
    "s_inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
}


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def build_batch(tokenizer, prompts, device):
    ids = [tokenizer.encode(p) for p in prompts]
    max_len = max(len(x) for x in ids)
    pad = tokenizer.eos_token_id
    input_ids = torch.full((len(ids), max_len), pad, dtype=torch.long)
    mask = torch.zeros((len(ids), max_len), dtype=torch.long)
    for i, x in enumerate(ids):
        input_ids[i, :len(x)] = torch.tensor(x)
        mask[i, :len(x)] = 1
    last = torch.tensor([len(x) - 1 for x in ids])
    return input_ids.to(device), mask.to(device), last


@torch.no_grad()
def logit_diff(model, input_ids, mask, last, io_ids, s_ids):
    logits = model(input_ids=input_ids, attention_mask=mask).logits
    rows = torch.arange(len(last))
    final = logits[rows, last.to(logits.device)]
    return (final[rows, io_ids] - final[rows, s_ids]).float().cpu().numpy()


def per_head_deltas(model, batch, io_ids, s_ids, method, means=None, source=None):
    input_ids, mask, last = batch
    base = logit_diff(model, input_ids, mask, last, io_ids, s_ids)
    L, H = model.config.num_hidden_layers, model.config.num_attention_heads
    deltas = {}
    for layer in range(L):
        for head in range(H):
            h = register_head_intervention(model, layer, head, method, means=means, source=source)
            try:
                abl = logit_diff(model, input_ids, mask, last, io_ids, s_ids)
            finally:
                h.remove()
            deltas[(layer, head)] = base - abl
    return base, deltas


def layer_means(model, batch):
    input_ids, mask, _ = batch
    acts = capture_out_proj_inputs(model, input_ids, mask)
    m = mask.bool()
    return {l: a[m].mean(dim=0) for l, a in acts.items()}


def resample_sources(model, tokenizer, exs, device, rng):
    """For each example, the ABC prompt of a different same-template example."""
    by_t = {}
    for i, e in enumerate(exs):
        by_t.setdefault(e["template"], []).append(i)
    src_prompts = []
    for i, e in enumerate(exs):
        pool = [j for j in by_t[e["template"]] if j != i] or [i]
        src_prompts.append(exs[rng.choice(pool)]["abc_prompt"])
    ids, mask, _ = build_batch(tokenizer, src_prompts, device)
    return capture_out_proj_inputs(model, ids, mask), src_prompts


def means_of(deltas):
    return {h: float(np.mean(v)) for h, v in deltas.items()}


def topk(d, k):
    return sorted(d, key=lambda h: d[h], reverse=True)[:k]


def rank_of(d, head):
    order = sorted(d, key=lambda h: d[h], reverse=True)
    return order.index(head) + 1


def selection_stats(all_heads, num_heads, disc, held):
    dm, hm = means_of(disc), means_of(held)
    cands = topk(dm, TOP_K)
    per_ex = np.mean([held[h] for h in cands], axis=0)
    cm, lo, hi = bootstrap_mean_ci(list(per_ex), N_BOOT, seed=3)
    rnd = [mean_effect_for_head_set(hm, d)
           for d in sample_random_head_sets(all_heads, TOP_K, N_CONTROL_DRAWS, seed=CONTROL_SEED)]
    try:
        lay = [mean_effect_for_head_set(hm, d) for d in
               sample_layer_matched_head_sets(cands, num_heads, N_CONTROL_DRAWS, seed=CONTROL_SEED + 1)]
        lay_p = empirical_percentile(cm, lay)
    except ValueError:
        lay, lay_p = [], None
    return {
        "candidate_heads": [list(h) for h in cands],
        "heldout_effect_mean": cm, "heldout_effect_ci95": [lo, hi],
        "random_control_mean": float(np.mean(rnd)),
        "p_random_ge_candidate": empirical_percentile(cm, rnd),
        "layer_matched_control_mean": float(np.mean(lay)) if lay else None,
        "p_layer_matched_ge_candidate": lay_p,
        "rank_stability_spearman_disc_vs_held": spearman_rank_correlation(dm, hm),
    }


def spearman_boot(da, db, n, seed):
    """Bootstrap over prompts: resample example indices, recompute per-head means."""
    heads = sorted(da)
    A = np.stack([da[h] for h in heads])  # [heads, examples]
    B = np.stack([db[h] for h in heads])
    rng = np.random.default_rng(seed)
    m = A.shape[1]
    vals = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        a = dict(zip(heads, A[:, idx].mean(1)))
        b = dict(zip(heads, B[:, idx].mean(1)))
        vals.append(spearman_rank_correlation(a, b))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default=None, help="default: results/naacl_ioi")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / "naacl_ioi"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model.replace('/', '_')}_ioi_ablation_methods.json"
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

    rng = random.Random(RESAMPLE_SEED)
    per_split = {}
    for name, se in splits.items():
        batch = build_batch(tok, [e["prompt"] for e in se], args.device)
        io = torch.tensor([tok.encode(e["io"])[0] for e in se])
        s = torch.tensor([tok.encode(e["s"])[0] for e in se])
        src, _ = resample_sources(model, tok, se, args.device, rng)
        per_split[name] = {"batch": batch, "io": io, "s": s, "src": src}

    disc_means = layer_means(model, per_split["discovery"]["batch"])

    results = {"baseline": {}, "methods": {}}
    deltas = {}
    for method in ("zero", "mean", "resample"):
        deltas[method] = {}
        for name, d in per_split.items():
            base, dl = per_head_deltas(model, d["batch"], d["io"], d["s"], method,
                                       means=disc_means, source=d["src"])
            deltas[method][name] = dl
            results["baseline"][name] = {"mean_logit_diff": float(base.mean()),
                                         "frac_io_preferred": float((base > 0).mean()),
                                         "n": int(len(base))}
        print(f"done {method}", flush=True)

    full = {m: {h: np.concatenate([deltas[m]["discovery"][h], deltas[m]["heldout"][h]])
                for h in all_heads} for m in deltas}
    full_means = {m: means_of(full[m]) for m in full}

    for m in deltas:
        st = selection_stats(all_heads, H, deltas[m]["discovery"], deltas[m]["heldout"])
        st["top10_all_prompts"] = [list(h) for h in topk(full_means[m], 10)]
        if args.model == "gpt2":
            st["wang_head_ranks"] = {grp: {f"{l}.{h}": rank_of(full_means[m], (l, h)) for (l, h) in hs}
                                     for grp, hs in WANG_HEADS.items()}
            st["wang_head_effects"] = {f"{l}.{h}": full_means[m][(l, h)]
                                       for hs in WANG_HEADS.values() for (l, h) in hs}
        results["methods"][m] = st

    pairs = {}
    for a, b in (("zero", "mean"), ("zero", "resample"), ("mean", "resample")):
        pairs[f"{a}_vs_{b}"] = {
            "spearman_all_heads": spearman_rank_correlation(full_means[a], full_means[b]),
            "spearman_ci95_bootstrap_prompts": spearman_boot(full[a], full[b], N_SPEARMAN_BOOT, seed=7),
            "top5_overlap": len(set(topk(full_means[a], 5)) & set(topk(full_means[b], 5))),
            "top10_overlap": len(set(topk(full_means[a], 10)) & set(topk(full_means[b], 10))),
        }
    results["cross_method"] = pairs
    results["per_head_mean_effect"] = {m: {f"{l}.{h}": v for (l, h), v in fm.items()}
                                       for m, fm in full_means.items()}
    results["config"] = {
        "model": args.model, "n_prompts": len(exs), "split": [half, len(exs) - half],
        "metric": "logit(IO) - logit(S) at final prompt position; effect = clean - ablated",
        "location": "pre-output-projection (c_proj/o_proj input)",
        "mean_reference": "discovery-split clean prompts, all non-padding positions",
        "resample_source": "ABC variant of a different same-template example, same split",
        "top_k": TOP_K, "n_control_draws": N_CONTROL_DRAWS, "n_boot": N_BOOT,
        "n_spearman_boot": N_SPEARMAN_BOOT,
        "seeds": {"data": DATA_SEED, "split": SPLIT_SEED, "resample": RESAMPLE_SEED,
                  "control": CONTROL_SEED},
        "device": args.device, "dtype": "float32",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "git_commit": git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "templates_source": "Wang et al. 2023, Appendix E, Figure 14",
    }
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}")
    print(json.dumps({"baseline": results["baseline"], "cross_method": pairs,
                      "selection": {m: {k: v for k, v in results["methods"][m].items()
                                        if k not in ("wang_head_effects",)}
                                    for m in results["methods"]}}, indent=1))


if __name__ == "__main__":
    main()
