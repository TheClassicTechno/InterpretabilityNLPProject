#!/usr/bin/env python3
"""
Diagnostic (post hoc): is a model's 0% teacher-forced exact match on the
arithmetic control task a capability floor or an answer-format artifact?

The arithmetic answer is " <a+b>". For tokenizers that split digits and the
leading space into separate tokens (e.g. Qwen2.5: " 24" -> " ", "2", "4"),
exact match also requires predicting the bare space token. This script
reports, per model, how often the first answer token is predicted, how often
all remaining answer tokens are predicted given the gold prefix, and the
most frequent argmax at the first answer position. Same data and seeds as
the main pipeline; no ablation.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import get_arithmetic_control_tasks
from src.metrics import _build_batch, is_valid_qa, teacher_forced_exact_match_batch
from src.model_loader import ModelLoader

ARITHMETIC_SEED = 1


@torch.no_grad()
def diagnose(model_name, n=200, max_operand=20):
    loader = ModelLoader(model_name).load()
    model, tok, device = loader.get_model(), loader.get_tokenizer(), loader.get_device()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    qs, ans = get_arithmetic_control_tasks(tok, n=n, seed=ARITHMETIC_SEED, max_operand=max_operand)
    ids, mask, spans = _build_batch(tok, qs, ans, device)
    logits = model(input_ids=ids, attention_mask=mask).logits

    first_ok = rest_ok = 0
    first_pred_counts = {}
    for i, (start, end) in enumerate(spans):
        preds = [logits[i, k - 1].argmax().item() for k in range(start, end)]
        gold = [ids[i, k].item() for k in range(start, end)]
        first_ok += preds[0] == gold[0]
        rest_ok += preds[1:] == gold[1:]
        t = tok.convert_ids_to_tokens(preds[0])
        first_pred_counts[t] = first_pred_counts.get(t, 0) + 1

    # Same questions, answer written WITHOUT the leading space ("24" not " 24").
    unspaced = [(q, a.strip()) for q, a in zip(qs, ans) if is_valid_qa(tok, q, a.strip())]
    uq, ua = [q for q, _ in unspaced], [a for _, a in unspaced]
    unspaced_em = teacher_forced_exact_match_batch(model, tok, uq, ua, device) if uq else []

    return {
        "model": model_name,
        "n": len(qs),
        "example_answer_tokens": tok.tokenize(ans[0]),
        "first_answer_token_correct": first_ok,
        "remaining_answer_tokens_all_correct_given_gold_prefix": rest_ok,
        "top_first_position_argmax": sorted(first_pred_counts.items(), key=lambda kv: -kv[1])[:5],
        "unspaced_answer_n_valid": len(uq),
        "unspaced_answer_exact_match": float(sum(unspaced_em) / len(unspaced_em)) if unspaced_em else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["gpt2", "distilgpt2", "Qwen/Qwen2.5-1.5B-Instruct"])
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "neurips" / "experiment_results" / "arithmetic_format_diagnostic.json"))
    args = parser.parse_args()
    out = {"note": "post hoc diagnostic; teacher-forced argmax on the arithmetic control task (seed 1, N=200, max operand 20)",
           "results": [diagnose(m) for m in args.models]}
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))
