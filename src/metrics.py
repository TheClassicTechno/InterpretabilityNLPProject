"""
Continuous and exact-match metrics based on teacher forcing.

The original accuracy metric (`str(answer).strip() in response`) had two
problems: (1) it is a substring match, so answer "8" matches inside a
generated "18" or "80"; (2) it is binary, so it saturates at 0% or 100%
and cannot detect graded ablation effects (the "floor effect" problem
this project is about). Both metrics here operate via a single teacher-
forced forward pass (no autoregressive generation), which is both more
correct and much cheaper.
"""

import torch
from typing import List, Tuple


def tokenize_qa(tokenizer, question: str, answer: str) -> Tuple[List[int], List[int]]:
    """
    Tokenize `question` and `question + answer`, and verify that the
    question's tokenization is an exact token-level prefix of the
    combined tokenization (BPE is not strictly prefix-stable, so this
    can occasionally fail at token boundaries).

    Returns (prefix_ids, full_ids). Raises ValueError if the prefix
    property does not hold or the answer contributes zero tokens.
    """
    prefix_ids = tokenizer(question)["input_ids"]
    full_ids = tokenizer(question + answer)["input_ids"]

    if len(full_ids) <= len(prefix_ids):
        raise ValueError(
            f"Answer produced no additional tokens: question={question!r} answer={answer!r}"
        )
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError(
            f"Tokenization of question is not a prefix of question+answer "
            f"(BPE merge crossed the boundary): question={question!r} answer={answer!r}"
        )
    return prefix_ids, full_ids


def is_valid_qa(tokenizer, question: str, answer: str) -> bool:
    try:
        tokenize_qa(tokenizer, question, answer)
        return True
    except ValueError:
        return False


def _build_batch(tokenizer, questions: List[str], answers: List[str], device):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    all_ids = []
    answer_spans = []  # (start, end) exclusive, indices into the padded sequence
    for q, a in zip(questions, answers):
        prefix_ids, full_ids = tokenize_qa(tokenizer, q, a)
        all_ids.append(full_ids)
        answer_spans.append((len(prefix_ids), len(full_ids)))

    max_len = max(len(ids) for ids in all_ids)
    batch_size = len(all_ids)

    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    for i, ids in enumerate(all_ids):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, : len(ids)] = 1

    return input_ids.to(device), attention_mask.to(device), answer_spans


@torch.no_grad()
def logprob_of_continuation_batch(
    model, tokenizer, questions: List[str], answers: List[str], device="cpu"
) -> List[float]:
    """
    Teacher-forced sum log-probability (natural log, base e) of the gold
    answer continuation for each (question, answer) pair, in ONE batched
    forward pass. Position t's log-prob is read from the logits produced
    at position t-1 (standard next-token teacher forcing).

    Returns a list of floats, one per example.
    """
    if not questions:
        return []

    input_ids, attention_mask, answer_spans = _build_batch(tokenizer, questions, answers, device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    log_probs = torch.log_softmax(outputs.logits, dim=-1)

    results = []
    for i, (start, end) in enumerate(answer_spans):
        total = 0.0
        for t in range(start, end):
            token_id = input_ids[i, t].item()
            total += log_probs[i, t - 1, token_id].item()
        results.append(total)
    return results


@torch.no_grad()
def teacher_forced_exact_match_batch(
    model, tokenizer, questions: List[str], answers: List[str], device="cpu"
) -> List[bool]:
    """
    Whether the greedy argmax token at every answer position matches the
    gold token, under teacher forcing (single batched forward pass).
    This is a stricter, unambiguous replacement for the old substring
    match, evaluated in the same forward pass structure as the log-prob
    metric so the two are directly comparable.
    """
    if not questions:
        return []

    input_ids, attention_mask, answer_spans = _build_batch(tokenizer, questions, answers, device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    preds = outputs.logits.argmax(dim=-1)

    results = []
    for i, (start, end) in enumerate(answer_spans):
        correct = all(
            preds[i, t - 1].item() == input_ids[i, t].item() for t in range(start, end)
        )
        results.append(correct)
    return results
