# Responses to reviewers: InterpScience 2026, Submission 112 (camera-ready)

Every revision listed here is present in `neurips/neurips_2026.tex` / `neurips_2026.pdf` as built on 2026-10-30.
Section numbers refer to the camera-ready PDF.

## Response to Reviewer QC4F

Thank you for the careful and constructive review, and for your comments on the control design and reproducibility.

**Alternative ablation modalities.** We added mean-ablation for the full RQ2 pipeline on GPT-2 and DistilGPT2 (§4.4, Table 2).
The reference means come from the discovery split only. Your concern was warranted. Under mean-ablation the protocol still
separates reselected heads from both matched controls (p = 0.001), but on GPT-2 the selected heads are disjoint from the
zero-ablation set, and the five zero-selected heads' held-out effect falls from 0.412 to 0.005 nats (post hoc). Their zero-ablation
effect is therefore consistent with an off-distribution replacement rather than the removal of information the task needs. We
revised the abstract, introduction, discussion, and conclusion accordingly and added a fourth recommendation: report the
replacement value and check at least one on-distribution baseline. We then tested whether the mean itself was the problem, with a
resample-ablation analysis specified before it was run: each head's output is replaced by its output on another copying example
with the same token layout (§4.4, Table 2). On GPT-2 the zero-selected heads' held-out effect is 0.020 nats (p = 0.346 against
random head sets; 0.005 and 0.011 with two other source assignments), so resample agrees with mean. On DistilGPT2 resample selects
exactly the mean-ablation heads. On IOI we also compare zero, mean, and resample ablation (§4.5). Noise injection and optimal
ablation remain untested, and §5 says so.

**Model diversity and libraries.** We added Qwen2.5-1.5B-Instruct, a different architecture family (App. C, Table 4). The projection
confound recurs (r = 0.108, 0/5 overlap), and the matched-control separation holds under both baselines. §5 now states that DistilGPT2
is distilled from GPT-2 and so is not an independent replication, and that nothing above 1.5B parameters was tested. On libraries, §2
now explains that the confound depends on where the intervention is placed, not on the library. TransformerLens and nnsight expose both
the valid pre-projection location and the mixed block output, and we state explicitly that we did not measure how often the naive
placement occurs.

**Task simplicity.** §3 now explains why the tasks are deliberately simple. We added IOI on GPT-2, using the published templates
(§4.5, App. D). Under all three baselines the selected heads exceed both controls, but zero-ablation shares only 2 of its top-5 heads
with mean or resample ablation. Greater-Than was not tested.

## Response to Reviewer E5js

Thank you for the evaluation and for drawing the distinction between reproducible intervention effects and task-specific mechanisms.

**Arithmetic as a control.** We agree, and we tested the concern directly. §3 and §5 now state that arithmetic differs from copying in
template, computation, tokenization, and baseline, so a contrast between them cannot isolate a copying-specific computation. We added a
format-matched control, ordered-sequence continuation (`1 2 3 4 . 5 6 7 → 8`), specified in writing before running. It has the same
template, a single-token answer, and a similar baseline (GPT-2: 96.9% vs 99.5%), but the answer cannot be copied from the prompt. On
GPT-2 the candidates harm this task as much as copying (Δ = 0.447 vs 0.412 nats), and specificity disappears (p = 0.567; mean-ablation
candidates p = 0.882). On DistilGPT2 the effect remains larger on copying (p = 0.001). We report both outcomes. We also corrected a
transcription error in the submitted version: the random-control specificity range is [−0.135, 0.165], not [−0.130, 0.163].

**Task specificity on GPT-2.** We kept both the primary (p = 0.078) and the N = 500 (p = 0.274) results. The wording now calls the
arithmetic contrast inconclusive rather than suggestive. A new 20-partition sensitivity analysis shows that the arithmetic p-value
crosses 0.05 in 9 of 20 partitions, while the matched-control separation holds in all 20 (§4.3, App. E).

**Model family.** Addressed as in the response to QC4F (Qwen2.5 replication; DistilGPT2 described as non-independent).

## Other camera-ready changes

- §4.1: removed the "comparable in magnitude" comparison (0.067 vs 0.412 nats).
- §4.2: a format check shows that GPT-2's arithmetic floor survives removing the answer's leading space; Qwen's does not (0.0% → 23.0%).
- De-anonymized author block, workshop footer (`final` option), and public code link. The naive hook is identified as this project's
  own original implementation.
- `scripts/verify_paper_numbers.py` checks 140 printed values against the result files.
