# Camera-ready design doc: InterpScience @ NeurIPS 2026 (Submission 112)

Created 2026-10-30. This doc expands `camera_ready_plan.md` and is the spec the implementation follows.
Target file: **`neurips/neurips_2026.tex`**. Deadline: 2026-11-15 AoE.

## 0. Decisions (user, 2026-10-30)

| # | Decision |
|---|---|
| 1 | Report the mean-ablation robustness result in the camera-ready, including the post hoc 0.412 → 0.005 nats |
| 2 | Port the Qwen2.5-1.5B-Instruct replication and the GPT-2 IOI (zero/mean/resample) results |
| 3 | Run both new experiments: C1 (split stability) and C2 (format-matched control task) |
| 4 | Author block: Juli Huang, Stanford University, `julih@stanford.edu` |
| 5 | (added later 2026-10-30) Run resample ablation on the copying task (C3). Noise ablation, Greater-Than, and >1.5B models stay stated limitations |

## 1. Invariants (must hold at the end)

1. The original result files are never modified: `gpt2_results.json`, `distilgpt2_results.json`, `gpt2_n500_results.json`,
   `*_meanablation_results.json`, `mean_vs_zero_posthoc.json`, `arithmetic_format_diagnostic.json`, `results/naacl_ioi/*.json`.
   Their SHA-256 hashes are checked before and after.
2. New results go to new files: `neurips/experiment_results/<model>_camera_ready_robustness.json`. The script refuses to overwrite.
3. The current (uncommitted) code reproduces `gpt2_results.json` exactly, apart from `timestamp_utc` and `git_commit`.
   This is checked before any ported number is trusted.
4. Every number in the `.tex` is read from a JSON field, checked by a script, and listed in `claim_evidence_map.md`.
5. Main text ≤ 9 pages. References, appendix, and checklist are excluded.
6. All citations exist and were already verified (NAACL bib) or are verified now (Gould et al. 2024; checked via web search 2026-10-30:
   ICLR 2024, arXiv 2312.09230).
7. No claim exceeds evidence. In particular, the paper no longer says the zero-selected GPT-2 heads have a baseline-independent effect.

## 2. Scientific framing after the changes

The three original findings stay; the RQ2 claim is qualified:

- **RQ0 (unchanged, strengthened):** the post-projection slice is not a head ablation. r = 0.057 / 0.001 / 0.108 on GPT-2 / DistilGPT2 / Qwen2.5, with 0/5 overlap on each.
- **RQ1 (qualified):** accuracy is uninformative at the arithmetic floor. For GPT-2/DistilGPT2 the floor persists without the leading
  space (0.0%), so it is not just answer format; for Qwen2.5 it *is* format (23.0% unspaced). Log-prob stays graded.
- **RQ2 (qualified):** under zero-ablation, the selected heads' held-out effect exceeds matched controls. Under mean-ablation the protocol
  still separates reselected heads from controls, but on GPT-2 the selected heads are disjoint and the zero-selected heads' effect vanishes
  (0.412 → 0.005). The selected heads and their effect depend on the replacement value. Specificity is not robust on GPT-2;
  C2 tests it against a format-matched control.
- **New 4th lesson:** report the replacement value and at least one on-distribution baseline.

## 3. Item specs

Each item lists the exact edit. "Port" = text adapted from `NAACL/naacl_submission/acl_latex.tex`, with each number re-verified against JSON.

### Writing-only

- **B1 (§4.1):** replace "comparable in magnitude to the candidate heads' own held-out effect (0.412 nats)" with
  "about one sixth of the candidate heads' held-out effect (0.412 nats)". Ratio 0.067/0.412 = 0.163 (computed from JSON).
- **A5 (specificity wording):** abstract, §1 (3), §4.3, Fig. 1C caption, Table 1 note. Use "inconclusive / not robust", not "positive trend";
  keep p = 0.078 and 0.274. In App. B, note that DistilGPT2's p = 0.001 is a single run with no rerun, measured against the same mismatched control.
- **A7 (§3 Controls + §4.3):** add one sentence saying the random and layer-matched controls ask whether the effect is unusual among heads, the
  cross-task contrast asks whether the effect differs between tasks, and passing the first does not imply the second.
- **A3 (§2):** add a paragraph "Scope of the failure", adapted from CR passage B + the NAACL "Where the confound can arise" paragraph
  (TransformerLens `hook_z` vs `hook_attn_out`; nnsight exposes both). Cite `nanda2022transformerlens`, `fiottokaufman2025nnsight`.
  Prevalence is not measured.
- **A4a (§3):** add a "Choice of tasks" paragraph (CR passage C), with the copying task, arithmetic, and the new sequence-continuation task.
- **A6 (§5):** add a "Task-control mismatch" paragraph (CR passage A), pointing to C2 as the format-matched follow-up.
- **B2:** `\usepackage[dblblindworkshop,final]{neurips_2026}`; author block with name, affiliation, and email.
  The footer becomes "40th Conference … Workshop: Interpretability as a Science." The checklist stays (`\input{checklist.tex}`).
- **B3:** reword the "pre-existing, previously unaudited implementation" passages so they no longer depend on anonymity; the claim stays the same.
  Replace "anonymized supplementary archive" with the public repository URL (as in the arXiv version).
- **Related work for baselines:** one sentence citing `li2024optimal` (optimal ablation) and mean-ablation practice in `wang2023interpretability`. Copy the
  verified bib entries from `NAACL/naacl_submission/custom.bib`, converted to the existing `references.bib` key style.

### Ported experiments (EXP-PORT)

- **A1 Mean-ablation:** new §4.4, "Robustness to the ablation baseline", plus Table "tab:mean" (GPT-2 + DistilGPT2 columns). Numbers come from
  `{gpt2,distilgpt2}_meanablation_results.json` → `mean_ablation.*` and from `mean_vs_zero_posthoc.json`, with the post hoc value labelled as post hoc.
  Design facts to state: means are computed over all non-padding positions of the discovery split only, then applied to every split; selection,
  controls, and seeds are identical. Replace the §5 "Zero-ablation is off-distribution … we did not run" paragraph with an
  "Ablation baseline" limitation (noise and optimal ablation are untested).
- **A2 Qwen2.5:** add App. "Qwen2.5-1.5B-Instruct replication" (table) and 2–3 sentences in §5 "Model scope". Numbers come from
  `qwen2.5-1.5b-instruct_meanablation_results.json`. Cite `qwen2025qwen25`. State the cost: 11,976 s, 9.0 GiB, per provenance.
- **A4b IOI:** add §4.5 (short) + App. table, from `results/naacl_ioi/*.json`. First, check that the scripts exist and that the JSONs carry
  their config. If the scripts reproduce in reasonable time (zero/mean/resample on GPT-2), rerun to scratch and compare.
- **RQ1 format diagnostic:** one sentence in §4.2, from `arithmetic_format_diagnostic.json`.

### New experiments (EXP-NEW): pre-specified before running

Both run in one new script, `scripts/run_camera_ready_robustness.py --model {gpt2,distilgpt2}`, under zero-ablation (primary). It reuses
`ValidatedHeadAblator`, `src/controls.py`, the same data seeds (copying 0, arithmetic 1), and the same control-draw seed (42).
Output: `neurips/experiment_results/<model>_camera_ready_robustness.json`. Qwen is excluded on cost (the full pipeline took 3.3 h), and the paper says so.

**Sanity check (both):** per-head per-example Δ is computed once on all 200 copying examples. Restricted to the original partition
(first 100 / last 100), its means must match the saved `per_head_delta_discovery` / `per_head_delta_heldout` within 1e-4 nats. Otherwise the run aborts.

#### C1: Sensitivity of head selection to the discovery/held-out partition

| Field | Spec |
|---|---|
| Question | Does the RQ2 conclusion depend on the one fixed 50/50 partition? |
| H1 | The candidates' held-out effect exceeds both matched controls for any partition |
| Competing | The primary result reflects a lucky partition |
| Design | 20 random 100/100 partitions of the 200 copying examples (`np.random.default_rng(1000+i).permutation(200)`, i = 0..19). For each, rerun selection (top-5 on discovery), held-out candidate Δ, random- and layer-matched-control p (1,000 draws each, same seeds), rank stability ρ, and arithmetic specificity p. No new forward passes are needed after the single 200-example sweep. |
| Report | Median and range of the held-out Δ; number of partitions with p ≤ 0.05 against each control; specificity p distribution; overlap of each partition's top-5 with the primary top-5; how often each head is selected |
| Falsifier | Candidates fail p ≤ 0.05 against either control in > 2 of 20 partitions → "partition-sensitive" |
| Caveat | Partitions overlap, so they are not independent replications. This is a sensitivity analysis, not a new test. |

#### C2: Format-matched control task (ordered-sequence continuation)

| Field | Spec |
|---|---|
| Question | Is the candidates' larger effect on copying than on arithmetic explained by format, tokenization, or baseline differences? |
| Control task | `x1 x2 x3 x4 . x5 x6 x7` → ` x8`, where x1..x8 are 8 consecutive items of an ordered sequence: numerals 1–99, English months, capital letters A–Z, number words one–twenty. Every non-wrapping window is used, after filtering to single-token items and the `is_valid_qa` boundary check. This is deterministic: no seed, and no duplicates. |
| Why matched | Same surface template as copying (4 items, ".", 3 items, single-token space-prefixed answer) and a similar baseline (pilot, unablated only: GPT-2 numerals 98.9% / −0.42 nats). Different rule: the answer never appears in the context, so it cannot be copied in context. It is solved by ordinal knowledge, a behavior attributed to successor heads (Gould et al., ICLR 2024). |
| Pilot disclosure | Before freezing, only unablated baseline acc/log-prob of the four sequence families were measured, on GPT-2 and DistilGPT2 (GPT-2: numerals n=92 98.9%, months n=5 100%, letters n=19 84.2%, number words n=13 100%). No head effects were inspected. All four families are kept, with no post-pilot filtering. |
| Primary statistic | S_seq = mean held-out copying Δ − mean sequence-control Δ for the primary zero-selected candidates, against the same 1,000 random draws; one-sided Monte Carlo p. Also reported: the candidates' Δ on the sequence task alone versus random draws (does ablating them generically damage this task?). |
| Secondary | The same contrast for the mean-ablation candidates under mean-ablation, with means taken from the copying discovery split, as in the existing pipeline. |
| Interpretation | p ≥ 0.05 → the copying-vs-arithmetic advantage is not specific relative to a format-matched task, which further weakens specificity. p < 0.05 → the effect is larger on in-context copying than on format-matched sequence continuation. That is evidence of a task-dependent effect, not of an induction mechanism. Both outcomes are reported regardless. |
| Confounds | The sequence task is dominated by numerals (92/129). The baseline is close but not identical. The two tasks share the "." token and short-sequence structure. Induction-like heads might still help predict sequence order. |

#### C3: Resample ablation on the copying task (added 2026-10-30, decision 5; written before any code or run)

| Field | Spec |
|---|---|
| Question | Is the GPT-2 collapse of the zero-selected heads' effect under mean-ablation (0.412 → 0.005 nats) caused by zero being off-distribution, or by the mean being an unusual replacement value? |
| H1 (paper's current reading) | Zero-ablation inflates these heads' effect; an on-distribution replacement also removes most of it |
| Competing | The position-pooled mean is itself an atypical value; an activation the head actually produces on another copying example restores the effect |
| Intervention | Same pre-projection location. Head h's slice of the `c_proj` input is replaced, at every position, by its value on a *source* copying example (the source's own question + answer, a separate forward pass). The source carries no information about the target's words. |
| Source matching | Copying prompts differ only in how many tokens the first word takes (1, 2 or 3); every later word, the period, and the answer are one token. Source must have the identical per-word token signature (so every position plays the same role) and come from the **same split** as the target, and must not be the target. Drawn uniformly with `np.random.default_rng(seed)` (primary seed 2, as in IOI). The single discovery example whose first word is 3 tokens has no partner in either split and is excluded from the resample analysis (discovery N = 99; held-out N = 100). |
| Primary statistic (R1) | Mean held-out resample Δ of the five **zero-selected** candidates, 95% bootstrap CI (2,000 resamples, seed 3), against the resample random-control distribution (1,000 draws, seed 42) and the layer-matched control for that set (seed 43); one-sided Monte Carlo p. |
| Pre-specified reading of R1 | (A) p_random > 0.05 → resample agrees with mean: the zero-selected heads' effect is not reproduced by on-distribution replacement. (B) p_random ≤ 0.05 and Δ ≥ 0.206 (half the zero effect) → resample largely restores the effect; the mean-ablation collapse reflects the mean value and §4.4 must be revised. (C) p_random ≤ 0.05 and Δ < 0.206 → partial: the effect exceeds random heads but is less than half the zero effect. The threshold is arbitrary but fixed now; the numbers are reported regardless. |
| Secondary (R2) | Full RQ2 under resample: top-5 on resample discovery Δ, held-out Δ vs random and layer-matched controls, rank stability ρ, overlap with zero- and mean-selected sets. |
| Secondary (R3) | Agreement of per-head Δ (Pearson, Spearman, top-5 overlap) resample vs zero and resample vs mean, on held-out (N = 100 for all three) and discovery (resample N = 99 vs saved N = 100). |
| Source sensitivity (R4) | Repeat R1 and R2 with source seeds 3 and 4; report the range of the R1 Δ and of the reselected heads. The paper uses seed 2. |
| Sanity checks (abort on failure) | (i) Resampling a head from the target itself changes no log-prob (max abs Δ < 1e-5) for 5 heads; (ii) unablated per-example copying log-probs equal the saved baseline within 1e-4 nats; (iii) source and target batches have identical shapes and attention masks. |
| Not done | Arithmetic or sequence-control specificity under resample (needs separately matched sources; out of scope). Qwen (cost). |
| Models / output | GPT-2 and DistilGPT2. `scripts/run_resample_copying.py --model {gpt2,distilgpt2}` → `neurips/experiment_results/<model>_resample_copying.json` (never overwrites). Zero and mean per-head Δ are read from the saved result files, not recomputed. |

## 4. Paper structure after the changes (target main text ≤ 9 pp)

1 Intro (findings updated: 3 → baseline-qualified; add finding 4: replacement value) · 2 Background (+ scope paragraph, + baseline-choice sentence) ·
3 Design (+ task rationale, sequence control, controls distinction, mean/resample definitions) · 4 Results: 4.1 RQ0, 4.2 RQ1 (+ format
diagnostic), 4.3 RQ2 (+ C1 summary sentence, + C2 result), 4.4 Robustness to the ablation baseline (Table), 4.5 IOI (short) · 5 Discussion and
limitations (rewritten paragraphs) · 6 Conclusion · Appendix: A RQ0 figure, B DistilGPT2, C Qwen2.5, D IOI table, E C1 partition table, F reproducibility.

The abstract stays at about its current length: the mean-ablation and specificity sentences are replaced rather than added to.

## 5. Verification protocol

1. `pytest tests/`: all pass (output recorded).
2. Hashes of the original JSONs: identical before and after.
3. A regression rerun of `gpt2_results.json` to scratch: identical except the timestamp and commit.
4. `scripts/verify_camera_ready_numbers.py` (scratch or repo): for each number in a list of (tex string, JSON path, rounding),
   assert the formatted JSON value appears in the tex.
5. `pdflatex` + `bibtex` ×2: no undefined references or citations; check the overfull boxes; page count of the main text ≤ 9.
6. Update `claim_evidence_map.md`, `reproducibility_provenance.md` (new runs: command, time, memory, hashes), and `checklist.tex`.
7. Write `neurips/reviewer_responses.md`: claim only edits present in the final PDF.

## 6. Execution checklist

- [x] V0 hashes, tests, and the regression check
- [x] C1/C2 code (dataset function + test + script), then run GPT-2 and DistilGPT2
- [x] Writing-only items B1, A5, A7, A3, A4a, A6, B2, B3, baseline related work
- [x] A1 mean-ablation section + table; reframe the abstract, intro, and conclusion
- [x] A2 Qwen appendix; A4b IOI; RQ1 diagnostic
- [x] C1/C2 results integrated
- [x] Number-verification script, build, page check
- [x] Claim map, provenance, checklist, reviewer responses

Status 2026-10-30 (resumed session): all items above implemented. Final checks: 29 tests pass; original result hashes unchanged;
GPT-2 regression and IOI reruns reproduce the saved files apart from timestamp/commit; 121/121 printed values verified; clean build,
main text ends on p. 8. Code, results, and paper were committed and merged into `main` of the public repository the paper cites.
Remaining, needs the author: upload the PDF and post the reviewer responses.
