# Camera-ready revision plan: InterpScience @ NeurIPS 2026 (Submission 112)

Created 2026-10-30. Camera-ready deadline: **2026-11-15 AoE**. Nothing has been changed yet; this file is the plan.

## 0. Which file is the paper

**`neurips/neurips_2026.tex`** (last modified 2026-09-01) is the workshop paper. It is the only `.tex` in the repo using
`\usepackage[dblblindworkshop]{neurips_2026}` + `\workshoptitle{Interpretability as a Science}`, and its text matches
the abstract on OpenReview. Other copies, which are **not** the workshop paper:

| File | What it is |
|---|---|
| `arxivconversion/arxiv_submission/neurips_2026.tex`, `arxivconversion/neurips_for_interpretabilty/neurips_2026.tex` | arXiv preprint variant of the same text (`[preprint]`, GitHub URL, no checklist, flat figure paths) |
| `NAACL/naacl_submission/acl_latex.tex` | Later, extended ACL-format version. It already contains mean-ablation, Qwen2.5, and IOI results that we can port here (see §2) |

Comment sources: `neurips/creviewercomments.txt` (Reviewers **QC4F** and **E5js**; AC: accept) and
`neurips/camerareadycomments.txt` (revision advice, abbreviated **CR** below; it adds independent issues CR-6.x).

Constraints: long paper ≤ 9 pages of main text (references/appendix excluded); camera-ready **must** be in NeurIPS format.
The current PDF has 15 pages in total, and the main text is roughly 6.

## 1. Key finding from the existing results (read before editing)

These experiments were already run for the NAACL version and saved in `neurips/experiment_results/` and `results/naacl_ioi/`.
They have **not** been reported in the workshop paper. One result materially affects the GPT-2 story:

- `mean_vs_zero_posthoc.json`: the 5 zero-selected GPT-2 heads have held-out Δ **0.412 nats under zero-ablation**
  but only **0.005 nats under mean-ablation**, which is below the mean-ablation random-control mean (0.016).
  DistilGPT2: 0.586 → 0.182. Qwen2.5-1.5B: 0.208 → 0.155.
- Under mean-ablation, *re-selected* heads still beat both controls (p=0.001), but they are a **disjoint** set on GPT-2 (0/5 overlap).
- `arithmetic_format_diagnostic.json`: for **Qwen2.5** the 0% arithmetic "floor" is a format artifact (its tokenizer splits
  `" 24"` into a bare space + digits; 23% exact match without the space). For **GPT-2/DistilGPT2** the gold answer is a single
  token and exact match stays **0.0% without the leading space**, so their floor is not explained by this format choice.
  *(Corrected 2026-10-30: an earlier version of this note wrongly said GPT-2 gets later answer tokens right; for GPT-2 that
  diagnostic field is vacuous because the answer is one token.)*

**Consequence:** if we answer QC4F's mean-ablation request (which we should), RQ2's wording in the abstract, intro, RQ2,
discussion, and conclusion must say that the GPT-2 zero-ablation effect of these heads is **baseline-dependent**. The RQ0 and
the matched-control *protocol* claims are unaffected. The RQ1 "floor" sentence must say the floor is a property of the prompt
format, not a pure capability floor. Silently omitting either result would be selective reporting.

## 2. Item list

Type key: **PAPER** = writing-only edit · **EXP-PORT** = result already exists, needs verification + integration ·
**EXP-NEW** = new experiment/code · **VERIFY** = checks/build.

### A. Reviewer-driven items

| ID | Comment (source) | Type | Plan | Section(s) |
|---|---|---|---|---|
| A1 | Zero-ablation is off-distribution; compare mean / resample / noise (QC4F; CR Exp 1) | **EXP-PORT** + PAPER | (a) Re-verify `gpt2_/distilgpt2_meanablation_results.json` and `mean_vs_zero_posthoc.json` against current code: run tests, then confirm the uncommitted `src/ablation.py` refactor reproduces the original `gpt2_results.json` byte for byte apart from the timestamp. (b) Add a new subsection, "Robustness to the ablation baseline", with a compact table: reselected candidates vs controls under mean; zero-vs-mean per-head r/ρ; top-5 overlap; the post hoc 0.412→0.005 number labelled *post hoc*. (c) Replace the "we did not run" limitation paragraph. Noise ablation stays untested and is stated as such. | new §4.4, §5, Table |
| A2 | Only two models from one lineage; larger architectures (QC4F, E5js; CR Phase A) | **EXP-PORT** + PAPER | Port the Qwen2.5-1.5B-Instruct replication (`qwen2.5-1.5b-instruct_meanablation_results.json`) as an appendix table plus 1–2 sentences in the main text. Rewrite "Model scope": DistilGPT2 is distilled from GPT-2, so it is not independent; Qwen is a different family; >1.5B is untested. Include CR passage D, edited so it no longer says "only zero-ablation". | §5, new App. |
| A3 | Whether post-projection hook errors are prevalent in TransformerLens / nnsight (QC4F; CR 4B) | PAPER | Add CR passage B ("Scope of the intervention-semantic failure") to §2. Optionally add one sentence, carried over from NAACL, on where such hooks are reachable (e.g. TransformerLens `attn_out` vs `hook_z`), with citations already verified in the NAACL bib. Do not claim prevalence. | §2 |
| A4 | Tasks too simple; test IOI / Greater-Than (QC4F; CR 4C, Exp 4) | **EXP-PORT** + PAPER | (a) Add CR passage C ("Choice of diagnostic tasks") to §3. (b) Port the GPT-2 IOI result (`results/naacl_ioi/gpt2_ioi_ablation_methods.json`, `gpt2_ioi_minimal_pair.json`): candidates beat both controls under zero, mean, and resample (p=0.001); zero vs mean top-5 overlap is 2/5. This is short main-text text plus an appendix. Greater-Than stays untested and is stated as such. Verify the scripts `scripts/run_ioi_*.py` reproduce the JSONs. | §3, §4.4/App. |
| A5 | Task specificity weak on GPT-2; keep p=0.078 / p=0.274 (E5js; CR 6.2) | PAPER | Remove the remaining positive-leaning wording ("positive trend", "suggestive" in Fig. 1C caption/abstract → "inconclusive / not robust"). Keep both p-values. Note that the DistilGPT2 specificity p=0.001 is a single run with no rerun and uses the same mismatched control. | Abstract, §1, §4.3, Fig 1, §5 |
| A6 | Arithmetic is an imperfect control (E5js; CR 4A, Exp 2) | PAPER (+ optional EXP-NEW) | Add CR passage A ("Task-control mismatch") to §5. Integrate the format-diagnostic finding from §1 into RQ1/§5. **Optional EXP-NEW:** a format-matched control task (same template, a different contextual rule, e.g. copy w₁ instead of w₄). This has not been run anywhere, needs `src/dataset.py` + runner changes and its own seed and result file, and must be a pre-committed, report-regardless analysis. | §3, §4.2, §5 |
| A7 | Distinguish head-vs-head controls from the cross-task control (CR 6.3) | PAPER | Add one sentence in §3 Controls / §4.3: random and layer-matched controls test "is the effect unusual among heads"; the arithmetic contrast tests "does it differ across tasks"; passing the first does not imply the second. | §3, §4.3 |

### B. Independent issues (CR §6)

| ID | Issue | Type | Plan |
|---|---|---|---|
| B1 | §4.1 says 0.067 nats disagreement is "comparable in magnitude" to 0.412 (≈6× smaller) (CR 6.1) | PAPER | Reword: report 0.067 directly and say it is ~16% of the candidate held-out effect. Do not call it comparable. |
| B2 | Anonymous author + "Submitted to … Do not distribute" footer (CR 6.4) | PAPER | `\usepackage[dblblindworkshop,final]{neurips_2026}` gives the footer "40th Conference … Workshop: Interpretability as a Science." Author: Juli Huang, Stanford University (as on OpenReview). Add an Acknowledgments section if wanted. |
| B3 | De-anonymize the supplement pointer | PAPER | Replace "anonymized supplementary archive" with the public GitHub URL, as in the arXiv version. Remove the "previously unaudited implementation" phrasing only if the provenance can now be named. |
| B4 | Keep the original result JSONs intact (CR 6.5) | VERIFY | Do not overwrite `gpt2_results.json`, `distilgpt2_results.json`, `gpt2_n500_results.json`. New results go in separate files, and each is added to `claim_evidence_map.md` + `reproducibility_provenance.md`. |
| B5 | The abstract and conclusion must keep correct claim boundaries after A1/A5 | PAPER | Rewrite the RQ2 sentences: the held-out effect beats matched controls *under zero-ablation*; it is baseline-dependent on GPT-2; task specificity is not robust. |

### C. Optional new experiments (not required by reviewers)

| ID | Experiment | Type | Cost / note |
|---|---|---|---|
| C1 | Alternative discovery/held-out splits (CR Exp 3): re-split with e.g. 10 seeds, report top-5 stability + p-values | EXP-NEW | Cheap (per-head Δ already batched; ~10×100 s on CPU). Addresses the "single split" limitation that both QC4F and CR mention. Recommended if time allows. |
| C2 | Format-matched control task (A6 optional) | EXP-NEW | Medium effort. Design must be fixed before running. |
| C3 | Greater-Than, noise ablation, >1.5B models | — | **Not planned.** List as limitations. |

### D. Final verification

| ID | Check | Type |
|---|---|---|
| D1 | `pytest tests/` passes (the ablation-semantics, resample, and Qwen adapter tests) | VERIFY |
| D2 | Regenerate figures only via `scripts/make_figures.py`; the original figures must stay unchanged | VERIFY |
| D3 | Every new number in the paper is traceable to a JSON field; update `claim_evidence_map.md` | VERIFY |
| D4 | Build the PDF; main text ≤ 9 pages; no undefined refs/citations; no overfull boxes; all citations real (desk-reject rule) | VERIFY |
| D5 | Update `checklist.tex` answers affected by the changes (experiments, limitations, compute) | PAPER |
| D6 | Draft reviewer responses (`neurips/reviewer_responses.md`) that claim **only** changes actually present in the final PDF | PAPER |
| D7 | Optionally sync the arXiv copy afterwards (separate decision) | PAPER |

## 3. Proposed execution order (one item at a time, verify after each)

1. D1 + regression check (A1a): confirm the current code still reproduces the original results before trusting any ported number.
2. B1, A5, A7, A3, B2, B3: writing-only fixes with no dependency on new numbers.
3. A1: mean-ablation subsection + table. Then B5 (abstract/intro/conclusion reframing).
4. A6 (passage A + format-diagnostic) and A4 (task rationale + IOI).
5. A2 (Qwen appendix + model-scope rewrite).
6. Optional C1 (and C2) if approved.
7. D2–D6, then the final build and audit.

## 4. Decisions needed from you

1. **Report the mean-ablation result (A1) in the camera-ready?** Recommended: yes. It is the direct answer to QC4F, and
   omitting a known result that weakens the GPT-2 story would be selective. It changes the RQ2 framing (see §1).
2. **Port Qwen (A2) and IOI (A4) into this paper**, or only reference them as limitations? Recommended: port them, with compact
   main-text sentences and appendix tables; this fits within 9 pages.
3. **Run the optional new experiments C1 (split stability) / C2 (format-matched control)?** Recommended: C1 yes (cheap), C2 only if time allows.
4. Author block: "Juli Huang, Stanford University" + the email address? Any acknowledgments?
