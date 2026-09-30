# Claim -> Evidence Map

Every quantitative claim in `neurips_2026.tex` must appear in this table.
Source of truth: `neurips/experiment_results/gpt2_results.json`, produced by
`scripts/run_neurips_experiments.py` (seeds
induction=0/arithmetic=1/control_sampling=42, torch 2.8.0, transformers
4.57.3, CPU). Figures: `scripts/make_figures.py` reads only this JSON.

| Claim | JSON path | Value | Figure/Table |
|---|---|---|---|
| Naive vs. correct ablation are ~uncorrelated | `rq0_naive_vs_correct.pearson_r`, `.spearman_rho` | r=0.057, rho=0.110 | Fig. 2 (rq0_naive_vs_correct.png) |
| 0/5 top-head overlap, naive vs. correct | `rq0_naive_vs_correct.top5_head_overlap` (of `top_k_candidates`=5) | 0/5 | Fig. 2 |
| Mean abs. per-head delta difference (naive vs. correct) | `rq0_naive_vs_correct.mean_abs_delta_diff` | 0.0670 nats | Fig. 2 |
| Copying-task baseline accuracy (ceiling) | `baselines.induction_full.accuracy_mean` (+ `accuracy_ci95`) | 0.995 [0.985, 1.000], n=200 | Fig. 1A, Table 1 |
| Copying-task baseline log-prob | `baselines.induction_full.logprob_mean` (+ `logprob_ci95`) | -0.825 [-0.901, -0.759] nats | Fig. 1A |
| Arithmetic-control baseline accuracy (floor) | `baselines.arithmetic_control.accuracy_mean` (+ ci) | 0.000 [0.000, 0.000], n=200 | Fig. 1A, Table 1 |
| Arithmetic-control baseline log-prob | `baselines.arithmetic_control.logprob_mean` (+ ci) | -7.442 [-7.530, -7.350] nats | Fig. 1A |
| Candidate heads (top-5 by discovery delta) | `rq2_candidate_vs_controls.candidate_heads` | [(0,0),(0,7),(1,3),(11,0),(0,10)] | Table 1 |
| Discovery -> held-out rank stability | `rq2_candidate_vs_controls.rank_stability_spearman_discovery_vs_heldout` | rho=0.974 (n=144 heads) | text only |
| Candidate held-out delta | `rq2_candidate_vs_controls.candidate_heldout_delta_mean` (+ ci95) | 0.412 [0.353, 0.474] nats | Fig. 1B, Table 1 |
| Random-control held-out delta distribution | `rq2_candidate_vs_controls.random_control_heldout_delta` | mean=0.029 [-0.033, 0.113] (5th/95th pct, 1000 draws) | Fig. 1B, Table 1 |
| Layer-matched control held-out delta distribution | `rq2_candidate_vs_controls.layer_matched_control_heldout_delta` | mean=0.051 [0.001, 0.098] (1000 draws) | Fig. 1B, Table 1 |
| Monte Carlo p: candidate beats random control | `...random_control_heldout_delta.empirical_p_candidate_beats_random` | p=1/1001=0.001 (0/1000 draws >= candidate, add-one corrected) | Table 1 |
| Monte Carlo p: candidate beats layer-matched control | `...layer_matched_control_heldout_delta.empirical_p_candidate_beats_layer_matched` | p=1/1001=0.001 (0/1000 draws >= candidate, add-one corrected) | Table 1 |
| Candidate specificity (copying-task delta - arithmetic delta) | `rq2_candidate_vs_controls.specificity.candidate` | 0.139 nats | Fig. 1C, Table 1 |
| Random-control specificity distribution | `rq2_candidate_vs_controls.specificity.{random_control_mean,random_control_p5,random_control_p95}` | mean=0.011 [-0.130, 0.163] | Fig. 1C |
| Monte Carlo p: candidate specificity beats random control | `rq2_candidate_vs_controls.specificity.empirical_p_candidate_beats_random` | p=78/1001=0.078 (not below 0.05) | Table 1, Discussion |
| Full per-head delta vectors (all 144 heads, 3 splits) | `rq0_naive_vs_correct.per_head_delta_{correct,naive}`, `rq2_candidate_vs_controls.per_head_delta_{discovery,heldout,control}` | -- | Appendix (raw JSON referenced) |
| Model/experiment config (versions, seeds, N per split) | `config.*` | see JSON | Reproducibility / Experimental setup section |

## DistilGPT2 replication (Appendix B, Table 2)

Source: `neurips/experiment_results/distilgpt2_results.json`, same
pipeline/seeds/procedure as the GPT-2 run, model swapped only.

| Claim | JSON path | Value |
|---|---|---|
| Copying-task baseline accuracy | `baselines.induction_full.accuracy_mean` (+ci) | 0.900 [0.855, 0.940] |
| Copying-task baseline log-prob | `baselines.induction_full.logprob_mean` (+ci) | -0.957 [-1.070, -0.849] |
| Arithmetic baseline accuracy (floor) | `baselines.arithmetic_control.accuracy_mean` (+ci) | 0.000 [0.000, 0.000] |
| Arithmetic baseline log-prob | `baselines.arithmetic_control.logprob_mean` (+ci) | -7.379 [-7.500, -7.250] |
| RQ0 naive vs correct Pearson r / top-5 overlap | `rq0_naive_vs_correct.pearson_r`, `.top5_head_overlap` | r=0.001, 0/5 |
| Candidate heads | `rq2_candidate_vs_controls.candidate_heads` | [(0,1),(0,9),(5,0),(3,2),(0,7)] |
| Rank stability (disc. vs held-out) | `rq2_candidate_vs_controls.rank_stability_spearman_discovery_vs_heldout` | rho=0.957 |
| Candidate held-out delta | `rq2_candidate_vs_controls.candidate_heldout_delta_mean` (+ci) | 0.586 [0.511, 0.661] |
| Random/layer control held-out delta + p | `rq2_candidate_vs_controls.{random_control_heldout_delta,layer_matched_control_heldout_delta}` | 0.080 [-0.026, 0.231], 0.054 [-0.017, 0.128]; both p=0.001 |
| Specificity (candidate vs random control) | `rq2_candidate_vs_controls.specificity` | candidate=0.492, random mean=0.023, p=0.001 |

## One-time N=500 power check on GPT-2 (pre-committed, not re-run if unfavorable)

Source: `neurips/experiment_results/gpt2_n500_results.json` (n_induction=500,
n_arithmetic=500, arithmetic_max_operand=30 -- all else identical to the
primary N=200 run). Requested explicitly as a single additional run to
sharpen the specificity estimate; reported as-is regardless of outcome,
per the project's rule against optional stopping / p-hacking.

| Claim | JSON path | Value |
|---|---|---|
| Candidate heads | `rq2_candidate_vs_controls.candidate_heads` | [(0,7),(1,3),(0,0),(11,0),(0,10)]; same set as primary run | Traceability only |
| Candidate held-out delta | `rq2_candidate_vs_controls.candidate_heldout_delta_mean` (+ ci95) | 0.448 [0.402, 0.501] nats | Discussion |
| Random-control held-out delta + p | `rq2_candidate_vs_controls.random_control_heldout_delta` | mean=0.032; p=0.001 | Discussion |
| Layer-matched held-out delta + p | `rq2_candidate_vs_controls.layer_matched_control_heldout_delta` | mean=0.062; p=0.001 | Discussion |
| Discovery/held-out rank stability | `rq2_candidate_vs_controls.rank_stability_spearman_discovery_vs_heldout` | rho=0.983 | Traceability only |
| Candidate specificity | `rq2_candidate_vs_controls.specificity.candidate` | 0.056 nats | Discussion |
| Monte Carlo specificity p-value | `rq2_candidate_vs_controls.specificity.empirical_p_candidate_beats_random` | p=274/1001=0.274 (not significant; weaker than primary run) | Abstract, Introduction, Discussion |

## Pre-existing exploratory evidence (not re-run by this pipeline)

These are cited only as historical/motivating context for why the original
metric (substring match) and original tasks (N=4-8) were floor-affected and
statistically underpowered -- they are NOT used as evidence for any RQ0-RQ2
claim above.

| Claim | Source file | Value |
|---|---|---|
| Original hand-written arithmetic task, GPT-2, substring-match accuracy | `results/day1_lite/results.json` | 0.0% (N=8) |
| Original hand-written GSM8K sample, GPT-2, substring-match accuracy | `results/day2_gsm8k/results.json` | 0.0% (N=5) |
| Original per-head ablation on floor task produced all-zero importance scores | `results/day2_full_ablation/results.json` | all 144 heads: importance=0.0 |

## Rule

No number goes into `neurips_2026.tex` unless it appears in this table with
a JSON path. If a sentence needs a number not yet in this table, compute it
first (or explicitly mark the claim as qualitative), then add the row.
