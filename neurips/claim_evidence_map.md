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
| Random-control specificity distribution | `rq2_candidate_vs_controls.specificity.{random_control_mean,random_control_p5,random_control_p95}` | mean=0.011 [-0.135, 0.165] (JSON: -0.1345, 0.1653). **Corrected 2026-10-12:** the NeurIPS/arXiv text printed [-0.130, 0.163], which matches no result file; the figure always used the JSON values | Fig. 1C |
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

## Mean-ablation robustness (ARR/NAACL version, `NAACL/naacl_submission/acl_latex.tex` Section 5.4, Table 2)

Source: `gpt2_meanablation_results.json`, `distilgpt2_meanablation_results.json`
(field prefix `mean_ablation.`), and post hoc `mean_vs_zero_posthoc.json`.

| Claim | JSON path | GPT-2 | DistilGPT2 |
|---|---|---|---|
| Zero vs. mean per-head delta correlation (discovery) | `mean_ablation.mean_vs_zero.pearson_r_discovery` / `.spearman_rho_discovery` | r=0.424, rho=0.495 | r=0.553, rho=0.871 |
| Top-5 overlap, mean- vs zero-selected | `mean_ablation.mean_vs_zero.top5_overlap_with_zero_candidates` | 0/5 | 1/5 |
| Mean-ablation candidate heads | `mean_ablation.rq2_candidate_vs_controls.candidate_heads` | L10H11, L10H0, L7H6, L0H9, L0H3 | L2H11, L3H2, L3H10, L4H6, L5H1 |
| Candidate held-out delta [CI] | `mean_ablation.rq2_candidate_vs_controls.candidate_heldout_delta_mean` (+`_ci95`) | 0.176 [0.157, 0.194] | 0.445 [0.388, 0.508] |
| Random control mean [p5, p95], p | `...random_control_heldout_delta` | 0.016 [-0.034, 0.057], p=0.001 | 0.060 [-0.012, 0.172], p=0.001 |
| Layer-matched mean [p5, p95], p | `...layer_matched_control_heldout_delta` | 0.009 [-0.071, 0.058], p=0.001 | 0.030 [-0.024, 0.067], p=0.001 |
| Rank stability rho | `...rank_stability_spearman_discovery_vs_heldout` | 0.948 | 0.960 |
| Specificity candidate / random mean [p5, p95], p | `...specificity` | 0.168 / 0.021 [-0.087, 0.125], p=0.014 | 0.462 / 0.064 [-0.095, 0.231], p=0.001 |
| Zero-selected heads, held-out delta under zero / mean (post hoc) | `mean_vs_zero_posthoc.json` `<model>.heldout_delta_under_{zero,mean}_mean_over_candidates` | 0.412 / 0.005 | 0.586 / 0.182 |
| Mean-ablation wall time | `reproducibility_provenance.md` | 324.57 s | 107.79 s |

## Qwen2.5-1.5B-Instruct replication (ARR/NAACL version, Section 5.5, Table 4)

Source: `qwen2.5-1.5b-instruct_meanablation_results.json` (zero-ablation fields at top level, mean-ablation under `mean_ablation.`), `arithmetic_format_diagnostic.json`, `mean_vs_zero_posthoc.json`.

| Claim | JSON path | Value |
|---|---|---|
| Copying / arithmetic accuracy [CI] | `baselines.*.accuracy_mean` (+ci) | 0.990 [0.975, 1.000] / 0.000 [0.000, 0.000] |
| Copying / arithmetic log-prob [CI] | `baselines.*.logprob_mean` (+ci) | -0.38 [-0.43, -0.34] / -5.31 [-5.44, -5.17] |
| RQ0 Pearson / Spearman / top-5 overlap | `rq0_naive_vs_correct.*` | 0.108 / 0.176 / 0/5 |
| Zero candidates | `rq2_candidate_vs_controls.candidate_heads` | L8H3, L14H3, L18H8, L13H4, L2H3 |
| Zero: candidate held-out delta [CI] | `rq2_...candidate_heldout_delta_mean` | 0.208 [0.176, 0.243] |
| Zero: random / layer-matched mean [p5,p95], p | `rq2_...random_control_heldout_delta`, `...layer_matched...` | 0.012 [-0.010, 0.046] / 0.004 [-0.012, 0.019], p=0.001 each |
| Zero: rank stability; specificity cand / random mean, p | `rq2_...rank_stability...`, `rq2_...specificity` | 0.927; 0.376 / -0.007, p=0.004 |
| Zero vs mean r / rho / overlap | `mean_ablation.mean_vs_zero.*` | 0.925 / 0.756 / 5/5 |
| Mean: candidate held-out delta [CI] | `mean_ablation.rq2_...candidate_heldout_delta_mean` | 0.155 [0.129, 0.182] |
| Mean: random / layer-matched mean [p5,p95], p | `mean_ablation.rq2_...` | 0.006 [-0.007, 0.031] / 0.003 [-0.008, 0.015], p=0.001 each |
| Mean: rank stability; specificity cand, p | `mean_ablation.rq2_...` | 0.850; 0.349, p=0.007 |
| Arithmetic exact match, unspaced answer (post hoc) | `arithmetic_format_diagnostic.json results[*].unspaced_answer_exact_match` | GPT-2 0.0, DistilGPT2 0.0, Qwen 0.23 |
| Qwen user CPU time / peak RSS | `reproducibility_provenance.md` | 11,975.83 s / 9.0 GiB |

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


## IOI comparison of ablation baselines (NAACL revision, Section 5.6, Table 3)
Source: `results/naacl_ioi/gpt2_ioi_ablation_methods.json` (script `scripts/run_ioi_ablation_methods.py`, GPT-2 small, N=200, CPU float32)

| Paper statement | JSON field | Value |
|---|---|---|
| IO preferred on 97% / 100% of prompts; mean logit difference 3.44 / 3.53 | `baseline.{discovery,heldout}.frac_io_preferred`, `.mean_logit_diff` | 0.97 / 1.00; 3.439 / 3.529 |
| Spearman zero–mean 0.74 [0.71, 0.76] | `cross_method.zero_vs_mean.spearman_all_heads`, `spearman_ci95_bootstrap_prompts` | 0.737 [0.711, 0.761] |
| Spearman zero–resample 0.47 [0.42, 0.51] | `cross_method.zero_vs_resample` | 0.473 [0.423, 0.508] |
| Spearman mean–resample 0.65 [0.60, 0.68] | `cross_method.mean_vs_resample` | 0.650 [0.602, 0.677] |
| Top-5 / top-10 overlap: zero–mean 2/4, zero–resample 2/4, mean–resample 4/9 | `cross_method.*.top5_overlap`, `top10_overlap` | as stated |
| Top-5 heads per method (Table 3, rank order) | `methods.{zero,mean,resample}.candidate_heads` | as stated |
| Held-out effect 1.55 [1.42, 1.70] / 0.92 [0.84, 1.01] / 0.93 [0.84, 1.02] | `methods.*.heldout_effect_mean`, `heldout_effect_ci95` | 1.554 / 0.924 / 0.927 |
| 68% larger under zero | 1.554 / 0.924 | 1.68 |
| p = .001 vs random and layer-matched, all methods | `methods.*.p_random_ge_candidate`, `p_layer_matched_ge_candidate` | 0.000999 |
| Rank stability 0.94 / 0.97 / 0.91 | `methods.*.rank_stability_spearman_disc_vs_held` | 0.938 / 0.968 / 0.911 |
| Ranks of Wang et al. heads (Table 3) | `methods.*.wang_head_ranks` | as stated |
| Head identities of the published circuit | Wang et al. (2023), arXiv:2211.00593, Section 3 and Appendix (verified from the PDF text) | 9.6, 9.9, 10.0; 10.7, 11.10; 7.3, 7.9, 8.6, 8.10; induction 5.5 |

## Synthesis table (Table 4, round 2)
Every value is copied from rows already traced above. Hook placement: r = 0.057 / 0.001 / 0.108 (GPT-2 / DistilGPT2 / Qwen RQ0 rows). Metric: 0.0% and 99.5% (RQ1 rows). Controls: p = 0.001, 0.078, 0.274 (RQ2 rows). Replacement: 0.412 → 0.005 (mean_vs_zero_posthoc.json) and IOI top-5 overlap 2/5 (IOI rows). No new number.

## Camera-ready workshop version (`neurips/neurips_2026.tex`, 2026-10-30)

All numbers are checked mechanically by `scripts/verify_paper_numbers.py` (142/142 passed at build).
Workshop locations of results that already have rows above: mean-ablation → §4.4 + Table 2 (`tab:mean`);
Qwen2.5 → §4.1, §4.2, §4.4, App. C + Table 4 (`tab:qwen`); IOI → §4.5 + App. D + Table 5 (`tab:ioi`).

| Claim | JSON path | Value | Location |
|---|---|---|---|
| Random-control specificity range (corrected) | `gpt2_results.json` `rq2_candidate_vs_controls.specificity.random_control_p5/p95` | [-0.135, 0.165] (the submitted workshop PDF printed [-0.130, 0.163]) | §4.3, Table 1 |
| GPT-2 floor persists without leading space | `arithmetic_format_diagnostic.json` `results[model=gpt2].unspaced_answer_exact_match` | 0.0 | §4.2 |
| Qwen arithmetic unspaced exact match | same, `model=Qwen/Qwen2.5-1.5B-Instruct` | 0.23 | §4.2, App. C |
| Mean-ablation GPT-2 specificity | `gpt2_meanablation_results.json` `mean_ablation.rq2_candidate_vs_controls.specificity.{candidate,random_control_mean,empirical_p_candidate_beats_random}` | 0.168 vs 0.021, p=0.014 | §4.4 |
| IOI: 7.9 in top-5 under mean and resample only (zero: rank 6) | `results/naacl_ioi/gpt2_ioi_ablation_methods.json` `methods.*.candidate_heads`, `per_head_mean_effect` | — | §4.5 (the NAACL text says "every baseline"; that is wrong for zero) |
| IOI zero held-out CI upper bound | `methods.zero.heldout_effect_ci95[1]` | 1.6946 → 1.69 (NAACL printed 1.70) | Table 5 |
| C2 sequence-task baseline (GPT-2 / DistilGPT2) | `<model>_camera_ready_robustness.json` `c2_sequence_control.baseline.{n,accuracy_mean,logprob_mean}` | n=129; 0.969 / -0.572; 0.876 / -0.695 | §3, §4.3 |
| C2 candidates' sequence-task Δ vs random | `c2_sequence_control.primary_zero_candidates.{candidate_control_task_delta,random_control_task_delta_mean,empirical_p_candidate_control_delta_ge_random}` | GPT-2 0.447 vs 0.051, p=0.001; DistilGPT2 0.113 vs 0.107, p=0.280 | §4.3 |
| C2 specificity vs sequence control (zero candidates) | `...primary_zero_candidates.{candidate_specificity,random_specificity_mean,empirical_p_candidate_beats_random}` | GPT-2 -0.035 vs -0.023, p=0.567; DistilGPT2 0.472, p=0.001 | Abstract, §1, §4.3 |
| C2 specificity (mean-ablation candidates) | `...secondary_mean_candidates.{candidate_specificity,empirical_p_candidate_beats_random}` | GPT-2 -0.138, p=0.882; DistilGPT2 0.382, p=0.001 | §4.3 |
| C1 partitions with original top-5 | `c1_partition_sensitivity.overlap_with_primary_counts["5"]` | GPT-2 19/20; DistilGPT2 20/20 | §4.3, Table 6 |
| C1 head controls | `c1_partition_sensitivity.n_partitions_p_{random,layer_matched}_le_0.05` | 20/20 both models | Abstract, §1, §4.3, Table 6 |
| C1 held-out Δ median [range] | `c1_partition_sensitivity.candidate_heldout_delta` | GPT-2 0.441 [0.368, 0.469]; DistilGPT2 0.578 [0.535, 0.643] | §4.3, Table 6 |
| C1 rank stability median [range] | `c1_partition_sensitivity.rank_stability_spearman` | GPT-2 0.966 [0.937, 0.979]; DistilGPT2 0.961 [0.923, 0.981] | Table 6 |
| C1 arithmetic specificity | `c1_partition_sensitivity.{arith_specificity,p_arith_specificity,n_partitions_p_arith_specificity_le_0.05}` | GPT-2 9/20, median p=0.054 [0.039, 0.650]; DistilGPT2 20/20, p=0.001 | §4.3, Table 6 |
| C1/C2 run cost | `/usr/bin/time -l` (see provenance) | GPT-2 301.43 s, 1.90 GiB; DistilGPT2 82.08 s, 1.63 GiB | §3 |
| Qwen run cost | `reproducibility_provenance.md` | 11,975.83 CPU-s, 9.0 GiB | App. C |
| C3 zero-selected heads under resample (held-out) | `<model>_resample_copying.json` `primary.r1_zero_selected_under_resample.{heldout_delta_mean,heldout_delta_ci95,random_control,layer_matched_control}` | GPT-2 0.020 [0.006, 0.035], random mean 0.015, p=0.346 / 0.430; DistilGPT2 0.201 [0.173, 0.232], p=0.075 / 0.001 | Abstract, §1, §4.4, Table 2 |
| C3 source-seed sensitivity (GPT-2) | `gpt2_resample_copying.json` `r4_source_sensitivity.r1_heldout_delta_by_seed` | seeds 2/3/4: 0.020 / 0.005 / 0.011; reading A for all seeds, both models | §4.4 |
| C3 resample-selected heads vs controls | `primary.r2_resample_selected.{heads,heldout_delta_mean,heldout_delta_ci95,random_control,layer_matched_control,overlap_with_zero_selected,overlap_with_mean_selected}` | GPT-2 10.0, 10.11, 10.1, 10.6, 11.2: 0.179 [0.159, 0.200], p=0.001/0.001, overlap 0/5 zero, 2/5 mean; DistilGPT2 0.586 [0.516, 0.659], 1/5 zero, 5/5 mean | §4.4, Table 2 |
| C3 per-head agreement (discovery-split Pearson, resample N=99) | `primary.r3_agreement.discovery_resample_vs_{zero,mean}.pearson`; zero–mean from `<model>_meanablation_results.json` `mean_ablation.mean_vs_zero.pearson_r_discovery` | GPT-2 resample–zero 0.212, resample–mean 0.724, zero–mean 0.424; DistilGPT2 0.473 / 0.946 / 0.553 (held-out, also saved: GPT-2 0.279 / 0.716; DistilGPT2 0.452 / 0.940) | §4.4, Table 2 |
| C3 run cost | `/usr/bin/time -l` (see provenance) | GPT-2 914.99 s, 2.14 GiB; DistilGPT2 325.98 s, 2.10 GiB | §3 |
