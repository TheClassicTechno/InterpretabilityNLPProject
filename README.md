# LLM NLP Interpretability Project

Code, saved results, and figures for the paper **"When Do Attention-Head
Ablations Support Causal Claims? Projection-Level Confounds, Floor Effects,
and Matched Controls"** (Juli Huang, Stanford University).

The project studies whether attention-head ablation in GPT-2-family models
supports the causal claims it is typically used to make. It started as a
straightforward "which heads matter" ablation study; auditing it turned up a
real implementation bug and a floor-effect problem that made the original
conclusions unsupported (see [What changed](#what-changed) below). It was
then rebuilt around that finding.

## What the paper shows

Working with GPT-2 small, on a controlled induction-style copying task and
an off-target arithmetic control task:

1. **A naive attention-head ablation hook doesn't ablate a head.** A
   forward hook registered on the whole attention module's output zeroes a
   channel slice *after* the block's output projection has already linearly
   mixed every head together. Fixing this (a pre-hook on the projection's
   input) gives a per-head effect vector that is essentially uncorrelated
   with the naive one (Pearson r=0.057) and shares zero of its top-5
   "important" heads.
2. **Binary accuracy loses information at a floor and near a ceiling.**
   GPT-2 scores 0.0% on arithmetic, where accuracy cannot register further
   degradation, and 99.5% on copying, where it can register a large drop but
   may hide confidence changes that leave the predicted token unchanged.
   Log-probability of the gold continuation stays graded in both regimes and
   is what every causal claim below is based on.
3. **With a discovery/held-out split and matched controls, five heads show a
   held-out-replicated effect, but task specificity is not robust.** The
   per-head effect ranking is stable across splits (Spearman rho=0.974 over
   all 144 heads), and the top-5 heads' held-out effect (0.412 nats) exceeds
   both 1,000-draw random-head and layer-matched control distributions
   (Monte Carlo p=0.001 each). Whether the effect is specific to the copying
   task is only suggestive in the primary run (p=0.078) and weaker in a
   pre-committed N=500 rerun (p=0.274).

On DistilGPT2, the intervention-semantic and matched-control findings
replicate and the arithmetic floor recurs; copying accuracy is lower (90.0%),
so GPT-2's near-ceiling example does not replicate.

`neurips/claim_evidence_map.md` lists, for every quantitative statement in
the paper, the exact JSON field it was read from.

## Environment

The reported runs used Python 3.9.6, PyTorch 2.8.0, transformers 4.57.3,
NumPy 2.0.2, and Matplotlib 3.9.4 (pinned in `requirements.txt`):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The experiments run on CPU or CUDA. The reported measurements used one
10-core Apple M5 CPU with 24 GB unified memory and no GPU. No model weights
or datasets are stored in this repository: GPT-2 and DistilGPT2 are
downloaded from Hugging Face on first use, and both tasks are generated in
memory by `src/dataset.py` from the seeds recorded in each result file.
Exact model revisions and weight hashes are in
`neurips/experiment_results/reproducibility_provenance.md`.

## Verify the intervention and the statistics

```bash
python -m unittest discover -s tests -v
```

All 13 tests should pass in under a second, with no model download. They
check that the corrected pre-hook zeroes exactly one head's slice of the
output-projection input and nothing else, that removing the hook restores
the unablated output, that the naive hook changes the logits while leaving
that input untouched, and that the layer-matched control and the
finite-sample Monte Carlo correction are built correctly.

## Reproduce each reported run

Primary GPT-2 run (199 s on the machine above):

```bash
python scripts/run_neurips_experiments.py \
  --model gpt2 \
  --output neurips/experiment_results/reproduced_gpt2_results.json \
  --n-induction 200 \
  --n-arithmetic 200 \
  --arithmetic-max-operand 20
```

DistilGPT2 replication (58 s):

```bash
python scripts/run_neurips_experiments.py \
  --model distilgpt2 \
  --output neurips/experiment_results/reproduced_distilgpt2_results.json \
  --n-induction 200 \
  --n-arithmetic 200 \
  --arithmetic-max-operand 20
```

Larger GPT-2 check, N=500 (429 s). This run is reported even though its
task-specificity statistic is weaker than in the primary run:

```bash
python scripts/run_neurips_experiments.py \
  --model gpt2 \
  --output neurips/experiment_results/reproduced_gpt2_n500_results.json \
  --n-induction 500 \
  --n-arithmetic 500 \
  --arithmetic-max-operand 30
```

The saved reference outputs are `gpt2_results.json`,
`distilgpt2_results.json`, and `gpt2_n500_results.json` in
`neurips/experiment_results/`, each with an independent `verification_*.json`
rerun. Seeds and resampling counts are fixed in the script and written to the
output JSON. To compare a rerun with the saved result, ignoring only the
timestamp and Git state:

```bash
jq -S 'del(.config.timestamp_utc, .config.git_commit)' \
  neurips/experiment_results/gpt2_results.json > /tmp/reference.json
jq -S 'del(.config.timestamp_utc, .config.git_commit)' \
  neurips/experiment_results/reproduced_gpt2_results.json > /tmp/reproduced.json
diff -u /tmp/reference.json /tmp/reproduced.json
```

The experiment is deterministic for the recorded software, model revision,
and seeds. Note that the `config.git_commit` recorded in the saved results
(`e2528cd`) predates the commit of the experiment code; see the provenance
file for the SHA-256 of the exact code that produced them.

## Rebuild the figures

```bash
python scripts/make_figures.py
```

This reads only `neurips/experiment_results/gpt2_results.json` and writes
`neurips/figures/main_figure.png` and `neurips/figures/rq0_naive_vs_correct.png`.

## What changed

The project originally concluded, from GPT-2 scoring 0% on arithmetic and
GSM8K, that "no single head is responsible" for reasoning and that this was
"not a size issue" (GPT-2 vs. DistilGPT2 behaving the same way). Both claims
were unsupported: with baseline accuracy already at 0%, an ablation cannot
make it worse, so a null result there is a floor-effect artifact, not
evidence that no head matters. On auditing the ablation code itself, the
original hook also turned out to be ablating the wrong thing entirely (point
1 above). The current `src/` code and `scripts/run_neurips_experiments.py`
replace that analysis. The original hand-written 4-8-example task sets and
substring-match accuracy check (`SimpleDataset` in `src/dataset.py`,
`get_model_accuracy` in `src/baseline.py`) are kept only for reference and
are not the basis for any claim in the paper.

## Repository layout

- `src/ablation.py` -- `ValidatedHeadAblator`, the corrected pre-projection
  head ablation used for every reported effect, plus the naive
  post-projection hook reproduced solely for the RQ0 comparison.
- `src/dataset.py`, `src/metrics.py`, `src/controls.py` -- task generation,
  teacher-forced exact match and log-probability, bootstrap intervals,
  matched-control sampling, and Monte Carlo p-values.
- `src/legacy_invalid_ablation.py` -- the original, invalid `HeadAblator`,
  kept for history only.
- `scripts/` -- the paper's experiment pipeline and figure generation.
- `tests/` -- executable checks of the intervention semantics and controls.
- `neurips/experiment_results/` -- saved result files, verification reruns,
  and the reproducibility provenance record.
- `neurips/figures/` -- the paper's figures.
- `neurips/claim_evidence_map.md` -- claim-to-JSON-field map.
- `results/` -- outputs from the original (pre-audit) exploratory runs;
  kept for provenance, not used as evidence for any claim in the paper.
- `run_experiments.py`, `config.py`, `inspect_results.py`, `quick_test.py`,
  `src/experiment.py`, `src/experiment_lite.py`, `src/baseline.py`,
  `src/analysis.py`, `src/visualization.py` -- the original exploratory
  pipeline.

## Third-party assets

The pretrained models are third-party assets under their own terms: GPT-2
under the MIT License and DistilGPT2 under the Apache License 2.0. Hugging
Face `transformers` is used under the Apache License 2.0.
