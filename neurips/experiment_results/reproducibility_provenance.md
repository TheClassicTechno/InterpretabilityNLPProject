# Reproducibility provenance

Recorded on 2026-09-01. Every result file in this directory was produced by
the code as packaged, on the machine described below.

## Environment

- Hardware: one Apple M5 CPU with 10 cores and 24 GB unified memory. No GPU.
- Software: Python 3.9, PyTorch 2.8.0, and transformers 4.57.3.
- Local storage used by the virtual environment, both model caches, and
  result JSON files: about 1.5 GB.

## Source file identity

SHA-256 of the code as packaged:

| File | SHA-256 |
|---|---|
| `scripts/run_neurips_experiments.py` | `a09559c0fed85266c89b957c2f708f7f6be88e2ee6a40332a6b9dab7c1f7a31f` |
| `scripts/make_figures.py` | `0210d51172386d132d8bdc5bb5d8c5929fea3cf2448f59a4755b0169099b07c9` |
| `src/ablation.py` | `60b518e23d9a8ca15a0119b94fb6ade5655384b5e09d5198fce1c5aed631a5d9` |
| `src/controls.py` | `8881524825c86b00b9e78ed408b2e41cb98e0b35e0f61b215e186b27a2e9173a` |
| `src/dataset.py` | `8b84b09ccd4624be9b6c99968e477fe0506bc20b8edf698207eefbe91bfbf1d6` |
| `src/metrics.py` | `772fb28194c3c7cbd3679f33c140395347144f6a6c8bd42c2ae6bbd57a75d7c3` |
| `src/model_loader.py` | `9ce98416799a76369f1fc3325e968fcf2cf26a4542d306ea682d9c4b19bc34bd` |
| `src/__init__.py` | `35314410ae2fe26917a3456e1d05e686b62e65d749714ab26b748e7dfa5c10f5` |

## Model identity

GPT-2:

- Hugging Face revision: `607a30d783dfa663caf39e06633721c8d4cfcd7e`
- `model.safetensors` SHA-256:
  `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`
- `config.json` SHA-256:
  `0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4`
- `tokenizer.json` SHA-256:
  `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`

DistilGPT2:

- Hugging Face revision: `2290a62682d06624634c1f46a6ad5be0f47f38aa`
- `model.safetensors` SHA-256:
  `e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b`
- `config.json` SHA-256:
  `4ec5947c1d59fee6212cdf3b0ec1a53eac02092554c5ff0a733488cbd2c64f3a`
- `tokenizer.json` SHA-256:
  `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`

## Measured runs

All six runs used the cached model revisions above with offline loading.
The three reported configurations were each run twice: once to produce the
result file used in the paper, and once again, independently, as a
verification rerun.

| Run | Configuration | Wall time | Maximum resident set size |
|---|---|---:|---:|
| GPT-2 primary | 200 induction, 200 arithmetic, maximum operand 20 | 198.99 s | 2,083,405,824 bytes (1.94 GiB) |
| DistilGPT2 replication | 200 induction, 200 arithmetic, maximum operand 20 | 57.62 s | 2,034,941,952 bytes (1.90 GiB) |
| GPT-2 larger run | 500 induction, 500 arithmetic, maximum operand 30 | 428.56 s | 3,458,056,192 bytes (3.22 GiB) |
| GPT-2 primary, verification | 200 induction, 200 arithmetic, maximum operand 20 | 177.97 s | 2,271,215,616 bytes (2.12 GiB) |
| DistilGPT2, verification | 200 induction, 200 arithmetic, maximum operand 20 | 59.72 s | 1,898,774,528 bytes (1.77 GiB) |
| GPT-2 larger run, verification | 500 induction, 500 arithmetic, maximum operand 30 | 446.18 s | 3,657,842,688 bytes (3.41 GiB) |

The commands were:

```bash
python scripts/run_neurips_experiments.py --model gpt2 \
  --output neurips/experiment_results/gpt2_results.json \
  --n-induction 200 --n-arithmetic 200 --arithmetic-max-operand 20

python scripts/run_neurips_experiments.py --model distilgpt2 \
  --output neurips/experiment_results/distilgpt2_results.json \
  --n-induction 200 --n-arithmetic 200 --arithmetic-max-operand 20

python scripts/run_neurips_experiments.py --model gpt2 \
  --output neurips/experiment_results/gpt2_n500_results.json \
  --n-induction 500 --n-arithmetic 500 --arithmetic-max-operand 30
```

each repeated with the output redirected to the matching
`verification_*.json` file.

## Verification outcome

Each verification file is byte-identical to the result file it verifies
after removing only `config.timestamp_utc`. No scientific field differs in
any of the three pairs, including per-example values and the full 144-head
effect vectors. This covers the primary GPT-2 configuration as well as the
two secondary ones.

To check a pair yourself:

```bash
diff <(jq -S 'del(.config.timestamp_utc)' neurips/experiment_results/gpt2_results.json) \
     <(jq -S 'del(.config.timestamp_utc)' neurips/experiment_results/verification_gpt2_results.json)
```

## Recorded Git commit

Each result file records `config.git_commit` =
`e2528cd4f2bc49b447942fede669a88f21cb91a4`. That is the commit the working
tree was based on when the runs were made; the experiment code itself was
uncommitted at the time, so checking out `e2528cd` does **not** give the code
that produced these results. The code that did is identified by the SHA-256
table under "Source file identity" above, and those exact files were first
committed to this repository together with these result files.

## File hashes

Hashes of the result files in this directory:

| File | SHA-256 |
|---|---|
| `gpt2_results.json` | `d5b71a4cd9b0234d50fc2eae914e4077c243e3502dad7e382c4e6c2dc78e38aa` |
| `distilgpt2_results.json` | `9758bd930397592e70c152e609dfcc3e906f51fc5c7c046365a588f3f354d0d7` |
| `gpt2_n500_results.json` | `95a7509649622bed0d01f2bbd22c465faedffec04496d32eb68a6e64959aafa9` |
| `verification_gpt2_results.json` | `9a54ee954e67e96ac2fd1b6f8b7dfaa30fc1496f25b7ba48450ad635c938d862` |
| `verification_distilgpt2_results.json` | `44778801f73aa0a8de23516b4838497f5e8dba78c6d2b48ea97da51833bd11f5` |
| `verification_gpt2_n500_results.json` | `e372bd626bffffa27e53865507e1236d3190f2e02dea5e9d801d7997ae66c09f` |

The anonymized review package prepared for the NeurIPS 2026 workshop
contained copies of these six files with only `config.git_commit` removed; no
other field differs.

## Mean-ablation robustness runs (added 2026-10-11, ARR/NAACL submission)

Design fixed before running: mean-ablation replaces a head's slice of the
`c_proj` input with its mean over all non-padding token positions of the
induction discovery split; the same means are applied to every split. Each
run was performed once and is reported regardless of outcome. Same machine,
environment, seeds, and model revisions as above.

| Run | Configuration | Wall time | Maximum resident set size |
|---|---|---:|---:|
| GPT-2 mean-ablation | 200 induction, 200 arithmetic, maximum operand 20 | 324.57 s | 2,114,846,720 bytes (1.97 GiB) |
| DistilGPT2 mean-ablation | 200 induction, 200 arithmetic, maximum operand 20 | 107.79 s | 2,089,025,536 bytes (1.95 GiB) |
| GPT-2 zero-mode regression after refactor | 200 induction, 200 arithmetic, maximum operand 20 | 188.20 s | 2,545,270,784 bytes (2.37 GiB) |

```bash
python scripts/run_neurips_experiments.py --model gpt2 --ablation mean
python scripts/run_neurips_experiments.py --model distilgpt2 --ablation mean
python scripts/posthoc_mean_vs_zero.py   # post hoc, reads the two files above
```

Checks:

- The zero-mode regression rerun (written outside this directory) equals
  `gpt2_results.json` after removing `config.timestamp_utc` and
  `config.git_commit`, so the refactor (`rq2_statistics` helper) did not
  change the zero-ablation pipeline.
- Each `*_meanablation_results.json` recomputes the full zero-ablation
  pipeline; with `config.timestamp_utc`, `config.git_commit`,
  `config.ablation_baseline`, and `mean_ablation` removed it equals the
  corresponding primary result file.
- `mean_vs_zero_posthoc.json` is a post hoc summary (no forward passes).

| File | SHA-256 |
|---|---|
| `gpt2_meanablation_results.json` | `7ab186b88833131f170ef8e0a73c3c2daa387317616824005d224b07828ae3d1` |
| `distilgpt2_meanablation_results.json` | `bb30b37597eeeaffd458859df11bfaec04313645072b902242baf7a53433890e` |
| `mean_vs_zero_posthoc.json` (GPT-2/DistilGPT2 only; superseded below) | `2149b050fd0f4e5601791e78800add07b2d917ac3bec9d0e12a05ee98c5fa1f4` |

Source files used for these runs:

| File | SHA-256 |
|---|---|
| `src/ablation.py` | `a0fb65fdaf7785ba6790e65ae62253da210603e96851ce500aab0a3d0d427ece` |
| `scripts/run_neurips_experiments.py` | `62423b9aeb92f0dd7129daf1759fd699ce8f7860fb92c5ad7616777222eb89b2` |
| `scripts/posthoc_mean_vs_zero.py` | `2ec63fcfb743dac28613f01d8c19795e36a8a1c50896b72e3f21971c6bfd4ddd` |
| `tests/test_ablation_semantics.py` | `55470d1877d38dba811c58d69a4fc707ba5fe65718e343741f79a2c187c5f3a1` |

## Qwen2.5-1.5B-Instruct replication and arithmetic-format diagnostic

The architecture adapter (`attention_modules` in `src/ablation.py`) places
pre-projection hooks on `o_proj` for Llama/Qwen2-family models and leaves
GPT-2-family behavior unchanged: after the change, a DistilGPT2 zero-mode rerun
equaled `distilgpt2_results.json` after removing `config.timestamp_utc` and
`config.git_commit`. Three tests on a tiny random Qwen2 configuration assert
the same intervention semantics as for GPT-2 (20 tests in total).

| Run | Configuration | User CPU time | Maximum resident set size |
|---|---|---:|---:|
| Qwen2.5-1.5B-Instruct, zero + mean ablation | 200 induction, 200 arithmetic, maximum operand 20 | 11,975.83 s | 9,667,149,824 bytes (9.0 GiB) |

Wall-clock time (77,218 s) is not meaningful because the machine was
suspended for much of the run. The run was performed once and not repeated.

```bash
python scripts/run_neurips_experiments.py --model Qwen/Qwen2.5-1.5B-Instruct --ablation mean \
  --output neurips/experiment_results/qwen2.5-1.5b-instruct_meanablation_results.json
python scripts/posthoc_mean_vs_zero.py          # now also summarizes Qwen
python scripts/diagnose_arithmetic_format.py    # post hoc, all three models
```

`arithmetic_format_diagnostic.json` (post hoc): teacher-forced exact match on
the arithmetic task with the answer written without its leading space is 0.0
for GPT-2 and DistilGPT2 and 0.23 for Qwen2.5-1.5B-Instruct, whose tokenizer
splits " 24" into a bare space token plus digits.

`mean_vs_zero_posthoc.json` was regenerated to add Qwen; its GPT-2 and
DistilGPT2 values are unchanged.

| File | SHA-256 |
|---|---|
| `qwen2.5-1.5b-instruct_meanablation_results.json` | `89b9cd002dc7bc0b296504e6d9fc97ddb45c89fab3968c855822ffb0d2d31793` |
| `mean_vs_zero_posthoc.json` (regenerated) | `0d389c079a1f20b6026426e25112cae447a5a2b01be7baf2918ba1f6eec97c60` |
| `arithmetic_format_diagnostic.json` | `e591f6a47d37f978bc68c84aaa6fc8ee7cd85fb3e404da1853af53dfb72cee3a` |
| `src/ablation.py` (with adapter; used for the Qwen run) | `39fb03ba614c7676cfc471593c7462d6d084da34688c32de1a1bd23d636b2390` |
| `tests/test_ablation_semantics.py` | `b24ba9cd651f80eaa89ceb77f9a1b42ad57c5bd6fd5171a159858f32441665f8` |
| `scripts/diagnose_arithmetic_format.py` | `025edd5e42c1b66809f8898f8222c601360a5bb880ea940625a2aa9a736ed222` |
| `scripts/posthoc_mean_vs_zero.py` | `173b014868119562a196a6e624950c3cdb8a1a70cec69f0035c36d448a75ee50` |

## Camera-ready analyses (added 2026-10-30, NeurIPS InterpScience camera-ready)

Design fixed in writing before running: `neurips/camera_ready_design.md` (Sec. 3, C1 partition sensitivity, C2 sequence-continuation control).
Same machine and environment as above (torch 2.8.0, transformers 4.57.3, CPU, float32). Measured with `/usr/bin/time -l`.

| Run | Command | Wall time | Peak RSS | Output |
|---|---|---|---|---|
| C1/C2, GPT-2 | `python scripts/run_camera_ready_robustness.py --model gpt2` | 301.43 s (in-script 298.77 s) | 2,042,281,984 B (1.90 GiB) | `gpt2_camera_ready_robustness.json` |
| C1/C2, DistilGPT2 | `python scripts/run_camera_ready_robustness.py --model distilgpt2` | 82.08 s (in-script 79.97 s) | 1,745,158,144 B (1.63 GiB) | `distilgpt2_camera_ready_robustness.json` |

Built-in sanity check: per-head Δ recomputed on all 200 copying examples, restricted to the original partition, matches the saved
`per_head_delta_discovery` / `per_head_delta_heldout` with max abs deviation 0.0 (`config.sanity_max_abs_dev_vs_saved`) for both models.

Verification performed 2026-10-30 before the camera-ready numbers were trusted:

- Regression: `scripts/run_neurips_experiments.py --model gpt2 --output <scratch>/gpt2_results.json` (default configuration) with the current code reproduces
  `gpt2_results.json` exactly apart from `config.timestamp_utc` and `config.git_commit`.
- IOI reproduction: `scripts/run_ioi_ablation_methods.py` rerun to a scratch directory reproduces
  `results/naacl_ioi/gpt2_ioi_ablation_methods.json` exactly apart from `config.timestamp_utc`.
- Hashes of all pre-existing result files (sections above and `results/naacl_ioi/*.json`) were recorded before the camera-ready work
  and are unchanged after it.
- `pytest tests/`: 29 passed (includes `tests/test_sequence_control.py`).
- `scripts/verify_paper_numbers.py`: 121/121 printed values match the result files (140/140 after C3 below).

| File | SHA-256 |
|---|---|
| `gpt2_camera_ready_robustness.json` | `3576c6b78fb8460d0bbbf1e9ea94904abaacde5126f2c466e9568a0a2971c235` |
| `distilgpt2_camera_ready_robustness.json` | `cfd0c4f3b185be420dc13400432dcc76d1110c9fd940b52d247e8aa21b8a2699` |
| `scripts/run_camera_ready_robustness.py` | `01e89072b42737ab150bbb5bb8837d64edc99876584d95d1b6a99a46bdeaa774` |
| `src/dataset.py` (adds the sequence-continuation task) | `ce5711a7e84c1a212ae9660f37c5a8852348ab061d2ea7a4ea54508dfc1d8ea3` |
| `src/ablation.py` | `e788f80e448453e118bf4dc8732e5967425355b4748c350a6ecbd99e012aa9c6` |
| `tests/test_sequence_control.py` | `07f027c55b9f2b822dd15142dff27983da6cb503ada628a4f98eb2d4d9c15a0c` |
| `scripts/verify_paper_numbers.py` | `c86c4b25cfb7670c2e220d3763db8b92c094079927db164bd7e27311ce6753de` |

## C3: resample ablation on the copying task (added 2026-10-30)

Design committed and pushed before any code was written or run: `neurips/camera_ready_design.md` (Sec. 3, C3), commit `584f871`.
Same machine and environment. Measured with `/usr/bin/time -l`.

| Run | Command | Wall time | Peak RSS | Output |
|---|---|---|---|---|
| C3, GPT-2 | `python scripts/run_resample_copying.py --model gpt2` | 914.99 s (in-script 911.26 s) | 2,297,217,024 B (2.14 GiB) | `gpt2_resample_copying.json` |
| C3, DistilGPT2 | `python scripts/run_resample_copying.py --model distilgpt2` | 325.98 s (in-script 323.72 s) | 2,258,468,864 B (2.10 GiB) | `distilgpt2_resample_copying.json` |

Built-in sanity checks, both models: unablated per-example log-probs equal the saved baseline (max abs deviation 0.0);
resampling a head from the target itself changes nothing (max abs Δ 0.0); source and target batches have identical shapes,
attention masks, and answer spans. Discovery example 63 (first word splits into 3 tokens) has no same-layout partner and is
excluded (discovery N = 99, held-out N = 100). Each run was performed once. Pre-specified reading of R1: (A) for all three
source seeds on both models. After the run: `pytest tests/` 33 passed; `scripts/verify_paper_numbers.py` 140/140.

| File | SHA-256 |
|---|---|
| `gpt2_resample_copying.json` | `80a580129cede50246818e8c804689359c21b0186ea3d223ff1ba319bd5972de` |
| `distilgpt2_resample_copying.json` | `02467e4859150a9a9d145d85c973e1cbfe07769bf7ca70e431ab079ef91bf5db` |
| `scripts/run_resample_copying.py` | `75c747353a52c263931e97a74a2b7224f07ee3c2fdfa1b9994a310ec3634f989` |
| `src/ablation.py` (adds source matching) | `3bbab0c5dc3ec9d28d990605cc6780706bc16cb32db5d1ad6c519f5d37f7e56e` |
| `tests/test_resample_ablation.py` | `e54aa9232cba9e57aecbbf1d02b772bb909703781fd6d25c9754a234bbae29b1` |
| `scripts/verify_paper_numbers.py` (adds C3 checks) | `e34dab97bb4c9ced2e00fce9edafbd3d953174c929a24a4a48fbc04d0d469ccb` |
