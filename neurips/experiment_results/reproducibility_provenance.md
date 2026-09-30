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
