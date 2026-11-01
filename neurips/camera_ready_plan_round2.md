# Camera-ready, round 2: remaining items from the external review audit

Paper: `neurips/neurips_2026.tex` (built PDF `neurips/neurips_2026.pdf`). Baseline commit: `7776156`.
Source of items: an external audit of the camera-ready PDF against reviews QC4F and E5js (pasted by the author, 2026-11-01).

Rules carried over from round 1: no new claim without a result file; original result JSONs are never overwritten
(new outputs use "x" mode); every printed number stays checked by `scripts/verify_paper_numbers.py`.

## Items

| # | Item (audit wording) | Type | What changes | Verified by |
|---|---|---|---|---|
| R1 | "Mean-ablation is called on-distribution; a pooled mean need not be a value a head produces" | WRITING | §4.4 opening: describe mean and resample as alternative replacement values; say only resample uses activations the heads actually produce. Replace "on-distribution baseline" (abstract, intro (iv), §4.5, §5 Broader implication) with wording that does not claim mean is in-distribution. | grep: no "on-distribution"/"stay on the model's activation distribution" applied to mean; build |
| R2 | "0.0% accuracy with bootstrap CI [0.0%, 0.0%] needs qualification" | CODE + WRITING | New script `scripts/binomial_floor_intervals.py` computes exact Clopper-Pearson 95% intervals for every 0/N accuracy from the saved JSONs (GPT-2, DistilGPT2, GPT-2 N=500, Qwen2.5), writes `binomial_floor_intervals.json` (new file, "x" mode). Unit test for the interval function against closed forms. Paper: §3 Statistics notes that a percentile bootstrap is degenerate when every example fails and that exact binomial intervals are given instead; §4.2 and Tables 1, 3, 4 report the binomial interval for the floor rows. | test passes; verifier checks the new numbers; build |
| R3 | "Check 'fully reproducible' against the released GitHub repo" | VERIFY (+ WRITING if needed) | Fresh clone of the public repo into scratch: (a) every file named in App. F and the README exists; (b) tests pass; (c) rerun the primary GPT-2 and DistilGPT2 pipelines from the clone and diff against the saved JSONs (ignoring timestamp and git commit); (d) the figure script reproduces both figures. Replace "fully reproducible" in §1 with the precise, checked statement. Record the result in the provenance file. | diff output |
| R4 | "Is the checklist required? Is the template/instructions satisfied?" | VERIFY | Check the workshop call and NeurIPS rules for the checklist and template requirements; confirm style option `dblblindworkshop,final`, page count, single PDF. No paper change unless a requirement is unmet. | documented below |

## Not planned (stated limitations, unchanged)

- Prevalence of the naive hook in practice (TransformerLens / nnsight users): not measured; §2 already says so. The audit
  agrees no experiment is needed unless we claim the error is widespread, and we do not.
- Models above 1.5B parameters, noise and optimal ablation, Greater-Than: stated in §5.

## Status

- [x] R1 wording: no "on-distribution" remains; mean and resample are "alternative replacement values".
- [x] R2 exact intervals: 0/200 -> [0.0%, 1.8%]; 3 new tests; 2 new verifier checks (144/144).
- [x] R3 fresh clone at `7776156`: 47/47 named files present, 33/33 tests, GPT-2 and DistilGPT2 reruns identical to saved files
  (ignoring timestamp, Git commit), both PNG figures byte-identical. Sec. 1 and Sec. 3 now state exactly this.
- [x] R4 workshop call requires NeurIPS format only; checklist allowed and outside the page limit; footer correct; main text ends p. 9.
