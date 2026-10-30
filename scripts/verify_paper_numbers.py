#!/usr/bin/env python3
"""
Mechanical check that numbers printed in the camera-ready paper
(neurips/neurips_2026.tex) equal the saved result files.

Each check formats a value read from a JSON field and asserts that the
formatted string occurs in the .tex. It catches transcription and rounding
errors; it does not check that a number is used in the right sentence
(that is what neurips/claim_evidence_map.md is for). Exit code 1 on failure.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "neurips" / "experiment_results"
TEX = (ROOT / "neurips" / "neurips_2026.tex").read_text()


def load(path):
    return json.loads(Path(path).read_text())


checks = []  # (label, expected substring)


def f3(x):
    return f"{x:.3f}"


def neg(s):
    # the tex writes negative numbers as $-0.135$ or -0.135 inside math
    return s.replace("-", "$-") + "$" if s.startswith("-") else s


def add(label, s):
    checks.append((label, s))


def p3(x):
    return f"{x:.3f}"


# ---- primary GPT-2, DistilGPT2, N=500 ---------------------------------------
for name in ["gpt2", "distilgpt2"]:
    d = load(RES / f"{name}_results.json")
    b, r0, r2 = d["baselines"], d["rq0_naive_vs_correct"], d["rq2_candidate_vs_controls"]
    add(f"{name} rq0 pearson", f3(r0["pearson_r"]))
    add(f"{name} copy acc", f3(b["induction_full"]["accuracy_mean"]))
    add(f"{name} copy acc ci", f"[{f3(b['induction_full']['accuracy_ci95'][0])}, {f3(b['induction_full']['accuracy_ci95'][1])}]")
    add(f"{name} copy lp", f"{b['induction_full']['logprob_mean']:.2f}".replace("-", "$-") + "$")
    add(f"{name} arith lp", f"{b['arithmetic_control']['logprob_mean']:.2f}".replace("-", "$-") + "$")
    add(f"{name} cand held", f3(r2["candidate_heldout_delta_mean"]))
    add(f"{name} cand held ci", f"[{f3(r2['candidate_heldout_delta_ci95'][0])}, {f3(r2['candidate_heldout_delta_ci95'][1])}]")
    add(f"{name} random mean", f3(r2["random_control_heldout_delta"]["mean"]))
    add(f"{name} layer mean", f3(r2["layer_matched_control_heldout_delta"]["mean"]))
    add(f"{name} rank rho", f3(r2["rank_stability_spearman_discovery_vs_heldout"]))
    add(f"{name} spec cand", f3(r2["specificity"]["candidate"]))
    add(f"{name} spec random mean", f3(r2["specificity"]["random_control_mean"]))
    add(f"{name} spec random range",
        f"[{neg(f3(r2['specificity']['random_control_p5']))}, {f3(r2['specificity']['random_control_p95'])}]")
    add(f"{name} p random", f"$p$(candidate $\\geq$ random control, held-out $\\Delta$) $={p3(r2['random_control_heldout_delta']['empirical_p_candidate_beats_random'])}$")
    add(f"{name} p spec", f"$p$(candidate $\\geq$ random control, specificity) $={p3(r2['specificity']['empirical_p_candidate_beats_random'])}$")

g = load(RES / "gpt2_results.json")
r0, r2 = g["rq0_naive_vs_correct"], g["rq2_candidate_vs_controls"]
add("gpt2 rq0 spearman", f"\\rho={f3(r0['spearman_rho'])}")
add("gpt2 rq0 mean abs diff", f"${f3(r0['mean_abs_delta_diff'])}$ nats")
add("gpt2 rq0 overlap is 0", "0/5" if r0["top5_head_overlap"] == 0 else "MISMATCH")
add("gpt2 random range", f"$[{f3(r2['random_control_heldout_delta']['p5'])},{f3(r2['random_control_heldout_delta']['p95'])}]$")
add("gpt2 layer range", f"$[{f3(r2['layer_matched_control_heldout_delta']['p5'])},{f3(r2['layer_matched_control_heldout_delta']['p95'])}]$")
add("gpt2 spec range (text)", f"$[{f3(r2['specificity']['random_control_p5'])},{f3(r2['specificity']['random_control_p95'])}]$")
add("gpt2 spec p (text)", f"$p={p3(r2['specificity']['empirical_p_candidate_beats_random'])}$")
assert [list(h) for h in r2["candidate_heads"]] == [[0, 0], [0, 7], [1, 3], [11, 0], [0, 10]]

n5 = load(RES / "gpt2_n500_results.json")["rq2_candidate_vs_controls"]
add("n500 spec", f"${f3(n5['specificity']['candidate'])}$ nats")
add("n500 spec p", f"$p={p3(n5['specificity']['empirical_p_candidate_beats_random'])}$")
assert n5["random_control_heldout_delta"]["empirical_p_candidate_beats_random"] < 0.0011
assert n5["layer_matched_control_heldout_delta"]["empirical_p_candidate_beats_layer_matched"] < 0.0011

# ---- mean ablation ------------------------------------------------------------
for name in ["gpt2", "distilgpt2"]:
    m = load(RES / f"{name}_meanablation_results.json")["mean_ablation"]
    mv, mr = m["mean_vs_zero"], m["rq2_candidate_vs_controls"]
    add(f"{name} mean r", f3(mv["pearson_r_discovery"]))
    add(f"{name} mean rho", f3(mv["spearman_rho_discovery"]))
    add(f"{name} mean overlap", f"{mv['top5_overlap_with_zero_candidates']}/5")
    add(f"{name} mean cand held", f"{f3(mr['candidate_heldout_delta_mean'])} & [{f3(mr['candidate_heldout_delta_ci95'][0])}, {f3(mr['candidate_heldout_delta_ci95'][1])}]")
    rc, lc = mr["random_control_heldout_delta"], mr["layer_matched_control_heldout_delta"]
    add(f"{name} mean random", f"{f3(rc['mean'])} & [{neg(f3(rc['p5']))}, {f3(rc['p95'])}]")
    add(f"{name} mean layer", f"{f3(lc['mean'])} & [{neg(f3(lc['p5']))}, {f3(lc['p95'])}]")
    add(f"{name} mean rho stab", f3(mr["rank_stability_spearman_discovery_vs_heldout"]))
    add(f"{name} mean ps", f"{p3(rc['empirical_p_candidate_beats_random'])} / {p3(lc['empirical_p_candidate_beats_layer_matched'])} / {p3(mr['specificity']['empirical_p_candidate_beats_random'])}")
gm = load(RES / "gpt2_meanablation_results.json")["mean_ablation"]["rq2_candidate_vs_controls"]
assert [list(h) for h in gm["candidate_heads"]] == [[10, 11], [10, 0], [7, 6], [0, 9], [0, 3]]
add("gpt2 mean spec cand", f3(gm["specificity"]["candidate"]))

ph = load(RES / "mean_vs_zero_posthoc.json")
for name in ["gpt2", "distilgpt2", "qwen2.5-1.5b-instruct"]:
    add(f"{name} posthoc zero", f3(ph[name]["heldout_delta_under_zero_mean_over_candidates"]))
    add(f"{name} posthoc mean", f3(ph[name]["heldout_delta_under_mean_mean_over_candidates"]))

# ---- Qwen ------------------------------------------------------------------------
q = load(RES / "qwen2.5-1.5b-instruct_meanablation_results.json")
qb, q0, q2 = q["baselines"], q["rq0_naive_vs_correct"], q["rq2_candidate_vs_controls"]
qm = q["mean_ablation"]
add("qwen rq0", f"{f3(q0['pearson_r'])} / {f3(q0['spearman_rho'])}")
add("qwen copy acc", f"{f3(qb['induction_full']['accuracy_mean'])} & [{f3(qb['induction_full']['accuracy_ci95'][0])}, {f3(qb['induction_full']['accuracy_ci95'][1])}]")
add("qwen arith lp", f"{qb['arithmetic_control']['logprob_mean']:.2f}".replace("-", "$-") + "$")
add("qwen mean r/rho", f"{f3(qm['mean_vs_zero']['pearson_r_discovery'])} / {f3(qm['mean_vs_zero']['spearman_rho_discovery'])}")
for label, r in [("zero", q2), ("mean", qm["rq2_candidate_vs_controls"])]:
    add(f"qwen {label} held", f"{f3(r['candidate_heldout_delta_mean'])} & [{f3(r['candidate_heldout_delta_ci95'][0])}, {f3(r['candidate_heldout_delta_ci95'][1])}]")
    rc, lc, sp = r["random_control_heldout_delta"], r["layer_matched_control_heldout_delta"], r["specificity"]
    add(f"qwen {label} random", f"{f3(rc['mean'])} & [{neg(f3(rc['p5']))}, {f3(rc['p95'])}]")
    add(f"qwen {label} layer", f"{f3(lc['mean'])} & [{neg(f3(lc['p5']))}, {f3(lc['p95'])}]")
    add(f"qwen {label} rho", f3(r["rank_stability_spearman_discovery_vs_heldout"]))
    add(f"qwen {label} spec", f3(sp["candidate"]))
    add(f"qwen {label} spec rand", f"{neg(f3(sp['random_control_mean']))} & [{neg(f3(sp['random_control_p5']))}, {f3(sp['random_control_p95'])}]")
    add(f"qwen {label} ps", f"{p3(rc['empirical_p_candidate_beats_random'])} / {p3(lc['empirical_p_candidate_beats_layer_matched'])} / {p3(sp['empirical_p_candidate_beats_random'])}")
assert qm["mean_vs_zero"]["top5_overlap_with_zero_candidates"] == 5 and q0["top5_head_overlap"] == 0

fmt = {r["model"]: r for r in load(RES / "arithmetic_format_diagnostic.json")["results"]}
assert fmt["gpt2"]["unspaced_answer_exact_match"] == 0.0
add("qwen unspaced", f"{100 * fmt['Qwen/Qwen2.5-1.5B-Instruct']['unspaced_answer_exact_match']:.1f}\\%")

# ---- IOI --------------------------------------------------------------------------
ioi = load(ROOT / "results" / "naacl_ioi" / "gpt2_ioi_ablation_methods.json")
for mth in ["zero", "mean", "resample"]:
    r = ioi["methods"][mth]
    add(f"ioi {mth} held", f"{r['heldout_effect_mean']:.2f} [{r['heldout_effect_ci95'][0]:.2f}, {r['heldout_effect_ci95'][1]:.2f}]")
    assert r["p_random_ge_candidate"] < 0.0011 and r["p_layer_matched_ge_candidate"] < 0.0011
cm = ioi["cross_method"]
add("ioi zero-mean rho", f"zero vs.\\ mean ${cm['zero_vs_mean']['spearman_all_heads']:.2f}$")
add("ioi zero-res rho", f"zero vs.\\ resample ${cm['zero_vs_resample']['spearman_all_heads']:.2f}$")
add("ioi mean-res rho", f"mean vs.\\ resample\n${cm['mean_vs_resample']['spearman_all_heads']:.2f}$")
ratio = ioi["methods"]["zero"]["heldout_effect_mean"] / ioi["methods"]["mean"]["heldout_effect_mean"]
add("ioi 68%", f"{round(100 * (ratio - 1))}\\% larger")
add("ioi frac io", f"{round(100 * ioi['baseline']['discovery']['frac_io_preferred'])}\\% (discovery)")

# ---- camera-ready robustness (C1/C2) ---------------------------------------------
# ---- camera-ready robustness (C1/C2) ---------------------------------------------
for name in ["gpt2", "distilgpt2"]:
    cr = load(RES / f"{name}_camera_ready_robustness.json")
    assert cr["config"]["sanity_max_abs_dev_vs_saved"] <= 1e-4
    c1, c2 = cr["c1_partition_sensitivity"], cr["c2_sequence_control"]
    add(f"{name} c1 held", f"{f3(c1['candidate_heldout_delta']['median'])} [{f3(c1['candidate_heldout_delta']['min'])}, {f3(c1['candidate_heldout_delta']['max'])}]")
    add(f"{name} c1 rho", f"{f3(c1['rank_stability_spearman']['median'])} [{f3(c1['rank_stability_spearman']['min'])}, {f3(c1['rank_stability_spearman']['max'])}]")
    a = c1["arith_specificity"]
    add(f"{name} c1 spec", f"{f3(a['median'])} [{neg(f3(a['min']))}, {f3(a['max'])}]")
    ps = c1["p_arith_specificity"]
    add(f"{name} c1 spec p", f"{f3(ps['median'])} [{f3(ps['min'])}, {f3(ps['max'])}]")
    add(f"{name} c1 n spec", f"{c1['n_partitions_p_arith_specificity_le_0.05']}/20")
    add(f"{name} c1 controls", f"{c1['n_partitions_p_random_le_0.05']}/20 / {c1['n_partitions_p_layer_matched_le_0.05']}/20")
    add(f"{name} c1 overlap5", f"{c1['overlap_with_primary_counts'].get('5', 0)}/20")
    pz, sm = c2["primary_zero_candidates"], c2["secondary_mean_candidates"]
    add(f"{name} c2 seq delta", f3(pz["candidate_control_task_delta"]))
    add(f"{name} c2 seq rand", f"mean {f3(pz['random_control_task_delta_mean'])}")
    add(f"{name} c2 seq p", f"$p={p3(pz['empirical_p_candidate_control_delta_ge_random'])}$")
    add(f"{name} c2 spec", neg(f3(pz["candidate_specificity"])) if pz["candidate_specificity"] < 0 else f3(pz["candidate_specificity"]))
    add(f"{name} c2 spec p", f"$p={p3(pz['empirical_p_candidate_beats_random'])}$")
    add(f"{name} c2 mean spec", neg(f3(sm["candidate_specificity"])) if sm["candidate_specificity"] < 0 else f3(sm["candidate_specificity"]))
    add(f"{name} c2 mean p", f"$p={p3(sm['empirical_p_candidate_beats_random'])}$")
    add(f"{name} c2 baseline acc", f"{100 * c2['baseline']['accuracy_mean']:.1f}\\%")
    assert c2["baseline"]["n"] == 129
gcr = load(RES / "gpt2_camera_ready_robustness.json")
add("gpt2 c2 rand spec mean", neg(f3(gcr["c2_sequence_control"]["primary_zero_candidates"]["random_specificity_mean"])))
add("gpt2 c2 baseline lp", f"{gcr['c2_sequence_control']['baseline']['logprob_mean']:.2f}".replace("-", "$-") + "$")
assert gcr["c1_partition_sensitivity"]["head_selection_counts"] == {"1_3": 20, "0_7": 20, "11_0": 20, "0_0": 20, "0_10": 19, "7_6": 1}

failed = [(lab, s) for lab, s in checks if s not in TEX]
for lab, s in failed:
    print(f"MISSING  {lab}: {s!r}")
print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
sys.exit(1 if failed else 0)
