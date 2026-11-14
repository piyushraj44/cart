"""Benchmarks for the ClearCart intelligence layer (FSPM, MCAI, CDI, counterfactual rewrites).

Run from the project root:  .venv/bin/python -m ml.benchmark_intelligence

Writes ml/artifacts/intelligence_benchmark.json and report/figures/intel_*.png.

IMPORTANT - what these numbers are (and are not)
  * No labelled dataset of journeys, choice sets or composite deception levels exists. Every
    experiment below is either pure maths (A1, A3), a property/sensitivity experiment on SYNTHETIC
    journeys built from real held-out TEST-split strings (A2, A4), a SYNTHETIC benchmark whose
    labels come from how the cases were constructed (A4, B), a description of the real demo inputs
    (C, whose journeys are themselves synthetic and flagged as such), or a model-estimated rewrite
    experiment on real TEST-split rows (D). None of them is real-world validation.
  * All text that is scored comes from data/processed/text_dataset.csv rows with split == 'test'
    unless it is explicitly listed as AUTHORED (accept-button labels, the neutral "No thanks",
    the screenshot headings).
  * Seeds: numpy SeedSequence(2026), one child stream per experiment.
"""
from __future__ import annotations

import ml  # noqa: F401  (sets OMP_NUM_THREADS=1 before torch/xgboost load - required on macOS)

import difflib
import itertools
import json
import platform
import re
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ml.common import ARTIFACTS, DATA_PROCESSED, FIGURES, ROOT

OUT_JSON = ARTIFACTS / "intelligence_benchmark.json"
SEED = 2026
SCHEMES = ("fibonacci", "linear", "equal")
COLORS = {"fibonacci": "#2a78d6", "linear": "#eb6834", "equal": "#1baf7a"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ACCENT = "#eda100"

_children = np.random.SeedSequence(SEED).spawn(8)
RNG = {name: np.random.default_rng(s) for name, s in zip(("A2", "A4", "A4boot", "B", "B_shot", "D", "spare1", "spare2"), _children)}


# ============================================================================ utilities
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
                         "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 9,
                         "legend.frameon": False, "axes.titlesize": 10, "axes.titleweight": "bold"})
    return plt


def r(x, n=4):
    if x is None:
        return None
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else round(float(x), n)
    return x


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def load_test() -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / "text_dataset.csv", keep_default_na=False)
    t = df[df["split"] == "test"].reset_index(drop=True)
    assert len(t) > 0
    return t


class Ctx:
    def __init__(self):
        from backend.auditor import Auditor
        from ml.model import TextClassifier
        t0 = time.perf_counter()
        self.clf = TextClassifier.load(ARTIFACTS)
        self.clf.predict(["warm-up"])
        self.load_s = time.perf_counter() - t0
        # false-positive gates off: the saved intelligence artefacts were produced before the gates existed
        # (ml/eval_false_positives.py), so this keeps them reproducible.
        self.auditor = Auditor(self.clf, quote_gate=False, ood_gate=False)
        self.tau = float(self.clf.b.threshold)
        self.test = load_test()
        self._stage_cache: dict[tuple, tuple] = {}

    def stage_pressure(self, strings: tuple[str, ...]) -> tuple[float, int]:
        """Stage pressure exactly as the engine computes it for a text stage:
        segment_text -> Auditor.audit(..., 'text') -> pressure.content_pressure(findings)."""
        from backend.extractors import segment_text
        from backend.intelligence.pressure import category_maxima, content_pressure, noisy_or
        if strings not in self._stage_cache:
            cands = segment_text("\n".join(strings))
            audit = self.auditor.audit(cands, "text")
            p, _src = content_pressure(audit.findings)
            best, _ = category_maxima(audit.findings)
            # [0] current engine definition; [2] legacy noisy-OR over the same category maxima (comparison only)
            self._stage_cache[strings] = (float(p), sum(f.detection_method == "ml_classifier" for f in audit.findings),
                                          float(noisy_or(list(best.values()))))
        return self._stage_cache[strings]


def auroc(pos, neg):
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg]))


def auroc_ci(pos, neg, rng, n=1000):
    pos, neg = np.asarray(pos), np.asarray(neg)
    vals = []
    for _ in range(n):
        a = pos[rng.integers(0, len(pos), len(pos))]
        b = neg[rng.integers(0, len(neg), len(neg))]
        if np.ptp(np.r_[a, b]) == 0:
            vals.append(0.5)
            continue
        vals.append(auroc(a, b))
    return [r(float(np.percentile(vals, 2.5))), r(float(np.percentile(vals, 97.5)))]


# ============================================================================ A. FSPM
def exp_A1():
    from backend.intelligence.fspm import normalised_weights
    rows = []
    for n in range(2, 9):
        pos = list(range(1, n + 1))
        row = {"n_stages": n}
        for s in SCHEMES:
            w = normalised_weights(s, pos)
            row[s] = {"final_stage_share": r(w[-1]), "first_stage_share": r(w[0]),
                      "last_two_share": r(w[-1] + w[-2]), "weights": [r(x) for x in w]}
        rows.append(row)
    return {"synthetic": False, "kind": "pure maths (positions 1..n, submission order)", "rows": rows}


def sample_journeys(ctx: Ctx, rng, n_journeys: int, n_stages: int = 4, per_stage: int = 3):
    texts = ctx.test["text"].tolist()
    rows = ctx.test["row_id"].tolist()
    out = []
    for _ in range(n_journeys):
        stages = []
        for _ in range(n_stages):
            idx = rng.choice(len(texts), per_stage, replace=False)
            stages.append(tuple(texts[i] for i in idx))
        out.append(stages)
    return out


def exp_A2_A3(ctx: Ctx):
    from backend.intelligence.fspm import normalised_weights, weighted_score
    rng = RNG["A2"]
    journeys = sample_journeys(ctx, rng, 500)
    P = np.array([[ctx.stage_pressure(st)[0] for st in j] for j in journeys])        # 500 x 4
    P_legacy = np.array([[ctx.stage_pressure(st)[2] for st in j] for j in journeys])
    pos = [1, 2, 3, 4]
    diffs = {s: [] for s in SCHEMES}
    for p in P:
        esc, desc = sorted(p), sorted(p, reverse=True)
        for s in SCHEMES:
            diffs[s].append(weighted_score(list(esc), pos, s) - weighted_score(list(desc), pos, s))
    n_distinct = int(sum(np.ptp(p) > 0 for p in P))
    A2 = {"synthetic": True, "kind": "property/sensitivity experiment (no ground truth)",
          "design": ("500 synthetic 4-stage journeys; each stage = 3 distinct TEST-split strings drawn uniformly at random "
                     "(any label) and joined with newlines; stage pressure = content_pressure(Auditor.audit(segment_text(stage),'text').findings), "
                     "i.e. exactly the per-stage quantity the engine uses for a text stage. Positions 1..4 (submission order). "
                     "Escalating = the journey's 4 pressures sorted ascending; de-escalating = sorted descending."),
          "n_journeys": len(P), "journeys_with_non_constant_pressure": n_distinct,
          "stage_pressure_summary": {"mean": r(P.mean()), "median": r(float(np.median(P))), "share_zero": r(float((P == 0).mean())),
                                     "share_ge_0_5": r(float((P >= 0.5).mean())),
                                     "n_distinct_values_rounded_3dp": int(len(np.unique(np.round(P, 3))))},
          "legacy_noisy_or_stage_pressure_summary": {"mean": r(P_legacy.mean()), "median": r(float(np.median(P_legacy))),
                                                     "share_ge_0_99": r(float((P_legacy >= 0.99).mean())),
                                                     "share_zero": r(float((P_legacy == 0).mean())),
                                                     "n_distinct_values_rounded_3dp": int(len(np.unique(np.round(P_legacy, 3))))},
          "per_scheme": {}}
    for s in SCHEMES:
        d = np.array(diffs[s])
        A2["per_scheme"][s] = {"mean_escalating_minus_deescalating": r(d.mean()), "median": r(float(np.median(d))),
                               "max": r(d.max()), "frac_escalating_gt_deescalating": r(float((d > 1e-9).mean())),
                               "frac_equal": r(float((np.abs(d) <= 1e-9).mean())), "max_abs_diff": r(float(np.abs(d).max()), 12)}
    A2["note"] = ("Equal weighting is order-invariant by construction (every permutation gives the same score); the measured "
                  "max |difference| for 'equal' is floating-point noise only. Ties for fibonacci/linear occur only when all four "
                  "stage pressures are identical.")

    # ---------------- A3 single-stage perturbation
    w_exp = {s: normalised_weights(s, pos) for s in SCHEMES}
    marg = {s: [[] for _ in pos] for s in SCHEMES}
    for p in P:
        for s in SCHEMES:
            for i in range(4):
                lo, hi = p.copy(), p.copy()
                lo[i], hi[i] = 0.0, 1.0
                marg[s][i].append(weighted_score(list(hi), pos, s) - weighted_score(list(lo), pos, s))
    A3 = {"synthetic": True, "kind": "property experiment (linearity check)",
          "design": "Same 500 sampled journeys as A2. For position i, set p_i=0 and p_i=1 (others at their sampled values); "
                    "marginal = score(p_i=1) - score(p_i=0). Theory: marginal = 100 * w_i exactly.",
          "per_scheme": {}}
    for s in SCHEMES:
        A3["per_scheme"][s] = {
            "theoretical_100w": [r(100 * w, 6) for w in w_exp[s]],
            "mean_marginal": [r(float(np.mean(m)), 6) for m in marg[s]],
            "max_abs_deviation_from_100w": r(max(float(np.max(np.abs(np.array(m) - 100 * w_exp[s][i]))) for i, m in enumerate(marg[s])), 12),
        }
    return A2, A3, P, diffs


def exp_A4(ctx: Ctx):
    from backend.intelligence.fspm import weighted_score
    rng = RNG["A4"]
    t = ctx.test
    benign = t[t["label"] == "benign"][["row_id", "text"]].values.tolist()
    dark = t[t["label"] != "benign"][["row_id", "text"]].values.tolist()
    n_per = 100
    groups = {"late": [], "early": [], "none": []}
    used_dark = Counter()

    def benign_stage(k=3):
        idx = rng.choice(len(benign), k, replace=False)
        return [benign[i] for i in idx]

    def dark_stage():
        b = benign_stage(2)
        d = dark[rng.integers(len(dark))]
        used_dark[d[0]] += 1
        items = b + [d]
        order = rng.permutation(3)
        return [items[i] for i in order]

    for g in groups:
        for _ in range(n_per):
            if g == "late":
                st = [benign_stage(), benign_stage(), dark_stage(), dark_stage()]
            elif g == "early":
                st = [dark_stage(), dark_stage(), benign_stage(), benign_stage()]
            else:
                st = [benign_stage() for _ in range(4)]
            groups[g].append(st)
    pressures = {g: np.array([[ctx.stage_pressure(tuple(x[1] for x in stage))[0] for stage in j] for j in js]) for g, js in groups.items()}
    legacy = {g: np.array([[ctx.stage_pressure(tuple(x[1] for x in stage))[2] for stage in j] for j in js]) for g, js in groups.items()}
    position_sets = {"submission_order_1234": [1, 2, 3, 4], "canonical_product_to_payment_2345": [2, 3, 4, 5]}
    out = {"synthetic": True, "labels_from_construction": True,
           "kind": "synthetic labelled benchmark - measures whether the weighting behaves as designed, NOT real-world validity",
           "design": ("300 synthetic 4-stage journeys (100 per group). Every stage has 3 real TEST-split strings. 'late': stages 3-4 each "
                      "contain 1 random dark (label != benign) string + 2 benign strings, stages 1-2 are 3 benign strings; 'early': "
                      "the mirror image; 'none': benign strings only. Labels come from this construction. Stage pressure as in A2. "
                      "Benign strings can be false positives of the classifier and dark strings can be missed, so 'none' is not 0 by construction."),
           "n_per_group": n_per, "distinct_dark_rows_used": len(used_dark),
           "stage_pressure_mean_by_group_and_stage": {g: [r(float(x)) for x in pressures[g].mean(axis=0)] for g in groups},
           "share_of_none_journeys_with_any_nonzero_stage": r(float((pressures["none"].max(axis=1) > 0).mean())),
           "position_sets": {}}
    scores_out = {}
    for pname, pos in position_sets.items():
        res = {"positions": pos, "per_scheme": {}}
        for s in SCHEMES:
            sc = {g: np.array([weighted_score(list(p), pos, s) for p in pressures[g]]) for g in groups}
            if pname == "submission_order_1234":
                scores_out[s] = sc
            anyd = np.r_[sc["late"], sc["early"]]
            res["per_scheme"][s] = {
                "mean_score": {g: r(float(sc[g].mean()), 2) for g in groups},
                "sd_score": {g: r(float(sc[g].std(ddof=1)), 2) for g in groups},
                "auroc_late_vs_early": r(auroc(sc["late"], sc["early"])),
                "auroc_late_vs_early_ci95": auroc_ci(sc["late"], sc["early"], RNG["A4boot"]),
                "auroc_late_vs_none": r(auroc(sc["late"], sc["none"])),
                "auroc_late_vs_none_ci95": auroc_ci(sc["late"], sc["none"], RNG["A4boot"]),
                "auroc_anydark_vs_none": r(auroc(anyd, sc["none"])),
                "auroc_anydark_vs_none_ci95": auroc_ci(anyd, sc["none"], RNG["A4boot"]),
                "auroc_early_vs_none": r(auroc(sc["early"], sc["none"])),
            }
        out["position_sets"][pname] = res
    out["bootstrap"] = "95% percentile CI over 1000 bootstrap resamples of each group (seeded)."
    lg = {}
    for s in SCHEMES:
        sc = {g: np.array([weighted_score(list(p), [1, 2, 3, 4], s) for p in legacy[g]]) for g in groups}
        lg[s] = {"mean_score": {g: r(float(sc[g].mean()), 2) for g in groups},
                 "auroc_late_vs_early": r(auroc(sc["late"], sc["early"])),
                 "auroc_late_vs_none": r(auroc(sc["late"], sc["none"])),
                 "auroc_anydark_vs_none": r(auroc(np.r_[sc["late"], sc["early"]], sc["none"]))}
    out["legacy_noisy_or_comparison_submission_order"] = {
        "note": "Same journeys, stage pressure = noisy-OR of the same category maxima (the pre-change definition). Comparison only.",
        "per_scheme": lg}
    return out, scores_out


def fig_A1(A1):
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    ns = [row["n_stages"] for row in A1["rows"]]
    for ax, key, title in ((axes[0], "final_stage_share", "Weight on the final stage"),
                           (axes[1], "first_stage_share", "Weight on the first stage")):
        for s in SCHEMES:
            ax.plot(ns, [row[s][key] for row in A1["rows"]], "o-", color=COLORS[s], label=s, lw=1.8, ms=4)
        ax.set_title(title, loc="left")
        ax.set_xlabel("number of stages n (positions 1..n)")
        ax.set_ylabel("share of total weight")
        ax.set_ylim(0, 0.72)
        ax.set_xticks(ns)
    axes[0].legend(loc="upper right")
    fig.suptitle("FSPM weight concentration (pure maths)", x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout()
    fig.savefig(FIGURES / "intel_fspm_weights.png", dpi=200)
    plt.close(fig)


def fig_A2_A4(diffs, A4, scores_A4):
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    data = [np.array(diffs[s]) for s in SCHEMES]
    bp = ax.boxplot(data, tick_labels=list(SCHEMES), patch_artist=True, widths=0.55, showfliers=True,
                    flierprops={"markersize": 2, "markeredgecolor": INK2, "alpha": 0.5},
                    medianprops={"color": INK})
    for patch, s in zip(bp["boxes"], SCHEMES):
        patch.set_facecolor(COLORS[s]); patch.set_alpha(0.55); patch.set_edgecolor(COLORS[s])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title("A2  Order sensitivity (synthetic, n=500)", loc="left")
    ax.set_ylabel("score(escalating) - score(de-escalating)")
    ax = axes[1]
    groups = ["late", "early", "none"]
    x = np.arange(len(groups))
    w = 0.26
    per = A4["position_sets"]["submission_order_1234"]["per_scheme"]
    for k, s in enumerate(SCHEMES):
        m = [per[s]["mean_score"][g] for g in groups]
        se = [scores_A4[s][g].std(ddof=1) / np.sqrt(len(scores_A4[s][g])) * 1.96 for g in groups]
        ax.bar(x + (k - 1) * w, m, w * 0.92, color=COLORS[s], label=f"{s} (AUROC late vs early {per[s]['auroc_late_vs_early']:.2f})",
               yerr=se, error_kw={"elinewidth": 0.8, "ecolor": INK2, "capsize": 2})
    ax.set_xticks(x, ["late-stage dark", "early-stage dark", "benign only"])
    ax.set_ylabel("mean FSPM score (0-100) +/- 95% CI")
    ax.set_title("A4  Synthetic labelled journeys (n=100/group)", loc="left")
    ax.legend(fontsize=7.5, loc="upper right")
    fig.text(0.01, 0.005, "SYNTHETIC: journeys built from real TEST-split strings; labels come from construction, not real-world ground truth.",
             fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURES / "intel_fspm_order.png", dpi=200)
    plt.close(fig)


# ============================================================================ B. MCAI
FACTORS = ("weight", "size", "fill", "link", "grey", "opacity")
ACCEPT_TEXT = "Yes, add protection to my order"     # AUTHORED
NEUTRAL_DECLINE = "No thanks"                        # AUTHORED


def build_choice_html(factors: set, decline_text: str) -> str:
    acc = {"font-size": "20px" if "size" in factors else "16px", "font-weight": "700" if "weight" in factors else "400",
           "opacity": "1", "border": "1px solid #111111"}
    dec = {"font-size": "13px" if "size" in factors else "16px", "font-weight": "400",
           "opacity": "0.55" if "opacity" in factors else "1", "border": "1px solid #111111"}
    if "fill" in factors:
        acc.update({"background-color": "#0f766e", "color": "#ffffff"})
        dec.update({"background-color": "transparent"})
    else:
        acc.update({"background-color": "#ffffff", "color": "#111111"})
        dec.update({"background-color": "#ffffff"})
    dec["color"] = "#b0b0b0" if "grey" in factors else "#111111"
    css = lambda d: "; ".join(f"{k}: {v}" for k, v in d.items())
    dec_el = (f'<a href="#" class="dec">{decline_text}</a>' if "link" in factors
              else f'<button type="button" class="dec">{decline_text}</button>')
    return (f"<!doctype html><html><head><style>\nbody{{background-color:#ffffff}}\n.acc{{{css(acc)}}}\n.dec{{{css(dec)}}}\n</style></head>"
            f"<body><main><div class=\"offer\"><p>Add purchase protection to this order?</p>"
            f"<button type=\"button\" class=\"acc\">{ACCEPT_TEXT}</button> {dec_el}</div></main></body></html>")


def exp_B(ctx: Ctx):
    from scipy.stats import spearmanr
    from backend.intelligence.mcai import compute_mcai, is_decline
    rng = RNG["B"]
    t = ctx.test
    mis = t[(t["label"] == "misdirection")]
    mis_decl = mis[mis["text"].map(is_decline)][["row_id", "text"]].values.tolist()
    cases = []
    for level in range(0, 6):
        for subset in itertools.combinations(FACTORS, level):
            for wording in ("neutral", "misdirection_test_split"):
                if wording == "neutral":
                    d_txt, d_row = NEUTRAL_DECLINE, None
                else:
                    d_row, d_txt = mis_decl[rng.integers(len(mis_decl))]
                html = build_choice_html(set(subset), d_txt)
                res = compute_mcai("html", ctx.clf.predict, html=html)
                c = {"level": level, "factors": list(subset), "wording": wording, "decline_text": d_txt, "decline_row_id": d_row,
                     "detected": False}
                if res.available and res.choice_sets:
                    cs = res.choice_sets[0]
                    dims = {d.name: d for d in cs.dimensions}
                    c.update({"detected": bool(cs.accept is not None and cs.accept.text == ACCEPT_TEXT and cs.decline.text == d_txt),
                              "n_choice_sets": len(res.choice_sets),
                              "visual": dims["visual"].score, "semantic": dims["semantic"].score, "mcai": cs.mcai,
                              "coverage": cs.coverage,
                              "visual_features": {f.feature: f.asymmetry for f in dims["visual"].features}})
                cases.append(c)
    ok = [c for c in cases if c["detected"]]
    lv = np.array([c["level"] for c in ok])
    out = {"synthetic": True, "labels_from_construction": True,
           "kind": "synthetic labelled benchmark (designed asymmetry level = number of visual asymmetry factors applied)",
           "design": ("All subsets of size 0..5 of 6 visual asymmetry factors {weight: accept 700 vs decline 400; size: 20px vs 13px; "
                      "fill: accept filled #0f766e/white text vs decline transparent; link: decline is an <a> text link instead of a <button>; "
                      "grey: decline text #b0b0b0 (low contrast) instead of #111111; opacity: decline opacity 0.55} = 63 designs, each with "
                      "two decline wordings: the AUTHORED neutral 'No thanks' and a random TEST-split misdirection row that the decline lexicon "
                      "recognises. Accept label (AUTHORED) is constant. Declared HTML/CSS only (hex colours). Interaction dimension is unavailable "
                      "(no declared step counts), so MCAI = visual (w 0.4) + semantic (w 0.3), renormalised."),
           "n_cases": len(cases), "n_pair_correctly_detected": len(ok),
           "eligible_misdirection_decline_rows": len(mis_decl),
           "misdirection_rows_used": sorted({c["decline_row_id"] for c in cases if c["decline_row_id"] is not None}),
           "accept_text_authored": ACCEPT_TEXT}
    for key in ("visual", "mcai"):
        vals = np.array([c[key] for c in ok], float)
        rho, pval = spearmanr(lv, vals)
        out[f"spearman_level_vs_{key}"] = {"rho": r(float(rho)), "p_value": float(pval), "n": len(ok)}
        for wording in ("neutral", "misdirection_test_split"):
            sel = [c for c in ok if c["wording"] == wording]
            rho_w, p_w = spearmanr([c["level"] for c in sel], [c[key] for c in sel])
            out[f"spearman_level_vs_{key}_{wording}"] = {"rho": r(float(rho_w)), "p_value": float(p_w), "n": len(sel)}
        means = [float(np.mean([c[key] for c in ok if c["level"] == L])) for L in range(6)]
        out[f"per_level_mean_{key}"] = [r(m) for m in means]
        out[f"per_level_min_max_{key}"] = [[r(min(c[key] for c in ok if c["level"] == L)), r(max(c[key] for c in ok if c["level"] == L))] for L in range(6)]
        out[f"adjacent_level_mean_violations_{key}"] = int(sum(means[i + 1] < means[i] - 1e-12 for i in range(5)))
        # discordant pairs across levels
        disc = tot = 0
        for a, b in itertools.combinations(ok, 2):
            if a["level"] == b["level"]:
                continue
            lo, hi = (a, b) if a["level"] < b["level"] else (b, a)
            tot += 1
            disc += hi[key] < lo[key] - 1e-12
        out[f"discordant_pairs_{key}"] = {"discordant": int(disc), "pairs_with_different_level": int(tot), "share": r(disc / tot)}
        # subset monotonicity: adding one factor (same wording / same decline text) should not lower the score
        idx = {(tuple(c["factors"]), c["wording"]): c for c in ok}
        viol, edges, by_factor = 0, 0, Counter()
        examples = []
        for (fs, wording), c in idx.items():
            for f in FACTORS:
                if f in fs:
                    continue
                sup = tuple(x for x in FACTORS if x in set(fs) | {f})
                c2 = idx.get((sup, wording))
                if c2 is None:
                    continue
                # the visual dimension does not read the wording; MCAI does, so MCAI edges need identical decline text
                if key == "mcai" and c2["decline_text"] != c["decline_text"]:
                    continue
                edges += 1
                if c2[key] < c[key] - 1e-12:
                    viol += 1; by_factor[f] += 1
                    if len(examples) < 5:
                        examples.append({"from": list(fs), "added": f, "wording": wording, "before": r(c[key]), "after": r(c2[key])})
        out[f"subset_monotonicity_{key}"] = {"violations": viol, "edges_checked": edges, "violations_by_added_factor": dict(by_factor),
                                             "examples": examples,
                                             "note": ("edge = (design S, design S + one factor), same wording type; for MCAI the decline "
                                                      "text must also be identical (misdirection texts are drawn per case, so few MCAI "
                                                      "edges exist for that wording)")}
    sem = {w: [c["semantic"] for c in ok if c["wording"] == w] for w in ("neutral", "misdirection_test_split")}
    out["semantic_mean_by_wording"] = {w: r(float(np.mean(v))) for w, v in sem.items()}
    out["mcai_mean_by_wording"] = {w: r(float(np.mean([c["mcai"] for c in ok if c["wording"] == w]))) for w in sem}
    single = {}
    for f in FACTORS:
        c = idx.get(((f,), "neutral"))
        single[f] = r(c["visual"]) if c else None
    out["visual_single_factor_neutral"] = single
    out["cases"] = [{k: (r(v) if isinstance(v, float) else v) for k, v in c.items() if k != "visual_features"} for c in cases]
    return out, ok


def _font(size, bold=False):
    from PIL import ImageFont
    path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def render_choice(spec: dict):
    """AUTHORED synthetic screenshot of a heading + accept + decline option."""
    from PIL import Image, ImageDraw
    W, H = 900, 420
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((60, 40), spec["heading"], fill=(17, 17, 17), font=_font(30, True))

    def option(text, y, style):
        f = _font(style["size"], style.get("bold", False))
        tw = d.textlength(text, font=f)
        if style.get("fill") or style.get("outline"):
            pad_x, pad_y = 28, 16
            box = (60, y, 60 + tw + 2 * pad_x, y + style["size"] + 2 * pad_y)
            d.rounded_rectangle(box, radius=8, fill=style.get("fill"), outline=style.get("outline"), width=2)
            d.text((60 + pad_x, y + pad_y - 2), text, fill=style["color"], font=f)
            return box[3]
        d.text((60, y), text, fill=style["color"], font=f)
        return y + style["size"] + 6

    y = option(spec["accept"], 130, spec["accept_style"])
    option(spec["decline"], y + 40, spec["decline_style"])
    return img


def exp_B_screenshots(ctx: Ctx):
    from backend.extractors import ocr_available, ocr_lines
    from backend.intelligence.mcai import compute_mcai, is_decline
    if not ocr_available():
        return {"synthetic": True, "skipped": True, "reason": "Tesseract OCR not available on this machine."}
    t = ctx.test
    mis = t[(t["label"] == "misdirection") & t["text"].map(is_decline)][["row_id", "text"]].values.tolist()
    rng = RNG["B_shot"]
    m1, m2 = (mis[i] for i in rng.choice(len(mis), 2, replace=False))
    big = {"size": 30, "bold": True, "fill": (15, 118, 110), "color": (255, 255, 255)}
    grey_small = {"size": 15, "color": (170, 170, 170)}
    outline = {"size": 22, "bold": False, "outline": (17, 17, 17), "color": (17, 17, 17)}
    greyfill = {"size": 22, "bold": False, "fill": (229, 231, 235), "color": (17, 17, 17)}
    specs = [
        {"id": "S1", "designed": "asymmetric", "heading": "Protect your order?", "accept": "Yes, add protection", "decline": "No thanks",
         "accept_style": big, "decline_style": grey_small, "decline_row_id": None},
        {"id": "S2", "designed": "asymmetric", "heading": "Protect your order?", "accept": "Yes, add protection", "decline": m1[1],
         "accept_style": big, "decline_style": grey_small, "decline_row_id": int(m1[0])},
        {"id": "S3", "designed": "asymmetric", "heading": "Join our rewards club", "accept": "Join now and save", "decline": m2[1],
         "accept_style": big, "decline_style": {"size": 13, "color": (190, 190, 190)}, "decline_row_id": int(m2[0])},
        {"id": "S4", "designed": "symmetric", "heading": "Protect your order?", "accept": "Accept", "decline": "No thanks",
         "accept_style": outline, "decline_style": outline, "decline_row_id": None},
        {"id": "S5", "designed": "symmetric", "heading": "Cookie preferences", "accept": "Accept all", "decline": "Reject all",
         "accept_style": greyfill, "decline_style": greyfill, "decline_row_id": None},
        {"id": "S6", "designed": "symmetric", "heading": "Protect your order?", "accept": "Yes, add protection", "decline": m1[1],
         "accept_style": outline, "decline_style": outline, "decline_row_id": int(m1[0])},
        # SUPPLEMENTARY (added after S1-S3 showed OCR misses the light-grey decline): same asymmetric layout with a
        # darker / larger decline, to obtain an asymmetric measurement that OCR can read.
        {"id": "S7", "designed": "asymmetric", "supplementary": True, "heading": "Protect your order?", "accept": "Yes, add protection",
         "decline": "No thanks", "accept_style": big, "decline_style": {"size": 18, "color": (120, 120, 120)}, "decline_row_id": None},
        {"id": "S8", "designed": "asymmetric", "supplementary": True, "heading": "Protect your order?", "accept": "Yes, add protection",
         "decline": m1[1], "accept_style": big, "decline_style": {"size": 16, "color": (90, 90, 90)}, "decline_row_id": int(m1[0])},
    ]
    outdir = ARTIFACTS / "intel_synthetic_screens"
    outdir.mkdir(exist_ok=True)
    rows = []
    for sp in specs:
        img = render_choice(sp)
        img.save(outdir / f"{sp['id']}.png")
        cands = ocr_lines(img)
        lines = [c.text for c in cands if c.extra.get("unit") != "block"]
        res = compute_mcai("screenshot_ocr", ctx.clf.predict, cands=cands, image=img)
        row = {"id": sp["id"], "designed": sp["designed"], "supplementary": sp.get("supplementary", False),
               "accept_text": sp["accept"], "decline_text": sp["decline"],
               "decline_row_id": sp["decline_row_id"], "ocr_lines": lines, "mcai_available": res.available}
        if res.available:
            cs = max(res.choice_sets, key=lambda c: c.mcai if c.mcai is not None else -1)
            dims = {d.name: d for d in cs.dimensions}
            norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
            sim = lambda a, b: difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()
            row.update({
                "n_choice_sets": len(res.choice_sets),
                "detected_accept": cs.accept.text if cs.accept else None, "detected_decline": cs.decline.text,
                "accept_found": bool(cs.accept and sim(cs.accept.text, sp["accept"]) >= 0.85),
                "decline_found": sim(cs.decline.text, sp["decline"]) >= 0.85,
                "exact_ocr_text": bool(cs.accept and norm(cs.accept.text) == norm(sp["accept"]) and norm(cs.decline.text) == norm(sp["decline"])),
                "visual_available": dims["visual"].available, "visual": dims["visual"].score,
                "visual_reason": dims["visual"].reason,
                "visual_features": {f.feature: {"accept": f.accept_value, "decline": f.decline_value, "asym": f.asymmetry} for f in dims["visual"].features},
                "semantic": dims["semantic"].score, "mcai": cs.mcai, "coverage": cs.coverage})
        else:
            row.update({"accept_found": False, "decline_found": False, "reason": res.reason})
        row["both_options_found"] = bool(row.get("accept_found") and row.get("decline_found"))
        rows.append(row)
    main6 = [x for x in rows if not x["supplementary"]]
    asym = [x["visual"] for x in main6 if x["designed"] == "asymmetric" and x.get("visual") is not None]
    sym = [x["visual"] for x in main6 if x["designed"] == "symmetric" and x.get("visual") is not None]
    sup = [x["visual"] for x in rows if x["supplementary"] and x.get("visual") is not None]
    return {"synthetic": True, "labels_from_construction": True,
            "design": ("6 AUTHORED PIL-rendered screenshots (900x420, Arial): 3 asymmetric (large filled bold accept button vs small grey "
                       "unboxed decline text) and 3 symmetric (identical styling). Decline wording: AUTHORED 'No thanks'/'Reject all' or "
                       "TEST-split misdirection rows (ids listed). Run through backend.extractors.ocr_lines + compute_mcai('screenshot_ocr'). "
                       "Images saved in ml/artifacts/intel_synthetic_screens/."),
            "cases": rows,
            "main_6": {"n_both_options_found": sum(x["both_options_found"] for x in main6),
                       "n_asymmetric_with_visual": len(asym), "n_symmetric_with_visual": len(sym),
                       "mean_visual_asymmetric": r(float(np.mean(asym))) if asym else None,
                       "mean_visual_symmetric": r(float(np.mean(sym))) if sym else None},
            "supplementary_S7_S8": {"n_both_options_found": sum(x["both_options_found"] for x in rows if x["supplementary"]),
                                    "visual": [r(v) for v in sup]},
            "match_rule": "an option counts as found if the OCR text matches the rendered text with difflib ratio >= 0.85 "
                          "after lower-casing and removing non-alphanumerics; exact_ocr_text reports exact matches",
            "note": ("text_height_px / text_box_area_px are OCR word-box measurements: glyph shapes (descenders such as 'p' in "
                     "'Accept') and string length change them even for identically styled options, so symmetric pairs need "
                     "not score exactly 0 on screenshots.")}


def fig_B(ok, B):
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), sharex=True)
    jit = RNG["spare1"]
    for ax, key, title in ((axes[0], "visual", "Visual dimension D_visual (0-1)"), (axes[1], "mcai", "MCAI (0-100)")):
        for wording, col, mk in (("neutral", COLORS["fibonacci"], "o"), ("misdirection_test_split", COLORS["linear"], "s")):
            sel = [c for c in ok if c["wording"] == wording]
            xs = np.array([c["level"] for c in sel]) + (-0.12 if wording == "neutral" else 0.12) + jit.uniform(-0.06, 0.06, len(sel))
            ax.scatter(xs, [c[key] for c in sel], s=12, color=col, alpha=0.65, marker=mk,
                       label="decline 'No thanks' (authored)" if wording == "neutral" else "decline = TEST-split misdirection row")
        means = B[f"per_level_mean_{key}"]
        ax.plot(range(6), means, "-", color=INK, lw=1.4, label="mean per level", zorder=3)
        rho = B[f"spearman_level_vs_{key}"]["rho"]
        ax.set_title(f"{title}   Spearman rho = {rho:.2f}", loc="left")
        ax.set_xlabel("designed asymmetry level (number of visual factors)")
        ax.set_xticks(range(6))
    axes[0].legend(fontsize=7, loc="upper left")
    fig.text(0.01, 0.005, f"SYNTHETIC labelled HTML choice sets (n={len(ok)}); levels come from construction.", fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURES / "intel_mcai_levels.png", dpi=200)
    plt.close(fig)


# ============================================================================ C. CDI on demo inputs
def _cdi_summary(cdi: dict) -> dict:
    s = cdi.get("sensitivity") or {}
    return {"cdi": cdi["cdi"], "band": cdi["band"], "coverage": cdi["coverage"],
            "components": {c["key"]: {"available": c["available"], "value": c["value"], "effective_weight": c["effective_weight"],
                                      "contribution": c["contribution"], "provenance": c["provenance"]} for c in cdi["components"]},
            "one_at_a_time": s.get("one_at_a_time"), "dirichlet_p05": s.get("dirichlet_p05"), "dirichlet_p95": s.get("dirichlet_p95"),
            "dirichlet_samples": s.get("samples")}


def exp_C(ctx: Ctx):
    try:
        from backend.intelligence.engine import IntelligenceEngine
    except Exception as e:  # counterfactual/temporal may not exist yet
        return {"skipped": True, "reason": f"backend.intelligence.engine could not be imported: {type(e).__name__}: {e}"}
    from fastapi.testclient import TestClient
    import backend.main as main
    main.STATE.update({"clf": ctx.clf, "auditor": ctx.auditor, "intel": IntelligenceEngine(ctx.clf),
                       "metrics": json.loads((ARTIFACTS / "metrics.json").read_text()), "ocr": False})
    client = TestClient(main.app, raise_server_exceptions=True)     # not used as a context manager: lifespan (2nd model load) skipped
    out = {"kind": "description of the real demo inputs through the live API code path (POST endpoints via TestClient)",
           "synthetic_note": "samples.json and journeys.json are demo inputs; journeys are flagged synthetic (authored order/markup + verbatim TEST-split strings).",
           "samples": [], "journeys": []}
    samples = json.loads((ROOT / "data" / "samples" / "samples.json").read_text())
    for s in samples:
        if s["input_type"] == "text":
            resp = client.post("/api/audit/text", json={"text": s["content"]})
        elif s["input_type"] == "html":
            resp = client.post("/api/audit/html", data={"html": s["content"]})
        else:
            continue
        body = resp.json()
        row = {"id": s["id"], "input_type": s["input_type"], "http_status": resp.status_code,
               "provenance": s.get("provenance"), "n_findings": len(body.get("findings", []))}
        intel = body.get("intelligence")
        if intel is None:
            row["intelligence"] = None
            row["warnings"] = body.get("warnings")
        else:
            row.update(_cdi_summary(intel["cdi"]))
            row["mcai"] = intel["mcai"]["mcai"]
            row["counterfactual"] = {"original_cdi": intel["counterfactual"]["original_cdi"],
                                     "redesigned_cdi": intel["counterfactual"]["redesigned_cdi"],
                                     "risk_reduction_pct": intel["counterfactual"]["risk_reduction_pct"]}
        out["samples"].append(row)
    journeys = json.loads((ROOT / "data" / "samples" / "journeys.json").read_text())
    for j in journeys:
        resp = client.post("/api/audit/journey", data={"stages": json.dumps(j["stages"])})
        body = resp.json()
        row = {"id": j["id"], "synthetic": j.get("synthetic"), "http_status": resp.status_code}
        if resp.status_code == 200:
            row.update(_cdi_summary(body["cdi"]))
            row["fspm_scores"] = body["fspm"]["scores"]
            row["fspm_positions_mode"] = body["fspm"]["positions_mode"]
            row["stage_pressures"] = [st["pressure"] for st in body["stages"]]
            row["mcai"] = body["mcai"]["mcai"]
            row["counterfactual"] = {"original_cdi": body["counterfactual"]["original_cdi"],
                                     "redesigned_cdi": body["counterfactual"]["redesigned_cdi"],
                                     "risk_reduction_pct": body["counterfactual"]["risk_reduction_pct"]}
            row["temporal_events"] = len(body["temporal"]["events"])
        else:
            row["detail"] = body
        out["journeys"].append(row)
    return out


# ============================================================================ D. counterfactual rewrites
_OFFER = re.compile(r"\b(guarantee\w*|free|refund\w*|warrant\w*|discount\w*|money[-\s]?back|off)\b", re.I)


def independent_fact_check(orig: str, red: str) -> bool:
    """Independent of counterfactual.facts_preserved: every digit run and offer word in the rewrite occurs in the original."""
    o_nums, r_nums = Counter(re.findall(r"\d+", orig.replace(",", ""))), Counter(re.findall(r"\d+", red.replace(",", "")))
    if any(o_nums[k] < v for k, v in r_nums.items()):
        return False
    o_off = {m.lower() for m in _OFFER.findall(orig)}
    return all(m.lower() in o_off for m in _OFFER.findall(red))


def run_rewrites(ctx: Ctx, rows: pd.DataFrame):
    from backend.extractors import segment_text
    from backend.intelligence.counterfactual import plan_edits
    edits_all, n_rows_with_findings = [], 0
    for row in rows.itertuples():
        audit = ctx.auditor.audit(segment_text(row.text), "text")
        ml_f = [f for f in audit.findings if f.detection_method == "ml_classifier"]
        if not ml_f:
            continue
        n_rows_with_findings += 1
        for e in plan_edits(audit.findings, ctx.clf.predict):
            edits_all.append({"row_id": int(row.row_id), "gold_label": row.label, "category": e.category, "action": e.action,
                              "original": e.original, "redesigned": e.redesigned, "before": e.original_score,
                              "after": e.redesigned_score, "facts_preserved": e.facts_preserved,
                              "independent_fact_check": independent_fact_check(e.original, e.redesigned) if e.redesigned else None,
                              "explanation": e.explanation})
    return edits_all, n_rows_with_findings


def summarise_edits(edits, tau):
    per = {}
    cats = sorted({e["category"] for e in edits})
    for c in cats + ["ALL"]:
        es = [e for e in edits if c == "ALL" or e["category"] == c]
        rw = [e for e in es if e["action"] == "rewrite"]
        below = [e for e in rw if e["after"] is not None and e["after"] < tau]
        per[c] = {"n_findings": len(es), "n_rewrite": len(rw), "pct_safe_rewrite": r(100 * len(rw) / len(es), 1) if es else None,
                  "actions": dict(Counter(e["action"] for e in es)),
                  "n_rewrite_below_threshold": len(below),
                  "pct_rewrites_below_threshold": r(100 * len(below) / len(rw), 1) if rw else None,
                  "mean_p_dark_before_rewritten": r(float(np.mean([e["before"] for e in rw]))) if rw else None,
                  "mean_p_dark_after_rewritten": r(float(np.mean([e["after"] for e in rw]))) if rw else None,
                  "facts_preserved_violations": sum(1 for e in rw if not e["facts_preserved"]),
                  "independent_fact_check_failures": sum(1 for e in rw if e["independent_fact_check"] is False)}
    return per


def exp_D(ctx: Ctx):
    try:
        import backend.intelligence.counterfactual  # noqa: F401
    except Exception as e:
        return {"skipped": True, "reason": f"backend.intelligence.counterfactual not importable: {type(e).__name__}: {e}"}, None
    t = ctx.test
    dark = t[t["label"] != "benign"]
    benign = t[t["label"] == "benign"]
    t0 = time.perf_counter()
    edits, n_rows_f = run_rewrites(ctx, dark)
    b_edits, b_rows_f = run_rewrites(ctx, benign)
    rng = RNG["D"]
    rw = [e for e in edits if e["action"] == "rewrite"]
    ex_idx = rng.choice(len(rw), min(10, len(rw)), replace=False) if rw else []
    examples = [{k: rw[i][k] for k in ("row_id", "gold_label", "category", "original", "redesigned", "before", "after", "facts_preserved")}
                for i in sorted(ex_idx)]
    nsr = [e for e in edits if e["action"] == "no_safe_rewrite"]
    reasons = Counter(e["explanation"].split(":")[0].split(".")[0] for e in nsr)
    from backend.intelligence.counterfactual import NEUTRAL_DECLINE as CF_NEUTRAL
    nd = ctx.clf.predict([CF_NEUTRAL])[0]
    full = pd.read_csv(DATA_PROCESSED / "text_dataset.csv", keep_default_na=False)
    nt = full[full["text"].str.lower().str.match(r"^\s*no,? thank")]
    out = {"synthetic": False, "kind": "model-estimated rewrite experiment on real TEST-split rows",
           "threshold_tau": ctx.tau,
           "neutral_decline_text": CF_NEUTRAL,
           "neutral_decline_score": {"p_dark": r(nd["p_dark"]), "p_misdirection": r(nd["probabilities"]["misdirection"]),
                                     "flagged_at_tau": bool(nd["p_dark"] >= ctx.tau)},
           "dataset_rows_starting_no_thanks_by_split_label": {f"{a}/{b}": int(n) for (a, b), n in nt.groupby(["split", "label"]).size().items()},
           "design": ("Every TEST-split row with label != benign is audited alone (segment_text -> Auditor.audit(...,'text')); "
                      "plan_edits(findings, clf.predict) proposes one edit per finding; redesigned texts are re-scored by the same "
                      "calibrated classifier. Categories below are the FINDING (predicted) categories that select the rewrite rule."),
           "dark_rows": int(len(dark)), "dark_rows_with_ml_findings": n_rows_f, "dark_rows_without_findings": int(len(dark) - n_rows_f),
           "per_category": summarise_edits(edits, ctx.tau),
           "per_gold_label_rewrite_rate": {g: r(100 * sum(e["action"] == "rewrite" for e in edits if e["gold_label"] == g) /
                                              max(1, sum(1 for e in edits if e["gold_label"] == g)), 1)
                                           for g in sorted({e["gold_label"] for e in edits})},
           "no_safe_rewrite_reason_counts": dict(reasons),
           "examples": examples,
           "false_positive_benign_rows": {"benign_rows": int(len(benign)), "benign_rows_flagged": b_rows_f,
                                          "per_category": summarise_edits(b_edits, ctx.tau) if b_edits else {},
                                          "edits": [{k: e[k] for k in ("row_id", "category", "action", "original", "redesigned", "before", "after", "facts_preserved")}
                                                    for e in b_edits]},
           "seconds": r(time.perf_counter() - t0, 1),
           "circularity_caveat": ("The same classifier that flagged the original scores the rewrite, so a drop in P(dark) shows that the "
                                  "rewrite removes the cues the model relies on - not that consumers would be less pressured, and not "
                                  "independent validation of the redesign.")}
    return out, edits


def fig_D(D, edits, tau):
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    cat_col = {"urgency": "#2a78d6", "scarcity": "#eb6834", "social_proof": "#1baf7a", "misdirection": "#eda100"}
    rw = [e for e in edits if e["action"] == "rewrite"]
    for c, col in cat_col.items():
        sel = [e for e in rw if e["category"] == c]
        if sel:
            ax.scatter([e["before"] for e in sel], [e["after"] for e in sel], s=14, color=col, alpha=0.7, label=f"{c} (n={len(sel)})")
    ax.plot([0, 1], [0, 1], "--", color=GRID, lw=1)
    ax.axhline(tau, color="#dc2626", lw=0.9, ls=":", label=f"threshold {tau:.2f}")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xlabel("calibrated P(dark) - original"); ax.set_ylabel("calibrated P(dark) - rewrite")
    ax.set_title("Rewritten findings (model-estimated)", loc="left")
    ax.legend(fontsize=7, loc="upper left")
    ax = axes[1]
    per = D["per_category"]
    cats = [c for c in ("urgency", "scarcity", "social_proof", "misdirection") if c in per]
    x = np.arange(len(cats))
    a = [per[c]["pct_safe_rewrite"] or 0 for c in cats]
    b = [per[c]["pct_rewrites_below_threshold"] or 0 for c in cats]
    ax.bar(x - 0.2, a, 0.38, color=COLORS["fibonacci"], label="% findings with a safe rewrite")
    ax.bar(x + 0.2, b, 0.38, color=COLORS["equal"], label="% of rewrites scored below threshold")
    for i, c in enumerate(cats):
        ax.text(x[i] - 0.2, a[i] + 1.5, f"n={per[c]['n_findings']}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, cats); ax.set_ylim(0, 112); ax.set_ylabel("%")
    ax.set_title("Coverage and effect by finding category", loc="left")
    ax.legend(fontsize=7, loc="upper right")
    fig.text(0.01, 0.005, "Real TEST-split rows. Circular: the same classifier scores the rewrite - not a measure of consumer impact.",
             fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURES / "intel_counterfactual.png", dpi=200)
    plt.close(fig)


# ============================================================================ main
CAVEATS = [
    "No ground truth exists for journey pressure, choice asymmetry or composite deception; nothing here is real-world validation.",
    "A1 and A3 are pure mathematics of the weighting schemes; A3 confirms the score is linear in each stage pressure (marginal = 100*w_i).",
    "A2 and A4 journeys are SYNTHETIC: random combinations of real held-out TEST-split strings. A4 labels come from the construction.",
    "B choice sets and screenshots are SYNTHETIC and AUTHORED (markup, CSS, accept labels, 'No thanks', headings); levels come from the construction. "
    "Decline wordings marked misdirection_test_split are verbatim TEST-split rows.",
    "B uses hex colours on purpose; see the bug note about rgb()/rgba() backgrounds written with spaces (mcai.html_visual_features).",
    "C describes the shipped demo inputs; demo journeys are themselves synthetic and flagged as such.",
    "D is circular: the same classifier scores originals and rewrites, so it measures removal of model cues, not consumer impact.",
    "Stage pressures use the text threshold (clf.b.threshold); below-threshold elements contribute 0 by design (pressure.py).",
    "All Fibonacci/linear/equal weights and CDI/MCAI weights are design choices (backend/intelligence/config.py), not learned.",
    "The classifier scores the neutral decline 'No thanks' as misdirection (see D.neutral_decline_score): every dataset row starting "
    "'No thanks' is labelled misdirection. This inflates the MCAI semantic 'guilt' sub-score for neutral declines (B) and keeps "
    "counterfactual misdirection rewrites above threshold (D).",
    "The classifier also flags neutral stock/deadline statements (e.g. '1 in stock.'), i.e. it scores the content, not only the "
    "pressure wording; scarcity rewrites therefore rarely fall below threshold (D).",
]


def main():
    t_start = time.perf_counter()
    FIGURES.mkdir(parents=True, exist_ok=True)
    ctx = Ctx()
    from backend.intelligence.config import CONFIG_VERSION
    from backend.intelligence.config import PRESSURE_INTENSITY_WEIGHT
    result = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "config_version": CONFIG_VERSION,
              "stage_pressure_definition": ("p = lam*max_k c_k + (1-lam)*sum_k c_k/4 over urgency/scarcity/social_proof/misdirection, "
                                            f"c_k = max calibrated P(dark) of flagged findings of category k; lam = {PRESSURE_INTENSITY_WEIGHT} "
                                            "(backend/intelligence/pressure.py)"),
              "seed": SEED, "python": platform.python_version(), "classifier_threshold": ctx.tau,
              "test_split_rows": int(len(ctx.test)), "test_split_label_counts": ctx.test["label"].value_counts().to_dict(),
              "caveats": CAVEATS, "experiments": {}}
    E = result["experiments"]

    print("A1 ..."); A1 = exp_A1(); E["A1_weight_concentration"] = A1; fig_A1(A1)
    print("A2/A3 ..."); A2, A3, _P, diffs = exp_A2_A3(ctx); E["A2_order_sensitivity"] = A2; E["A3_single_stage_perturbation"] = A3
    print("A4 ..."); A4, sc4 = exp_A4(ctx); E["A4_synthetic_discrimination"] = A4
    fig_A2_A4(diffs, A4, sc4)
    print("B ..."); B, okB = exp_B(ctx); E["B_mcai_synthetic_html"] = B; fig_B(okB, B)
    print("B screenshots ...")
    try:
        E["B_mcai_synthetic_screenshots"] = exp_B_screenshots(ctx)
    except Exception as e:
        E["B_mcai_synthetic_screenshots"] = {"synthetic": True, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    print("C ...")
    try:
        E["C_cdi_demo_inputs"] = exp_C(ctx)
    except Exception as e:
        E["C_cdi_demo_inputs"] = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    print("D ...")
    D, edits = exp_D(ctx)
    E["D_counterfactual_rewrites"] = D
    if edits is not None:
        fig_D(D, edits, ctx.tau)
    result["runtime_seconds"] = round(time.perf_counter() - t_start, 1)
    result["model_load_seconds"] = round(ctx.load_s, 1)
    OUT_JSON.write_text(json.dumps(jsonable(result), indent=2, ensure_ascii=False))
    print(f"wrote {OUT_JSON} in {result['runtime_seconds']} s")


if __name__ == "__main__":
    main()
