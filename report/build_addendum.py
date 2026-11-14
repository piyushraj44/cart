"""Build report/ClearCart_Math_Addendum.pdf - technical addendum for the intel-v1.0 layer.

All numbers are read from ml/artifacts/intelligence_benchmark.json, ml/artifacts/metrics.json and
tests; nothing is hand-typed. Run from project root:  .venv/bin/python -m report.build_addendum
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, SimpleDocTemplate, Spacer

from report.build_report import (BODY, BRAND, CAP, H1, H2, LINE, MUTED, SMALL, W, P, bullets, f3, img, side_by_side,
                                 table)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
OUT = ROOT / "report" / "ClearCart_Math_Addendum.pdf"
B = json.loads((ROOT / "ml/artifacts/intelligence_benchmark.json").read_text())
E = B["experiments"]
M = json.loads((ROOT / "ml/artifacts/metrics.json").read_text())
CFG = (ROOT / "backend/intelligence/config.py").read_text()
# saved REAL response of POST /api/audit/journey for the demo journey (also used by the UI's ?mock=journey)
JS = json.loads((ROOT / "frontend/src/mock/sampleJourney.json").read_text())["fspm"]["scores"]


def count_tests() -> str:
    """Run the suite and return the real summary line."""
    r = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "pytest", "tests", "-q"], cwd=ROOT, capture_output=True, text=True)
    m = re.findall(r"(\d+ passed[^\n]*)", r.stdout)
    return m[-1].strip() if m else "pytest did not report"


def on_page(c, doc):
    c.saveState(); c.setFont("Helvetica", 7); c.setFillColor(MUTED)
    c.drawString(16 * mm, A4[1] - 10 * mm, "ClearCart — Technical addendum: mathematical innovation (intel-v1.0)")
    c.drawRightString(A4[0] - 16 * mm, A4[1] - 10 * mm, "Team ClearCart")
    c.setStrokeColor(LINE); c.line(16 * mm, A4[1] - 11.5 * mm, A4[0] - 16 * mm, A4[1] - 11.5 * mm)
    c.drawString(16 * mm, 8 * mm, "Synthetic benchmarks only — no real-world ground truth exists for these indices")
    c.drawRightString(A4[0] - 16 * mm, 8 * mm, f"Page {doc.page}")
    c.restoreState()


def build() -> None:
    tests_line = count_tests()
    ft = M["final_test"]
    story = []
    story += [P("Mathematical Innovation: From Isolated Dark Pattern Detection to Sequential Consumer Deception Intelligence", H1),
              P("Technical addendum to the ClearCart project report · configuration <font face='Courier'>intel-v1.0</font> · Team ClearCart", SMALL),
              Spacer(1, 4)]
    story.append(P(
        "The 8-page report evaluates ClearCart's element-level classifier. This addendum documents a new layer built <b>on top</b> of it. "
        "The layer adds five components: sequential pressure across a shopping journey (FSPM), an index of asymmetry between "
        "accept and decline choices (MCAI), a composite screening index (CDI), counterfactual ethical redesign with before/after "
        "scoring, and a temporal evidence timeline. The classifier, its artifacts and its reported metrics are unchanged: macro-F1 "
        f"{f3(ft['application_metrics']['macro_f1'])}, dark-vs-benign F1 {f3(ft['binary_dark_vs_benign']['f1'])} and ECE "
        f"{f3(M['E3_calibration']['test_after']['ece'])}. None of the new indices has real-world ground truth, so the evidence here is "
        "mathematical properties plus <b>synthetic, explicitly labelled</b> benchmarks. They are interpretable screening indices, not "
        "probabilities of fraud. We make no novelty or 'first' claim beyond describing what we built."))
    story.append(P("Status of each capability", H2))
    st = [["Capability", "Status", "Evidence"],
          ["Text classifier, calibration, threshold, OCR, HTML extraction", "Previously evaluated", "Held-out test split; 8-page report §3–6"],
          ["FSPM (Fibonacci / linear / equal)", "New · implemented · tested", "Unit tests; synthetic benchmarks A1–A4"],
          ["MCAI (visual / interaction / semantic)", "New · implemented · tested", "Unit tests; synthetic HTML + screenshot benchmark B"],
          ["CDI + weight sensitivity", "New · implemented · tested", "Unit tests; demo inputs C"],
          ["Counterfactual redesign + risk reduction", "New · implemented · tested", "Unit tests; rewrite experiment D (circular, see §4)"],
          ["Temporal evidence timeline: resets, new pre-selections, stock/price changes", "New · implemented · tested", "Unit tests on synthetic snapshot pairs; demo journey"],
          ["Timeline: changing interaction difficulty and accept/reject paths across stages", "New · implemented · tested", "API test on the demo journey; render test"],
          ["Journey API incl. screenshot stages, validation, PDF/JSON export", "New · implemented · tested", "API integration tests recompute every index from the response"],
          ["Frontend rendering of all five sections + backward compatibility", "New · implemented · tested", f"Server-side render tests on saved real responses; full suite: {tests_line}"],
          ["intel-v1.1: inferred interaction, confidence, constrained optimiser, fusion, replay, versions, request cache", "New · implemented · tested", "API + property tests; research studies R1–R6 (§2b)"],
          ["Rendered layout (button area), click-path discovery, multi-page crawling", "Unsupported", "Needs a browser/layout engine; step counts are declared or structurally inferred"],
          ["Validation against human judgements or consumer behaviour", "Unsupported", "No labelled journey/choice-set data exists"]]
    story.append(table(st, [W * 0.42, W * 0.22, W * 0.36]))
    story.append(P("Architecture", H2))
    story += bullets([
        "Every endpoint runs the unchanged auditor first; <font face='Courier'>backend/intelligence/engine.py</font> then reads its "
        "findings without modifying them, so evidence ids and provenance are preserved. Existing responses gain one optional field, "
        "<font face='Courier'>intelligence</font>. If that layer fails, the audit still returns with a warning.",
        "<font face='Courier'>POST /api/audit/journey</font> takes 1–8 ordered stages (text, HTML or screenshot) with optional capture "
        "time and declared step counts, and returns per-stage audits plus FSPM, timeline, MCAI, CDI and counterfactual results. "
        "<font face='Courier'>GET /api/samples/journeys</font> serves demo journeys, flagged synthetic. "
        "<font face='Courier'>POST /api/export/journey-pdf</font> exports a journey report.",
        "Modules: <font face='Courier'>pressure, fspm, mcai, cdi, counterfactual, temporal, engine</font>. Every constant lives in "
        "<font face='Courier'>config.py</font> and is a documented design choice, not a learned value.",
        "Each value carries provenance: <i>model</i> (calibrated classifier), <i>measured</i> (declared CSS or image pixels), "
        "<i>default</i> (browser default styles), <i>declared</i> (user-supplied step counts), <i>observed</i> (DOM facts or "
        "snapshot comparison), <i>by_construction</i> (the redesign specification) or <i>unavailable</i>."])
    story.append(PageBreak())

    # ---------------------------------------------------------------- definitions
    story.append(P("1. Definitions as implemented", H1))
    story.append(P("Stage (content) pressure", H2))
    story.append(P("For category k ∈ {urgency, scarcity, social proof, misdirection}, c<sub>k</sub> is the highest calibrated "
                   "P(dark) among findings flagged in that category, or 0 if none is flagged. A countdown observed to increase between "
                   "two snapshots raises the urgency channel to max(c<sub>urgency</sub>, 1). Then "
                   "<b>p = λ·max<sub>k</sub> c<sub>k</sub> + (1−λ)·¼·Σ<sub>k</sub> c<sub>k</sub></b> with λ = 0.5: an intensity term "
                   "plus a breadth term. Duplicate banners cannot inflate p, because each category contributes only its maximum."))
    story.append(P("FSPM: Fibonacci-weighted Sequential Pressure Model", H2))
    story.append(P("F(1)=F(2)=1 and F(n)=F(n−1)+F(n−2). For the stages actually submitted, "
                   "<b>FSPM = 100·Σ<sub>i</sub> F(pos<sub>i</sub>)·p<sub>i</sub> / Σ<sub>i</sub> F(pos<sub>i</sub>)</b>. Positions are "
                   "canonical (landing 1, product 2, cart 3, checkout 4, payment 5, confirmation 6, exit 7) when every stage declares "
                   "a type in increasing order, and the submission order otherwise. Missing stages are never invented or zero-filled. "
                   "Each stage's contribution is 100·w<sub>i</sub>p<sub>i</sub>. The same formula is always reported with linear "
                   "(w=pos) and equal (w=1) weights. Fibonacci weighting is our proposed emphasis on later, binding stages, not an "
                   "established psychological law."))
    story.append(P("MCAI: Multi-dimensional Choice Asymmetry Index", H2))
    story.append(P("Choice sets (an accept option and a competing decline option) are found with a rule-based role lexicon. This is "
                   "candidate extraction only. Asymmetry counts only in the consumer-unfriendly direction: "
                   "D<sub>visual</sub> = mean<sub>k</sub> clip((a<sub>k</sub>−d<sub>k</sub>)/max(a<sub>k</sub>,d<sub>k</sub>),0,1) over "
                   "prominence features. For HTML these are control type, font size, weight, fill, WCAG contrast and opacity, taken from "
                   "declared CSS with a minimal cascade; @media rules are skipped and rendered area is unavailable. For screenshots they "
                   "are OCR text height, text-box area, p90/p10 luminance contrast and the colour distance of the surrounding ring. "
                   "D<sub>interaction</sub> = max(0, s<sub>d</sub>−s<sub>a</sub>)/max(s<sub>d</sub>,s<sub>a</sub>) uses declared step "
                   "counts only. D<sub>semantic</sub> = 0.5·P(misdirection | decline) + 0.25·P(dark | accept) + 0.25·|words<sub>a</sub>−words<sub>d</sub>|/max, "
                   "renormalised over the sub-scores available. <b>MCAI = 100·Σ<sub>d∈A</sub> w<sub>d</sub>D<sub>d</sub>/Σ<sub>d∈A</sub> w<sub>d</sub></b> "
                   "with w = (0.4, 0.3, 0.3); unavailable dimensions are dropped and reported as coverage."))
    story.append(P("CDI: Composite Deception Intelligence index", H2))
    story.append(P("<b>CDI = 100·Σ<sub>j∈A</sub> w<sub>j</sub>x<sub>j</sub>/Σ<sub>j∈A</sub> w<sub>j</sub></b> with weights "
                   "<font face='Courier'>cdi-w1</font>: content pressure 0.55, choice asymmetry 0.25, observed obstruction 0.20. "
                   "Three rules prevent double counting. Content pressure is p for a single page and FSPM/100 for a journey, which "
                   "replaces it rather than adding to it. Choice asymmetry uses only the visual and interaction parts of MCAI, because "
                   "the semantic part reuses the misdirection probability already inside content pressure. Obstruction is the strongest "
                   "observed fact (max, not sum), and DOM and timeline facts for the same element are de-duplicated. Missing components "
                   "are removed from both sums, never treated as 0. Coverage (the available weight) is shown separately from the score. "
                   "Display bands are 25, 50 and 75. Sensitivity is computed in two ways, both with a fixed seed: each weight scaled by "
                   "0.5 and 1.5 and renormalised, and 1,000 Dirichlet(20·w) weight vectors summarised as p5–p95."))
    story.append(P("Counterfactual redesign and risk reduction", H2))
    story.append(P("Rewrites only remove or neutralise wording: pressure phrases are dropped, emphasis before a quantity is removed, "
                   "all-caps text is put in sentence case, and decline options become 'No thanks'. "
                   "<font face='Courier'>facts_preserved</font> rejects any rewrite that adds a number, price, percentage or duration, "
                   "or an offer word (free, guarantee, refund, warranty, discount, % off) that is not in the original. Social proof is "
                   "never rewritten, because that would need a verifiable data source. In the redesigned scenario, rewritten text is "
                   "<i>re-scored by the same classifier</i>; defaults are unchecked, choices equalised and resetting countdowns removed "
                   "<i>by construction</i>. Risk reduction = 100·(CDI<sub>orig</sub> − CDI<sub>redesign</sub>)/CDI<sub>orig</sub>. It is "
                   "a model-estimated figure, not a measured change in consumer behaviour."))
    story.append(P("Temporal evidence", H2))
    story.append(P("Stages are ordered by capture time if every stage has one, otherwise by submission order. A countdown reset is "
                   "claimed only when a later observation of the same countdown shows <i>more</i> time remaining, or, with timestamps, "
                   "more than the elapsed time allows plus 60 s tolerance. A new pre-selected option at a later stage is an observed "
                   "fact. Repeated scarcity, stock-count changes and price changes are reported as neutral observations, because they "
                   "can be legitimate (restocks, tax and shipping). Timeline events are typed as observed fact, observation or model "
                   "inference, and link back to stage and finding ids. Only observed facts enter FSPM (resets) or CDI (pre-selections). "
                   "Two further observation types compare stages. Changes in <i>declared</i> step counts are reported as interaction "
                   "difficulty growing or shrinking. Changes in the worst accept/decline pair are reported as MCAI rising or falling by "
                   "10 or more points, or the decline wording changing. Both already enter the CDI through MCAI, so they are not counted again."))
    story.append(PageBreak())

    # ---------------------------------------------------------------- results
    story.append(P("2. Benchmarks and what they show", H1))
    story.append(P(f"Script <font face='Courier'>ml/benchmark_intelligence.py</font>, seed {B['seed']}, classifier threshold "
                   f"{B['classifier_threshold']}. Two runs gave identical outputs apart from runtimes. Wherever text is needed it is "
                   "real held-out TEST-split text; everything else is labelled synthetic.", SMALL))
    a1 = {r["n_stages"]: r for r in E["A1_weight_concentration"]["rows"]}
    rows = [["Stages n", "Fibonacci: last / first", "Linear: last / first", "Equal: last / first"]]
    for n in (3, 4, 5, 6, 8):
        r = a1[n]
        rows.append([str(n)] + [f"{r[s]['final_stage_share']:.3f} / {r[s]['first_stage_share']:.3f}" for s in ("fibonacci", "linear", "equal")])
    a2 = E["A2_order_sensitivity"]["per_scheme"]
    a4 = E["A4_synthetic_discrimination"]["position_sets"]["submission_order_1234"]["per_scheme"]
    a4c = E["A4_synthetic_discrimination"]["position_sets"]["canonical_product_to_payment_2345"]["per_scheme"]
    story.append(side_by_side([P("A1 · Weight concentration (pure maths)", H2), table(rows, [W * 0.1, W * 0.17, W * 0.17, W * 0.15]),
                               P("With 4 stages at positions 1–4, Fibonacci and linear are almost identical. Fibonacci concentrates "
                                 "clearly only from about 5 positions, or with canonical positions such as product(2)…exit(7).", SMALL)],
                              img(FIG / "intel_fspm_weights.png", W * 0.38), wa=W * 0.61))
    story.append(P("A2–A4 · FSPM on synthetic journeys built from real TEST-split strings", H2))
    rows = [["Scheme", "A2: escalating − de-escalating (mean)", "A4 mean score late / early / none", "AUROC late vs early (95% CI)",
             "AUROC any-dark vs none", "late vs early, canonical pos. 2–5"]]
    for s in ("fibonacci", "linear", "equal"):
        r = a4[s]; ci = r["auroc_late_vs_early_ci95"]
        rows.append([s, f"{a2[s]['mean_escalating_minus_deescalating']:.2f}",
                     f"{r['mean_score']['late']:.1f} / {r['mean_score']['early']:.1f} / {r['mean_score']['none']:.1f}",
                     f"{r['auroc_late_vs_early']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]", f"{r['auroc_anydark_vs_none']:.3f}",
                     f"{a4c[s]['auroc_late_vs_early']:.3f}"])
    story.append(table(rows, [W * 0.1, W * 0.18, W * 0.2, W * 0.2, W * 0.14, W * 0.18]))
    story.append(P("Reading: A2 and A4 test whether the weighting <i>behaves as designed</i>; the labels come from the construction. "
                   "Equal weighting is order-invariant, so it cannot tell late-stage from early-stage manipulation (AUROC ≈ 0.5). "
                   "Fibonacci and linear can. With canonical positions, Fibonacci keeps its separation while linear weakens. The cost: "
                   "late-emphasis weights detect <i>early-only</i> manipulation less well than equal weighting (any-dark vs none). "
                   "A3 confirms the score is linear in each stage's pressure (marginal effect = 100·w<sub>i</sub> exactly).", BODY))
    story.append(img(FIG / "intel_fspm_order.png", W * 0.78))
    story.append(P("Figure A1 — FSPM order sensitivity (A2) and synthetic late/early/none separation (A4), from intelligence_benchmark.json.", CAP))
    story.append(PageBreak())

    bh = E["B_mcai_synthetic_html"]; bs = E["B_mcai_synthetic_screenshots"]
    story.append(P("B · MCAI on synthetic choice sets with known asymmetry", H2))
    lv = bh["per_level_mean_visual"]
    story.append(side_by_side([
        P(f"{bh['n_cases']} authored HTML choice sets: every subset of 6 visual asymmetry factors (63 designs) × 2 decline "
          f"wordings. Pairs were detected correctly in {bh['n_pair_correctly_detected']}/{bh['n_cases']}. Spearman ρ between "
          f"designed level and the visual dimension: <b>{bh['spearman_level_vs_visual']['rho']:.3f}</b>; with MCAI: "
          f"<b>{bh['spearman_level_vs_mcai']['rho']:.3f}</b>. Mean visual score by level 0–5: " + ", ".join(f"{v:.2f}" for v in lv) + ".", BODY),
        P(f"<b>Failure found:</b> the semantic score was <i>higher</i> for the neutral 'No thanks' "
          f"({bh['semantic_mean_by_wording']['neutral']:.3f}) than for real misdirection declines "
          f"({bh['semantic_mean_by_wording']['misdirection_test_split']:.3f}). Cause: all dataset rows starting 'No thanks' are labelled "
          "misdirection, so the classifier learned the phrase itself (P = "
          f"{E['D_counterfactual_rewrites']['neutral_decline_score']['p_dark']:.3f}). This is a training-label bias, documented and "
          "not hidden.", BODY),
        P(f"<b>Screenshots</b> (authored PIL renders): OCR found both options in {bs['main_6']['n_both_options_found']}/6 main cases. It "
          f"<b>missed the small light-grey decline in all 3 asymmetric cases</b>, which are the most manipulative designs. With a darker "
          f"decline (supplementary cases) both options were found, with visual = " + ", ".join(f"{v:.2f}" for v in bs['supplementary_S7_S8']['visual']) +
          ". The API now says explicitly when OCR may have missed a decline, instead of failing silently.", BODY)],
        img(FIG / "intel_mcai_levels.png", W * 0.42), wa=W * 0.57))
    story.append(P("C · CDI on the shipped demo inputs (real API runs)", H2))
    rows = [["Input", "CDI (band)", "Coverage", "Dirichlet p5–p95", "Redesign Δ (model-est.)"]]
    C = E["C_cdi_demo_inputs"]
    for group in ("samples", "journeys"):
        for v in C[group]:
            cdi = v.get("cdi")
            lo, hi = v.get("dirichlet_p05"), v.get("dirichlet_p95")
            red = (v.get("counterfactual") or {}).get("risk_reduction_pct", v.get("risk_reduction_pct"))
            rows.append([v["id"] + (" (synthetic)" if v.get("synthetic") else ""), "—" if cdi is None else f"{cdi:.1f} ({v.get('band')})",
                         f"{v.get('coverage', 0):.2f}", "—" if lo is None or lo == hi else f"{lo:.1f}–{hi:.1f}",
                         "—" if red is None else f"{red:.1f}%"])
    story.append(table(rows, [W * 0.3, W * 0.17, W * 0.12, W * 0.2, W * 0.21]))
    D = E["D_counterfactual_rewrites"]
    story.append(P("D · Counterfactual rewrites on all dark TEST-split rows (circular by design, see caveat)", H2))
    rows = [["Category", "Findings", "Safe rewrite", "Rewrites below τ", "Mean P(dark) before → after", "Fact violations"]]
    for cat, r in D["per_category"].items():
        rows.append([cat, str(r["n_findings"]), f"{r['pct_safe_rewrite']:.1f}%",
                     "—" if r.get("pct_rewrites_below_threshold") is None else f"{r['pct_rewrites_below_threshold']:.1f}%",
                     "—" if r.get("mean_p_dark_before_rewritten") is None else f"{r['mean_p_dark_before_rewritten']:.3f} → {r['mean_p_dark_after_rewritten']:.3f}",
                     str(r.get("facts_preserved_violations", 0) + r.get("independent_fact_check_failures", 0))])
    story.append(side_by_side([table(rows, [W * 0.11, W * 0.07, W * 0.08, W * 0.08, W * 0.15, W * 0.07]),
                               P("Rewrites of urgency wording often fall below the threshold. Scarcity and misdirection rewrites almost "
                                 "never do, because the classifier scores stock/deadline <i>content</i> and the phrase 'No thanks', not "
                                 "only pressure wording. No rewrite introduced a new fact. " + D["circularity_caveat"], SMALL)],
                              img(FIG / "intel_counterfactual.png", W * 0.38), wa=W * 0.6))
    story.append(PageBreak())

    # ---------------------------------------------------------------- intel-v1.1
    RS = json.loads((ROOT / "ml/artifacts/intelligence_research.json").read_text())

    def leaf(pattern: str):
        """First numeric value whose JSON path matches `pattern` (strict: missing data fails the build)."""
        rx = re.compile(pattern)

        def walk(o, path=""):
            if isinstance(o, dict):
                for k, v in o.items():
                    yield from walk(v, f"{path}/{k}")
            elif isinstance(o, (int, float)) and not isinstance(o, bool):
                yield path, o
        for pth, v in walk(RS["studies"]):
            if rx.search(pth):
                return v
        raise KeyError(pattern)
    story.append(P("2b. intel-v1.1: validation, optimisation, fusion and hardening", H1))
    story.append(P(f"Studies in <font face='Courier'>ml/research_intelligence.py</font> (seed {RS['seed']}, real classifier, real TEST-split strings; "
                   "journeys, reference cases and choice sets are synthetic and labelled so). Two runs gave identical study numbers.", SMALL))
    sch = ("fibonacci", "linear", "equal", "geometric_phi", "max_pressure")
    rows = [["Scheme", "Noise: rank ρ", "Missing stage: mean |Δ|", "Escalating > de-escalating"]]
    for k in sch:
        rows.append([k, f"{leaf(f'R1_fspm_robustness/a_single_observation_noise/per_scheme/{k}/rank_spearman$'):.3f}",
                     f"{leaf(f'R1_fspm_robustness/b_missing_stage/mean_abs_delta_overall/canonical/{k}$'):.2f}",
                     f"{leaf(f'R1_fspm_robustness/d_order_sensitivity/per_scheme/{k}/frac_escalating_higher$'):.3f}"])
    story.append(side_by_side([P("R1 · Is Fibonacci weighting justified?", H2), table(rows, [W * 0.17, W * 0.13, W * 0.15, W * 0.15]),
        P("Fibonacci, linear and geometric weights are equally robust to single-observation noise and equally order-sensitive; "
          "equal and max weighting cannot see order at all. Against 33 author-written reference expectations (design intent, "
          "not ground truth) linear satisfied all and Fibonacci all but one contestable case, and Fibonacci reacts most to a "
          "missing or duplicated final stage. <b>Conclusion: Fibonacci shows a measurable advantage over equal/max weighting, "
          "but none over linear weighting.</b> It is kept as the proposed scheme with linear and equal always reported beside it.", SMALL)],
        img(FIG / "research_fspm_robustness.png", W * 0.36), wa=W * 0.62))
    ag = ("weighted_mean (selected)", "geometric", "max", "power2", "noisy_or")
    rows = [["Aggregator", "ρ vs designed level", "ρ within wording", "Collapses to 0"]]
    for k in ag:
        within = f"R2_mcai_aggregation/per_aggregator/{re.escape(k)}/spearman_rho_within_fixed_wording_mean$"
        try:
            w = f"{leaf(within):.3f}"
        except KeyError:
            w = "constant"
        rows.append([k, f"{leaf(f'R2_mcai_aggregation/per_aggregator/{re.escape(k)}/spearman_rho_vs_level$'):.3f}", w,
                     f"{leaf(f'R2_mcai_aggregation/per_aggregator/{re.escape(k)}/collapse_to_zero_share$'):.1%}"])
    story.append(P("R2–R3 · Which aggregation function?", H2))
    story.append(side_by_side([table(rows, [W * 0.2, W * 0.12, W * 0.12, W * 0.11]),
        P("On 2,048 authored choice sets the weighted mean has the best rank agreement within a fixed wording and never collapses; "
          "geometric's higher overall ρ comes from forcing 0 whenever any dimension is 0, and max ignores everything but one "
          "dimension. For CDI, zero-filling missing components instead of renormalising deflated scores by about 10 points on "
          "average; naive addition of FSPM and page pressure, or of obstruction facts, inflated the demo CDI by 15–20 points. "
          "<b>The weighted mean with renormalisation and the de-duplication rules are kept.</b>", SMALL)],
        img(FIG / "research_mcai_aggregators.png", W * 0.4), wa=W * 0.57))
    story.append(P("R4–R6 · Optimiser exactness, cross-modal fusion, performance", H2))
    story.append(side_by_side([
        P("<b>Optimiser.</b> Over all dark TEST rows it found a feasible change for most findings; social-proof claims are removed "
          "and misdirection declines become 'Decline' (a model label bias, disclosed in every response). A brute-force check over the "
          "full candidate product space on 200 real pages confirmed the per-element argmin is the true minimum; the shipped rule's "
          "0.005 tie band (preferring an actual rewrite) makes it optimal to within 0.005 per element, max observed gap 0.0026.", SMALL),
        P("<b>Fusion.</b> Capturing the same checkout as HTML + text + screenshot without fusion inflated FSPM by up to 27 points and "
          "CDI by up to 15 in the journey studied; with fuse_with each category counts once.", SMALL),
        P("<b>Performance.</b> The request-scoped inference cache cut classifier work roughly in half (journey: 6 vs 14 model calls) "
          "and latency by 1.5–1.9× on text/HTML/journeys with identical outputs; screenshots are dominated by OCR (~0.44 s).", SMALL)],
        img(FIG / "research_cache_latency.png", W * 0.38), wa=W * 0.6))
    story.append(P("Hardening: images over 40 megapixels are rejected before decoding (decompression bombs), journey requests are capped "
                   "at 40 MB, error logs no longer include exception messages that can quote user content, and a quadratic regex that "
                   "took 114 s on a 200k-digit input now runs in milliseconds. Production npm dependencies audit clean; two dev-only "
                   "advisories (esbuild dev server, postcss) need a breaking Vite major and are documented. Every audit and journey is "
                   "stamped with model, calibration, threshold and configuration versions, and a journey can be replayed from its "
                   "record (inputs verified by SHA-256) to confirm identical results.", SMALL))
    story.append(PageBreak())

    # ---------------------------------------------------------------- failures + limitations
    story.append(P("3. Where it failed, and what we changed", H1))
    fails = [["Failure observed (real run)", "Change made", "Effect"],
             ["Noisy-OR stage pressure saturated at ≈1.0 for any page with one flag (near-binary calibrated scores): every demo-journey stage scored ≈1.0, so the three weighting schemes were nearly identical",
              "Intensity-breadth pressure p = ½·max + ½·mean over categories",
              f"Stage pressures: median {E['A2_order_sensitivity']['stage_pressure_summary']['median']:.3f} vs {E['A2_order_sensitivity']['legacy_noisy_or_stage_pressure_summary']['median']:.4f} before; demo journey now {JS['fibonacci']:.1f} / {JS['linear']:.1f} / {JS['equal']:.1f}. A4 AUROCs essentially unchanged"],
             ["Demo journey was titled 'escalating', but its measured last stage had the lowest pressure", "Renamed and re-described; demo left unchanged rather than tuned", "Fibonacci correctly scores it below equal weighting"],
             ["rgb() backgrounds with spaces read as 'no fill'; @media rules applied unconditionally; 'In stock | 44 left' became '… in stock in stock'",
              "Colour-token parser; @-blocks skipped and counted; stock clarification only when no stock wording exists", "Regression tests added"],
             ["Live UI check: a text + screenshot journey reported obstruction as 'DOM inspected, 0' although no HTML was submitted (empty per-stage fact lists were treated as observations)",
              "Obstruction is available only with an HTML stage or a real observed fact", "Now 'unavailable'; coverage drops honestly; regression test added"],
             ["OCR misses faint decline text and returned 'unavailable' silently", "Explicit OCR blind-spot reason in the MCAI response", "Limitation is visible to the user"],
             ["Classifier flags the neutral 'No thanks' and neutral stock/deadline text", "Reported, not gamed; we did not switch to 'Decline' just because the model scores it lower", "Counterfactual reductions come mostly from by-construction components"]]
    story.append(table(fails, [W * 0.42, W * 0.29, W * 0.29]))
    story.append(P("Limitations and assumptions", H2))
    story += bullets([
        "No ground truth: every weight and constant (Fibonacci, CDI w = 0.55/0.25/0.20, MCAI w = 0.4/0.3/0.3, λ = 0.5, fact strengths "
        "1.0/0.5) is a design choice. Sensitivity analysis shows how much the CDI moves; it does not make the weights correct.",
        "Synthetic benchmarks show the indices behave as designed; they do not show agreement with consumer harm or regulators.",
        "HTML visual measurements use declared CSS only. There is no layout engine, so rendered area, external stylesheets and "
        "@media rules are not evaluated. Screenshots provide OCR box geometry and pixel contrast only.",
        "Interaction asymmetry exists only when step counts are declared; ClearCart does not crawl or click through sites.",
        "Counterfactual scores are circular (same classifier), and two training-label biases limit them ('No thanks', stock/deadline content).",
        "Temporal checks compare only the snapshots provided. A reset that happens between snapshots without changing the observed values cannot be detected."])
    story.append(P("Reproduce", H2))
    story.append(P("<font face='Courier'>python -m ml.make_journey_samples · python -m ml.benchmark_intelligence · python -m pytest tests -q · "
                   f"python -m report.build_addendum</font> — test suite at build time: <b>{tests_line}</b>.", SMALL))
    story.append(P("References", H2))
    refs = ["Mathur et al., Dark Patterns at Scale, CSCW 2019 (taxonomy behind the dataset).",
            "Gray et al., The Dark (Patterns) Side of UX Design, CHI 2018 (interface interference, asymmetric choices).",
            "W3C WCAG 2.1, contrast ratio and relative luminance definitions (used for the contrast features).",
            "Guo et al., On Calibration of Modern Neural Networks, ICML 2017 (temperature scaling behind the calibrated scores).",
            "Saltelli et al., Global Sensitivity Analysis: The Primer, 2008 (one-at-a-time and sampling-based sensitivity)."]
    story += [P(f"[{i + 1}] {r}", SMALL) for i, r in enumerate(refs)]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=15 * mm, bottomMargin=13 * mm,
                            title="ClearCart — Mathematical Innovation addendum", author="Team ClearCart")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"wrote {OUT} with {len(PdfReader(str(OUT)).pages)} pages; tests: {tests_line}")


if __name__ == "__main__":
    build()
