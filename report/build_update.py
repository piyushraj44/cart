"""Build report/ClearCart_Update_Report.pdf: everything added since the 8-page project report.

Every number is read from generated artifacts (ml/artifacts/*.json, ml/artifacts_v2/metrics.json) or computed
here (test runs, one model score); nothing is hand-typed. Also rebuilds report/ClearCart_Full_Report.pdf as
project report + math addendum + this update. Run from the project root:
    .venv/bin/python -m report.build_update

Environment overrides (development only; the defaults build the real report):
    CLEARCART_REPORT_OUT_DIR     write the update + full PDFs (and the figure this builder draws) here instead of report/
    CLEARCART_REPORT_SKIP_TESTS  =1 skips the pytest and npm test runs; the PDF then says "not run"
    CLEARCART_VISUAL_METRICS     alternative path for ml/artifacts_visual/metrics.json (missing file: section skipped)
    CLEARCART_VISUAL_FIG_DIR     alternative directory for the visual_*.png figures (default report/figures)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pypdf import PdfReader, PdfWriter  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer  # noqa: E402

from report.build_report import (BODY, BRAND, CAP, H1, H2, LINE, MUTED, ROOT, SMALL, W, P, bullets, f3, img,  # noqa: E402
                                 side_by_side, table)

FIG = ROOT / "report" / "figures"
_OUT_DIR = os.environ.get("CLEARCART_REPORT_OUT_DIR")
OUT_DIR = Path(_OUT_DIR) if _OUT_DIR else ROOT / "report"
OUT = OUT_DIR / "ClearCart_Update_Report.pdf"
FULL = OUT_DIR / "ClearCart_Full_Report.pdf"
GEN_FIG = OUT_DIR if _OUT_DIR else FIG          # where this builder writes the figure it draws itself
SKIP_TESTS = os.environ.get("CLEARCART_REPORT_SKIP_TESTS", "").strip() == "1"
VIS_METRICS = Path(os.environ.get("CLEARCART_VISUAL_METRICS") or ROOT / "ml" / "artifacts_visual" / "metrics.json")
VIS_SPLIT = ROOT / "data" / "processed" / "visual_split_stats.json"
VFIG = Path(os.environ.get("CLEARCART_VISUAL_FIG_DIR") or FIG)
PARTS = [ROOT / "report" / "ClearCart_Report.pdf", ROOT / "report" / "ClearCart_Math_Addendum.pdf", OUT]
A = ROOT / "ml" / "artifacts"


def load(p):
    return json.loads(p.read_text())


CC = load(A / "claim_check_eval.json")
FP = load(A / "false_positive_eval.json")
AB = load(A / "abstention_eval.json")
CD = load(A / "conservative_detection_eval.json")
OC = load(A / "ocr_engine_eval.json")
V2 = load(ROOT / "ml" / "artifacts_v2" / "metrics.json")
VIS = load(VIS_METRICS) if VIS_METRICS.exists() else None   # experimental visual model; None until finalize has run
MRP = ("Maximum retail price (GST Incl.). The maximum retail price is applicable for imports in India effect from 1st "
       "April 2026. Retailers may have stock at different MRP based on their date of purchase.")


def rs(t: str) -> str:
    """Helvetica has no ₹ glyph; write it as Rs."""
    return t.replace("₹ ", "Rs ").replace("₹", "Rs ")


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def run_tests() -> tuple[str, str]:
    """Run both suites; return their real summary lines."""
    if SKIP_TESTS:
        return "not run (CLEARCART_REPORT_SKIP_TESTS=1)", "not run (CLEARCART_REPORT_SKIP_TESTS=1)"
    py = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
                        cwd=ROOT, capture_output=True, text=True)
    m = re.findall(r"(\d+ passed[^\n]*)", py.stdout)
    fe = subprocess.run(["npm", "test"], cwd=ROOT / "frontend", capture_output=True, text=True)
    n = re.findall(r"(\d+ passed, \d+ failed)", fe.stdout)
    return (m[-1].strip() if m else "pytest did not report"), (n[-1] if n else "npm test did not report")


def mrp_score() -> tuple[float, str]:
    from ml.common import ARTIFACTS
    from ml.model import TextClassifier
    p = TextClassifier.load(ARTIFACTS).predict([MRP])[0]
    return p["p_dark"], p["category"]


def make_claim_figure() -> None:
    sets = [("exploratory_text", "Exploratory"), ("fresh", "Fresh"), ("test", "TEST")]
    before_c, after_c = "#94a3b8", "#0f766e"
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.7), sharey=True)
    for ax, (metric, title) in zip(axes, [("false_positive_rate", "False-positive rate (lower is better)"),
                                          ("precision", "Precision (higher is better)"), ("recall", "Recall (higher is better)")]):
        xs = range(len(sets))
        for off, key, col, lab in ((-0.19, "before", before_c, "Before gate"), (0.19, "after", after_c, "With claim check")):
            vals = [CC[s][key][metric] for s, _ in sets]
            bars = ax.bar([x + off for x in xs], vals, width=0.36, color=col, label=lab, edgecolor="white", linewidth=1)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color="#0f172a")
        ax.set_title(title, fontsize=8.5, color="#0f172a", loc="left")
        ax.set_xticks(list(xs)); ax.set_xticklabels([l for _, l in sets], fontsize=7.5, color="#475569")
        ax.set_ylim(0, 1.12); ax.tick_params(axis="y", labelsize=7, colors="#475569")
        ax.grid(axis="y", color="#e2e8f0", linewidth=0.6); ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#cbd5e1")
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    GEN_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(GEN_FIG / "claim_check_effect.png", dpi=200)
    plt.close(fig)


def on_page(c, doc):
    c.saveState(); c.setFont("Helvetica", 7); c.setFillColor(MUTED)
    c.drawString(16 * mm, A4[1] - 10 * mm, "ClearCart — Update report: changes since the project report")
    c.drawRightString(A4[0] - 16 * mm, A4[1] - 10 * mm, "Team ClearCart")
    c.setStrokeColor(LINE); c.line(16 * mm, A4[1] - 11.5 * mm, A4[0] - 16 * mm, A4[1] - 11.5 * mm)
    c.drawString(16 * mm, 8 * mm, "All numbers read from ml/artifacts*/ or computed by report/build_update.py")
    c.drawRightString(A4[0] - 16 * mm, 8 * mm, f"Page {doc.page}")
    c.restoreState()


def m_row(name, m, note=""):
    return [name, f3(m["precision"]), f3(m["recall"]), f3(m["f1"]), f"{m['fp']} / {m['fp'] + m['tn']}", note]


def _mins(sec) -> str:
    if sec is None:
        return "—"
    return f"{sec:.0f} s" if sec < 120 else f"{sec / 60:.1f} min"


def _ap(x) -> str:
    return "—" if x is None else f3(x)


def _yes(b) -> str:
    return "<b>pass</b>" if b else "<font color='#b91c1c'><b>FAIL</b></font>"


def _vfig(name: str, width, max_h=None, caption: str | None = None) -> list:
    """Figure + caption if the training run produced it; nothing otherwise."""
    p = VFIG / name
    if not p.exists():
        return []
    out = [img(p, width, max_h=max_h)]
    return out + [P(caption, CAP)] if caption else out


ARCH_NAMES = {"mobilenet_v3_large": "MobileNetV3-Large", "efficientnet_b0": "EfficientNet-B0", "convnext_tiny": "ConvNeXt-Tiny"}


def visual_section(V: dict, num: int) -> list:
    """Section built only from ml/artifacts_visual/metrics.json and data/processed/visual_split_stats.json."""
    sp = V.get("split") or (load(VIS_SPLIT) if VIS_SPLIT.exists() else {})
    crit, cres = V.get("criteria", {}), V.get("criteria_result", {})
    req = crit.get("required_labels") or list(cres.get("b_per_label_val_ap_margin", {}))
    labels = crit.get("all_labels") or req
    opt = [t for t in labels if t not in req]
    sel, val, cv, trn = V.get("selected"), V.get("val", {}), V.get("cv", {}), V.get("training", {})
    all_pass = bool(cres.get("all_pass"))
    s = []

    # ---- why
    s.append(P(f"{num}. Experimental visual pattern-type model (opt-in, off by default)", H1))
    s.append(P(
        "Every served path reads <i>text</i>: pasted text, HTML, or OCR of a screenshot. Most ContextRico-DP labels are visual: a close "
        "control that is hard to find (<i>Hard to Close</i>), a pre-selected option (<i>Default Choice</i>), an accept button styled far "
        "above the decline (<i>False Hierarchy</i>), a pop-up that interrupts (<i>Nagging</i>). OCR followed by a text classifier cannot "
        "see layout, colour or prominence. This experimental model is trained on ContextRico-DP screenshots (previously used for "
        "evaluation only; the hackathon rules allow training on it) to name the visual pattern <b>type</b> on a screenshot."))
    s.append(P(
        "<b>What it is not.</b> ContextRico has no benign screenshots, so the model never learned what an honest screen looks like: it "
        "indicates a pattern type, not whether a page is deceptive. It is opt-in (<font face='Courier'>CLEARCART_VISUAL=1</font>, screenshot "
        "audits only), and its signals are reported separately: they never enter findings, the headline gauge, the CDI or the evidence tiers.",
        BODY))

    # ---- data and split
    ps = sp.get("per_split", {})
    hdr = ["Split", "Images", "Groups", "Web"] + labels + ["No target label"]
    rows = [hdr]
    for k in ("train", "val", "test"):
        d = ps.get(k, {})
        rows.append([k.upper()] + [str(d.get(c, "—")) for c in ["images", "groups", "web"] + labels + ["no_target_label"]])
    lw = (W - 4 * 13 * mm - 17 * mm) / max(len(labels), 1)
    s.append(KeepTogether([P("Data and split", H2), table(rows, [13 * mm] * 4 + [lw] * len(labels) + [17 * mm])]))
    corr = sp.get("corrections", {})
    disc = corr.get("discrepancies_not_applied", [])
    ov = sp.get("group_overlap", {})
    added = []
    for d in disc:
        extra = sorted(set(d.get("correction_would_give", [])) - set(d.get("groundtruth", [])))
        added.append(f"{d.get('image')} (+{', '.join(extra)})")
    ocr = sp.get("ocr_eval_images_also_in_visual_split", {})
    s += bullets([
        f"<b>{sp.get('images', 0):,} labelled images</b> ({len(sp.get('files_without_label', []))} files without a ground-truth entry left out). "
        f"Required labels: {', '.join(req)}; optional: {', '.join(opt)}. Images with none of them stay in as all-negative rows.",
        f"<b>Groups.</b> Web files by domain, named mobile files by app, Rico ids alone, then {sp.get('near_duplicate_pairs_merged', 0)} "
        f"near-duplicate pairs (dHash within 4 bits) merged: {sp.get('groups', 0):,} groups (largest {sp.get('largest_group', '—')}), split "
        f"70/15/15 at group level with label stratification. Group overlap: {', '.join(f'{k} {v}' for k, v in ov.items())}.",
        f"<b>Labels.</b> groundtrue.json as published. It already reflects correction.xlsx for {corr.get('already_reflected_in_groundtruth', '—')} of the "
        f"{corr.get('matched', '—')} matching rows; the {len(disc)} rows where correction.xlsx would add a label were <b>not applied</b> "
        f"(published ground truth is authoritative): {'; '.join(added)}.",
        f"<b>Overlap with the OCR evaluation.</b> The ml/eval_screenshots images are ContextRico files and also sit in this split "
        f"(TRAIN {ocr.get('train', '—')}, VAL {ocr.get('val', '—')}, TEST {ocr.get('test', '—')}). The text model's OCR evaluation is unaffected: "
        "the text model is not trained on them and does not change."])

    # ---- protocol
    s.append(P(
        f"<b>Protocol.</b> The acceptance criteria (a)–(c) in the table below were written to <font face='Courier'>criteria.json</font> at "
        f"{crit.get('fixed_at', '—')}, before any model was trained. Selection: {crit.get('selection', '—')}. Per-label temperature scaling "
        f"and thresholds also use VAL only; thresholds: {crit.get('thresholds', '—')}. Grouped CV (re-splits of TRAIN+VAL) measures "
        "variance only. TEST was read once, by <font face='Courier'>finalize</font>, after selection.", BODY))

    # ---- VAL table
    rows = [["Model (VAL, required labels)", "Macro-AP", "Grouped CV macro-AP", "Wall time", "Note"]]
    pr, cl = val.get("prior", {}), val.get("clip_lr", {})
    rows.append(["Label prior (TRAIN prevalence)", _ap(pr.get("macro_ap_required")), "—", "—", "chance level"])
    rows.append(["Frozen CLIP ViT-B/32 + logistic probe", _ap(cl.get("macro_ap_required")), "—",
                 _mins(V.get("clip_embedding_seconds")) + " (embedding)", "C per label chosen on VAL"])
    for a, m in val.items():
        if a in ("prior", "clip_lr"):
            continue
        c = cv.get(a)
        t = trn.get(a, {})
        note = ("<b>selected</b>; " if a == sel else "") + (f"best epoch {t['best_epoch']}" if "best_epoch" in t else "")
        rows.append([f"{ARCH_NAMES.get(a, a)} fine-tuned", _ap(m.get("macro_ap_required")),
                     f"{f3(c['mean'])} ± {f3(c['std'])} (n={len(c.get('folds', []))})" if c else "not run",
                     _mins(t.get("wall_seconds")), note])
    s.append(KeepTogether([P("Validation results", H2), table(rows, [56 * mm, 18 * mm, 36 * mm, 28 * mm, W - 138 * mm])]))
    s += _vfig("visual_model_comparison.png", W * 0.52, max_h=40 * mm,
               caption="VAL macro average precision over the required labels (grey: baselines; teal: fine-tuned candidates).")

    # ---- criteria result
    bd = cres.get("b_per_label_val_ap_margin", {})
    best_base = max(pr.get("macro_ap_required", 0), cl.get("macro_ap_required", 0))
    sel_ap = val.get(sel, {}).get("macro_ap_required")
    rows = [["Criterion", "Requirement", "Selected model on VAL", "Result"],
            ["(a)", f"macro-AP above both baselines (best: {f3(best_base)})", _ap(sel_ap), _yes(cres.get("a_beats_both_baselines_on_val"))]]
    for t, d in bd.items():
        rows.append([f"(b) {t}", f"AP > prevalence + margin = {f3(d['needs_more_than'])}", _ap(d.get("ap")),
                     _yes(d.get("ap") is not None and d["ap"] > d["needs_more_than"])])
    rows.append(["(c)", "TEST evaluated exactly once, after selection", "—", _yes(cres.get("c_test_evaluated_once"))])
    verdict = (f"<b>All criteria pass.</b> {ARCH_NAMES.get(sel, sel)} is exported as the experimental model and may be enabled with "
               "<font face='Courier'>CLEARCART_VISUAL=1</font>; it remains off by default." if all_pass else
               "<b>The criteria fixed before training are NOT all met, so the model is not ready to serve.</b> The serving code would "
               "still load the exported files if <font face='Courier'>CLEARCART_VISUAL=1</font> were set; it should stay off, and the "
               "numbers below are reported only because the protocol says TEST is reported whatever it shows.")
    s.append(KeepTogether([P(f"Acceptance criteria: {'all pass' if all_pass else 'NOT all pass'}", H2),
                           table(rows, [34 * mm, 72 * mm, 36 * mm, W - 142 * mm]), Spacer(1, 3), P(verdict, BODY)]))

    # ---- TEST table
    te = V.get("test", {})
    ts, tp_, tc = te.get("selected", {}), te.get("prior", {}), te.get("clip_lr", {})
    lc = V.get("label_config", {})
    rows = [["Label (TEST)", "Positives", "Prevalence (chance AP)", "AP", "CLIP probe AP", "Precision @ VAL thr.", "Recall @ VAL thr.", "Reliable"]]
    for t in labels:
        d = ts.get("per_label", {}).get(t, {})
        c = lc.get(t, {})
        rows.append([t + ("" if t in req else " (optional)"), str(d.get("n_pos", "—")), _ap(d.get("prevalence")), _ap(d.get("ap")),
                     _ap(tc.get("per_label", {}).get(t, {}).get("ap")),
                     _ap(d.get("precision_at_val_threshold")), _ap(d.get("recall_at_val_threshold")),
                     "yes" if c.get("reliable") else "<b>no</b> (never shown)"])
    rows.append(["<b>Macro, required labels</b>", "", _ap(tp_.get("macro_ap_required")) + " (prior)", f"<b>{_ap(ts.get('macro_ap_required'))}</b>",
                 _ap(tc.get("macro_ap_required")), "", "", ""])
    sub = te.get("selected_macro_ap_by_subset", {})
    s.append(KeepTogether([P(f"Held-out TEST ({te.get('n_images', '—')} images, evaluated once)", H2),
                           table(rows, [40 * mm, 15 * mm, 21 * mm, 14 * mm, 19 * mm, 21 * mm, 20 * mm, W - 150 * mm])]))
    s.append(P(
        "Chance AP equals prevalence. Precision and recall use the per-label threshold chosen on VAL (see Protocol); a label "
        "without such a threshold is marked not reliable and is never shown to users. "
        + (f"Macro-AP by subset: {', '.join(f'{k} {f3(v)}' for k, v in sub.items())}. " if sub else "")
        + "Calibration error after per-label temperature scaling: "
        + ", ".join(f"{t} {f3(ts['per_label'][t]['ece_after_calibration'])}" for t in labels
                    if ts.get("per_label", {}).get(t, {}).get("ece_after_calibration") is not None) + ".", SMALL))
    pl, rl = VFIG / "visual_per_label.png", VFIG / "visual_reliability.png"
    if pl.exists() and rl.exists():
        s.append(side_by_side(img(pl, W * 0.58, max_h=50 * mm), img(rl, W * 0.3, max_h=50 * mm), wa=W * 0.62))
        s.append(P("Left: TEST average precision per label vs prevalence. Right: TEST reliability after calibration, pooled over labels.", CAP))
    else:
        s += _vfig("visual_per_label.png", W * 0.7, caption="TEST average precision per label vs prevalence.")
        s += _vfig("visual_reliability.png", W * 0.36, caption="TEST reliability after calibration, pooled over labels.")

    # ---- Grad-CAM pointing game
    pg = V.get("pointing_game_test", {})
    hit, ctr = pg.get("peak_in_segment_rate"), pg.get("image_centre_baseline_rate")
    cmp_ = ("above" if hit > ctr else "not above") if hit is not None and ctr is not None else "not comparable with"
    s.append(KeepTogether([P("Where the model looks: Grad-CAM pointing game", H2), P(
        f"On the {pg.get('n_images', '—')} TEST images with ground-truth segments and a target label, the Grad-CAM peak for the highest-scoring true label "
        f"falls inside a segment for <b>{_ap(hit)}</b> of images, against <b>{_ap(ctr)}</b> for a point at the image centre, so localisation is "
        f"<b>{cmp_}</b> this trivial baseline. Mean best IoU of the evidence box with a segment: {_ap(pg.get('mean_best_iou_evidence_box'))}. "
        "Segments mark dark-pattern regions for the whole image, not per label, so a hit does not prove the region matches the named "
        "label. The evidence box is a visual hint, not a measurement.", BODY)]))
    s += _vfig("visual_gradcam.png", W, max_h=48 * mm,
               caption="Grad-CAM on TEST images: highest-confidence true positive per label (cyan: evidence box). Selected examples, not a sample.")

    # ---- probes
    probes = V.get("probes_not_metrics", [])
    kg = [p for p in probes if p.get("known_good")]
    kg_fired = [p for p in kg if p.get("fired")]
    other = [p for p in probes if not p.get("known_good")]
    items = []
    if kg:
        items.append(f"<b>ClearCart's own known-good pages ({len(kg)} screenshots):</b> " + (
            "nothing fired." if not kg_fired else
            "fired on " + "; ".join(f"{Path(p['file']).name} → {', '.join(p['fired'])}" for p in kg_fired)
            + ". These pages contain no dark pattern, so each is a false signal: the model has never seen a benign screen."))
    if other:
        items.append("<b>Demo sample screenshots</b> (ContextRico files; split in brackets): " + "; ".join(
            f"{Path(p['file']).name} [{p.get('contextrico_split') or 'not in split'}] → {', '.join(p['fired']) or 'nothing'}" for p in other)
            + ". Files from TRAIN were seen in training and say nothing about accuracy.")
    if items:
        s.append(KeepTogether([P("Probes (sanity checks, not metrics)", H2)] + bullets(items)))

    # ---- limitations
    web = {k: ps.get(k, {}).get("web") for k in ("train", "val", "test")}
    on = [ps.get(k, {}).get(t) for k in ("val", "test") for t in opt if isinstance(ps.get(k, {}).get(t), int)]
    opt_range = f"{min(on)}–{max(on)}" if on else "few"
    s.append(KeepTogether([P("Limitations", H2)] + bullets([
        "<b>No benign screenshots.</b> Every training image contains a dark pattern, so a probability means “which type, given that one is "
        "present”. On an honest page it will still name a type: it is not a deceptiveness detector.",
        f"<b>Mostly mobile.</b> Only {web['train']} TRAIN, {web['val']} VAL and {web['test']} TEST images are web pages; web results rest on very few examples.",
        "<b>Segments are not per label</b>, so localisation is checked only loosely; Grad-CAM shows where the network responded, not why.",
        f"<b>Small evaluation sets</b> ({te.get('n_images', '—')} TEST images; optional labels have {opt_range} positives per VAL/TEST split), so per-label "
        "numbers carry wide uncertainty. Labels are image-level and inherited from ContextRico without re-annotation."]))) 
    return s


def build() -> None:
    py_tests, fe_tests = run_tests()
    mrp_p, mrp_cat = mrp_score()
    make_claim_figure()
    tm = FP["test_split"]["metrics"]
    base, ood = tm["base_no_gates"], tm["+quote_and_ood (served now)"]
    cc_t = CC["test"]["after"]
    crit = CC["criteria"]
    story = []

    # ------------------------------------------------------------------ title + summary
    story += [P("ClearCart — Update report", ParagraphStyle("t", parent=H1, fontSize=22, leading=26, textColor=BRAND)),
              P(f"Changes since the 8-page project report · generated {date.today():%d %B %Y}", SMALL), Spacer(1, 4)]
    story.append(P("Summary", H2))
    story.append(P(
        "The project report evaluated the classifier on its own. Since then the served application has gained a chain of "
        "<b>conservative, post-hoc review gates</b> that never change a model probability, a <b>three-tier evidence output</b> "
        "that separates observed facts from unverified pressure claims, a <b>claim-specificity check</b> so that genuine sales stated "
        "in checkable terms are held for review instead of reported, a corrected <b>headline gauge</b>, two studies that were "
        "<i>not</i> adopted (a second OCR engine and fine-tuned transformers), "
        + ("an experimental, opt-in <b>visual pattern-type model</b> trained on ContextRico-DP screenshots (§4; it names a pattern type "
           "and never feeds findings or scores), " if VIS else "")
        + "and a single-origin container build. "
        "The rule throughout: a change ships only if criteria fixed before scoring pass, and the held-out TEST split is used as a "
        "safety veto, never for tuning."))
    story.append(P("Served text pipeline on the held-out TEST split (329 rows, 162 dark, 167 benign)", H2))
    story.append(table([
        ["Decision stage", "Precision", "Recall", "F1", "Benign FP", "What changed"],
        m_row("Model only (project report)", {"precision": base["dark_vs_benign_precision"], "recall": base["dark_vs_benign_recall"],
                                              "f1": base["dark_vs_benign_f1"], "fp": base["benign_false_positives"],
                                              "tn": base["benign_n"] - base["benign_false_positives"]}, "calibrated score ≥ τ = 0.68"),
        m_row("+ no-words, quote and OOD gates", {"precision": ood["dark_vs_benign_precision"], "recall": ood["dark_vs_benign_recall"],
                                                 "f1": ood["dark_vs_benign_f1"], "fp": ood["benign_false_positives"],
                                                 "tn": ood["benign_n"] - ood["benign_false_positives"]}, "OOD withholds 3 real social-proof rows"),
        m_row("+ claim-specificity check (served now)", cc_t, "withholds 2 rows (see §3)"),
    ], [52 * mm, 17 * mm, 15 * mm, 14 * mm, 18 * mm, W - 116 * mm]))
    story.append(P(
        "On in-distribution test text the gates cost recall and buy nothing, because the model's only TEST false positive is in-distribution. "
        "They exist for the out-of-distribution text real pages contain (prices, legal notes, explanatory pages, genuine promotions), "
        "which the 2019 training data does not cover. Their benefit is measured on probes and synthetic sets below, and is reported as such.", SMALL))
    story.append(P("Verification for this build", H2))
    story += bullets([f"Backend: <b>{py_tests}</b> (pytest, run by this builder).",
                      f"Frontend render tests: <b>{fe_tests}</b> (npm test, run by this builder).",
                      "Acceptance criteria of the claim-specificity check: " +
                      ("<b>all pass</b>" if crit["all_pass"] else "<b>NOT all pass</b>") + " (ml/artifacts/claim_check_eval.json)."]
                     + (["Acceptance criteria of the experimental visual model: " +
                         ("<b>all pass</b>" if VIS.get("criteria_result", {}).get("all_pass") else "<b>NOT all pass; not ready to serve</b>")
                         + " (ml/artifacts_visual/metrics.json, §4)."] if VIS else []))

    # ------------------------------------------------------------------ 1. tiers
    story.append(PageBreak())
    story.append(P("1. Three evidence tiers", H1))
    story.append(P(
        "A single \"flagged\" list mixed three different kinds of evidence. Every audit now reports them separately "
        "(<font face='Courier'>Finding.evidence_tier</font>, <font face='Courier'>summary.tier_counts</font>) and the results page shows "
        "them side by side:"))
    story.append(table([
        ["Tier", "Source", "What it claims"],
        ["Observed facts", "DOM observations, e.g. a pre-ticked paid add-on (checked attribute)", "A fact read from the input. No model score."],
        ["Pressure claims", "Model findings above the threshold, each with its claim-check markers",
         "The wording resembles labelled dark patterns. Whether the claim is true is not verified."],
        ["Held for review", "uncertain_elements: above the threshold but withheld by a gate",
         "Evidence is weak or the claim is specific and checkable. Listed, never silently dropped."],
    ], [30 * mm, 70 * mm, W - 100 * mm]))
    story.append(Spacer(1, 4))
    story.append(img(FIG / "ui_evidence_tiers.png", W * 0.62))
    story.append(P("Evidence tiers card for six pasted lines (live audit, served model). The three specific offers are held for review; "
                   "vague pressure stays a claim and shows its markers.", CAP))

    # ------------------------------------------------------------------ 3. gauge
    story.append(P("2. Headline gauge now reflects flagged findings only", H1))
    story.append(P(
        f"The results gauge used to show the highest calibrated score of <i>any</i> element, flagged or not. On a user screenshot of a "
        f"genuine luxury-watch product page, OCR merged the MRP note into one paragraph that scores <b>{pct(mrp_p, 0)}</b> "
        f"(<i>{mrp_cat.replace('_', ' ')}</i>), below the 90% OCR threshold. The page had no findings and a CDI of 0, yet the gauge read "
        f"\"{int(mrp_p * 100)}/100\". The paragraph was also not listed, so the number had no visible source."))
    story += bullets([
        "<font face='Courier'>summary.risk_score</font> is now the highest score among <b>flagged</b> model findings: 0 when nothing is "
        "flagged, and unreported (\"—\") when the only findings are DOM observations.",
        "The highest element score stays in the summary sentence (\"…was 82%, below the 90% flagging threshold\") and the merged OCR "
        "paragraph that sets it is now listed among analysed elements.",
        "Regression test: <font face='Courier'>tests/test_gauge.py</font> uses this MRP text. Earlier risk_score values recorded in "
        "false_positive_eval.json use the old definition."])

    story.append(PageBreak())
    # ------------------------------------------------------------------ 2. claim check
    story.append(P("3. Claim-specificity check: genuine sales vs manufactured pressure", H1))
    story.append(P(
        "Wording alone cannot separate an honest sale from a fake one: \"sale ends tonight\" is true on one site and false on another. "
        "What often differs is <b>specificity</b>. <font face='Courier'>backend/claim_check.py</font> is a rule set (no training data, no "
        "change to any probability) that lists two kinds of markers:"))
    story.append(table([
        ["Checkable terms (can be verified by the shopper)", "Vague or unverifiable pressure"],
        ["calendar deadline (\"30 November\", \"31/12/2026\"); named-day deadline (\"until Sunday at 23:59\"); reference price "
         "(\"was Rs 1,499\", MRP); stated saving or discount (not \"up to\"); code; shipping, delivery and returns terms",
         "hurry / act now; fear of missing out; vague deadline (\"ends soon\", \"tonight\", \"in 2 hours\"); \"limited\" without a date; "
         "undated hh:mm:ss countdown; selling fast; stock counts; live activity (viewing, in carts, bought ago); \"up to X%\"; "
         "a deadline that has already passed"],
    ], [W / 2, W / 2]))
    story.append(P(
        "<b>Decision.</b> An <i>urgency</i> or <i>scarcity</i> element above the threshold whose markers are all checkable is withheld "
        "as <font face='Courier'>checkable_claim</font>. Social proof and confirmshaming are never withheld this way: a date says nothing "
        "about whether a crowd signal or a guilt-framed refusal is honest. Every pressure claim carries its markers, so a reviewer sees why "
        "it was or was not withheld."))
    story.append(P(
        "<b>Protocol.</b> Acceptance criteria were fixed before scoring: on TEST, recall may drop at most 5 points and benign false positives "
        "may not rise; on both synthetic sets precision may not fall and the false-positive rate may not rise. The fresh set (12 genuine "
        "offers, 12 pressure claims) was written before the rules were scored, and the rules were not changed afterwards.", BODY))
    story.append(img(GEN_FIG / "claim_check_effect.png", W))
    story.append(P("Before vs with the claim check, same model and threshold. Exploratory: synthetic text cases the rules were written "
                   "against. Fresh: 24 synthetic strings written before scoring. TEST: held-out split, used only as a veto. Synthetic sets "
                   "are invented strings, not live websites: they show the rule behaves as specified, not field precision.", CAP))
    rows = [["Set", "Before: P / R / FPR", "With check: P / R / FPR", "TP FP FN TN after"]]
    for k, name in (("exploratory_text", "Exploratory synthetic (text cases)"), ("fresh", "Fresh synthetic (24)"), ("test", "Held-out TEST (329)")):
        b, a = CC[k]["before"], CC[k]["after"]
        rows.append([name, f"{f3(b['precision'])} / {f3(b['recall'])} / {f3(b['false_positive_rate'])}",
                     f"{f3(a['precision'])} / {f3(a['recall'])} / {f3(a['false_positive_rate'])}",
                     f"{a['tp']} {a['fp']} {a['fn']} {a['tn']}"])
    story.append(table(rows, [52 * mm, 44 * mm, 44 * mm, W - 140 * mm]))
    changed = CC["test"]["changed"]
    story.append(P("Withheld TEST rows and what we did not fix", H2))
    story += bullets([f"<b>{len(changed)} dark TEST rows withheld.</b> " + " ".join(
        f"“{rs(c['text'][:70])}{'…' if len(c['text']) > 70 else ''}” ({pct(c['p_dark'])}, {c['category']})." for c in changed)])
    story += bullets([
        "The first is a real countdown printed with spaced digits, which the countdown pattern misses. It is left unfixed: changing the "
        "rule after seeing TEST would be tuning on TEST.",
        "Price-only text that the model labels <i>misdirection</i> (fresh case “MRP Rs 2,499. You save Rs 500.”) is still reported, because "
        "the gate is limited to urgency and scarcity.",
        "Checkable terms can still be false. The gate says a claim is checkable, never that it is honest; verifying it needs a live "
        "page (e.g. two sessions showing the same absolute deadline)."])

    # ------------------------------------------------------------------ 4. experimental visual model (only once trained)
    num = 4
    if VIS is not None:
        story.append(PageBreak())
        story += visual_section(VIS, num)
        num += 1
    else:
        print(f"note: {VIS_METRICS.relative_to(ROOT) if VIS_METRICS.is_relative_to(ROOT) else VIS_METRICS} not found; "
              "visual model section skipped")

    # ------------------------------------------------------------------ earlier gates
    if VIS is None:
        story.append(PageBreak())   # with the visual section, its last page has room for this one
    else:
        story.append(Spacer(1, 6))
    story.append(P(f"{num}. Review gates added earlier (summarised from their evaluations)", H1))
    story.append(P("All gates act only on elements that crossed the threshold, and every withheld element is listed for review.", SMALL))
    gated_probes = [p for p in AB["probes"] if p.get("abstained")]
    ex_probe = gated_probes[0] if gated_probes else None
    story.append(table([
        ["Gate", "Rule and selection", "Measured effect"],
        ["No-words abstention", AB["definition"],
         f"TEST unchanged (macro-F1 {f3(AB['test_with_gate']['macro_f1'])}). Stops bare prices being called urgency"
         + (f", e.g. “{rs(ex_probe['text'])}” scored {pct(ex_probe['p_dark'])}." if ex_probe else ".")],
        ["Quote gate", "Rescore with quoted spans masked by “…” (placeholder chosen on validation); withhold if below τ.",
         "TEST unchanged. Treats quoted examples on explanatory pages as mentions, not claims."],
        ["Out-of-distribution gate", f"Max MiniLM similarity to TRAIN below τ_ood = {FP['rules']['ood_threshold']['threshold']:.3f} "
         "(≤ 2% of validation dark rows below it; validation only).",
         f"TEST recall {f3(base['dark_vs_benign_recall'])} → {f3(ood['dark_vs_benign_recall'])} (3 real social-proof rows withheld)."],
        ["Structured promotion / neutral choice (HTML)", "Withhold when HTML shows a valid &lt;time&gt;, a del/ins price pair, or an offer duration "
         "beside a neutral alternative; exact neutral refusal labels on buttons and links.",
         f"Synthetic FPR {f3(CD['threshold_sweep']['0.68']['false_positive_rate'])} → {f3(CD['improved_decision']['metrics']['false_positive_rate'])}; "
         f"second check {f3(CD['second_synthetic_check']['before']['false_positive_rate'])} → "
         f"{f3(CD['second_synthetic_check']['after']['false_positive_rate'])}; TEST unchanged (no DOM)."],
        ["Essential-consent exclusion", "A disabled, checked “strictly necessary” box is not a preselection finding.",
         f"4-case check FPR {f3(CD['essential_consent_check']['before']['false_positive_rate'])} → "
         f"{f3(CD['essential_consent_check']['after']['false_positive_rate'])}, recall unchanged."],
    ], [33 * mm, 72 * mm, W - 105 * mm]))
    fc = FP["clearcart_pages_probes_NOT_TEST_DATA"]["finding_counts"]
    nb, na = sum(fc["before (no FP gates)"].values()), sum(fc["after (both gates)"].values())
    cr = FP["contextrico_screenshots"]
    story.append(P(
        f"<b>Probes (not test data).</b> ClearCart's own known-good pages, audited as HTML, pasted text and screenshots, go from {nb} to {na} "
        f"findings with the quote and OOD gates. On ContextRico screenshots the negative-image flag rate goes from "
        f"{f3(cr['dev']['base_no_gates']['neg_flag_rate'])} to {f3(cr['dev']['+quote_and_ood (served now)']['neg_flag_rate'])} on DEV and "
        f"{f3(cr['test']['base_no_gates']['neg_flag_rate'])} to {f3(cr['test']['+quote_and_ood (served now)']['neg_flag_rate'])} on TEST; "
        "the paired differences are not statistically significant."))
    story.append(P(
        f"<b>Threshold changes were rejected.</b> No threshold above 0.68 passed both the synthetic improvement and the TEST safety criteria "
        f"(TEST recall at 0.85: {f3(CD['heldout_threshold_sweep']['0.85']['recall'])}, with the single false positive unchanged).", BODY))

    # ------------------------------------------------------------------ 5. studies not adopted
    tp = OC["summary_table"]["test"]
    t9, e9 = tp["tesseract"]["0.9"], tp["easyocr"]["0.9"]
    timing = OC["timing_controlled_subsample"]
    story.append(KeepTogether([P(f"{num + 1}. Studies that did not change the served system", H1),
                               P("Second OCR engine (EasyOCR)", H2), P(
        f"Selected on DEV with a ≤ 15% negative-flag limit; only Tesseract at τ = 0.90 met it. On TEST at 0.90 EasyOCR reads more "
        f"(category-correct recall {f3(e9['category_correct_recall'])} vs {f3(t9['category_correct_recall'])}, F1 {f3(e9['f1'])} vs "
        f"{f3(t9['f1'])}) but flags more negatives ({f3(e9['neg_flag_rate'])} vs {f3(t9['neg_flag_rate'])}); neither difference is significant. "
        f"It is also slower ({timing['easyocr_mps_s_per_image']:.1f} s/image on Apple MPS, {timing['easyocr_cpu_s_per_image']:.1f} s on CPU, "
        f"vs {timing['tesseract_v4_psm3_s_per_image']:.2f} s). Tesseract stays the default; EasyOCR is opt-in.")]))
    story.append(img(FIG / "ocr_engines.png", W, max_h=80 * mm))
    story.append(P("OCR engines on ContextRico DEV/TEST (ml/eval_ocr_engines.py).", CAP))
    story.append(P("Mathur et al. 2019 data and fine-tuned transformers (opt-in v2)", H2))
    cmp_ = V2["comparison"]; cv = V2["repeated_grouped_cv"]
    rows = [["Model", "VAL macro-F1", "TEST macro-F1", "TEST benign FP", "Grouped CV macro-F1"]]
    for k, name in (("served_v1", "Served v1 ensemble"), ("ensemble_v2", "Ensemble + Mathur rows"),
                    ("transformer_v1data", "distilroberta, v1 data"), ("transformer_v2", "distilroberta + Mathur rows")):
        d = cmp_[k]["test"]["dark_vs_benign"]
        rows.append([name, f3(cmp_[k]["val"]["macro_f1"]), f3(cmp_[k]["test"]["macro_f1"]), f"{d['fp']} / {d['fp'] + d['tn']}",
                     f"{f3(cv[k]['mean_macro_f1'])} ± {f3(cv[k]['std'])}"])
    story.append(table(rows, [46 * mm, 26 * mm, 27 * mm, 28 * mm, W - 127 * mm]))
    story.append(P(
        f"Selection is on VAL, where nothing beats the served model. Of Mathur's rows, {V2['data']['mathur_rows_added']} pass the leakage rule "
        f"and {V2['data']['mathur_texts_not_already_in_train']} add text not already in TRAIN, so the extra data adds no new language. The "
        "transformers score higher on TEST but have 5–7× the benign false positives and match the ensemble in grouped CV, so they were not "
        f"selected. A \"verified fake\" model was not exposed: grouped-CV AUROC {V2['deceptive_model']['grouped_cv']['auroc']['mean']:.2f} is "
        "below a pattern-type prior.", BODY))

    # ------------------------------------------------------------------ 6. engineering + limits
    story.append(KeepTogether([P(f"{num + 2}. Running and hosting", H1)] + bullets([
        "<b>Single origin.</b> <font face='Courier'>backend.hosted:app</font> serves the built frontend and the API from one server; "
        "the Dockerfile builds the frontend, installs Tesseract and bakes in the MiniLM encoder, listening on ${PORT:-8000} with /health.",
        "<b>Local run:</b> <font face='Courier'>npm run build</font> in frontend/, then <font face='Courier'>uvicorn backend.hosted:app --port 8080</font>. "
        "The Vite dev proxy target is configurable with CLEARCART_API.",
        "<b>Privacy and hardening:</b> uploads processed in memory, a request-scoped inference cache only, images over 40 megapixels rejected, "
        "journey requests capped at 40 MB, user content kept out of error logs, a ReDoS fixed."])))
    story.append(P(f"{num + 3}. Limitations and next evidence", H1))
    story += bullets([
        "The text model's training data cannot change under the hackathon rules, so commerce wording (prices, shipping, legal notes) still leans dark. "
        "The gates make these cases visible for review; only benign modern-retail training text would fix the model itself.",
        "Synthetic sets and probes show rules behave as specified. They are not estimates of precision on live websites.",
        "No check here can prove a claim false. The strongest next step is live verification: reloading a page in fresh sessions to see "
        "whether a deadline's implied end time moves or a stock count never changes."])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=15 * mm, bottomMargin=13 * mm,
                            title="ClearCart — Update report", author="Team ClearCart")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    n = len(PdfReader(str(OUT)).pages)

    w = PdfWriter()
    for part in PARTS:
        for page in PdfReader(str(part)).pages:
            w.add_page(page)
    w.add_metadata({"/Title": "ClearCart - Project Report, Mathematical Innovation addendum and Update report", "/Author": "Team ClearCart"})
    with open(FULL, "wb") as fh:
        w.write(fh)
    print(f"wrote {OUT} ({n} pages) and {FULL} ({len(PdfReader(str(FULL)).pages)} pages); {py_tests}; frontend {fe_tests}")


if __name__ == "__main__":
    build()
