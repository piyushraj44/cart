"""Measure the two false-positive gates (quote-aware scoring, out-of-distribution abstention).

Both gates are post-hoc and application-level (backend/auditor.py, rules in ml/model.py). The served model,
its probabilities and ml/artifacts/metrics.json are NOT changed; this script reports what the gates change:

  1. tau_ood: chosen on the VALIDATION split only (rule in ml/model.py); sensitivity of other quantiles shown.
  2. Frozen TEST split (329 rows), each row = one element: macro-F1, dark-vs-benign P/R/F1, benign FPR for
     base (no gates; must reproduce metrics.json), the existing no-words gate, + quote, + OOD, + both.
  3. Qualitative PROBES (known-good text, NOT test data): ClearCart's own pages rendered with react-dom/server
     (frontend/scripts/render_static_pages.mjs -> data/probes/clearcart_pages/*.html), audited as HTML and as
     pasted visible text through the API; the user-reported sentences; TEST-split benign rows.
  4. Screenshots: headless-Chrome captures of the About page audited through POST /api/audit/screenshot; and the
     ContextRico DEV/TEST protocol of ml/eval_screenshots.py (cached v4_psm3 OCR, tau 0.90) before/after.

Run: python -m ml.eval_false_positives   (writes ml/artifacts/false_positive_eval.json)
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

import ml  # noqa: F401
from ml.common import ARTIFACTS, DATA_PROCESSED, LABELS, ROOT
from ml.model import OOD_VAL_DARK_QUANTILE, TextClassifier, max_train_similarity, quote_masked

PAGES = ROOT / "data" / "probes" / "clearcart_pages"
OUT = ARTIFACTS / "false_positive_eval.json"
OCR_TAU = 0.90                     # backend.main.OCR_THRESHOLD (selected on ContextRico DEV)
USER_PROBES = [  # reported by the user / reproduced by the lead (About page text)
    "'Only 2 left!' or 'selling fast' claims that cannot be verified.",
    "Decline buttons that guilt or shame you, such as 'No, I don't want to save money'.",
    "HTML only",
]
# (word_gate, quote_gate, ood_gate)
VARIANTS = {
    "base_no_gates": (False, False, False),
    "served_before_fix (no-words gate)": (True, False, False),
    "+quote_gate": (True, True, False),
    "+ood_gate": (True, False, True),
    "+quote_and_ood (served now)": (True, True, True),
}


def _auditor(clf, quote: bool, ood: bool):
    from backend.auditor import Auditor
    return Auditor(clf, quote_gate=quote, ood_gate=ood)


def gated_decisions(clf, texts: list[str], preds: list[dict], tau: float, variant: tuple) -> list[dict]:
    """Final per-element decision for one variant: {'flagged', 'gate', ...} (uses the Auditor's own gate code)."""
    from backend.extractors import Candidate
    word, quote, ood = variant
    ps = [dict(p) for p in preds]
    for p in ps:
        p["flagged"] = p["p_dark"] >= tau and not (word and p["abstained"])
        p.pop("gate", None)
    _auditor(clf, quote, ood).apply_gates([Candidate(text=t, source_type="text") for t in texts], ps, tau)
    return ps


def block(y: np.ndarray, flagged: np.ndarray, cats: list[str]) -> dict:
    pred = np.array([LABELS.index(c) if f else 0 for f, c in zip(flagged, cats)])
    yb, pb = (y != 0).astype(int), (pred != 0).astype(int)
    p, r, f, _ = precision_recall_fscore_support(yb, pb, average="binary", zero_division=0)
    nb = int((yb == 0).sum())
    fp = int(((yb == 0) & (pb == 1)).sum())
    return {"macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
            "dark_vs_benign_precision": round(float(p), 4), "dark_vs_benign_recall": round(float(r), 4),
            "dark_vs_benign_f1": round(float(f), 4), "benign_false_positives": fp, "benign_n": nb,
            "benign_fpr": round(fp / nb, 4), "dark_detected": int(((yb == 1) & (pb == 1)).sum()),
            "missed_dark": int(((yb == 1) & (pb == 0)).sum()), "dark_n": int(yb.sum())}


def split_eval(clf, df: pd.DataFrame, tau: float) -> dict:
    texts = df.text.tolist()
    y = df.label.map(LABELS.index).values
    preds = clf.predict(texts)
    sims = max_train_similarity(clf.b, np.stack([p["_emb"] for p in preds]))
    out, changed = {}, {}
    base = None
    for name, v in VARIANTS.items():
        d = gated_decisions(clf, texts, preds, tau, v)
        fl = np.array([p["flagged"] for p in d])
        out[name] = block(y, fl, [p["category"] for p in d])
        if base is None:
            base = fl
        else:
            changed[name] = [{"text": texts[i], "label": df.label.iloc[i], "p_dark": round(d[i]["p_dark"], 4),
                              "category": d[i]["category"], "gate": d[i].get("gate") or ("no_words" if d[i]["abstained"] else None),
                              "max_train_similarity": round(float(sims[i]), 4),
                              "masked_p_dark": round(d[i]["masked_p_dark"], 4) if "masked_p_dark" in d[i] else None}
                             for i in np.where(base & ~fl)[0]]
    return {"metrics": out, "rows_unflagged_vs_base": changed,
            "quote_rule_applies_to_rows": int(sum(quote_masked(t) is not None for t in texts))}


def ood_section(clf) -> dict:
    df = pd.read_csv(DATA_PROCESSED / "text_dataset.csv", keep_default_na=False)
    va = df[df.split == "val"].reset_index(drop=True)
    _, emb = clf.raw_proba(va.text.tolist())
    sims = max_train_similarity(clf.b, emb)
    dark = np.sort(sims[(va.label != "benign").values])
    ben = sims[(va.label == "benign").values]
    sens = {}
    for q in (0.01, 0.02, 0.05):
        k = int(np.floor(q * len(dark)))
        t = float(dark[k])
        sens[f"{q:.0%}"] = {"threshold": round(t, 4), "val_dark_below": int((dark < t).sum()),
                            "val_benign_below": int((ben < t).sum())}
    return {"val_dark_quantiles": {str(q): round(float(np.quantile(dark, q)), 4) for q in (0, .01, .02, .05, .1, .5)},
            "val_benign_quantiles": {str(q): round(float(np.quantile(ben, q)), 4) for q in (0, .01, .02, .05, .1, .5)},
            "val_dark_lowest10": [round(float(s), 4) for s in dark[:10]],
            "sensitivity_not_used_for_selection": sens}


def visible_text(html: str) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "svg"]):
        t.decompose()
    return "\n".join(s.strip() for s in soup.get_text("\n").split("\n") if s.strip())


def _summ(j: dict) -> dict:
    return {"elements_analysed": j["summary"]["elements_analysed"], "risk_score": j["summary"]["risk_score"],
            "no_concern": j["summary"]["no_concern"],
            "findings": [{"evidence": f["evidence"], "category": f["category"], "confidence_pct": f["confidence_pct"],
                          "detection_method": f["detection_method"],
                          "nearest_train_similarity": f["similar_examples"][0]["similarity"] if f["similar_examples"] else None}
                         for f in j["findings"]],
            "uncertain_elements": [{"text": u["text"], "gate": u["gate"], "p_dark": u["p_dark"], "masked_p_dark": u["masked_p_dark"],
                                    "max_train_similarity": u["max_train_similarity"]} for u in j.get("uncertain_elements", [])]}


def api_probes(clf) -> dict:
    """Audit ClearCart's own pages through the real API with gates off (before) and on (after)."""
    from fastapi.testclient import TestClient

    from backend.main import STATE, app
    res = {}
    with TestClient(app) as c:
        served = STATE["auditor"]
        for label, (q, o) in {"before (no FP gates)": (False, False), "after (both gates)": (True, True)}.items():
            STATE["auditor"] = _auditor(STATE["clf"], q, o)
            r = {}
            for page in ("about", "model", "workspace"):
                html = (PAGES / f"{page}.html").read_text()
                r[f"{page}.html as HTML"] = _summ(c.post("/api/audit/html", data={"html": html}).json())
                r[f"{page}.html as pasted visible text"] = _summ(c.post("/api/audit/text", json={"text": visible_text(html)}).json())
            r["user-reported sentences as pasted text"] = _summ(c.post("/api/audit/text", json={"text": "\n".join(USER_PROBES)}).json())
            for shot in ("about_1280_viewport.png", "about_1440.png"):
                with open(PAGES / shot, "rb") as fh:
                    r[f"screenshot {shot}"] = _summ(c.post("/api/audit/screenshot", files={"file": (shot, fh, "image/png")}).json())
            res[label] = r
        STATE["auditor"] = served
    return res


def flag_counts(res: dict) -> dict:
    return {k: {src: len(v["findings"]) for src, v in r.items()} for k, r in res.items()}


def contextrico(clf) -> dict:
    from ml import eval_screenshots as es
    items, _ = es.build_eval_set()
    ocr = {it["image"]: es._ocr_one(it, "v4_psm3") for it in items}        # cached OCR (no re-run)
    texts = sorted({c["text"] for rec in ocr.values() for c in rec["candidates"]})
    preds = []
    for i in range(0, len(texts), 512):
        preds += clf.predict(texts[i:i + 512])
    out = {"protocol": "ml/eval_screenshots.py: v4_psm3 OCR (cached), image flagged if any OCR line/block has p_dark >= 0.90; "
                       "gated or abstained candidates count as not flagged (p_dark set to 0 for that candidate)",
           "n_unique_ocr_texts": len(texts)}
    variant_preds = {}
    for name, v in VARIANTS.items():
        d = dict(zip(texts, gated_decisions(clf, texts, preds, OCR_TAU, v)))
        sc = {t: {"p_dark": p["p_dark"] if p["flagged"] or p["p_dark"] < OCR_TAU else 0.0, "category": p["category"]} for t, p in d.items()}
        variant_preds[name] = es.image_preds(items, ocr, sc)
        out.setdefault("gated_ocr_texts_above_tau", {})[name] = sum(1 for p in d.values() if p["p_dark"] >= OCR_TAU and not p["flagged"])
    for split in ("dev", "test"):
        sub = [i for i in items if i["split"] == split]
        out[split] = {name: {k: m[k] for k in ("n_pos", "n_neg", "detection_recall", "category_correct_recall", "neg_flag_rate",
                                               "neg_flag_rate_ci95", "fp", "tp", "precision", "f1", "per_category_recall")}
                      for name, m in ((n, es.metrics(sub, variant_preds[n], OCR_TAU, ocr)) for n in VARIANTS)}
        out[split + "_paired_base_vs_both"] = es.paired(sub, variant_preds["base_no_gates"], OCR_TAU,
                                                        variant_preds["+quote_and_ood (served now)"], OCR_TAU)
        # effect of the two NEW gates alone (the no-words gate is on in both arms)
        out[split + "_paired_served_before_vs_now"] = es.paired(sub, variant_preds["served_before_fix (no-words gate)"], OCR_TAU,
                                                                variant_preds["+quote_and_ood (served now)"], OCR_TAU)
    # which DEV/TEST images the NEW gates changed (served before fix -> served now), and the OCR text responsible
    changes = []
    ref = "served_before_fix (no-words gate)"
    for it in items:
        b = es.flagged_cats(variant_preds[ref][it["image"]], OCR_TAU)
        a = es.flagged_cats(variant_preds["+quote_and_ood (served now)"][it["image"]], OCR_TAU)
        if b != a:
            lost = [c["text"] for c, c0 in zip(variant_preds["+quote_and_ood (served now)"][it["image"]], variant_preds[ref][it["image"]])
                    if c0["p_dark"] >= OCR_TAU and c["p_dark"] < OCR_TAU]
            changes.append({"image": it["image"], "split": it["split"], "is_positive": it["is_pos"], "gt_cats": it["gt_cats"],
                            "flags_before": sorted(b), "flags_after": sorted(a), "unflagged_texts": lost[:4]})
    out["images_changed_by_new_gates"] = changes
    return out


def main() -> None:
    clf = TextClassifier.load(ARTIFACTS)
    tau = float(clf.b.threshold)
    from backend.auditor import ood_config
    ood = ood_config(clf)
    df = pd.read_csv(DATA_PROCESSED / "text_dataset.csv", keep_default_na=False)
    te = df[df.split == "test"].reset_index(drop=True)
    va = df[df.split == "val"].reset_index(drop=True)
    reported = json.loads((ARTIFACTS / "metrics.json").read_text())
    test = split_eval(clf, te, tau)
    print(json.dumps(test["metrics"], indent=1))
    up = clf.predict(USER_PROBES)
    sims = max_train_similarity(clf.b, np.stack([p["_emb"] for p in up]))
    print("[probes] API audits of ClearCart pages ...", flush=True)
    api = api_probes(clf)
    print(json.dumps(flag_counts(api), indent=1))
    print("[contextrico] ...", flush=True)
    cr = contextrico(clf)
    for s in ("dev", "test"):
        for n, m in cr[s].items():
            print(f"  {s.upper()} {n:36s} neg_flag={m['neg_flag_rate']:.3f} det={m['detection_recall']:.3f} "
                  f"cat={m['category_correct_recall']:.3f} P={m['precision']:.3f} F1={m['f1']:.3f}")
    res = {
        "description": __doc__.strip().splitlines()[0],
        "model": "served v1 (ml/artifacts/clearcart_text_model.joblib), unchanged; text threshold tau = %.2f" % tau,
        "rules": {
            "quote_gate": ("For an element that crossed the threshold: if it contains >= 1 quoted span (\"..\", “..”, ‘..’, '..'; "
                           "mixed open/close marks allowed; an unclosed quote running to the end of the element counts; in-word "
                           "apostrophes never open/close) AND >= 4 words of >= 2 letters remain after masking, score the text with "
                           "every quoted span replaced by '…'. If the masked score is below the threshold, the element is not a "
                           "finding; it is listed in uncertain_elements as 'quoted example — treated as a mention, not a claim'. "
                           "Entirely-quoted or short elements are scored normally."),
            "quote_placeholder_selection": ("'…' chosen on VALIDATION: appending each of 9 candidate placeholders to the 328 val "
                                            "texts, '…' moved P(dark) least (mean |Δ| 0.0011, 0 decision flips)."),
            "quote_rule_revision": ("First version required matching quote characters and a closing quote. On the About-page "
                                    "screenshot probe the OCR text had mismatched (' … ’) and cut-off quotes, so the rule was "
                                    "generalised to mixed marks and end-of-element unclosed quotes. This revision was made after "
                                    "looking at that probe (not at the TEST split); the TEST numbers below use the final rule."),
            "ood_gate": ("max cosine similarity (MiniLM, L2-normalised) to the 1,645 TRAIN texts in the neighbour bank; an element "
                         "that crossed the threshold is withheld as 'uncertain — unlike the training data' if similarity < tau_ood."),
            "ood_threshold": ood,
            "ood_threshold_quantile": OOD_VAL_DARK_QUANTILE,
            "ood_threshold_selection": ("VALIDATION split only, fixed before any TEST evaluation: 2% (upper end of the 1–2% range) "
                                        "of validation DARK rows may fall below tau_ood. Applied unchanged to text, HTML and OCR."),
            "ood_validation_distribution": ood_section(clf),
            "order": "no-words gate (existing) -> quote gate -> OOD gate; only elements with p_dark >= threshold are ever gated",
        },
        "test_split": {**test, "n_rows": len(te),
                       "reported_in_metrics_json": {"macro_f1": reported["final_test"]["application_metrics"]["macro_f1"], **{f"dark_vs_benign_{k}": reported["final_test"]["binary_dark_vs_benign"][k] for k in ("precision", "recall", "f1", "fp", "false_positive_rate")}},
                       "base_reproduces_metrics_json": bool(abs(test["metrics"]["base_no_gates"]["macro_f1"] - reported["final_test"]["application_metrics"]["macro_f1"]) < 1e-3 and test["metrics"]["base_no_gates"]["benign_false_positives"] == reported["final_test"]["binary_dark_vs_benign"]["fp"])},
        "val_split_for_reference_used_for_selection": split_eval(clf, va, tau)["metrics"],
        "user_probes": [{"text": t, "p_dark": round(p["p_dark"], 4), "category": p["category"],
                         "max_train_similarity": round(float(s), 4), "quote_masked": quote_masked(t)} for t, p, s in zip(USER_PROBES, up, sims)],
        "clearcart_pages_probes_NOT_TEST_DATA": {"note": ("Known-good text: ClearCart's own pages rendered by "
                                                          "frontend/scripts/render_static_pages.mjs; screenshots by headless Chrome. "
                                                          "Every finding here is a false positive."),
                                                 "finding_counts": flag_counts(api), "details": api},
        "contextrico_screenshots": cr,
        "note": "metrics.json and all other artefacts are unchanged; this file only reports the post-hoc gates.",
    }
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
