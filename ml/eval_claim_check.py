"""Paired evaluation of the claim-specificity gate (backend/claim_check.py).

Run: .venv/bin/python -m ml.eval_claim_check

Acceptance criteria, fixed BEFORE any score from this script was inspected:
  1. Held-out grouped TEST split (329 rows): recall may drop by at most 5 percentage points, and benign
     false positives must not increase. TEST is used only as a safety veto, never to tune the rules.
  2. Synthetic sets: precision must not fall and FPR must not rise on the exploratory set
     (ml/eval_conservative_detection.CASES, which the rules were written against) and on FRESH_CASES below.

FRESH_CASES were written after the rules in backend/claim_check.py were fixed and before they were scored.
They are invented text, not real websites: a pass shows the rule behaves as specified, not field precision.
Positive label = the fixture states a vague or unverifiable pressure claim.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from backend.auditor import Auditor
from backend.extractors import Candidate
from ml.common import ARTIFACTS, DATA_PROCESSED
from ml.eval_conservative_detection import CASES as EXPLORATORY, metrics
from ml.model import TextClassifier

OUT = Path(__file__).parent / "artifacts" / "claim_check_eval.json"
MAX_RECALL_DROP = 0.05

FRESH_CASES = [
    ("fb01", 0, "Black Friday: 25% off all jackets until 1 December"),
    ("fb02", 0, "Was ₹1,499, now ₹999"),
    ("fb03", 0, "Free returns within 30 days"),
    ("fb04", 0, "Use code AUTUMN10 for 10% off at checkout"),
    ("fb05", 0, "Offer valid until 15 December 2026"),
    ("fb06", 0, "Free delivery on orders over £40"),
    ("fb07", 0, "Regular price $120. Sale price $90."),
    ("fb08", 0, "Dispatched within 2 working days"),
    ("fb09", 0, "Buy 2, get the third free - ends 6 January"),
    ("fb10", 0, "MRP ₹2,499. You save ₹500."),
    ("fb11", 0, "Winter sale ends Sunday at 23:59"),
    ("fb12", 0, "Members get 15% off, valid to 31/12/2026"),
    ("fd01", 1, "Hurry, sale ends soon!"),
    ("fd02", 1, "Up to 70% off - limited time only"),
    ("fd03", 1, "Only 4 left at this price"),
    ("fd04", 1, "Deal ends in 00:09:59"),
    ("fd05", 1, "Selling fast - 40% off today only"),
    ("fd06", 1, "Last chance to save ₹500!"),
    ("fd07", 1, "Don't miss out: 30% off ends tonight"),
    ("fd08", 1, "Almost gone! Was $80, now $40"),
    ("fd09", 1, "18 people have this in their cart"),
    ("fd10", 1, "Grab it before it's gone - 20% off until 30 November"),
    ("fd11", 1, "Act now, price goes up in 2 hours"),
    ("fd12", 1, "Offer expired 1 January 2020 - still claim 50% off now"),
]


def _cand(text: str) -> Candidate:
    return Candidate(text=text, source_type="text", char_start=0, char_end=len(text))


def _flags(clf, auditor: Auditor, texts: list[str], tau: float) -> tuple[list[bool], list[dict]]:
    cands = [_cand(t) for t in texts]
    preds = clf.predict(texts)
    for p in preds:
        p["flagged"] = p["p_dark"] >= tau and not p.get("abstained", False)
    auditor.apply_gates(cands, preds, tau)
    return [p["flagged"] for p in preds], preds


def _paired(clf, before: Auditor, after: Auditor, rows: list[tuple[str, int, str]], tau: float) -> dict:
    texts, y = [r[2] for r in rows], [r[1] for r in rows]
    fb, _ = _flags(clf, before, texts, tau)
    fa, pa = _flags(clf, after, texts, tau)
    return {
        "before": metrics(y, fb), "after": metrics(y, fa),
        "cases": [{"id": r[0], "label": r[1], "text": r[2], "p_dark": round(p["p_dark"], 4), "category": p["category"],
                   "flagged_before": b, "flagged_after": a, "gate_after": p.get("gate")}
                  for r, p, b, a in zip(rows, pa, fb, fa)],
    }


def evaluate() -> dict:
    clf = TextClassifier.load(ARTIFACTS)
    tau = clf.b.threshold
    before = Auditor(clf, claim_check_gate=False)
    after = Auditor(clf, claim_check_gate=True)

    exploratory = [(c[0], c[1], c[2]) for c in EXPLORATORY if c[3] is None]   # pasted-text cases only
    test = pd.read_csv(DATA_PROCESSED / "text_dataset.csv", keep_default_na=False)
    test = test[test.split == "test"].reset_index(drop=True)
    test_rows = [(f"t{i}", int(lbl != "benign"), str(t)) for i, (t, lbl) in enumerate(zip(test.text, test.label))]

    res = {"threshold": tau,
           "exploratory_text": _paired(clf, before, after, exploratory, tau),
           "fresh": _paired(clf, before, after, FRESH_CASES, tau),
           "test": _paired(clf, before, after, test_rows, tau)}
    res["test"]["changed"] = [c for c in res["test"]["cases"] if c["flagged_before"] != c["flagged_after"]]
    res["test"].pop("cases")
    tb, ta = res["test"]["before"], res["test"]["after"]
    res["criteria"] = {
        "test_recall_drop_ok": tb["recall"] - ta["recall"] <= MAX_RECALL_DROP + 1e-9,
        "test_fp_not_increased": ta["fp"] <= tb["fp"],
        **{f"{k}_precision_not_lower": res[k]["after"]["precision"] >= res[k]["before"]["precision"] - 1e-9
           for k in ("exploratory_text", "fresh")},
        **{f"{k}_fpr_not_higher": res[k]["after"]["false_positive_rate"] <= res[k]["before"]["false_positive_rate"] + 1e-9
           for k in ("exploratory_text", "fresh")},
    }
    res["criteria"]["all_pass"] = all(res["criteria"].values())
    return res


def main() -> None:
    res = evaluate()
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    for k in ("exploratory_text", "fresh", "test"):
        print(k, "before", res[k]["before"], "\n", " " * len(k), "after ", res[k]["after"])
    print("criteria", res["criteria"])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
