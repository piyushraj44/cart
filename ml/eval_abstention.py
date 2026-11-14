"""Measure the abstention gate (no words -> no classification) on the held-out TEST split.

The gate was added after a real-world false positive: price rows such as "₹ 2,990.00 ₹ 3,950.00"
scored ~0.95 "urgency". This script reports exactly what the gate changes on the original held-out
evaluation, with the served model and decision rule. It does NOT overwrite metrics.json.

Run: python -m ml.eval_abstention
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

import ml  # noqa: F401
from ml.common import ARTIFACTS, DATA_PROCESSED, LABELS
from ml.model import TextClassifier, has_lexical_evidence

PROBES = ["₹ 3,950.00", "₹ 2,990.00 ₹ 3,950.00", "£3.950.00 £39500 +2.980.00 +2.95000 +395000 +3:950.00",
          "$49.99", "4.6 (12,842 reviews)", "00:12:47", "OFFER ENDS 00:54:59", "1 day 08:15:25", "62% OFF"]


def app_pred(preds, tau, gate: bool) -> np.ndarray:
    out = []
    for p in preds:
        flagged = p["p_dark"] >= tau and not (gate and p["abstained"])
        out.append(LABELS.index(p["category"]) if flagged else LABELS.index("benign"))
    return np.array(out)


def block(y, pred) -> dict:
    yb, pb = (y != 0).astype(int), (pred != 0).astype(int)
    p, r, f, _ = precision_recall_fscore_support(yb, pb, average="binary", zero_division=0)
    return {"macro_f1": round(float(f1_score(y, pred, average="macro")), 4), "dark_vs_benign_precision": round(float(p), 4),
            "dark_vs_benign_recall": round(float(r), 4), "dark_vs_benign_f1": round(float(f), 4),
            "benign_false_positives": int(((yb == 0) & (pb == 1)).sum()), "missed_dark": int(((yb == 1) & (pb == 0)).sum())}


def main() -> None:
    clf = TextClassifier.load(ARTIFACTS)
    tau = clf.b.threshold
    df = pd.read_csv(DATA_PROCESSED / "text_dataset.csv", keep_default_na=False)
    gated_all = df[~df.text.map(has_lexical_evidence)]
    te = df[df.split == "test"].reset_index(drop=True)
    y = te.label.map(LABELS.index).values
    preds = clf.predict(te.text.tolist())
    gated = [i for i, p in enumerate(preds) if p["abstained"]]
    probe_preds = clf.predict(PROBES)
    res = {
        "definition": "abstain when the text has no alphabetic word of >= 2 letters (currency codes like Rs/INR/USD do not count)",
        "threshold_tau": tau,
        "rows_without_words_all_splits": {k: int(v) for k, v in gated_all.label.value_counts().items()},
        "rows_without_words_all_splits_examples": gated_all.text.head(12).tolist(),
        "test_rows_gated": [{"text": te.text[i], "label": te.label[i], "p_dark": round(preds[i]["p_dark"], 4),
                             "was_flagged_before": bool(preds[i]["p_dark"] >= tau)} for i in gated],
        "test_without_gate": block(y, app_pred(preds, tau, False)),
        "test_with_gate": block(y, app_pred(preds, tau, True)),
        "probes": [{"text": t, "p_dark": round(p["p_dark"], 4), "category": p["category"], "abstained": p["abstained"],
                    "flagged_now": p["flagged"]} for t, p in zip(PROBES, probe_preds)],
        "note": "The original evaluation (metrics.json) is unchanged; this file reports the effect of the post-hoc gate.",
    }
    (ARTIFACTS / "abstention_eval.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("rows_without_words_all_splits", "test_rows_gated", "test_without_gate", "test_with_gate")}, indent=1, ensure_ascii=False))
    for p in res["probes"]:
        print(p)


if __name__ == "__main__":
    main()
