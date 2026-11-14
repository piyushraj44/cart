"""Regression tests for the two false-positive gates (backend/auditor.py, rules in ml/model.py).

Gate 1: quote-aware scoring (a quoted example inside explanatory text is a mention, not a claim).
Gate 2: out-of-distribution abstention (max similarity to TRAIN texts below a validation-chosen threshold).
Plus: with the gates disabled, the reported held-out TEST metrics of the base model are reproduced exactly.
"""
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.auditor import Auditor, ood_config
from backend.extractors import Candidate, segment_text
from ml.common import ARTIFACTS, DATA_PROCESSED, LABELS
from ml.model import (OOD_VAL_DARK_QUANTILE, TextClassifier, max_train_similarity, quote_masked, quoted_spans)

ABOUT_SCARCITY = "'Only 2 left!' or 'selling fast' claims that cannot be verified."
ABOUT_SHAME = "Decline buttons that guilt or shame you, such as 'No, I don't want to save money'."


@pytest.fixture(scope="module")
def clf():
    return TextClassifier.load(ARTIFACTS)


@pytest.fixture(scope="module")
def gated(clf):
    return Auditor(clf)


@pytest.fixture(scope="module")
def ungated(clf):
    return Auditor(clf, quote_gate=False, ood_gate=False, claim_check_gate=False)


def _audit(aud, text):
    return aud.audit(segment_text(text), "text")


# ------------------------------------------------------------------ quote rule (pure function)
@pytest.mark.parametrize("text,n_spans", [
    (ABOUT_SCARCITY, 2), (ABOUT_SHAME, 1),
    ("Don't miss out! It's only 2 left", 0),                    # apostrophes are not quotes
    ("No thanks, I don’t want deals", 0),
    ("Juniors' Shoes and Men's boots", 0),
    ("Here for a limited time. 4.5\" Red Bottle", 0),           # inch mark, no pair
    ("as 'No, | don't want to save money’. Add-ons, insurance", 1),   # OCR: mixed quote marks
    ("Crowd signals such as 'people are viewing this", 1),      # OCR line cut mid-quotation
])
def test_quoted_spans(text, n_spans):
    assert len(quoted_spans(text)) == n_spans


def test_quote_rule_needs_context():
    assert quote_masked(ABOUT_SCARCITY) == "… or … claims that cannot be verified."
    assert quote_masked('"No thanks, I hate saving money"') is None       # entirely a quotation
    assert quote_masked("Hurry, 'only 2 left'!") is None                  # < 4 unquoted words
    assert quote_masked("Hurry! Only 2 left in stock.") is None           # no quotes


# ------------------------------------------------------------------ quote gate in the auditor
def test_quoted_mention_not_flagged(gated, ungated):
    text = ABOUT_SCARCITY + "\n" + ABOUT_SHAME
    before = _audit(ungated, text)
    assert len(before.findings) == 2                     # the reported false positives
    after = _audit(gated, text)
    assert after.findings == [] and after.summary.no_concern
    assert {u.text for u in after.uncertain_elements} == {ABOUT_SCARCITY, ABOUT_SHAME}
    for u in after.uncertain_elements:                   # recorded, not dropped
        assert u.gate == "quoted_mention" and "mention, not a claim" in u.gate_note
        assert u.p_dark >= 0.68 and u.masked_p_dark < 0.68 and u.masked_text
    assert any("mention, not a claim" in w for w in after.warnings)
    # a withheld 99% must not drive the page gauge
    assert after.summary.risk_score < 68


def test_same_words_unquoted_still_flagged(gated):
    for t in ["Only 2 left! or selling fast claims that cannot be verified.",
              "Decline buttons that guilt or shame you, such as No, I don't want to save money."]:
        a = _audit(gated, t)
        assert len(a.findings) == 1 and a.uncertain_elements == [], t


def test_entirely_quoted_or_short_scored_normally(gated):
    for t in ['"No thanks, I don\'t want to save money"', "Hurry, 'only 2 left'!"]:
        a = _audit(gated, t)
        assert len(a.findings) == 1, t


def test_quote_gate_keeps_claim_when_masked_text_is_still_dark(gated):
    a = _audit(gated, 'Hurry! Only 2 left in stock - use code "SAVE10" before it sells out')
    assert len(a.findings) == 1 and a.uncertain_elements == []


# ------------------------------------------------------------------ OOD gate
def test_ood_threshold_follows_validation_rule(clf):
    cfg = ood_config(clf)
    df = pd.read_csv(DATA_PROCESSED / "text_dataset.csv", keep_default_na=False)
    va = df[(df.split == "val") & (df.label != "benign")]
    sims = max_train_similarity(clf.b, clf.embed(va.text.tolist()))
    below = int((sims < cfg["threshold"]).sum())
    assert below <= OOD_VAL_DARK_QUANTILE * len(sims) < below + 1    # largest value with <= 2% strictly below
    saved = json.loads((ARTIFACTS / "false_positive_eval.json").read_text())["rules"]["ood_threshold"]["threshold"]
    assert cfg["threshold"] == pytest.approx(saved, abs=1e-6)
    assert len(clf.b.neighbor_texts) == int((df.split == "train").sum())  # the bank is the TRAIN split


def test_ood_element_abstained(gated, ungated):
    # a character counter from ClearCart's own audit page: 72% "urgency", similarity 0.34 to any training text
    t = "0 / 200,000 characters"
    assert len(_audit(ungated, t).findings) == 1
    a = _audit(gated, t)
    assert a.findings == []
    (u,) = a.uncertain_elements
    assert u.gate == "out_of_distribution" and u.max_train_similarity < u.ood_threshold
    assert "unlike the training data" in u.gate_note
    assert any("unlike anything in the training data" in w for w in a.warnings)


@pytest.mark.parametrize("t", ["Hurry! Only 2 left in stock.", "12 people are viewing this right now",
                               "No thanks, I don't want to save money", "Sale ends in 02:14:09 - order now!"])
def test_in_distribution_dark_text_still_flagged(gated, t):
    a = _audit(gated, t)
    assert len(a.findings) == 1 and a.uncertain_elements == [], t


def test_ood_applies_to_ocr_candidates(gated):
    c = Candidate(text="0 / 200,000 characters", source_type="screenshot_ocr", bbox={"x": 0, "y": 0, "width": 10, "height": 10},
                  extra={"unit": "line"})
    a = gated.audit([c], "screenshot_ocr", threshold=0.7)
    assert a.findings == [] and a.uncertain_elements[0].gate == "out_of_distribution"


def test_gates_never_add_findings(gated, ungated):
    df = pd.read_csv(DATA_PROCESSED / "text_dataset.csv", keep_default_na=False)
    texts = df[df.split == "test"].text.tolist()[:120]
    cands = [Candidate(text=t, source_type="text") for t in texts]
    on = {f.evidence for f in gated.audit(cands, "text").findings}
    off = {f.evidence for f in ungated.audit(cands, "text").findings}
    assert on <= off


# ------------------------------------------------------------------ base model metrics unchanged with gates off
def test_reported_test_metrics_reproduced_with_gates_disabled(clf, ungated):
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    rep = json.loads((ARTIFACTS / "metrics.json").read_text())["final_test"]
    df = pd.read_csv(DATA_PROCESSED / "text_dataset.csv", keep_default_na=False)
    te = df[df.split == "test"].reset_index(drop=True)
    preds = clf.predict(te.text.tolist())
    for p in preds:
        p["flagged"] = p["p_dark"] >= clf.b.threshold
    ungated.apply_gates([Candidate(text=t, source_type="text") for t in te.text], preds, clf.b.threshold)
    assert not any(p.get("gate") for p in preds)          # disabled gates change nothing
    y = te.label.map(LABELS.index).values
    pred = np.array([LABELS.index(p["category"]) if p["flagged"] else 0 for p in preds])
    assert f1_score(y, pred, average="macro") == pytest.approx(rep["application_metrics"]["macro_f1"], abs=5e-5)
    p, r, f, _ = precision_recall_fscore_support((y != 0).astype(int), (pred != 0).astype(int), average="binary")
    b = rep["binary_dark_vs_benign"]
    assert (round(p, 4), round(r, 4), round(f, 4)) == (b["precision"], b["recall"], b["f1"])
    assert int(((y == 0) & (pred != 0)).sum()) == b["fp"] == 1


def test_eval_artifact_reports_recall_cost():
    m = json.loads((ARTIFACTS / "false_positive_eval.json").read_text())["test_split"]
    assert m["base_reproduces_metrics_json"]
    base, now = m["metrics"]["base_no_gates"], m["metrics"]["+quote_and_ood (served now)"]
    assert now["dark_vs_benign_recall"] <= base["dark_vs_benign_recall"]
    assert now["benign_false_positives"] <= base["benign_false_positives"]


# ------------------------------------------------------------------ API: backward compatible schema
def test_api_response_has_optional_uncertain_elements():
    from backend.main import app
    with TestClient(app) as c:
        j = c.post("/api/audit/text", json={"text": ABOUT_SCARCITY + "\nHurry! Only 2 left in stock."}).json()
    assert [f["evidence"] for f in j["findings"]] == ["Hurry! Only 2 left in stock."]
    assert j["uncertain_elements"][0]["gate"] == "quoted_mention"
    for k in ("summary", "findings", "benign_elements", "warnings", "disclaimer"):
        assert k in j
    assert all(e.get("gate") is None for e in j["benign_elements"])
