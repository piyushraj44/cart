import json
import numpy as np
import pandas as pd
import pytest

from ml.common import ARTIFACTS, DATA_PROCESSED, LABELS
from ml.model import TextClassifier


@pytest.fixture(scope="module")
def clf():
    return TextClassifier.load(ARTIFACTS)


TEXTS = ["Hurry, only a few left!", "Add to cart", "LAST CHANCE - ENDS IN: 20 HRS", "No, I'll rather pay full price.",
         "45 added to bag in the past 24 hours", "Shipping policy"]


def test_artifacts_exist():
    assert (ARTIFACTS / "clearcart_text_model.joblib").exists()
    assert (ARTIFACTS / "metrics.json").exists()
    assert (ARTIFACTS / "minilm").is_dir()


def test_prediction_invariants(clf):
    for p in clf.predict(TEXTS):
        assert abs(sum(p["probabilities"].values()) - 1) < 1e-6
        assert 0 <= p["p_dark"] <= 1
        assert p["flagged"] == (p["p_dark"] >= clf.b.threshold)
        assert p["category"] in LABELS and p["category"] != "benign"


def test_deterministic(clf):
    a = clf.predict(TEXTS); b = clf.predict(TEXTS)
    for x, y in zip(a, b):
        assert x["p_dark"] == pytest.approx(y["p_dark"], abs=1e-9)
        assert x["category"] == y["category"]


def test_empty(clf):
    assert clf.predict([]) == []


def test_key_terms_in_text(clf):
    for t in TEXTS:
        for cat in ["urgency", "scarcity", "social_proof", "misdirection"]:
            for term in clf.key_terms(t, cat):
                assert term.lower() in t.lower()


def test_neighbors(clf):
    p = clf.predict(["Only 2 left in stock"])[0]
    nb = clf.neighbors(p["_emb"])
    assert len(nb) == 2
    train = pd.read_csv(DATA_PROCESSED / "text_dataset.csv")
    train = train[train.split == "train"]
    train_texts = set(train["text"])
    for n in nb:
        assert n["label"] in LABELS
        assert -1.01 <= n["similarity"] <= 1.01
        assert n["text"] in train_texts


def test_saved_model_matches_reported_test_f1(clf):
    from sklearn.metrics import f1_score
    df = pd.read_csv(DATA_PROCESSED / "text_dataset.csv")
    te = df[df.split == "test"]
    preds = clf.predict(te["text"].tolist())
    y_pred = [p["category"] if p["flagged"] else "benign" for p in preds]
    f1 = f1_score(te["label"], y_pred, labels=LABELS, average="macro")
    ref = json.loads((ARTIFACTS / "metrics.json").read_text())["final_test"]["application_metrics"]["macro_f1"]
    assert abs(f1 - ref) <= 0.002, f"recomputed {f1:.4f} vs metrics.json {ref}"
