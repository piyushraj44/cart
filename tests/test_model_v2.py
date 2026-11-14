"""Opt-in v2 model (ml/artifacts_v2, selected on VAL by ml/train_v2.py).

Checks that the v2 loader keeps TextClassifier's predict() contract, that the metrics saved in
ml/artifacts_v2/metrics.json are reproduced by re-scoring the frozen TEST split, and that the API
serves v2 only when CLEARCART_MODEL=v2 (the default stays the v1 model).
"""
import json

import numpy as np
import pandas as pd
import pytest

from ml.common import DATA_PROCESSED, LABELS, ROOT
from ml.model import load_classifier

V2 = ROOT / "ml" / "artifacts_v2"
pytestmark = pytest.mark.skipif(not (V2 / "selected.json").exists(), reason="v2 artefacts not built (run python -m ml.train_v2)")

TEXTS = ["Hurry, only a few left!", "Add to cart", "LAST CHANCE - ENDS IN: 20 HRS", "No, I'll rather pay full price.",
         "45 added to bag in the past 24 hours", "Shipping policy", "₹ 2,990.00 ₹ 3,950.00"]
CONTRACT_KEYS = {"text", "probabilities", "p_dark", "category", "category_share", "flagged", "abstained", "_emb"}


@pytest.fixture(scope="module")
def selected():
    return json.loads((V2 / "selected.json").read_text())["selected_model"]


@pytest.fixture(scope="module")
def clf2():
    return load_classifier("v2")


@pytest.fixture(scope="module")
def v2_metrics():
    return json.loads((V2 / "metrics.json").read_text())


def test_v2_artifacts_exist(selected):
    assert selected in ("ensemble_v2", "transformer_v2", "transformer_v1data")
    assert (V2 / "metrics.json").exists()


def test_v2_predict_contract(clf2):
    v1 = load_classifier("")
    p1, p2 = v1.predict(TEXTS), clf2.predict(TEXTS)
    assert len(p2) == len(TEXTS)
    for a, b in zip(p1, p2):
        assert set(b) == set(a) == CONTRACT_KEYS
        assert list(b["probabilities"]) == LABELS
        assert abs(sum(b["probabilities"].values()) - 1) < 1e-6
        assert 0 <= b["p_dark"] <= 1 and 0 <= b["category_share"] <= 1 + 1e-9
        assert b["category"] in LABELS and b["category"] != "benign"
        assert b["flagged"] == (b["p_dark"] >= clf2.b.threshold and not b["abstained"])
        assert isinstance(b["_emb"], np.ndarray) and b["_emb"].ndim == 1
    assert p2[-1]["abstained"] and not p2[-1]["flagged"]          # the abstention gate is shared
    assert clf2.predict([]) == []


def test_v2_deterministic(clf2):
    a, b = clf2.predict(TEXTS), clf2.predict(TEXTS)
    for x, y in zip(a, b):
        assert x["p_dark"] == pytest.approx(y["p_dark"], abs=1e-6)
        assert x["category"] == y["category"]


def test_v2_key_terms_and_neighbors(clf2):
    for t in TEXTS[:6]:
        for cat in ["urgency", "scarcity", "social_proof", "misdirection"]:
            terms = clf2.key_terms(t, cat)
            assert len(terms) <= 4
            for term in terms:
                assert term.lower() in t.lower()
    p = clf2.predict(["Only 2 left in stock"])[0]
    nb = clf2.neighbors(p["_emb"])
    assert len(nb) == 2
    df = pd.read_csv(DATA_PROCESSED / "text_dataset.csv")
    mr = pd.read_csv(DATA_PROCESSED / "mathur_rows.csv", dtype=str, keep_default_na=False)
    train_texts = set(df.loc[df.split == "train", "text"]) | set(mr.loc[mr.eligible == "True", "text"])
    heldout = set(df.loc[df.split != "train", "text"])
    for n in nb:
        assert n["label"] in LABELS and -1.01 <= n["similarity"] <= 1.01
        assert n["text"] in train_texts and n["text"] not in heldout


def test_v2_bundle_stamp(clf2, selected):
    assert clf2.b.meta["selected_model"] == selected
    assert 0 < clf2.b.threshold < 1 and clf2.b.temperature > 0


def test_v2_saved_metrics_match_recomputed_test(clf2, selected, v2_metrics):
    from sklearn.metrics import f1_score
    df = pd.read_csv(DATA_PROCESSED / "text_dataset.csv")
    te = df[df.split == "test"]
    preds = clf2.predict(te["text"].tolist())
    y_pred = [p["category"] if p["flagged"] else "benign" for p in preds]
    f1 = f1_score(te["label"], y_pred, labels=LABELS, average="macro")
    saved = v2_metrics["comparison"][selected]["test"]
    assert abs(f1 - saved["macro_f1_with_abstention_gate"]) <= 0.002, f"recomputed {f1:.4f} vs saved {saved}"
    y_noab = [p["category"] if p["p_dark"] >= clf2.b.threshold else "benign" for p in preds]
    assert abs(f1_score(te["label"], y_noab, labels=LABELS, average="macro") - saved["macro_f1"]) <= 0.002
    assert saved["n"] == len(te) == 329


def test_v2_selected_on_val_not_test(v2_metrics, selected):
    sel = v2_metrics["selection"]
    cands = ["ensemble_v2", "transformer_v1data", "transformer_v2"]
    assert selected == max(cands, key=lambda n: sel["val_macro_f1"][n])


def test_default_model_is_unchanged(monkeypatch):
    monkeypatch.delenv("CLEARCART_MODEL", raising=False)
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as c:
        v = c.post("/api/audit/text", json={"text": "Hurry, only 2 left!"}).json()["versions"]
        assert v["model_version"] == "v1" and v["model_name"] == "served_v1" and v["model"] == "ensemble_lr_xgb"
        assert v["threshold_text"] == "0.68"


def test_api_serves_v2_when_opted_in(monkeypatch, selected):
    monkeypatch.setenv("CLEARCART_MODEL", "v2")
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as c:
        r = c.post("/api/audit/text", json={"text": "Hurry, only 2 left! Offer ends in 10:00 minutes."})
        assert r.status_code == 200
        j = r.json()
        assert j["versions"]["model_version"] == "v2" and j["versions"]["model_name"] == selected
        for f in j["findings"]:
            for t in f["key_terms"]:
                assert t.lower() in f["evidence"].lower()
        info = c.get("/api/model-info").json()
        assert selected in info["name"] and info["test_metrics"]["test_rows"] == 329


# ------------------------------------------------------------------ every v2 artefact (not only the selected one)
@pytest.fixture(scope="module", params=["ensemble_v2", "transformer_v2", "transformer_v1data"])
def any_v2(request):
    return request.param, load_classifier(request.param)


def test_each_v2_model_contract_and_saved_metrics(any_v2, v2_metrics):
    from sklearn.metrics import f1_score
    name, clf = any_v2
    for p in clf.predict(TEXTS):
        assert set(p) == CONTRACT_KEYS and abs(sum(p["probabilities"].values()) - 1) < 1e-6
        assert p["flagged"] == (p["p_dark"] >= clf.b.threshold and not p["abstained"])
        for term in clf.key_terms(p["text"], p["category"]):
            assert term.lower() in p["text"].lower()
        assert len(clf.neighbors(p["_emb"])) == 2
    df = pd.read_csv(DATA_PROCESSED / "text_dataset.csv")
    te = df[df.split == "test"]
    preds = clf.predict(te["text"].tolist())
    f1 = f1_score(te["label"], [p["category"] if p["flagged"] else "benign" for p in preds], labels=LABELS, average="macro")
    assert abs(f1 - v2_metrics["comparison"][name]["test"]["macro_f1_with_abstention_gate"]) <= 0.002
