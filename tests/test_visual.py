"""EXPERIMENTAL opt-in visual pattern-type model (backend/visual.py): off by default, never changes the core audit.

Uses a RANDOM-weight model written to a temp dir (the trained artifacts may not exist), on CPU.
"""
import io
import json

import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from backend import visual
from backend.extractors import ocr_available
from backend.main import app
from ml.common import ROOT
from ml.visual_data import TARGETS
from ml.visual_model import build

FIRING = ["Hard to Close", "Nagging"]          # reliable, threshold 0.0 -> always reported
SHOT = "/api/audit/screenshot"


def _font(size):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _config(size=128):
    lc = {l: {"temperature": 1.5, "threshold": 0.0 if l in FIRING else 0.5, "reliable": l in FIRING} for l in TARGETS}
    lc["Default Choice"] = {"temperature": 1.0, "threshold": None, "reliable": False}
    return {"arch": "mobilenet_v3_large", "size": size, "labels": list(TARGETS), "label_config": lc,
            "provenance": "model (visual, experimental)", "model_sha256": "abcdef0123456789" * 4}


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("visual_model")
    torch.manual_seed(0)
    m = build("mobilenet_v3_large", len(TARGETS), pretrained=False)
    torch.save(m.state_dict(), d / "model.pt")
    (d / "config.json").write_text(json.dumps(_config()))
    return d


@pytest.fixture(scope="module")
def corrupt_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("visual_corrupt")
    (d / "model.pt").write_bytes(b"not a torch checkpoint")
    (d / "config.json").write_text(json.dumps(_config()))
    return d


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv("CLEARCART_VISUAL", raising=False)
    monkeypatch.delenv("CLEARCART_VISUAL_DIR", raising=False)
    visual.reset()
    yield
    visual.reset()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def shot():
    img = Image.new("RGB", (900, 320), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 40), "Only 2 left in stock", fill="black", font=_font(48))
    d.text((40, 140), "Hurry, sale ends today", fill="black", font=_font(48))
    d.text((40, 240), "Add to cart", fill="black", font=_font(40))
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue(), img.size


def _on(monkeypatch, d):
    monkeypatch.setenv("CLEARCART_VISUAL", "1")
    monkeypatch.setenv("CLEARCART_VISUAL_DIR", str(d))


def _post(client, raw):
    r = client.post(SHOT, files={"file": ("shot.png", raw, "image/png")})
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------- enabled() gating (no OCR needed)
def test_enabled_requires_flag_and_files(monkeypatch, model_dir, tmp_path):
    assert not visual.enabled()
    monkeypatch.setenv("CLEARCART_VISUAL_DIR", str(model_dir))
    assert not visual.enabled()                                   # files present, flag unset
    monkeypatch.setenv("CLEARCART_VISUAL", "0")
    assert not visual.enabled()
    monkeypatch.setenv("CLEARCART_VISUAL", "1")
    assert visual.enabled()
    monkeypatch.setenv("CLEARCART_VISUAL_DIR", str(tmp_path))     # flag set, files missing
    assert not visual.enabled()


def test_predict_contract(monkeypatch, model_dir):
    _on(monkeypatch, model_dir)
    img = Image.new("RGB", (300, 500), "white")
    ImageDraw.Draw(img).rectangle((40, 60, 200, 120), fill="red")
    out = visual.predict(img)
    assert [s["label"] for s in out] and {s["label"] for s in out} == set(FIRING)    # only reliable labels >= threshold
    assert [s["probability"] for s in out] == sorted((s["probability"] for s in out), reverse=True)
    for s in out:
        assert set(s) == {"label", "probability", "threshold", "bbox", "provenance"}
        assert 0.0 <= s["probability"] <= 1.0 and s["probability"] == round(s["probability"], 4)
        assert s["threshold"] == 0.0 and s["provenance"] == "model (visual, experimental)"
        b = s["bbox"]
        if b is not None:
            assert b["x"] >= 0 and b["y"] >= 0 and b["width"] >= 1 and b["height"] >= 1
            assert b["x"] + b["width"] <= 300 and b["y"] + b["height"] <= 500, b
    assert visual.predict(img) == out                             # deterministic on CPU
    assert visual.version() == "mobilenet_v3_large:abcdef012345"


def test_predict_failure_returns_empty(monkeypatch, corrupt_dir, caplog):
    _on(monkeypatch, corrupt_dir)
    assert visual.enabled()
    with caplog.at_level("WARNING", logger="clearcart.visual"):
        assert visual.predict(Image.new("RGB", (64, 64), "white")) == []
    assert any("visual model" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------- endpoint (needs OCR)
needs_ocr = pytest.mark.skipif(not ocr_available(), reason="tesseract missing")


@needs_ocr
def test_endpoint_off_on_and_core_unchanged(monkeypatch, client, shot, model_dir):
    raw, (w, h) = shot
    off = _post(client, raw)
    assert off.get("visual_signals") is None
    assert "visual_model" not in (off.get("versions") or {})

    _on(monkeypatch, model_dir)
    on = _post(client, raw)
    sig = on["visual_signals"]
    assert isinstance(sig, list) and {s["label"] for s in sig} == set(FIRING)
    for s in sig:
        assert 0.0 <= s["probability"] <= 1.0
        b = s["bbox"]
        if b is not None:
            assert b["x"] >= 0 and b["y"] >= 0 and b["x"] + b["width"] <= w and b["y"] + b["height"] <= h, (b, w, h)
    assert on["versions"]["visual_model"] == "mobilenet_v3_large:abcdef012345"

    # visual signals never enter the core audit, the summary, the intelligence layer or the warnings
    assert on["findings"] == off["findings"]
    assert on["summary"]["risk_score"] == off["summary"]["risk_score"]
    assert on["summary"]["tier_counts"] == off["summary"]["tier_counts"]
    assert on["summary"] == off["summary"]
    assert on["intelligence"]["cdi"] == off["intelligence"]["cdi"]
    assert on["warnings"] == off["warnings"]
    assert {k: v for k, v in on["versions"].items() if k != "visual_model"} == off["versions"]


@needs_ocr
def test_endpoint_corrupt_model_is_empty_not_500(monkeypatch, client, shot, corrupt_dir):
    raw, _ = shot
    off = _post(client, raw)
    _on(monkeypatch, corrupt_dir)
    on = _post(client, raw)
    assert on["visual_signals"] == []
    assert on["findings"] == off["findings"] and on["summary"] == off["summary"]


# ---------------------------------------------------------------------------- split hygiene
SPLIT = ROOT / "data/processed/visual_split.csv"
GT = ROOT / "data/raw/contextrico/groundtrue.json"


@pytest.mark.skipif(not SPLIT.exists(), reason="visual split not built: run `python -m ml.visual_data`")
def test_split_has_no_group_overlap():
    df = pd.read_csv(SPLIT)
    assert set(df.split) == {"train", "val", "test"}
    assert df.image.is_unique
    groups = {s: set(df[df.split == s].group) for s in ("train", "val", "test")}
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]


@pytest.mark.skipif(not (SPLIT.exists() and GT.exists()), reason="ContextRico raw data or visual split missing")
def test_split_labels_match_groundtruth():
    df = pd.read_csv(SPLIT, keep_default_na=False)
    gt = {k: set(v["labels"]) for k, v in json.loads(GT.read_text()).items()}
    assert len(df)
    for r in df.to_dict("records"):
        img = r["image"]
        assert img in gt, img
        labels = set(filter(None, r["labels"].split("|")))
        assert labels == gt[img], (img, labels, gt[img])
        for t in TARGETS:                                          # target columns agree with the ground truth too
            assert int(r[t]) == int(t in gt[img]), (img, t)
