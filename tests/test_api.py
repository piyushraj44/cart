import io
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
from backend.auditor import display_pct

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = json.loads((ROOT / "data/samples/samples.json").read_text())


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _sample(i):
    return next(s for s in SAMPLES if s["id"] == i)


def check_findings(res, content, text_mode=False):
    for f in res["findings"]:
        if f["detection_method"] == "dom_observation":
            assert f["model_score"] is None and f["confidence_pct"] is None
        else:
            assert f["confidence_pct"] <= 99
            assert f["confidence_pct"] == min(99, int(f["model_score"] * 100))
            assert f["key_terms"] is not None
            for t in f["key_terms"]:
                assert t.lower() in f["evidence"].lower()
        if text_mode:
            sl = f["source_location"]
            assert content[sl["char_start"]:sl["char_end"]] == f["evidence"]
            assert f["evidence"] in content
        else:
            # html: evidence is text content, whitespace-collapsed; verify per-token presence in visible text
            norm = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", content))
            import html as H
            norm = re.sub(r"\s+", " ", H.unescape(norm))
            assert f["evidence"] in norm or f["evidence"] in H.unescape(content), f["evidence"]


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["model_loaded"] is True and r["ocr_available"] is True


def test_model_info(client):
    r = client.get("/api/model-info")
    assert r.status_code == 200
    j = r.json()
    assert 0 < j["threshold"] < 1 and j["test_metrics"]["macro_f1"] and j["ocr_available"]
    assert {c["key"] for c in j["categories"]} >= {"urgency", "scarcity", "social_proof", "misdirection", "preselected_option"}


def test_samples(client):
    r = client.get("/api/samples")
    assert r.status_code == 200 and len(r.json()) == 4


@pytest.mark.parametrize("sid", ["text-deceptive", "text-benign"])
def test_text_samples(client, sid):
    s = _sample(sid)
    r = client.post("/api/audit/text", json={"text": s["content"]})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["input_type"] == "text"
    check_findings(j, s["content"], text_mode=True)
    if "benign" in sid:
        assert j["summary"]["no_concern"] is True and j["findings"] == [] and j["summary"]["overall_risk"] == "none"
    else:
        assert len(j["findings"]) >= 1 and not j["summary"]["no_concern"]


@pytest.mark.parametrize("sid", ["html-checkout", "html-benign"])
@pytest.mark.parametrize("mode", ["form", "file"])
def test_html_samples(client, sid, mode):
    s = _sample(sid)
    if mode == "form":
        r = client.post("/api/audit/html", data={"html": s["content"]})
    else:
        r = client.post("/api/audit/html", files={"file": ("page.html", s["content"].encode(), "text/html")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["input_type"] == "html"
    check_findings(j, s["content"])
    if "benign" in sid:
        assert j["summary"]["no_concern"] is True and j["findings"] == [] and j["summary"]["overall_risk"] == "none"
    else:
        assert len(j["findings"]) >= 1
        assert any(f["detection_method"] == "dom_observation" for f in j["findings"])
        blob = json.dumps(j["findings"]) + json.dumps(j["benign_elements"])
        assert "hidden note" not in blob.lower()
        assert "ignored by the extractor" not in blob


def test_text_errors(client):
    assert client.post("/api/audit/text", json={"text": ""}).status_code == 422
    assert client.post("/api/audit/text", json={"text": "  \n\t "}).status_code == 422
    assert client.post("/api/audit/text", json={"text": "!!! --- ???"}).status_code == 422
    assert client.post("/api/audit/text", json={"text": "a" * 200_001}).status_code == 413
    assert client.post("/api/audit/text", json={}).status_code == 422
    assert client.post("/api/audit/text", content="notjson", headers={"content-type": "application/json"}).status_code == 422
    assert client.post("/api/audit/text", json={"text": "a " * 100_000}).status_code == 200


def test_text_exact_limit(client):
    r = client.post("/api/audit/text", json={"text": ("Buy now\n" * 25000)[:200_000]})
    assert r.status_code == 200


def test_html_errors(client):
    p = "/api/audit/html"
    assert client.post(p, files={"file": ("x.txt", b"<p>hi there</p>", "text/plain")}).status_code == 415
    assert client.post(p, data={"html": ""}).status_code == 422
    assert client.post(p, data={"html": "   "}).status_code == 422
    assert client.post(p, data={"html": "just plain text here"}).status_code == 422
    assert client.post(p, files={"file": ("e.html", b"", "text/html")}).status_code == 422
    assert client.post(p, data={"html": "<p>" + "a" * (2 * 1024 * 1024) + "</p>"}).status_code in (400, 413)  # see test_html_form_between_1_and_2MB
    assert client.post(p, files={"file": ("big.html", b"<p>" + b"a" * (2 * 1024 * 1024) + b"</p>", "text/html")}).status_code == 413
    assert client.post(p).status_code == 422
    assert client.post(p, data={"html": "<div></div>"}).status_code == 422
    assert client.post(p, data={"html": "<script>var a=1</script><style>p{}</style>"}).status_code == 422


def _png(w=100, h=60, color="white"):
    b = io.BytesIO(); Image.new("RGB", (w, h), color).save(b, "PNG"); return b.getvalue()


def test_screenshot_errors(client):
    p = "/api/audit/screenshot"
    r = client.post(p, files={"file": ("a.png", b"this is not an image", "image/png")})
    assert r.status_code in (415, 422)
    assert client.post(p, files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")}).status_code == 415
    assert client.post(p, files={"file": ("a.png", b"\x89PNG" + b"0" * (8 * 1024 * 1024 + 10), "image/png")}).status_code == 413
    assert client.post(p).status_code == 422
    assert client.post(p, files={"file": ("blank.png", _png(), "image/png")}).status_code == 422  # no text


@pytest.mark.skipif(not (ROOT / "data/raw/contextrico/images/ContextRico-DP").exists(),
                    reason="ContextRico screenshots not downloaded: run `python -m ml.download_data --images`")
def test_screenshot_happy(client):
    g = json.loads((ROOT / "data/raw/contextrico/groundtrue.json").read_text())
    base = ROOT / "data/raw/contextrico/images/ContextRico-DP"
    chosen = None
    for k, v in g.items():
        if any(l in ("Low Stock Message", "Limited Time Message") for l in v["labels"]):
            for d in ("Mobile", "Web"):
                if (base / d / k).exists():
                    chosen = base / d / k; break
        if chosen:
            break
    assert chosen, "no ContextRico image found"
    raw = chosen.read_bytes()
    mime = "image/png" if chosen.suffix == ".png" else "image/jpeg"
    r = client.post("/api/audit/screenshot", files={"file": (chosen.name, raw, mime)})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["input_type"] == "screenshot_ocr"
    w, h = Image.open(chosen).size
    assert j["image_size"] == {"width": w, "height": h}
    locs = [f["source_location"] for f in j["findings"]] + [e["source_location"] for e in j["benign_elements"]]
    assert locs
    for sl in locs:
        b = sl["bbox"]
        assert b and b["x"] >= 0 and b["y"] >= 0 and b["x"] + b["width"] <= w and b["y"] + b["height"] <= h, (b, w, h)
    for f in j["findings"]:
        assert f["confidence_pct"] <= 99


def test_pdf_export(client):
    s = _sample("text-deceptive")
    j = client.post("/api/audit/text", json={"text": s["content"]}).json()
    r = client.post("/api/export/pdf", json=j)
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf" and r.content.startswith(b"%PDF")
    j0 = client.post("/api/audit/text", json={"text": _sample("text-benign")["content"]}).json()
    r = client.post("/api/export/pdf", json=j0)
    assert r.status_code == 200 and r.content.startswith(b"%PDF")
    r = client.post("/api/export/pdf", json={"bad": 1})
    assert r.status_code == 422


def test_perf_timing(client, capsys):
    import time
    s = _sample("text-deceptive")["content"]
    client.post("/api/audit/text", json={"text": s})
    t = time.perf_counter(); client.post("/api/audit/text", json={"text": s}); small = (time.perf_counter() - t) * 1000
    big = "\n".join(f"Line {i}: Only {i} left in stock, hurry! Buy product number {i} today" for i in range(300))
    t = time.perf_counter(); r = client.post("/api/audit/text", json={"text": big}); large = (time.perf_counter() - t) * 1000
    assert r.status_code == 200
    print(f"\nTIMING deceptive_warm_ms={small:.0f} 300line_ms={large:.0f}")
    assert small < 5000 and large < 30000


def test_html_form_between_1_and_2MB(client):
    html = "<p>Hurry</p>" + "<p>Only 2 left</p>" * 70000   # ~1.2MB, under the documented 2MB limit
    r = client.post("/api/audit/html", data={"html": html})
    assert r.status_code == 200, (r.status_code, r.text[:100])
    r = client.post("/api/audit/html", data={"html": "<p>" + "a" * (2 * 1024 * 1024 + 10) + "</p>"})
    assert r.status_code == 413, r.status_code


def test_html_long_paragraph_not_dropped(client):
    html = "<p>" + "Hurry, only 2 left in stock. " * 60 + "</p>"
    r = client.post("/api/audit/html", data={"html": html})
    assert r.status_code == 200, (r.status_code, r.text[:100])
