"""API integration tests for the intel-v1.0 journey endpoint, exports and backward compatibility.

These use the real trained model through FastAPI's TestClient. Every index in the response is
recomputed here independently from the response's own inputs, so the API cannot report a number
that does not follow from its formula.
"""
import ml  # noqa: F401  (OMP fix before torch/xgboost load)

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader

from backend.intelligence.config import CDI_WEIGHTS
from backend.intelligence.fspm import fibonacci
from backend.main import app
from backend.schemas import AuditResponse, JourneyResponse

ROOT = Path(__file__).resolve().parents[1]
JOURNEYS = json.loads((ROOT / "data/samples/journeys.json").read_text())
SAMPLES = {s["id"]: s for s in json.loads((ROOT / "data/samples/samples.json").read_text())}
SHOT = ROOT / "data/samples/screenshots/www.gasbike.net.png"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def demo(client):
    r = client.post("/api/audit/journey", data={"stages": json.dumps(JOURNEYS[0]["stages"])})
    assert r.status_code == 200, r.text[:300]
    return r.json()


def _post(client, stages, files=None):
    return client.post("/api/audit/journey", data={"stages": json.dumps(stages)}, files=files or {})


# ------------------------------------------------------------------ happy path + independent recomputation
def test_demo_journey_validates_against_schema(demo):
    JourneyResponse.model_validate(demo)            # JSON export == this object; it must round-trip
    assert len(demo["stages"]) == len(JOURNEYS[0]["stages"])
    assert all(st["synthetic"] for st in demo["stages"])
    assert any("synthetic" in w.lower() for w in demo["warnings"])


def test_fspm_recomputed_from_response(demo):
    f = demo["fspm"]
    pressures = [st["pressure"] for st in f["stages"]]
    positions = [st["position"] for st in f["stages"]]
    assert f["positions_mode"] == "canonical" and positions == [2, 3, 4, 7]
    for scheme, w in {"fibonacci": [fibonacci(p)[-1] for p in positions], "linear": positions, "equal": [1] * len(positions)}.items():
        expect = 100 * sum(wi * pi for wi, pi in zip(w, pressures)) / sum(w)
        assert f["scores"][scheme] == pytest.approx(expect, abs=0.02), scheme
        assert sum(st["contributions"][scheme] for st in f["stages"]) == pytest.approx(f["scores"][scheme], abs=0.05)
    top = max(range(len(f["stages"])), key=lambda i: f["stages"][i]["contributions"]["fibonacci"])
    assert f["top_stage_index"] == top


def test_cdi_recomputed_from_components(demo):
    c = demo["cdi"]
    avail = [x for x in c["components"] if x["available"]]
    wsum = sum(CDI_WEIGHTS[x["key"]] for x in avail)
    assert c["cdi"] == pytest.approx(100 * sum(CDI_WEIGHTS[x["key"]] * x["value"] for x in avail) / wsum, abs=0.02)
    assert c["coverage"] == pytest.approx(wsum, abs=1e-4)
    content = next(x for x in c["components"] if x["key"] == "content_pressure")
    assert content["value"] == pytest.approx(demo["fspm"]["scores"]["fibonacci"] / 100, abs=1e-3)   # FSPM replaces, not adds
    s = c["sensitivity"]
    assert s["dirichlet_p05"] - 1e-6 <= c["cdi"] <= s["dirichlet_p95"] + 1e-6


def test_counterfactual_reduction_formula(demo):
    cf = demo["counterfactual"]
    assert cf["original_cdi"] == pytest.approx(demo["cdi"]["cdi"])
    assert cf["risk_reduction_pct"] == pytest.approx(100 * (cf["original_cdi"] - cf["redesigned_cdi"]) / cf["original_cdi"], abs=0.06)
    assert "model-estimated" in cf["label"].lower()
    assert all(e["facts_preserved"] for e in cf["edits"])
    assert {e["stage_index"] for e in cf["edits"]} <= set(range(len(demo["stages"])))


def test_timeline_facts_and_new_journey_events(demo):
    ev = demo["temporal"]["events"]
    cats = {(e["kind"], e["category"]) for e in ev}
    assert ("observed_fact", "countdown_reset") in cats
    assert ("observed_fact", "new_preselected_option") in cats
    assert ("observation", "interaction_difficulty") in cats          # declared 1-vs-5 cancellation path
    assert ("observation", "choice_asymmetry") in cats                # MCAI change across stages
    assert ("observation", "decline_path_change") in cats
    # declared/MCAI events are observations, never deceptive facts (no double counting with CDI)
    assert not any(e["deceptive_signal"] for e in ev if e["category"] in ("interaction_difficulty", "choice_asymmetry", "decline_path_change"))
    t = demo["temporal"]
    assert len(t["pressure_progression"]) == len(t["progression_stage_indices"]) == len(demo["stages"])


def test_benign_journey_scores_zero(client):
    r = _post(client, JOURNEYS[1]["stages"])
    assert r.status_code == 200
    o = r.json()
    assert o["fspm"]["scores"]["fibonacci"] == 0 and o["cdi"]["cdi"] == 0
    assert not any(e["deceptive_signal"] for e in o["temporal"]["events"])


def test_incomplete_journey_keeps_canonical_positions(client):
    stages = [JOURNEYS[0]["stages"][0], JOURNEYS[0]["stages"][2]]         # product, checkout (cart missing)
    o = _post(client, stages).json()
    assert [s["position"] for s in o["fspm"]["stages"]] == [2, 4]
    assert len(o["fspm"]["stages"]) == 2                                   # missing stage not invented


# ------------------------------------------------------------------ screenshot stage end to end
def test_journey_with_screenshot_stage(client):
    stages = [
        {"name": "Product", "stage_type": "product", "input_type": "text", "content": SAMPLES["text-deceptive"]["content"]},
        {"name": "Checkout screenshot", "stage_type": "checkout", "input_type": "screenshot", "file_field": "stage_1_file"},
    ]
    r = _post(client, stages, files={"stage_1_file": ("gasbike.png", SHOT.read_bytes(), "image/png")})
    assert r.status_code == 200, r.text[:300]
    o = r.json()
    shot = o["stages"][1]
    assert shot["input_type"] == "screenshot" and shot["content"] is None
    assert shot["audit"]["input_type"] == "screenshot_ocr" and shot["audit"]["image_size"]
    assert any(f["category"] == "urgency" for f in shot["audit"]["findings"])       # countdown banner, as in single audit
    assert o["fspm"]["available"] and len(o["fspm"]["stages"]) == 2


# ------------------------------------------------------------------ validation errors
@pytest.mark.parametrize("payload,code", [
    ("[]", 422),
    ("not json", 422),
    (json.dumps([{"name": "x", "input_type": "text", "content": "Hurry"}] * 9), 422),
    (json.dumps([{"name": "x", "input_type": "text", "content": "Hurry", "stage_type": "basket"}]), 422),
    (json.dumps([{"name": "x", "input_type": "text", "content": "Hurry", "captured_at": "yesterday"}]), 422),
    (json.dumps([{"name": "x", "input_type": "screenshot", "file_field": "missing"}]), 422),
    (json.dumps([{"name": "x", "input_type": "text", "content": "   "}]), 422),
    (json.dumps([{"name": "x", "input_type": "text", "content": "ok",
                  "choice_path": {"label": "c", "accept_steps": 0, "decline_steps": 3}}]), 422),
])
def test_journey_validation(client, payload, code):
    r = client.post("/api/audit/journey", data={"stages": payload})
    assert r.status_code == code, r.text[:200]
    assert "detail" in r.json()


def test_missing_stages_field(client):
    assert client.post("/api/audit/journey", data={}).status_code == 422


# ------------------------------------------------------------------ exports
def _pdf_text(raw: bytes) -> str:
    return " ".join(p.extract_text() for p in PdfReader(io.BytesIO(raw)).pages)


def test_journey_pdf_export_contains_new_sections(client, demo):
    r = client.post("/api/export/journey-pdf", json=demo)
    assert r.status_code == 200 and r.content[:4] == b"%PDF"
    text = _pdf_text(r.content)
    for needle in ("Deception intelligence", "Fibonacci", "Choice asymmetry", "Counterfactual", "Temporal evidence timeline"):
        assert needle in text, needle


def test_single_audit_pdf_includes_intelligence(client):
    a = client.post("/api/audit/html", data={"html": SAMPLES["html-checkout"]["content"]}).json()
    assert a["intelligence"] is not None
    text = _pdf_text(client.post("/api/export/pdf", json=a).content)
    assert "Deception intelligence" in text and "CDI" in text


# ------------------------------------------------------------------ backward compatibility
def test_old_audit_json_without_intelligence_still_accepted(client):
    a = client.post("/api/audit/text", json={"text": SAMPLES["text-deceptive"]["content"]}).json()
    a.pop("intelligence")                                    # shape saved by the pre-intel-v1.0 app
    AuditResponse.model_validate(a)
    r = client.post("/api/export/pdf", json=a)
    assert r.status_code == 200 and r.content[:4] == b"%PDF"


@pytest.mark.parametrize("kind", ["text", "html", "screenshot"])
def test_every_input_type_gets_intelligence(client, kind):
    if kind == "text":
        a = client.post("/api/audit/text", json={"text": SAMPLES["text-deceptive"]["content"]}).json()
    elif kind == "html":
        a = client.post("/api/audit/html", data={"html": SAMPLES["html-checkout"]["content"]}).json()
    else:
        a = client.post("/api/audit/screenshot", files={"file": ("s.png", SHOT.read_bytes(), "image/png")}).json()
    intel = a["intelligence"]
    assert intel["config_version"] and intel["cdi"]["cdi"] is not None
    assert intel["fspm"]["available"] is False and intel["temporal"]["available"] is False   # single page: not faked
    vis = [d for cs in intel["mcai"]["choice_sets"] for d in cs["dimensions"] if d["name"] == "visual"]
    if kind == "text":
        assert all(not d["available"] and d["score"] is None for d in vis)                   # never fabricated from text


def test_obstruction_unavailable_without_dom_or_facts(client):
    """Regression: a text+screenshot journey must not claim 'DOM was inspected' (found in live UI check)."""
    stages = [
        {"name": "Product", "stage_type": "product", "input_type": "text", "content": "Only 5 item(s) left."},
        {"name": "Checkout", "stage_type": "checkout", "input_type": "screenshot", "file_field": "f"},
    ]
    o = _post(client, stages, files={"f": ("s.png", SHOT.read_bytes(), "image/png")}).json()
    obst = next(c for c in o["cdi"]["components"] if c["key"] == "observed_obstruction")
    assert obst["available"] is False and obst["value"] is None
    assert o["cdi"]["coverage"] < 1.0
