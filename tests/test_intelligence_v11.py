"""Integration tests for intel-v1.1 through the real API: versions, confidence, optimiser, fusion, replay,
temporal metrics/states, inference cache and security guards."""
import ml  # noqa: F401  (OMP fix before torch/xgboost load)

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import backend.main as main
from backend.main import app

ROOT = Path(__file__).resolve().parents[1]
JOURNEYS = json.loads((ROOT / "data/samples/journeys.json").read_text())
SAMPLES = {s["id"]: s for s in json.loads((ROOT / "data/samples/samples.json").read_text())}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def journey(client):
    return client.post("/api/audit/journey", data={"stages": json.dumps(JOURNEYS[0]["stages"])}).json()


def _html_audit(client):
    return client.post("/api/audit/html", data={"html": SAMPLES["html-checkout"]["content"]}).json()


# ------------------------------------------------------------------ versions + evidence ids
def test_versions_stamped_on_audits_and_journeys(client, journey):
    a = client.post("/api/audit/text", json={"text": "Hurry, only 2 left!"}).json()
    for v in (a["versions"], journey["versions"]):
        assert v["intel_config"] == "intel-v1.1" and v["model"] and v["calibration"].startswith("temperature=")
        assert v["threshold_text"] == "0.68" and v["threshold_ocr"] == "0.90"


def test_evidence_ids_are_stage_qualified(journey):
    refs = [r for st in journey["fspm"]["stages"] for r in st["pressure_sources"] if r["finding_id"]]
    assert refs and all(r["evidence_id"] == f"s{r['stage_index'] + 1}:{r['finding_id']}" for r in refs)


# ------------------------------------------------------------------ severity / confidence / coverage are separate
def test_cdi_confidence_separate_from_severity(client):
    cdi = _html_audit(client)["intelligence"]["cdi"]
    conf = cdi["confidence"]
    assert conf["level"] in ("high", "medium", "low", "none")
    assert conf["model_evidence"] == conf["evidence_by_provenance"].get("model", 0)
    assert conf["min_model_confidence"] <= conf["mean_model_confidence"] <= 1
    assert cdi["alternatives"]["weighted_mean (selected)"] == pytest.approx(cdi["cdi"], abs=0.02)


# ------------------------------------------------------------------ MCAI inferred interaction + attribution
def test_html_interaction_is_inferred_from_preticked_optouts(client):
    cs = _html_audit(client)["intelligence"]["mcai"]["choice_sets"][0]
    inter = next(d for d in cs["dimensions"] if d["name"] == "interaction")
    assert inter["provenance"] == "inferred"
    feats = {f["feature"]: f for f in inter["features"]}
    assert feats["preselected_optouts"]["decline_value"] == 2.0              # two pre-ticked cost/consent options
    assert inter["score"] == pytest.approx(2 / 3, abs=1e-4)
    assert sum(cs["dimension_contributions"].values()) == pytest.approx(cs["mcai"], abs=0.05)


# ------------------------------------------------------------------ optimiser
def test_optimiser_selects_min_score_feasible_candidate(client):
    cf = client.post("/api/audit/text", json={"text": SAMPLES["text-deceptive"]["content"]}).json()["intelligence"]["counterfactual"]
    assert cf["optimisation"]["method"].startswith("exact per-element")
    for e in cf["edits"]:
        feas = [c for c in e["candidates"] if c["feasible"]]
        assert all(not c["violated"] for c in feas)
        if e["action"] in ("rewrite", "remove_unverifiable_claim"):
            chosen = next(c for c in e["candidates"] if c["text"] == e["redesigned"] and c["variant"] != "original")
            assert chosen["feasible"]
            assert chosen["score"] <= min(c["score"] for c in feas) + 0.005 + 1e-9
    assert any(e["action"] == "remove_unverifiable_claim" for e in cf["edits"])     # social-proof claim removed
    assert cf["risk_reduction_pct"] == pytest.approx(100 * (cf["original_cdi"] - cf["redesigned_cdi"]) / cf["original_cdi"], abs=0.06)


# ------------------------------------------------------------------ temporal metrics and states
def test_temporal_states_metrics_and_deltas(journey):
    t = journey["temporal"]
    reset = next(e for e in t["events"] if e["category"] == "countdown_reset")
    assert reset["previous_state"] == "00:54:59 remaining" and reset["current_state"] == "00:58:40 remaining"
    m = t["metrics"]
    assert m["units"] == 4 and m["countdown_resets"] == 1 and m["new_preselections"] == 1
    assert len(m["pressure_deltas"]) == 3
    p = t["pressure_progression"]
    assert [d["delta"] for d in m["pressure_deltas"]] == pytest.approx([p[i] - p[i - 1] for i in range(1, 4)], abs=1e-3)
    assert all(0 <= v <= 1 for v in m["category_persistence"].values())


# ------------------------------------------------------------------ fusion
def _fusion_stages():
    st = [dict(JOURNEYS[0]["stages"][0]), dict(JOURNEYS[0]["stages"][2])]
    st.append({"name": "Checkout (text copy)", "input_type": "text", "fuse_with": 1,
               "content": "OFFER ENDS 00:58:40\nno thanks, I'll save money another day\nAdd shipping protection for $4.99"})
    return st


def test_fusion_counts_the_same_page_once(client):
    fused = client.post("/api/audit/journey", data={"stages": json.dumps(_fusion_stages())}).json()
    unfused_st = _fusion_stages(); unfused_st[2].pop("fuse_with")
    unfused = client.post("/api/audit/journey", data={"stages": json.dumps(unfused_st)}).json()
    assert len(fused["fspm"]["stages"]) == 2 and len(unfused["fspm"]["stages"]) == 3
    g = fused["fusion"][0]
    assert g["stage_indices"] == [1, 2] and g["duplicate_pairs"]
    assert [s["fused_into"] for s in fused["stages"]] == [None, None, 1]
    # merged pressure is a max over modalities, never a sum
    assert g["fused_pressure"] <= 1.0
    assert g["fused_pressure"] >= max(s["pressure"] for s in unfused["stages"][1:]) - 1e-6


def test_fuse_with_must_point_backwards(client):
    st = _fusion_stages(); st[1]["fuse_with"] = 2
    r = client.post("/api/audit/journey", data={"stages": json.dumps(st)})
    assert r.status_code == 422 and "EARLIER" in r.json()["detail"]


# ------------------------------------------------------------------ replay
def test_replay_reproduces_identical_results(client, journey):
    r = client.post("/api/audit/journey/replay", data={"replay": json.dumps(journey["replay"])})
    assert r.status_code == 200
    out = r.json()
    assert out["identical"] is True and out["differences"] == []
    assert out["journey"]["fspm"]["scores"] == journey["fspm"]["scores"]


def test_replay_rejects_tampered_input(client, journey):
    rec = json.loads(json.dumps(journey["replay"]))
    rec["inputs"][1]["content"] = rec["inputs"][1]["content"] + " Hurry!"
    r = client.post("/api/audit/journey/replay", data={"replay": json.dumps(rec)})
    assert r.status_code == 422 and "SHA-256" in r.json()["detail"]


def test_replay_with_screenshot_needs_the_identical_file(client):
    shot = (ROOT / "data/samples/screenshots/www.gasbike.net.png").read_bytes()
    st = [{"name": "Shot", "input_type": "screenshot", "file_field": "f"}]
    j = client.post("/api/audit/journey", data={"stages": json.dumps(st)}, files={"f": ("a.png", shot, "image/png")}).json()
    assert j["replay"]["inputs"][0]["content"] is None                      # image bytes are not stored
    ok = client.post("/api/audit/journey/replay", data={"replay": json.dumps(j["replay"])}, files={"f": ("a.png", shot, "image/png")})
    assert ok.status_code == 200 and ok.json()["identical"]
    other = (ROOT / "data/samples/screenshots/knockaround.com.png").read_bytes()
    bad = client.post("/api/audit/journey/replay", data={"replay": json.dumps(j["replay"])}, files={"f": ("a.png", other, "image/png")})
    assert bad.status_code == 422


# ------------------------------------------------------------------ inference cache
def test_request_cache_removes_duplicate_classifier_work(client, monkeypatch):
    clf = main.STATE["clf"]
    calls = {"n": 0, "texts": 0}
    real = clf.predict

    def counting(texts):
        calls["n"] += 1; calls["texts"] += len(texts)
        return real(texts)
    monkeypatch.setattr(clf, "predict", counting)
    client.post("/api/audit/html", data={"html": SAMPLES["html-checkout"]["content"]})
    with_cache = dict(calls)
    calls.update(n=0, texts=0)
    from contextlib import nullcontext
    monkeypatch.setattr(main, "request_scorer", lambda clf: nullcontext())
    client.post("/api/audit/html", data={"html": SAMPLES["html-checkout"]["content"]})
    assert with_cache["texts"] < calls["texts"]                       # strictly fewer texts sent to the model


# ------------------------------------------------------------------ security guards
def test_decompression_bomb_rejected_before_decoding(client):
    buf = io.BytesIO(); Image.new("L", (9000, 9000)).save(buf, "PNG")
    r = client.post("/api/audit/screenshot", files={"file": ("b.png", buf.getvalue(), "image/png")})
    assert r.status_code == 413 and "megapixel" in r.json()["detail"]


def test_oversized_journey_request_rejected(client):
    r = client.post("/api/audit/journey", data={"stages": "[]"},
                    headers={"content-length": str(main.MAX_JOURNEY_BYTES + 1)})
    assert r.status_code == 413


def test_long_digit_run_is_linear_time():
    """Regression (security audit): a 200k-digit input made segment_text backtrack quadratically (~114 s)."""
    import time
    from backend.extractors import _useful, segment_text
    t = time.perf_counter()
    segment_text("9" * 200_000)
    assert time.perf_counter() - t < 2.0
    # semantics unchanged: clock-like digit strings still count as useful, bare digits do not
    assert _useful("12:30") and _useful("00 : 54 : 59") and not _useful("123456") and _useful("Only 2 left")
