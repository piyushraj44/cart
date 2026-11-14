"""ClearCart FastAPI backend.

Run from the project root:
    .venv/bin/uvicorn backend.main:app --port 8000
"""
from __future__ import annotations

import io
import json
import logging
import os
from contextlib import asynccontextmanager

import ml  # noqa: F401  (sets OMP_NUM_THREADS before torch/xgboost load - see ml/__init__.py)
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from backend.auditor import Auditor
from backend.extractors import extract_html, ocr_available, ocr_lines, segment_text
from backend.recommendations import CATEGORY_LABELS
from backend.intelligence.config import CANONICAL_STAGES, CONFIG_VERSION
from backend.intelligence.schemas import JourneyStageInput, ReplayRecord
from backend.schemas import (AuditResponse, JourneyResponse, JourneyStageResult, ModelInfo, ReplayResult, SampleInput,
                             TextAuditRequest, VisualSignal)
from backend.scoring import request_scorer
from ml.common import ARTIFACTS, ROOT

ARTIFACTS_V2 = ROOT / "ml" / "artifacts_v2"

log = logging.getLogger("clearcart")
MAX_TEXT_CHARS = 200_000
MAX_HTML_BYTES = 2 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
# OCR config + threshold selected on ContextRico DEV images (ml/eval_screenshots.py); the text
# threshold (0.68) produced a 29-39% flag rate on screenshots without text-observable patterns.
OCR_THRESHOLD = 0.90
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from ml.model import load_classifier
    # Opt-in model selection: CLEARCART_MODEL=v2 serves the VAL-selected v2 model from ml/artifacts_v2
    # (see ml/train_v2.py). Unset / "v1" keeps the frozen default model in ml/artifacts.
    choice = os.environ.get("CLEARCART_MODEL", "").strip().lower()
    clf = load_classifier(choice)                    # loaded exactly once
    clf.predict(["warm-up"])                         # first call JIT/allocations
    STATE["clf"] = clf
    STATE["model_version"] = "v1" if choice in ("", "v1", "default") else "v2"
    STATE["model_name"] = "served_v1" if STATE["model_version"] == "v1" else str(clf.b.meta.get("selected_model", choice))
    STATE["auditor"] = Auditor(clf)
    from backend.intelligence.engine import IntelligenceEngine
    STATE["intel"] = IntelligenceEngine(clf)
    STATE["metrics"] = json.loads((ARTIFACTS / "metrics.json").read_text())
    if STATE["model_version"] == "v2":
        STATE["metrics_v2"] = json.loads((ARTIFACTS_V2 / "metrics.json").read_text())
    STATE["ocr"] = ocr_available()
    log.info("ClearCart model loaded; OCR available=%s", STATE["ocr"])
    yield
    STATE.clear()


APP_VERSION = "1.1.0"
app = FastAPI(title="ClearCart API", version=APP_VERSION, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])
# Images: reject absurd pixel dimensions BEFORE decoding (a few-KB PNG can declare a 40k x 40k canvas).
MAX_IMAGE_PIXELS = 40_000_000
MAX_JOURNEY_BYTES = 40 * 1024 * 1024          # whole multipart request for a journey (8 stages)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    # Privacy: log the exception type and stack frames only - exception messages can quote user content.
    import traceback
    log.error("unhandled %s on %s\n%s", type(exc).__name__, request.url.path, "".join(traceback.format_tb(exc.__traceback__)))
    return JSONResponse(status_code=500, content={"detail": "Internal error while analysing the input. Please try again."})


def _auditor() -> Auditor:
    if "auditor" not in STATE:
        raise HTTPException(503, "Model is not loaded yet.")
    return STATE["auditor"]


def _versions() -> dict[str, str]:
    """Everything needed to reproduce a score: model, calibration, thresholds and scoring configuration."""
    from backend.intelligence.config import CDI_WEIGHTS_VERSION
    b = STATE["clf"].b
    return {"app": APP_VERSION, "model": str(b.meta.get("selected_model", "")), "model_trained_at": str(b.meta.get("trained_at", "")),
            "model_name": str(STATE.get("model_name", "served_v1")), "model_version": str(STATE.get("model_version", "v1")),
            "calibration": f"temperature={b.temperature:.4f}", "threshold_text": f"{b.threshold:.2f}",
            "threshold_ocr": f"{OCR_THRESHOLD:.2f}", "intel_config": CONFIG_VERSION, "cdi_weights": CDI_WEIGHTS_VERSION}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "clf" in STATE, "ocr_available": bool(STATE.get("ocr"))}


# ---------------------------------------------------------------------------- shared input runners
def _run_text(text: str, where: str = ""):
    if not (text or "").strip():
        raise HTTPException(422, f"{where}The text is empty. Paste some webpage text to analyse.")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(413, f"{where}Text is too long ({len(text):,} chars). The limit is {MAX_TEXT_CHARS:,} characters.")
    cands = segment_text(text)
    if not cands:
        raise HTTPException(422, f"{where}No analysable text was found (the input contains no words).")
    return _auditor().audit(cands, "text"), cands


def _run_html(content: str, where: str = ""):
    if not (content or "").strip():
        raise HTTPException(422, f"{where}The HTML is empty.")
    if len(content.encode("utf-8")) > MAX_HTML_BYTES:
        raise HTTPException(413, f"{where}HTML is larger than the 2 MB limit.")
    if "<" not in content:
        raise HTTPException(422, f"{where}This does not look like HTML. Use the 'Paste text' option for plain text.")
    cands, warnings = extract_html(content)
    if not cands:
        raise HTTPException(422, f"{where}No visible text elements were found in the HTML.")
    return _auditor().audit(cands, "html", warnings), cands


def _run_shot(raw: bytes, where: str = ""):
    if not STATE.get("ocr"):
        raise HTTPException(503, "Screenshot analysis needs the Tesseract OCR engine. Install it (macOS: brew install tesseract) and restart the backend.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"{where}Image is larger than the 8 MB limit.")
    from PIL import Image, UnidentifiedImageError
    try:
        img = Image.open(io.BytesIO(raw))          # reads the header only
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(422, f"{where}The file could not be read as an image.")
    if img.width * img.height > MAX_IMAGE_PIXELS:
        raise HTTPException(413, f"{where}Image dimensions {img.width}x{img.height} exceed the {MAX_IMAGE_PIXELS // 1_000_000} megapixel limit.")
    try:
        img.load()
    except (OSError, Image.DecompressionBombError):
        raise HTTPException(422, f"{where}The file could not be read as an image.")
    from backend.extractors import default_ocr_engine, ocr_engine_lines   # CLEARCART_OCR_ENGINE (default tesseract)
    engine = default_ocr_engine()
    cands = ocr_engine_lines(img, engine)
    engine_name = {"tesseract": "Tesseract OCR", "easyocr": "EasyOCR", "union": "Tesseract + EasyOCR"}[engine]
    warnings = [f"Screenshot text was extracted with {engine_name} and classified by the text model. The classifier does not "
                "see layout or colours; the choice-asymmetry index only uses OCR box sizes and pixel contrast, and OCR can "
                "misread small or stylised text."]
    if not [c for c in cands if c.extra.get("unit") == "line"]:
        raise HTTPException(422, f"{where}No readable text was found in the screenshot.")
    warnings.append(f"Screenshots use a stricter flagging threshold ({int(OCR_THRESHOLD*100)}%) than text, chosen on held-out "
                    "ContextRico screenshots to limit false alarms from OCR noise such as clocks and timers.")
    audit = _auditor().audit(cands, "screenshot_ocr", warnings, image_size={"width": img.width, "height": img.height},
                             threshold=OCR_THRESHOLD)
    return audit, cands, img


def _with_intelligence(audit: AuditResponse, input_type: str, cands, html: str | None = None, image=None) -> AuditResponse:
    """Attach the intelligence layer. A failure here never breaks the validated core audit."""
    audit.versions = _versions()
    engine = STATE.get("intel")
    if engine is None:
        return audit
    try:
        audit.intelligence = engine.single(audit, input_type, cands, html=html, image=image)
    except Exception as exc:
        import traceback
        log.error("intelligence layer failed: %s\n%s", type(exc).__name__, "".join(traceback.format_tb(exc.__traceback__)))
        audit.warnings.append("The intelligence layer (CDI/MCAI/counterfactual) could not be computed for this input; "
                              "the classifier findings above are unaffected.")
    return audit


# ---------------------------------------------------------------------------- single-page audits
@app.post("/api/audit/text", response_model=AuditResponse)
def audit_text(req: TextAuditRequest):
    with request_scorer(STATE["clf"]):
        audit, cands = _run_text(req.text or "")
        return _with_intelligence(audit, "text", cands)


@app.post("/api/audit/html", response_model=AuditResponse)
async def audit_html(request: Request):
    # Parse the form ourselves: Starlette's default per-field limit (1 MB) would reject
    # pasted HTML between 1 and 2 MB with a 400 before our 2 MB check could return 413.
    try:
        form = await request.form(max_part_size=MAX_HTML_BYTES * 2)
    except Exception:
        raise HTTPException(413, "Pasted HTML is larger than the 2 MB limit.")
    file = form.get("file")
    html = form.get("html")
    if isinstance(file, str):
        file = None
    if file is not None and file.filename:
        name = file.filename.lower()
        if not name.endswith((".html", ".htm")):
            raise HTTPException(415, "Unsupported file type. Upload an .html or .htm file.")
        raw = await file.read(MAX_HTML_BYTES + 1)
        if len(raw) > MAX_HTML_BYTES:
            raise HTTPException(413, "HTML file is larger than the 2 MB limit.")
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            content = raw.decode("latin-1")
    elif html is not None:
        content = html
    else:
        raise HTTPException(422, "Provide an HTML file (field 'file') or pasted HTML (field 'html').")
    with request_scorer(STATE["clf"]):
        audit, cands = _run_html(content)
        return _with_intelligence(audit, "html", cands, html=content)


@app.post("/api/audit/screenshot", response_model=AuditResponse)
async def audit_screenshot(file: UploadFile = File(...)):
    if not STATE.get("ocr"):
        raise HTTPException(503, "Screenshot analysis needs the Tesseract OCR engine. Install it (macOS: brew install tesseract) and restart the backend.")
    if (file.content_type or "").lower() not in IMAGE_TYPES and not (file.filename or "").lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        raise HTTPException(415, "Unsupported image type. Upload a PNG, JPEG or WebP screenshot.")
    raw = await file.read(MAX_IMAGE_BYTES + 1)
    with request_scorer(STATE["clf"]):
        audit, cands, img = _run_shot(raw)
        audit = _with_intelligence(audit, "screenshot_ocr", cands, image=img)
    return _with_visual(audit, img)


def _with_visual(audit: AuditResponse, img) -> AuditResponse:
    """EXPERIMENTAL opt-in visual pattern-type signals (CLEARCART_VISUAL=1). Attached AFTER the core audit and the
    intelligence layer so they cannot affect findings, summary, CDI or warnings."""
    from backend import visual
    if not visual.enabled():
        return audit
    audit.visual_signals = [VisualSignal(**s) for s in visual.predict(img)]
    v = visual.version()
    if v and audit.versions is not None:
        audit.versions["visual_model"] = v
    return audit


# ---------------------------------------------------------------------------- journey audit
MAX_STAGES = 8


def _sha256(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


async def _journey_form(request: Request):
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > MAX_JOURNEY_BYTES:
        raise HTTPException(413, f"The journey request exceeds the {MAX_JOURNEY_BYTES // (1024 * 1024)} MB limit.")
    try:
        return await request.form(max_part_size=MAX_HTML_BYTES * 2)
    except Exception:
        raise HTTPException(413, "A journey stage exceeds the size limit (2 MB HTML/text, 8 MB image).")


def _parse_stages(raw: str | None, field: str) -> list[JourneyStageInput]:
    from pydantic import TypeAdapter, ValidationError
    if not isinstance(raw, str) or not raw.strip():
        raise HTTPException(422, f"Provide the journey as a JSON list in the form field '{field}'.")
    try:
        inputs = TypeAdapter(list[JourneyStageInput]).validate_json(raw)
    except ValidationError as e:
        raise HTTPException(422, "Invalid journey stages: " + "; ".join(
            f"{'/'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()[:5]))
    if not 1 <= len(inputs) <= MAX_STAGES:
        raise HTTPException(422, f"A journey needs between 1 and {MAX_STAGES} stages (got {len(inputs)}).")
    return inputs


async def _read_stage_files(form, inputs: list[JourneyStageInput]) -> dict[int, bytes]:
    files: dict[int, bytes] = {}
    for i, si in enumerate(inputs):
        if si.input_type != "screenshot":
            continue
        up = form.get(si.file_field or "")
        if up is None or isinstance(up, str):
            raise HTTPException(422, f"Stage {i + 1} ('{si.name}'): missing screenshot file (multipart field '{si.file_field}').")
        files[i] = await up.read(MAX_IMAGE_BYTES + 1)
    return files


def _journey_core(inputs: list[JourneyStageInput], files: dict[int, bytes]) -> JourneyResponse:
    """Audit every stage with the existing pipeline, then run the journey intelligence engine (one request cache)."""
    import time
    import uuid
    from datetime import datetime, timezone
    t0 = time.perf_counter()
    if "intel" not in STATE:
        raise HTTPException(503, "Model is not loaded yet.")
    stages, warnings, hashes = [], [], []
    with request_scorer(STATE["clf"]) as scorer:
        for i, si in enumerate(inputs):
            where = f"Stage {i + 1} ('{si.name}'): "
            if si.captured_at:
                try:
                    datetime.fromisoformat(si.captured_at.replace("Z", "+00:00"))
                except ValueError:
                    raise HTTPException(422, f"{where}captured_at must be an ISO-8601 timestamp.")
            if si.stage_type is not None and si.stage_type not in CANONICAL_STAGES:
                raise HTTPException(422, f"{where}stage_type must be one of {', '.join(CANONICAL_STAGES)}.")
            if si.fuse_with is not None and si.fuse_with >= i:
                raise HTTPException(422, f"{where}fuse_with must point to an EARLIER stage (0-based index < {i}).")
            html = img = None
            if si.input_type == "text":
                audit, cands = _run_text(si.content or "", where)
                hashes.append(_sha256((si.content or "").encode("utf-8")))
            elif si.input_type == "html":
                audit, cands = _run_html(si.content or "", where)
                html = si.content
                hashes.append(_sha256((si.content or "").encode("utf-8")))
            else:
                audit, cands, img = _run_shot(files[i], where)
                hashes.append(_sha256(files[i]))
            audit.versions = None                       # stamped once on the journey
            declared = [(si.choice_path.accept_steps, si.choice_path.decline_steps, si.choice_path.label)] if si.choice_path else None
            stages.append({"index": i, "name": si.name, "stage_type": si.stage_type, "input_type": si.input_type,
                           "captured_at": si.captured_at, "audit": audit, "cands": cands, "html": html, "image": img,
                           "declared": declared, "fuse_with": si.fuse_with})
        out = STATE["intel"].journey(stages)
        log.info("journey: %d stages, classifier %s", len(stages), scorer.stats())
    stage_results = [JourneyStageResult(
        index=s["index"], name=s["name"], stage_type=s["stage_type"], position=out["positions"][k],
        input_type=s["input_type"], captured_at=s["captured_at"], synthetic=inputs[k].synthetic,
        content=inputs[k].content if s["input_type"] != "screenshot" else None,
        audit=s["audit"], pressure=round(out["pressures"][k], 4), fused_into=out["fused_into"][k]) for k, s in enumerate(stages)]
    if any(r.synthetic for r in inputs):
        warnings.append("This journey contains authored (synthetic) demonstration content; see the stage badges.")
    if out["fusion"]:
        warnings.append(f"{sum(len(g.stage_indices) - 1 for g in out['fusion'])} stage(s) were merged with an earlier capture of "
                        "the same page, so their findings are counted once.")
    versions = _versions()
    replay = ReplayRecord(
        inputs=[si.model_copy(update={"content": None}) if si.input_type == "screenshot" else si for si in inputs],
        input_sha256=hashes, versions=versions,
        expected={"fspm": out["fspm"].scores, "cdi": out["cdi"].cdi,
                  "stage_pressures": [round(p, 4) for p in out["pressures"]],
                  "findings": [[[f.id, f.category, f.model_score] for f in s["audit"].findings] for s in stages]})
    return JourneyResponse(
        journey_id=uuid.uuid4().hex[:12], created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        processing_ms=int((time.perf_counter() - t0) * 1000), stages=stage_results, fspm=out["fspm"],
        temporal=out["temporal"], cdi=out["cdi"], mcai=out["mcai"], counterfactual=out["counterfactual"],
        config_version=CONFIG_VERSION, disclaimer=JOURNEY_DISCLAIMER, warnings=warnings, versions=versions,
        fusion=out["fusion"], replay=replay)


@app.post("/api/audit/journey", response_model=JourneyResponse)
async def audit_journey(request: Request):
    form = await _journey_form(request)
    inputs = _parse_stages(form.get("stages"), "stages")
    files = await _read_stage_files(form, inputs)
    return _journey_core(inputs, files)


@app.post("/api/audit/journey/replay", response_model=ReplayResult)
async def replay_journey(request: Request):
    """Reconstruct a journey from its replay record (+ the original screenshot files) and verify determinism."""
    form = await _journey_form(request)
    raw = form.get("replay")
    if not isinstance(raw, str) or not raw.strip():
        raise HTTPException(422, "Provide the journey's replay record as JSON in the form field 'replay'.")
    try:
        rec = ReplayRecord.model_validate_json(raw)
    except Exception:
        raise HTTPException(422, "The replay record is not valid (expected the 'replay' object of a journey response).")
    inputs = rec.inputs
    if not 1 <= len(inputs) <= MAX_STAGES or len(rec.input_sha256) != len(inputs):
        raise HTTPException(422, "The replay record is inconsistent (stage count / hashes).")
    files = await _read_stage_files(form, inputs)
    for i, si in enumerate(inputs):
        data = files[i] if si.input_type == "screenshot" else (si.content or "").encode("utf-8")
        if _sha256(data) != rec.input_sha256[i]:
            raise HTTPException(422, f"Stage {i + 1} ('{si.name}') does not match the original input (SHA-256 differs); "
                                     "replay needs the identical file/content.")
    journey = _journey_core(inputs, files)
    diffs: list[str] = []
    for k, v in rec.versions.items():
        if journey.versions.get(k) != v:
            diffs.append(f"version '{k}' changed: {v} -> {journey.versions.get(k)}")
    exp, got = rec.expected, journey.replay.expected
    for key in ("fspm", "cdi", "stage_pressures", "findings"):
        if exp.get(key) != got.get(key):
            diffs.append(f"'{key}' differs: original {json.dumps(exp.get(key))[:200]} vs replay {json.dumps(got.get(key))[:200]}")
    return ReplayResult(identical=not diffs, differences=diffs, journey=journey)


JOURNEY_DISCLAIMER = ("Journey indices (FSPM, MCAI, CDI) are interpretable screening indices built on ClearCart's calibrated "
                      "classifier and direct observations. They are not probabilities that a merchant is fraudulent and have "
                      "only been evaluated on synthetic, explicitly labelled benchmarks. Fibonacci weighting is a proposed "
                      "weighting, shown alongside equal and linear baselines.")


@app.get("/api/model-info", response_model=ModelInfo)
def model_info():
    clf = STATE.get("clf")
    if clf is None:
        raise HTTPException(503, "Model is not loaded yet.")
    if STATE.get("model_version") == "v2":
        return _model_info_v2(clf)
    m = STATE["metrics"]
    ft = m["final_test"]
    return ModelInfo(
        name="ClearCart text dark-pattern classifier", version=clf.b.meta.get("trained_at", "1.0"),
        architecture=(f"Soft-voting ensemble: TF-IDF (word 1-2g + char 2-5g) + Logistic Regression (weight {clf.b.weight_lr}) and "
                      f"frozen MiniLM embeddings + surface features + XGBoost (weight {round(1-clf.b.weight_lr, 2)}); "
                      f"temperature-scaled (T={clf.b.temperature:.3f}); flag if calibrated P(dark) >= {clf.b.threshold}"),
        embedder=clf.b.meta["embedder"] + " (pretrained, frozen - not fine-tuned)",
        categories=[{"key": k, "label": CATEGORY_LABELS[k], "method": "ML classifier"} for k in clf.labels if k != "benign"]
                   + [{"key": "preselected_option", "label": CATEGORY_LABELS["preselected_option"], "method": "DOM observation (HTML only)"}],
        threshold=clf.b.threshold, temperature=round(clf.b.temperature, 4),
        training_data=f"yamanalab/ec-darkpattern dataset.tsv - train split only ({clf.b.meta['n_train']} rows); grouped by page and text template to prevent leakage",
        test_metrics={
            "macro_f1": ft["application_metrics"]["macro_f1"], "accuracy": ft["application_metrics"]["accuracy"],
            "dark_vs_benign_precision": ft["binary_dark_vs_benign"]["precision"], "dark_vs_benign_recall": ft["binary_dark_vs_benign"]["recall"],
            "dark_vs_benign_f1": ft["binary_dark_vs_benign"]["f1"], "roc_auc_ovr_macro": ft["application_metrics"]["roc_auc_ovr_macro"],
            "test_rows": sum(ft["confusion_matrix"][i][j] for i in range(5) for j in range(5)),
            "calibration_ece": m["E3_calibration"]["test_after"]["ece"],
        },
        limitations=[
            "Trained on short English e-commerce strings from 2019 shopping sites; other languages and domains are out of scope.",
            "Obstruction, sneaking and forced-action patterns had too few examples (43 rows total) to be learned as categories.",
            "Scores estimate how closely text matches labelled dark-pattern examples; they cannot verify whether a deadline or stock claim is actually false.",
            "Screenshot support is OCR + text classification; visual layout and styling are not analysed.",
            "Preselected-option findings are rule-based DOM observations, not model predictions.",
        ],
        ocr_available=bool(STATE.get("ocr")),
    )


def _model_info_v2(clf) -> ModelInfo:
    """Model card for the opt-in v2 model; every number is read from ml/artifacts_v2/metrics.json."""
    name = STATE["model_name"]
    r = STATE["metrics_v2"]["comparison"][name]
    t = r["test"]
    if getattr(clf, "kind", "") == "transformer":
        arch = (f"Fine-tuned {clf.b.meta.get('base_model', 'transformer')} sequence classifier (class-weighted loss, early stopping on VAL); "
                f"temperature-scaled (T={clf.b.temperature:.3f}); flag if calibrated P(dark) >= {clf.b.threshold}. Key terms: "
                "leave-one-word-out occlusion on the transformer; similar examples: cosine similarity of its pooled embeddings.")
        embedder = f"{clf.b.meta.get('base_model', 'transformer')} (pretrained, fine-tuned on the training split)"
    else:
        arch = (f"Soft-voting ensemble: TF-IDF + Logistic Regression (weight {clf.b.weight_lr}) and frozen MiniLM + XGBoost "
                f"(weight {round(1 - clf.b.weight_lr, 2)}); temperature-scaled (T={clf.b.temperature:.3f}); flag if calibrated P(dark) >= {clf.b.threshold}")
        embedder = clf.b.meta["embedder"] + " (pretrained, frozen - not fine-tuned)"
    return ModelInfo(
        name=f"ClearCart text dark-pattern classifier ({name}, opt-in v2)", version=str(clf.b.meta.get("trained_at", "v2")),
        architecture=arch, embedder=embedder,
        categories=[{"key": k, "label": CATEGORY_LABELS[k], "method": "ML classifier"} for k in clf.labels if k != "benign"]
                   + [{"key": "preselected_option", "label": CATEGORY_LABELS["preselected_option"], "method": "DOM observation (HTML only)"}],
        threshold=clf.b.threshold, temperature=round(clf.b.temperature, 4),
        training_data=f"{clf.b.meta.get('trained_on', '')} ({clf.b.meta.get('n_train', '?')} rows); evaluated on the frozen ec-darkpattern TEST split",
        test_metrics={
            "macro_f1": t["macro_f1"], "accuracy": t["accuracy"],
            "dark_vs_benign_precision": t["dark_vs_benign"]["precision"], "dark_vs_benign_recall": t["dark_vs_benign"]["recall"],
            "dark_vs_benign_f1": t["dark_vs_benign"]["f1"], "roc_auc_ovr_macro": t["roc_auc_ovr_macro"],
            "test_rows": t["n"], "calibration_ece": t["ece_after_calibration"],
        },
        limitations=[
            "Opt-in v2 model (CLEARCART_MODEL=v2); the default served model is unchanged.",
            "Trained on short English e-commerce strings from 2019 shopping sites; other languages and domains are out of scope.",
            "Scores estimate how closely text matches labelled dark-pattern examples; they cannot verify whether a deadline or stock claim is actually false.",
            "The screenshot OCR threshold was selected for the v1 model and has not been re-tuned for v2.",
            "Preselected-option findings are rule-based DOM observations, not model predictions.",
        ],
        ocr_available=bool(STATE.get("ocr")),
    )


@app.get("/api/samples", response_model=list[SampleInput])
def samples():
    path = ROOT / "data" / "samples" / "samples.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


@app.post("/api/export/pdf")
def export_pdf(audit: AuditResponse):
    from backend.pdf_export import render_audit_pdf
    pdf = render_audit_pdf(audit)
    return Response(content=pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="clearcart-audit-{audit.audit_id}.pdf"'})


@app.get("/api/samples/journeys")
def journey_samples():
    """Demo journeys. Synthetic ones are flagged and carry provenance (real test-split strings vs authored text)."""
    path = ROOT / "data" / "samples" / "journeys.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


@app.post("/api/export/journey-pdf")
def export_journey_pdf(journey: JourneyResponse):
    from backend.pdf_export import render_journey_pdf
    pdf = render_journey_pdf(journey)
    return Response(content=pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="clearcart-journey-{journey.journey_id}.pdf"'})
