"""EXPERIMENTAL, OPT-IN visual pattern-type model for screenshot audits.

Enabled only when CLEARCART_VISUAL=1 AND the model files (config.json + model.pt) exist in CLEARCART_VISUAL_DIR
(default ml/artifacts_visual/). The model was trained on ContextRico screenshots, which ALL contain dark patterns, so its
output indicates a visual pattern TYPE, never whether a page is deceptive. Its signals are reported separately and never
enter findings, the risk summary, the intelligence layer or the warnings.

Serving runs on CPU for determinism. Any failure returns [] and logs a warning that contains no user content.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path

from ml.common import ROOT

log = logging.getLogger("clearcart.visual")
DEFAULT_DIR = ROOT / "ml" / "artifacts_visual"
PROVENANCE = "model (visual, experimental)"

_LOCK = threading.Lock()
_CACHE: dict = {}          # {"key": (dir, mtimes), "model": nn.Module, "cfg": dict}


def model_dir() -> Path:
    return Path(os.environ.get("CLEARCART_VISUAL_DIR") or DEFAULT_DIR)


def _files(d: Path) -> tuple[Path, Path]:
    return d / "config.json", d / "model.pt"


def enabled() -> bool:
    if os.environ.get("CLEARCART_VISUAL", "").strip() != "1":
        return False
    cfg, pt = _files(model_dir())
    return cfg.is_file() and pt.is_file()


def reset() -> None:
    """Drop the cached model (tests, or after replacing the artifacts)."""
    with _LOCK:
        _CACHE.clear()


def _load():
    """Lazy, thread-safe, cached load. Reloads if the directory or file mtimes change. Caller holds _LOCK."""
    d = model_dir()
    cfg_p, pt_p = _files(d)
    key = (str(d.resolve()), cfg_p.stat().st_mtime_ns, pt_p.stat().st_mtime_ns)
    if _CACHE.get("key") == key:
        return _CACHE["model"], _CACHE["cfg"]
    _CACHE.clear()
    import torch
    from ml.visual_model import build
    cfg = json.loads(cfg_p.read_text())
    labels = list(cfg["labels"])
    model = build(cfg["arch"], len(labels), pretrained=False)
    state = torch.load(pt_p, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _CACHE.update(key=key, model=model, cfg=cfg)
    return model, cfg


def config() -> dict | None:
    """The loaded config (None if disabled or unreadable)."""
    if not enabled():
        return None
    try:
        with _LOCK:
            return _load()[1]
    except Exception as exc:
        log.warning("visual model could not be loaded (%s)", type(exc).__name__)
        return None


def version() -> str | None:
    """Short version string for the response `versions` dict: arch + first 12 chars of the model sha256."""
    cfg = config()
    if cfg is None:
        return None
    sha = str(cfg.get("model_sha256") or "")
    if not sha:
        try:
            sha = hashlib.sha256(_files(model_dir())[1].read_bytes()).hexdigest()
        except OSError:
            sha = ""
    return f"{cfg.get('arch', '?')}:{sha[:12]}" if sha else str(cfg.get("arch", "?"))


def predict(image) -> list[dict]:
    """Calibrated visual pattern-type signals above each reliable label's threshold, highest probability first.

    Never raises: on any failure returns [] and logs a warning without user content.
    """
    try:
        import numpy as np
        import torch
        from ml.visual_data import letterbox
        from ml.visual_model import calibrated, evidence_box, gradcam, to_original, to_tensor
        with _LOCK:                                   # Grad-CAM uses hooks + model gradients: serialise access
            model, cfg = _load()
            size = int(cfg["size"])
            labels = list(cfg["labels"])
            lc = cfg.get("label_config") or {}
            temps = np.array([float((lc.get(l) or {}).get("temperature") or 1.0) for l in labels], dtype=np.float64)
            arr = np.array(letterbox(image, size)[None])          # writable copy for torch.from_numpy
            x = to_tensor(arr)
            with torch.no_grad():
                logits = model(x)[0].float().cpu().numpy().astype(np.float64)
            probs = calibrated(logits, temps)
            out = []
            w, h = image.size
            for i, label in enumerate(labels):
                c = lc.get(label) or {}
                thr = c.get("threshold")
                if not c.get("reliable") or thr is None:
                    continue
                p = float(probs[i])
                if not np.isfinite(p) or p < float(thr):
                    continue
                bbox = None
                try:
                    with torch.enable_grad():
                        cam = gradcam(model, x.clone(), i)
                    box = evidence_box(cam)
                    if box is not None:
                        bbox = to_original(box, w, h, size)
                except Exception as exc:          # evidence is optional; the signal still stands
                    log.warning("visual Grad-CAM failed (%s)", type(exc).__name__)
                finally:
                    model.zero_grad(set_to_none=True)
                out.append({"label": label, "probability": round(p, 4), "threshold": float(thr), "bbox": bbox,
                            "provenance": str(cfg.get("provenance") or PROVENANCE)})
        out.sort(key=lambda s: -s["probability"])
        return out
    except Exception as exc:
        log.warning("visual model prediction failed (%s)", type(exc).__name__)
        return []
