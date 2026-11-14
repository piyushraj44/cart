"""Experimental VISUAL pattern-type model on ContextRico-DP (multi-label, opt-in; never changes the served text model).

Stages (run in order, each as a module from the project root):
    .venv/bin/python -m ml.visual_data                              # split + image cache
    .venv/bin/python -m ml.train_visual criteria                     # fix acceptance criteria BEFORE any training
    .venv/bin/python -m ml.train_visual baselines                    # label prior + frozen CLIP linear probe (VAL only)
    .venv/bin/python -m ml.train_visual train --arch mobilenet_v3_large   # (also efficientnet_b0, convnext_tiny)
    .venv/bin/python -m ml.train_visual cv --arch <arch>             # 3 grouped re-splits of TRAIN+VAL (variance only)
    .venv/bin/python -m ml.train_visual finalize                     # select on VAL, calibrate, TEST once, figures, export

Selection, temperature scaling and thresholds use VAL only. TEST is read only by `finalize`, once, after selection.
ContextRico has no benign screenshots, so this model indicates a visual pattern TYPE; it is not a deceptive-vs-benign detector.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from ml.common import ROOT, SEED
from ml.visual_data import ART, CR_DIR, REQUIRED, SIZE, SPLIT_CSV, TARGETS, cache_images, iterative_split
from ml.visual_model import (ARCHS, build, calibrated, device, ece, evidence_box, fit_temperature, gradcam, head_params,
                             to_original, to_tensor)

RUNS = ART / "runs"
CRITERIA = ART / "criteria.json"
METRICS = ART / "metrics.json"
MODEL_PT = ART / "model.pt"
CONFIG = ART / "config.json"
FIG = ROOT / "report" / "figures"
MIN_PRECISION = 0.70
AP_MARGIN = 0.10


# ------------------------------------------------------------------ data
def load_data():
    df = pd.read_csv(SPLIT_CSV, keep_default_na=False)
    z = np.load(cache_images(df), allow_pickle=False)
    assert list(z["names"]) == list(df.image)
    return df, z["images"], df[TARGETS].to_numpy(dtype=np.float32)


def scores(y: np.ndarray, p: np.ndarray, labels=TARGETS) -> dict:
    out = {}
    for j, t in enumerate(labels):
        yt = y[:, j]
        out[t] = {"prevalence": round(float(yt.mean()), 4), "n_pos": int(yt.sum()),
                  "ap": round(float(average_precision_score(yt, p[:, j])), 4) if yt.any() else None,
                  "auroc": round(float(roc_auc_score(yt, p[:, j])), 4) if 0 < yt.sum() < len(yt) else None}
    req = [out[t]["ap"] for t in REQUIRED if out[t]["ap"] is not None]
    return {"per_label": out, "macro_ap_required": round(float(np.mean(req)), 4)}


def _save(name: str, obj: dict) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{name}.json").write_text(json.dumps(obj, indent=2))


def _load(name: str) -> dict:
    return json.loads((RUNS / f"{name}.json").read_text())


# ------------------------------------------------------------------ stage: criteria
def stage_criteria() -> None:
    if CRITERIA.exists():
        print("criteria already fixed at", json.loads(CRITERIA.read_text())["fixed_at"]); return
    ART.mkdir(parents=True, exist_ok=True)
    CRITERIA.write_text(json.dumps({
        "fixed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selection": "highest VAL macro average precision over the required labels among the fine-tuned candidates",
        "a": "the selected model beats BOTH baselines (label prior, frozen CLIP linear probe) on VAL macro-AP (required labels)",
        "b": f"for every required label, VAL AP > VAL prevalence + {AP_MARGIN}",
        "c": "TEST is evaluated exactly once, after selection, by `finalize`, and reported whatever it shows",
        "optional_labels": f"{', '.join(t for t in TARGETS if t not in REQUIRED)} are shown only if VAL AP > prevalence + {AP_MARGIN} "
                           f"and a threshold with VAL precision >= {MIN_PRECISION} exists",
        "thresholds": f"per label, highest-F1 VAL threshold with VAL precision >= {MIN_PRECISION}; otherwise the label is 'not reliable' and never shown",
        "required_labels": REQUIRED, "all_labels": TARGETS,
    }, indent=2))
    print("criteria fixed:", CRITERIA)


# ------------------------------------------------------------------ stage: baselines
def clip_embeddings(images: np.ndarray) -> np.ndarray:
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    name = "openai/clip-vit-base-patch32"
    dev = device()
    model = CLIPModel.from_pretrained(name).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(name)
    out = []
    with torch.no_grad():
        for i in range(0, len(images), 64):
            batch = [Image.fromarray(a) for a in images[i:i + 64]]
            px = proc(images=batch, return_tensors="pt")["pixel_values"].to(dev)
            f = model.get_image_features(pixel_values=px)
            f = f.pooler_output if hasattr(f, "pooler_output") else f
            out.append(torch.nn.functional.normalize(f, dim=-1).float().cpu().numpy())
    return np.concatenate(out)


def stage_baselines() -> None:
    df, images, Y = load_data()
    tr, va = (df.split == "train").to_numpy(), (df.split == "val").to_numpy()
    prior = np.tile(Y[tr].mean(0), (va.sum(), 1))
    _save("prior", {"model": "label prior (TRAIN prevalence)", "val": scores(Y[va], prior)})
    t0 = time.perf_counter()
    emb = clip_embeddings(images)
    np.save(ART / "cache" / "clip_b32.npy", emb)
    t_emb = time.perf_counter() - t0
    pv, chosen = np.zeros((va.sum(), len(TARGETS))), {}
    for j, t in enumerate(TARGETS):
        best = None
        for C in (0.01, 0.1, 1.0, 10.0):
            lr = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(emb[tr], Y[tr, j])
            ap = average_precision_score(Y[va, j], lr.predict_proba(emb[va])[:, 1])
            if best is None or ap > best[0]:
                best = (ap, C, lr)
        chosen[t] = best[1]
        pv[:, j] = best[2].predict_proba(emb[va])[:, 1]
    _save("clip_lr", {"model": "frozen CLIP ViT-B/32 image embedding + one-vs-rest logistic regression (balanced), C per label on VAL",
                      "C": chosen, "embedding_seconds": round(t_emb, 1), "val": scores(Y[va], pv)})
    print("prior VAL macro-AP", _load("prior")["val"]["macro_ap_required"], "| CLIP probe", _load("clip_lr")["val"]["macro_ap_required"])


# ------------------------------------------------------------------ stage: train
def augment(x_uint8: torch.Tensor) -> torch.Tensor:
    from torchvision.transforms import v2
    t = v2.Compose([v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                    v2.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.9, 1.0), fill=255)])
    return torch.stack([t(img) for img in x_uint8])


def _norm(x_uint8: torch.Tensor) -> torch.Tensor:
    return to_tensor(x_uint8.permute(0, 2, 3, 1).numpy())


def predict_logits(model, images: np.ndarray, bs: int = 32) -> np.ndarray:
    dev = device(); model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(images), bs):
            out.append(model(to_tensor(images[i:i + bs]).to(dev)).float().cpu().numpy())
    return np.concatenate(out)


def fit(arch: str, images: np.ndarray, Y: np.ndarray, tr: np.ndarray, va: np.ndarray, seed: int,
        epochs: int = 15, patience: int = 3, bs: int = 16) -> tuple[torch.nn.Module, dict]:
    torch.manual_seed(seed); np.random.seed(seed)
    dev = device()
    model = build(arch, len(TARGETS)).to(dev)
    head = head_params(model); hid = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in hid]
    opt = torch.optim.AdamW([{"params": head, "lr": 3e-4}, {"params": body, "lr": 3e-5}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ytr = Y[tr]
    pos = ytr.sum(0); pw = torch.tensor((len(ytr) - pos) / np.maximum(pos, 1), dtype=torch.float32, device=dev)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    Xtr = torch.from_numpy(images[tr]).permute(0, 3, 1, 2)
    Yt = torch.from_numpy(ytr)
    rng = np.random.default_rng(seed)
    hist, best, best_state, bad = [], -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.perf_counter(); tot = 0.0
        for idx in np.array_split(rng.permutation(len(Xtr)), int(np.ceil(len(Xtr) / bs))):
            xb = _norm(augment(Xtr[idx])).to(dev)
            loss = lossf(model(xb), Yt[idx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        sched.step()
        pv = 1 / (1 + np.exp(-predict_logits(model, images[va])))
        m = scores(Y[va], pv)["macro_ap_required"]
        hist.append({"epoch": ep, "train_loss": round(tot / len(Xtr), 4), "val_macro_ap": m, "seconds": round(time.perf_counter() - t0, 1)})
        print(arch, hist[-1], flush=True)
        if m > best:
            best, bad = m, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, {"history": hist, "best_epoch": max(hist, key=lambda h: h["val_macro_ap"])["epoch"], "best_val_macro_ap": best}


def stage_train(arch: str) -> None:
    df, images, Y = load_data()
    tr, va = (df.split == "train").to_numpy(), (df.split == "val").to_numpy()
    t0 = time.perf_counter()
    model, info = fit(arch, images, Y, tr, va, SEED)
    wall = time.perf_counter() - t0
    RUNS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), RUNS / f"{arch}.pt")
    lv = predict_logits(model, images[va])
    np.save(RUNS / f"{arch}_val_logits.npy", lv)
    _save(arch, {"model": f"{arch} ({ARCHS[arch]}) fine-tuned, letterbox {SIZE}", "seed": SEED, **info,
                 "wall_seconds": round(wall, 1), "val": scores(Y[va], 1 / (1 + np.exp(-lv)))})
    print(arch, "VAL macro-AP", _load(arch)["val"]["macro_ap_required"], f"({wall:.0f}s)")


def stage_cv(arch: str, seeds=(1, 2, 3)) -> None:
    df, images, Y = load_data()
    pool = df[df.split.isin(["train", "val"])].copy()
    res = []
    for s in seeds:
        sp = iterative_split(pool, seed=s, ratios={"train": 0.82, "val": 0.18})
        tr = df.index.isin(pool.index[sp == "train"]); va = df.index.isin(pool.index[sp == "val"])
        t0 = time.perf_counter()
        model, info = fit(arch, images, Y, tr, va, s)
        res.append({"seed": s, "val_macro_ap": info["best_val_macro_ap"], "best_epoch": info["best_epoch"],
                    "seconds": round(time.perf_counter() - t0, 1)})
        print("cv", arch, res[-1], flush=True)
    v = [r["val_macro_ap"] for r in res]
    _save(f"cv_{arch}", {"protocol": "3 grouped, label-stratified re-splits of TRAIN+VAL (82/18), different seeds; TEST untouched",
                         "folds": res, "mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)})


# ------------------------------------------------------------------ stage: finalize
def thresholds_for(p: np.ndarray, y: np.ndarray) -> tuple[float | None, dict]:
    best = None
    for t in np.unique(np.round(p, 4)):
        pred = p >= t
        if pred.sum() == 0:
            continue
        prec = precision_score(y, pred, zero_division=0)
        if prec < MIN_PRECISION:
            continue
        f1 = f1_score(y, pred, zero_division=0)
        if best is None or f1 > best[1]:
            best = (float(t), f1, prec, recall_score(y, pred, zero_division=0))
    if best is None:
        return None, {}
    return best[0], {"f1": round(best[1], 4), "precision": round(best[2], 4), "recall": round(best[3], 4)}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weights_file(arch: str) -> Path | None:
    import torchvision.models as tvm
    hub = Path(torch.hub.get_dir()) / "checkpoints"
    enum = {"mobilenet_v3_large": "MobileNet_V3_Large_Weights", "efficientnet_b0": "EfficientNet_B0_Weights",
            "convnext_tiny": "ConvNeXt_Tiny_Weights"}[arch]
    # the cached file is named after the weights URL (e.g. efficientnet_b0_rwightman-7f5810bc.pth), not "<arch>-*.pth"
    f = hub / Path(getattr(tvm, enum)[ARCHS[arch]].url).name
    return f if f.exists() else None


def segments_of() -> dict:
    gt = json.loads((CR_DIR / "groundtrue.json").read_text())
    return {k: v.get("segments") or [] for k, v in gt.items()}


def pointing_game(model, df, images, Y, P, idx, temps) -> dict:
    """On TEST images with ground-truth segments: does the Grad-CAM peak of the best-scoring TRUE target label fall in a segment?"""
    from PIL import Image
    seg = segments_of(); dev = device(); root = ROOT
    hits = center_hits = n = 0; ious = []
    for k, i in enumerate(idx):  # P holds TEST rows only, aligned with idx; i is the global row
        name = df.image.iloc[i]; s = seg.get(name) or []
        true = np.nonzero(Y[i])[0]
        if not s or not len(true):
            continue
        j = int(true[np.argmax(P[k, true])])
        with Image.open(root / df.path.iloc[i]) as im:
            W, H = im.size
        x = to_tensor(images[i:i + 1]).to(dev).requires_grad_(True)
        cam = gradcam(model, x, j)
        yy, xx = np.unravel_index(int(np.argmax(cam)), cam.shape)
        pt = to_original((xx, yy, 1, 1), W, H, SIZE)
        box = evidence_box(cam)
        ob = to_original(box, W, H, SIZE) if box else None
        inside = lambda px, py: any(a <= px <= a + w and b <= py <= b + h for a, b, w, h in s)
        hits += inside(pt["x"], pt["y"]); center_hits += inside(W / 2, H / 2); n += 1
        if ob:
            ious.append(max(_iou((ob["x"], ob["y"], ob["width"], ob["height"]), tuple(t)) for t in s))
    return {"n_images": n, "peak_in_segment_rate": round(hits / n, 4) if n else None,
            "image_centre_baseline_rate": round(center_hits / n, 4) if n else None,
            "mean_best_iou_evidence_box": round(float(np.mean(ious)), 4) if ious else None,
            "note": "segments are image-level dark-pattern regions from groundtrue.json (x, y, w, h); not per label"}


def _iou(a, b) -> float:
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax1, bx1) - max(a[0], b[0])), max(0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter) if inter else 0.0


def stage_finalize() -> None:
    crit = json.loads(CRITERIA.read_text())
    df, images, Y = load_data()
    va, te = (df.split == "val").to_numpy(), (df.split == "test").to_numpy()
    cands = [a for a in ARCHS if (RUNS / f"{a}.json").exists()]
    runs = {a: _load(a) for a in cands}
    sel = max(cands, key=lambda a: runs[a]["val"]["macro_ap_required"])
    prior, clip = _load("prior"), _load("clip_lr")
    vsel = runs[sel]["val"]
    a_pass = vsel["macro_ap_required"] > max(prior["val"]["macro_ap_required"], clip["val"]["macro_ap_required"])
    b_detail = {t: {"ap": vsel["per_label"][t]["ap"], "needs_more_than": round(vsel["per_label"][t]["prevalence"] + AP_MARGIN, 4)}
                for t in REQUIRED}
    b_pass = all(d["ap"] > d["needs_more_than"] for d in b_detail.values())

    dev = device()
    model = build(sel, len(TARGETS), pretrained=False).to(dev)
    model.load_state_dict(torch.load(RUNS / f"{sel}.pt", map_location=dev))
    lv = np.load(RUNS / f"{sel}_val_logits.npy")
    temps = np.array([fit_temperature(lv[:, j], Y[va, j]) for j in range(len(TARGETS))])
    pv_raw, pv = 1 / (1 + np.exp(-lv)), calibrated(lv, temps)
    labels_cfg = {}
    for j, t in enumerate(TARGETS):
        thr, at = thresholds_for(pv[:, j], Y[va, j])
        ap = vsel["per_label"][t]["ap"]; prev = vsel["per_label"][t]["prevalence"]
        reliable = thr is not None and (t in REQUIRED or ap > prev + AP_MARGIN)
        labels_cfg[t] = {"temperature": round(temps[j], 4), "threshold": thr, "reliable": bool(reliable), "val_at_threshold": at,
                         "ece_val_before": round(ece(pv_raw[:, j], Y[va, j]), 4), "ece_val_after": round(ece(pv[:, j], Y[va, j]), 4)}

    # ---- TEST, once
    lt = predict_logits(model, images[te])
    np.save(ART / "test_logits.npy", lt)  # lets `figures` redraw plots without re-running TEST inference
    pt = calibrated(lt, temps)
    test = scores(Y[te], pt)
    for j, t in enumerate(TARGETS):
        c = labels_cfg[t]
        if c["threshold"] is not None:
            pred = pt[:, j] >= c["threshold"]
            test["per_label"][t].update(precision_at_val_threshold=round(precision_score(Y[te, j], pred, zero_division=0), 4),
                                        recall_at_val_threshold=round(recall_score(Y[te, j], pred, zero_division=0), 4),
                                        f1_at_val_threshold=round(f1_score(Y[te, j], pred, zero_division=0), 4))
        test["per_label"][t]["ece_after_calibration"] = round(ece(pt[:, j], Y[te, j]), 4)
    by_subset = {s: scores(Y[te][df[te].subset.to_numpy() == s], pt[df[te].subset.to_numpy() == s])["macro_ap_required"]
                 for s in ("Mobile", "Web") if (df[te].subset == s).sum() >= 5}
    # baselines on TEST for comparison (computed once, here)
    tr = (df.split == "train").to_numpy()
    prior_test = scores(Y[te], np.tile(Y[tr].mean(0), (te.sum(), 1)))
    emb = np.load(ART / "cache" / "clip_b32.npy")
    pc = np.zeros((te.sum(), len(TARGETS)))
    for j, t in enumerate(TARGETS):
        pc[:, j] = LogisticRegression(C=clip["C"][t], class_weight="balanced", max_iter=5000).fit(emb[tr], Y[tr, j]).predict_proba(emb[te])[:, 1]
    clip_test = scores(Y[te], pc)
    pointing = pointing_game(model, df, images, Y, pt, np.nonzero(te)[0], temps)

    # ---- export
    torch.save(model.state_dict(), MODEL_PT)
    wf = weights_file(sel)
    cfg = {"arch": sel, "pretrained_weights": ARCHS[sel], "pretrained_file": wf.name if wf else None,
           "pretrained_sha256": sha256(wf) if wf else None, "size": SIZE, "labels": TARGETS, "label_config": labels_cfg,
           "model_sha256": sha256(MODEL_PT), "trained_on": "ContextRico-DP TRAIN split (data/processed/visual_split.csv)",
           "provenance": "model (visual, experimental)"}
    CONFIG.write_text(json.dumps(cfg, indent=2))
    probes = run_probes(model, labels_cfg, df)
    figures(df, images, Y, runs, prior, clip, test, pt, te, model, labels_cfg)
    metrics = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "criteria": crit,
        "split": json.loads((SPLIT_CSV.parent / "visual_split_stats.json").read_text()),
        "val": {"prior": prior["val"], "clip_lr": clip["val"], **{a: runs[a]["val"] for a in cands}},
        "training": {a: {k: runs[a][k] for k in ("history", "best_epoch", "wall_seconds")} for a in cands},
        "cv": {a: _load(f"cv_{a}") for a in cands if (RUNS / f"cv_{a}.json").exists()},
        "selected": sel, "clip_C": clip["C"], "clip_embedding_seconds": clip["embedding_seconds"],
        "criteria_result": {"a_beats_both_baselines_on_val": a_pass, "b_per_label_val_ap_margin": b_detail, "b_pass": b_pass,
                            "c_test_evaluated_once": True, "all_pass": bool(a_pass and b_pass)},
        "label_config": labels_cfg,
        "test": {"selected": test, "prior": prior_test, "clip_lr": clip_test, "selected_macro_ap_by_subset": by_subset,
                 "n_images": int(te.sum())},
        "pointing_game_test": pointing, "probes_not_metrics": probes, "export": cfg,
    }
    METRICS.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"selected": sel, "criteria": metrics["criteria_result"], "test_macro_ap": test["macro_ap_required"],
                      "pointing": pointing}, indent=1))


def run_probes(model, labels_cfg, df) -> list:
    """Sanity probes (not metrics): ClearCart's own known-good screenshots and the demo sample screenshots."""
    from PIL import Image
    from ml.visual_data import letterbox
    temps = np.array([labels_cfg[t]["temperature"] for t in TARGETS])
    out = []
    files = sorted((ROOT / "data" / "probes" / "clearcart_pages").glob("*.png")) + \
        sorted(p for p in (ROOT / "data" / "samples" / "screenshots").iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    split_of = dict(zip(df.image, df.split))
    for f in files:
        with Image.open(f) as im:
            arr = letterbox(im)[None]
        p = calibrated(predict_logits(model, arr), temps)[0]
        fired = [t for j, t in enumerate(TARGETS) if labels_cfg[t]["reliable"] and p[j] >= labels_cfg[t]["threshold"]]
        out.append({"file": str(f.relative_to(ROOT)), "known_good": "clearcart_pages" in str(f),
                    "contextrico_split": split_of.get(f.name), "fired": fired,
                    "probabilities": {t: round(float(p[j]), 3) for j, t in enumerate(TARGETS)}})
    return out


def figures(df, images, Y, runs, prior, clip, test, pt, te, model, labels_cfg) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    FIG.mkdir(parents=True, exist_ok=True)
    ink, muted, grid = "#0f172a", "#475569", "#e2e8f0"

    def style(ax):
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#cbd5e1"); ax.grid(axis="x", color=grid, lw=0.6); ax.set_axisbelow(True)
        ax.tick_params(colors=muted, labelsize=8)

    # model comparison (VAL)
    names = ["Label prior", "CLIP probe"] + list(runs)
    vals = [prior["val"]["macro_ap_required"], clip["val"]["macro_ap_required"]] + [runs[a]["val"]["macro_ap_required"] for a in runs]
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    cols = ["#94a3b8", "#94a3b8"] + ["#0f766e"] * len(runs)
    ax.barh(names[::-1], vals[::-1], color=cols[::-1], height=0.6)
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8, color=ink)
    ax.set_xlim(0, 1); ax.set_title("VAL macro average precision, required labels", fontsize=9, color=ink, loc="left"); style(ax)
    fig.tight_layout(); fig.savefig(FIG / "visual_model_comparison.png", dpi=200); plt.close(fig)

    # per-label TEST AP vs prevalence
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ys = np.arange(len(TARGETS))[::-1]
    ap = [test["per_label"][t]["ap"] or 0 for t in TARGETS]; pr = [test["per_label"][t]["prevalence"] for t in TARGETS]
    ax.barh(ys + 0.18, ap, height=0.34, color="#0f766e", label="Selected model AP")
    ax.barh(ys - 0.18, pr, height=0.34, color="#94a3b8", label="Prevalence (chance AP)")
    for y, a, p in zip(ys, ap, pr):
        ax.text(a + 0.01, y + 0.18, f"{a:.2f}", va="center", fontsize=7.5, color=ink)
        ax.text(p + 0.01, y - 0.18, f"{p:.2f}", va="center", fontsize=7.5, color=ink)
    ax.set_yticks(ys); ax.set_yticklabels(TARGETS); ax.set_xlim(0, 1.08)
    ax.set_title("TEST average precision per label vs chance", fontsize=9, color=ink, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left", bbox_to_anchor=(0.62, 1.0), ncol=1, borderaxespad=0.2); style(ax)
    fig.tight_layout(); fig.savefig(FIG / "visual_per_label.png", dpi=200); plt.close(fig)

    # reliability (TEST, pooled over labels, after calibration)
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    p, y = pt.flatten(), Y[te].flatten()
    edges = np.linspace(0, 1, 11); xs, ys2 = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() >= 5:
            xs.append(p[m].mean()); ys2.append(y[m].mean())
    ax.plot([0, 1], [0, 1], color="#cbd5e1", lw=1)
    ax.plot(xs, ys2, marker="o", ms=4, color="#0f766e", lw=2)
    ax.set_xlabel("Predicted probability", fontsize=8, color=muted); ax.set_ylabel("Observed frequency", fontsize=8, color=muted)
    ax.set_title("TEST reliability (all labels)", fontsize=9, color=ink, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=muted, labelsize=7.5)
    fig.tight_layout(); fig.savefig(FIG / "visual_reliability.png", dpi=200); plt.close(fig)

    # Grad-CAM examples: 6 TEST images, highest-confidence true positive per label where possible
    dev = device(); idx = np.nonzero(te)[0]; picks = []
    for j, t in enumerate(TARGETS[:6]):
        tp = [k for k in range(len(idx)) if Y[idx[k], j] == 1 and idx[k] not in [q[0] for q in picks]]
        if tp:
            k = max(tp, key=lambda k: pt[k, j]); picks.append((idx[k], j, pt[k, j]))
    fig, axes = plt.subplots(1, len(picks), figsize=(2.0 * len(picks), 2.6))
    for ax, (i, j, prob) in zip(np.atleast_1d(axes), picks):
        cam = gradcam(model, to_tensor(images[i:i + 1]).to(dev).requires_grad_(True), j)
        ax.imshow(images[i]); ax.imshow(cam, cmap="magma", alpha=0.35)
        b = evidence_box(cam)
        if b:
            ax.add_patch(Rectangle((b[0], b[1]), b[2], b[3], fill=False, ec="#22d3ee", lw=1.5))
        ax.set_title(f"{TARGETS[j]}\n{prob:.2f}", fontsize=7, color=ink); ax.axis("off")
    fig.tight_layout(); fig.savefig(FIG / "visual_gradcam.png", dpi=200, bbox_inches="tight", pad_inches=0.05); plt.close(fig)


def stage_figures() -> None:
    """Redraw report/figures/visual_*.png from what `finalize` saved (no selection, no new TEST scoring)."""
    m = json.loads(METRICS.read_text())
    df, images, Y = load_data()
    te = (df.split == "test").to_numpy()
    sel, labels_cfg = m["selected"], m["label_config"]
    temps = np.array([labels_cfg[t]["temperature"] for t in TARGETS])
    pt = calibrated(np.load(ART / "test_logits.npy"), temps)
    runs = {a: _load(a) for a in ARCHS if (RUNS / f"{a}.json").exists()}
    model = build(sel, len(TARGETS), pretrained=False).to(device())
    model.load_state_dict(torch.load(MODEL_PT, map_location=device()))
    model.eval()
    figures(df, images, Y, runs, _load("prior"), _load("clip_lr"), m["test"]["selected"], pt, te, model, labels_cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["criteria", "baselines", "train", "cv", "finalize", "figures"])
    ap.add_argument("--arch", choices=list(ARCHS))
    a = ap.parse_args()
    if a.stage in ("train", "cv") and not a.arch:
        ap.error("--arch is required")
    if a.stage != "criteria" and not CRITERIA.exists():
        raise SystemExit("run `criteria` first: acceptance criteria must be fixed before any model is trained")
    {"criteria": stage_criteria, "baselines": stage_baselines, "finalize": stage_finalize,
     "figures": stage_figures}.get(a.stage, lambda: None)()
    if a.stage == "train":
        stage_train(a.arch)
    if a.stage == "cv":
        stage_cv(a.arch)


if __name__ == "__main__":
    main()
