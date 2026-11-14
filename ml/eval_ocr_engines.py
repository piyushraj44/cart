"""Compare OCR engines for screenshot auditing: Tesseract (served v4_psm3) vs EasyOCR vs their union.

Run from the project root:  .venv/bin/python -m ml.eval_ocr_engines

Protocol (identical to ml/eval_screenshots.py, whose code is reused, not copied):
  * eval set + 50/50 DEV/TEST split by image (seed 42): ml.eval_screenshots.build_eval_set
  * text classifier: ml.model.TextClassifier.load(ml/artifacts) - the CURRENT served model, never retrained here
  * image-level decision rule + metrics: ml.eval_screenshots.metrics (flag if any OCR candidate has p_dark >= tau)
  * per-engine (and joint) operating point picked on DEV with ml.eval_screenshots.choose (same rule, same constants);
    TEST is only reported.
Engines
  tesseract : backend.extractors.ocr_lines, v4_psm3 (served config) - reuses the eval_screenshots OCR cache
  easyocr   : backend.extractors.ocr_lines_easyocr (EasyOCR 1.7.x, CRAFT detector + english_g2 recogniser)
  union     : backend.extractors.union_candidates(tesseract, easyocr) - EasyOCR candidates whose box duplicates a
              Tesseract candidate of the same unit (IoU >= 0.5 or >= 80% of the smaller box covered) are dropped.
Also: synthetic choice screens S1-S8 (ml/artifacts/intel_synthetic_screens, from ml/benchmark_intelligence.py) through
each engine + compute_mcai, and a controlled sequential timing subsample (CPU and MPS).

Writes ml/artifacts/ocr_engine_eval.json and report/figures/ocr_engines.png.
"""
from __future__ import annotations

import ml  # noqa: F401  (OMP fix - must precede torch/xgboost)

import difflib
import json
import os
import re
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ml import eval_screenshots as es
from ml.common import ARTIFACTS, FIGURES, ROOT

OUT_JSON = ARTIFACTS / "ocr_engine_eval.json"
FIG = FIGURES / "ocr_engines.png"
EASY_CACHE = es.CACHE / "easyocr_en"
SYNTH_DIR = ARTIFACTS / "intel_synthetic_screens"
ENGINES = ["tesseract", "easyocr", "union"]
ENGINE_PARAMS = {
    "tesseract": {"engine": "tesseract", **es.CONFIGS["v4_psm3"]},
    "easyocr": {"engine": "easyocr", "lang": ["en"], "merge_blocks": True, "canvas_size": 2560, "mag_ratio": 1.0},
    "union": {"engine": "union", "iou": 0.5, "containment": 0.8},
}
REPORT_TAUS = [0.68, 0.9]
N_TIMING = 20
EVAL_DEVICE = os.environ.get("CLEARCART_EASYOCR_DEVICE") or ("mps" if __import__("torch").backends.mps.is_available() else "cpu")


# --------------------------------------------------------------------------- OCR
def _rec(item, cands, secs, img, engine, device=None):
    return {"image": item["image"], "config": engine, "params": ENGINE_PARAMS[engine], "device": device,
            "seconds": round(secs, 4), "image_size": list(img.size), "image_mode": img.mode,
            "candidates": [{"text": c.text, "bbox": c.bbox, "ocr_confidence": c.ocr_confidence,
                            "unit": c.extra.get("unit")} for c in cands]}


def _open(item):
    from PIL import Image
    img = Image.open(ROOT / item["path"])
    img.load()
    return img


def run_easyocr(items) -> dict[str, dict]:
    from backend.extractors import ocr_lines_easyocr
    out, todo = {}, [i for i in items if not (EASY_CACHE / (i["image"] + ".json")).exists()]
    print(f"[easyocr] {len(items)} images, {len(todo)} not cached; device={EVAL_DEVICE}", flush=True)
    for k, it in enumerate(items):
        cp = EASY_CACHE / (it["image"] + ".json")
        if cp.exists():
            out[it["image"]] = json.loads(cp.read_text())
            continue
        img = _open(it)
        t0 = time.perf_counter()
        cands = ocr_lines_easyocr(img, device=EVAL_DEVICE)
        rec = _rec(it, cands, time.perf_counter() - t0, img, "easyocr", EVAL_DEVICE)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(rec, ensure_ascii=False))
        out[it["image"]] = rec
        if (k + 1) % 25 == 0:
            print(f"[easyocr] {k + 1}/{len(items)}", flush=True)
    return out


def _to_cands(rec):
    from backend.extractors import Candidate
    return [Candidate(text=c["text"], source_type="screenshot_ocr", bbox=c["bbox"], ocr_confidence=c["ocr_confidence"],
                      extra={"unit": c["unit"]}) for c in rec["candidates"]]


def build_union(items, tess, easy) -> dict[str, dict]:
    from backend.extractors import union_candidates
    out = {}
    for it in items:
        t, e = tess[it["image"]], easy[it["image"]]
        u = union_candidates(_to_cands(t), _to_cands(e))
        out[it["image"]] = {"image": it["image"], "config": "union", "params": ENGINE_PARAMS["union"],
                            "seconds": round(t["seconds"] + e["seconds"], 4), "image_size": t["image_size"],
                            "candidates": [{"text": c.text, "bbox": c.bbox, "ocr_confidence": c.ocr_confidence,
                                            "unit": c.extra.get("unit")} for c in u],
                            "n_added_from_easyocr": len(u) - len(t["candidates"])}
    return out


def timing_subsample(dev) -> dict:
    """Sequential, single-process timing on the first N_TIMING DEV images (sorted by name): Tesseract v4_psm3,
    EasyOCR on CPU and on MPS. Also checks that CPU and MPS EasyOCR return the same candidates."""
    import torch
    from backend.extractors import _easyocr_reader, ocr_lines, ocr_lines_easyocr
    sub = sorted(dev, key=lambda i: i["image"])[:N_TIMING]
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for d in devices:                                   # load + warm up outside the timed loop
        _easyocr_reader(d)
        ocr_lines_easyocr(_open(sub[0]), device=d)
    t_tess, t_easy, same = [], {d: [] for d in devices}, 0
    for it in sub:
        img = _open(it)
        t0 = time.perf_counter(); ocr_lines(img, **es.CONFIGS["v4_psm3"]); t_tess.append(time.perf_counter() - t0)
        outs = {}
        for d in devices:
            t0 = time.perf_counter(); outs[d] = ocr_lines_easyocr(img, device=d); t_easy[d].append(time.perf_counter() - t0)
        if len(devices) > 1:
            key = lambda cs: [(c.text, tuple(c.bbox.values()), c.extra["unit"]) for c in cs]
            same += key(outs["cpu"]) == key(outs["mps"])
    sizes = [list(_open(i).size) for i in sub]
    return {"n_images": len(sub), "images": [i["image"] for i in sub], "median_image_size": np.median(sizes, axis=0).tolist(),
            "note": ("sequential, one image at a time, warm models, OMP_NUM_THREADS=1 (set by ml/__init__ for the "
                     "torch+xgboost segfault), so EasyOCR-CPU uses one intra-op thread"),
            "tesseract_v4_psm3_s_per_image": round(float(np.mean(t_tess)), 3),
            **{f"easyocr_{d}_s_per_image": round(float(np.mean(v)), 3) for d, v in t_easy.items()},
            "union_cpu_s_per_image": round(float(np.mean(t_tess) + np.mean(t_easy["cpu"])), 3),
            **({"union_mps_s_per_image": round(float(np.mean(t_tess) + np.mean(t_easy["mps"])), 3),
                "easyocr_cpu_vs_mps_identical_outputs": f"{same}/{len(sub)}"} if "mps" in devices else {})}


# --------------------------------------------------------------------------- synthetic screens
def synthetic(clf) -> dict:
    from PIL import Image
    from backend.extractors import ocr_engine_lines
    from backend.intelligence.mcai import compute_mcai
    bench = json.loads((ARTIFACTS / "intelligence_benchmark.json").read_text())
    specs = bench["experiments"]["B_mcai_synthetic_screenshots"]["cases"]
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    sim = lambda a, b: difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()
    rows = []
    for sp in specs:
        img = Image.open(SYNTH_DIR / f"{sp['id']}.png"); img.load()
        row = {"id": sp["id"], "designed": sp["designed"], "supplementary": sp["supplementary"],
               "accept_text": sp["accept_text"], "decline_text": sp["decline_text"], "engines": {}}
        for eng in ENGINES:
            os.environ["CLEARCART_EASYOCR_DEVICE"] = EVAL_DEVICE
            cands = ocr_engine_lines(img, eng)
            texts = [c.text for c in cands]
            res = compute_mcai("screenshot_ocr", clf.predict, cands=cands, image=img)
            r = {"ocr_lines": [c.text for c in cands if c.extra.get("unit") == "line"],
                 "accept_read": max((sim(t, sp["accept_text"]) for t in texts), default=0) >= 0.85,
                 "decline_read": max((sim(t, sp["decline_text"]) for t in texts), default=0) >= 0.85,
                 "mcai_available": res.available}
            if res.available:
                cs = max(res.choice_sets, key=lambda c: c.mcai if c.mcai is not None else -1)
                dims = {d.name: d for d in cs.dimensions}
                r.update({"detected_accept": cs.accept.text if cs.accept else None, "detected_decline": cs.decline.text,
                          "both_options_found": bool(cs.accept and sim(cs.accept.text, sp["accept_text"]) >= 0.85
                                                     and sim(cs.decline.text, sp["decline_text"]) >= 0.85),
                          "visual_available": dims["visual"].available, "visual": dims["visual"].score,
                          "visual_features": {f.feature: {"accept": f.accept_value, "decline": f.decline_value,
                                                          "asym": f.asymmetry} for f in dims["visual"].features},
                          "mcai": cs.mcai})
            else:
                r.update({"both_options_found": False, "visual_available": False, "reason": res.reason})
            row["engines"][eng] = r
        rows.append(row)
    summ = {}
    for eng in ENGINES:
        e = [x["engines"][eng] for x in rows]
        asym = [x["engines"][eng]["visual"] for x in rows if x["designed"] == "asymmetric" and not x["supplementary"]
                and x["engines"][eng].get("visual") is not None]
        sym = [x["engines"][eng]["visual"] for x in rows if x["designed"] == "symmetric"
               and x["engines"][eng].get("visual") is not None]
        summ[eng] = {"decline_read": sum(r["decline_read"] for r in e), "accept_read": sum(r["accept_read"] for r in e),
                     "both_options_found": sum(r["both_options_found"] for r in e),
                     "mcai_visual_available": sum(bool(r.get("visual_available")) for r in e),
                     "faint_grey_decline_S1_S3_read": sum(x["engines"][eng]["decline_read"] for x in rows
                                                          if x["id"] in ("S1", "S2", "S3")),
                     "mean_visual_main_asymmetric_S1_S3": round(float(np.mean(asym)), 4) if asym else None,
                     "mean_visual_symmetric_S4_S6": round(float(np.mean(sym)), 4) if sym else None, "n": len(e)}
    return {"synthetic": True, "labels_from_construction": True,
            "source": "PNG files saved by ml/benchmark_intelligence.py (exp_B_screenshots); texts from intelligence_benchmark.json",
            "match_rule": "read = some OCR candidate matches the rendered text with difflib ratio >= 0.85 after lower-casing "
                          "and removing non-alphanumerics (same rule as benchmark B)",
            "S1_S3_decline_style": "S1/S2: 15px rgb(170,170,170) on white; S3: 13px rgb(190,190,190) on white",
            "summary": summ, "cases": rows}


# --------------------------------------------------------------------------- figure
def figure(dev_m, test_m, picks, timing):
    plt = es._style()
    keys = [("detection_recall", "Detection\nrecall"), ("category_correct_recall", "Cat.-correct\nrecall"),
            ("neg_flag_rate", "Neg. flag rate\n(lower=better)"), ("precision", "Precision"), ("f1", "F1"),
            ("auroc_max_p_dark", "AUROC\n(max p_dark)")]
    cols = es.COLORS[:3]
    fig = plt.figure(figsize=(14, 8.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[4, 4, 1.7])
    x = np.arange(len(keys)); w = 0.27
    for r, tau in enumerate(REPORT_TAUS):
        for c, (split, M) in enumerate([("DEV", dev_m), ("TEST", test_m)]):
            ax = fig.add_subplot(gs[r, c])
            for j, eng in enumerate(ENGINES):
                m = M[eng][str(tau)]
                v = [m[k] or 0 for k, _ in keys]
                xs = x + (j - 1) * w
                ax.bar(xs, v, w * 0.92, color=cols[j], label=eng, edgecolor=es.SURF, linewidth=2)
                for xi, vi in zip(xs, v):
                    ax.text(xi, vi + 0.012, f"{vi:.2f}", ha="center", va="bottom", fontsize=6, color=es.INK2, rotation=90)
            ax.set_xticks(x, [k[1] for k in keys], fontsize=7.5)
            ax.set_ylim(0, 1.18)
            n_p, n_n = M["tesseract"][str(tau)]["n_pos"], M["tesseract"][str(tau)]["n_neg"]
            ax.set_title(f"{split} ({n_p} pos / {n_n} neg images), flag if p_dark >= {tau}", loc="left", fontsize=10)
            if r == 0 and c == 0:
                ax.legend(ncol=3, loc="upper left", fontsize=8)
    ax = fig.add_subplot(gs[:, 2])
    labs = ["tesseract", "easyocr\nCPU", "easyocr\nMPS", "union\nCPU", "union\nMPS"]
    vals = [timing["tesseract_v4_psm3_s_per_image"], timing["easyocr_cpu_s_per_image"],
            timing.get("easyocr_mps_s_per_image", 0), timing["union_cpu_s_per_image"], timing.get("union_mps_s_per_image", 0)]
    ax.barh(range(len(vals)), vals, 0.66, color=[cols[0], cols[1], cols[1], cols[2], cols[2]], edgecolor=es.SURF, linewidth=2)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=8, color=es.INK2)
    ax.set_yticks(range(len(vals)), labs, fontsize=8)
    ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    ax.set_xlim(0, max(vals) * 1.3)
    ax.set_title(f"OCR s / image\n(sequential, {timing['n_images']} DEV imgs)", loc="left", fontsize=10)
    pick_txt = "  |  ".join(f"{e}: DEV pick tau={picks[e]['tau']}"
                            + ("" if picks[e].get("admissible_ranking") else " (none meets neg rate <= 0.15)")
                            for e in ENGINES)
    fig.suptitle(f"OCR engines on ContextRico-DP screenshots, same text model and decision rule.\n{pick_txt}",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def _short(m):
    return {k: m[k] for k in ("tau", "detection_recall", "category_correct_recall", "neg_flag_rate", "precision", "f1",
                              "auroc_max_p_dark", "mean_ocr_seconds_per_image", "mean_candidates_per_image")}


def main():
    t_start = time.perf_counter()
    items, info = es.build_eval_set()
    dev = [i for i in items if i["split"] == "dev"]
    test = [i for i in items if i["split"] == "test"]
    tess = {i["image"]: es._ocr_one(i, "v4_psm3") for i in items}          # cached by eval_screenshots
    easy = run_easyocr(items)
    ocr = {"tesseract": tess, "easyocr": easy, "union": build_union(items, tess, easy)}

    from ml.model import TextClassifier
    clf = TextClassifier.load(ARTIFACTS)
    default_tau = float(clf.b.threshold)
    scores = es.score_texts(clf, ocr)
    preds = {e: es.image_preds(items, ocr[e], scores) for e in ENGINES}

    taus = sorted(set(REPORT_TAUS) | {default_tau})
    dev_m = {e: {str(t): es.metrics(dev, preds[e], t, ocr[e]) for t in taus} for e in ENGINES}
    dev_sweeps = {e: [es.metrics(dev, preds[e], t, ocr[e]) for t in es.TAU_GRID] for e in ENGINES}
    with patch.dict(es.CONFIGS, ENGINE_PARAMS):
        picks = {e: es.choose({e: dev_m[e][str(default_tau)]}, {e: dev_sweeps[e]}, default_tau) for e in ENGINES}
        joint = es.choose({e: dev_m[e][str(default_tau)] for e in ENGINES}, dev_sweeps, default_tau)
    for e in ENGINES:
        print(f"[select] {e}: tau={picks[e]['tau']}  {picks[e]['reason'][:160]}", flush=True)
    print(f"[select] joint: {joint['config']}@{joint['tau']}", flush=True)

    # ---- TEST: reported once, after the DEV picks above ----
    test_m = {e: {str(t): es.metrics(test, preds[e], t, ocr[e]) for t in taus} for e in ENGINES}
    test_at_pick = {e: es.metrics(test, preds[e], picks[e]["tau"], ocr[e]) for e in ENGINES}
    served = ("tesseract", 0.9)
    paired = {f"{e}@{t}": es.paired(test, preds[served[0]], served[1], preds[e], t)
              for e, t in [(e, picks[e]["tau"]) for e in ("easyocr", "union")] + [("easyocr", 0.9), ("union", 0.9)]}
    for e in ENGINES:
        for t in REPORT_TAUS:
            m = test_m[e][str(t)]
            print(f"  TEST {e:9s}@{t}: det={m['detection_recall']:.3f} cat={m['category_correct_recall']:.3f} "
                  f"neg={m['neg_flag_rate']:.3f} P={m['precision']:.3f} F1={m['f1']:.3f} AUROC={m['auroc_max_p_dark']}",
                  flush=True)

    # which TEST positives differ (category-correct) between served tesseract@0.9 and easyocr@0.9
    diff_rows = []
    for i in [i for i in test if i["is_pos"]]:
        a = bool(es.flagged_cats(preds["tesseract"][i["image"]], 0.9) & set(i["gt_cats"]))
        b = bool(es.flagged_cats(preds["easyocr"][i["image"]], 0.9) & set(i["gt_cats"]))
        if a != b:
            win = "easyocr" if b else "tesseract"
            hits = sorted([c for c in preds[win][i["image"]] if c["p_dark"] >= 0.9 and c["category"] in i["gt_cats"]],
                          key=lambda c: -c["p_dark"])
            diff_rows.append({"image": i["image"], "gt_cats": i["gt_cats"], "only_correct_with": win,
                              "flagging_text": [h["text"][:120] for h in hits[:2]]})

    timing = timing_subsample(dev)
    print(f"[timing] {timing}", flush=True)
    synth = synthetic(clf)
    print(f"[synthetic] {json.dumps(synth['summary'])}", flush=True)
    figure(dev_m, test_m, picks, timing)

    import easyocr, importlib.metadata as md
    out = {
        "description": ("Second OCR engine (EasyOCR) vs Tesseract v4_psm3 on the ContextRico-DP DEV/TEST split of "
                        "ml/eval_screenshots.py, with the current text model and the same image-level decision rule. "
                        "Operating points are picked on DEV only; TEST is reported."),
        "provenance": {
            "easyocr": {"version": easyocr.__version__, "license": md.metadata("easyocr")["License"],
                        "source": "https://github.com/JaidedAI/EasyOCR (pip: easyocr)",
                        "weights": {"craft_mlt_25k.pth": {"role": "CRAFT text detector (Clova AI, MIT-licensed code/model, "
                                                                  "redistributed by EasyOCR)", "bytes": 83152330,
                                                          "sha256": "4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17"},
                                    "english_g2.pth": {"role": "EasyOCR English recogniser (generation 2, CRNN)",
                                                       "bytes": 15143997,
                                                       "sha256": "e2272681d9d67a04e2dff396b6e95077bc19001f8f6d3593c307b9852e1c29e8"}},
                        "cache_path": "~/.EasyOCR/model/ (downloaded on first Reader construction)",
                        "added_dependencies": {k: md.version(k) for k in
                                               ["torchvision", "opencv-python-headless", "scikit-image", "pyclipper",
                                                "python-bidi", "shapely", "ninja", "imageio", "tifffile", "lazy-loader"]},
                        "unchanged": {"torch": md.version("torch"), "numpy": md.version("numpy")}},
            "tesseract": "pytesseract + system Tesseract (unchanged); OCR cache from ml/eval_screenshots.py (v4_psm3)",
        },
        "dataset": {"name": "ContextRico-DP", "license": "CC-BY-4.0", "split_sizes": info["split_sizes"],
                    "negatives_note": info["negatives_note"]},
        "split": "reused from ml/eval_screenshots.build_eval_set (identical DEV/TEST images)",
        "decision_rule": "ml/eval_screenshots.metrics: image flagged if any candidate has p_dark >= tau",
        "selection_rule": es.SELECTION_RULES["joint_config_and_threshold"],
        "engines": ENGINE_PARAMS,
        "eval_device_easyocr": EVAL_DEVICE,
        "text_model_threshold": default_tau,
        "dev": {"metrics": dev_m, "threshold_sweep": dev_sweeps},
        "dev_picks_per_engine": picks,
        "dev_pick_joint_across_engines": joint,
        "test": {"metrics": test_m, "at_dev_pick": test_at_pick,
                 "paired_vs_served_tesseract_at_0.9": paired,
                 "positives_category_correct_with_only_one_engine_at_0.9": diff_rows},
        "summary_table": {split: {e: {str(t): _short(M[e][str(t)]) for t in REPORT_TAUS} for e in ENGINES}
                          for split, M in [("dev", dev_m), ("test", test_m)]},
        "dev_negative_flag_diagnostic_easyocr_at_0.9": es.negative_flag_diagnostic(dev, preds["easyocr"], 0.9),
        "dev_negative_flag_diagnostic_tesseract_at_0.9": es.negative_flag_diagnostic(dev, preds["tesseract"], 0.9),
        "union_added_easyocr_candidates_per_image": round(float(np.mean([r["n_added_from_easyocr"] for r in ocr["union"].values()])), 2),
        "timing_controlled_subsample": timing,
        "synthetic_screens": synth,
        "caveats": [
            "mean_ocr_seconds_per_image in the metric blocks is NOT comparable across engines: Tesseract times come from "
            "the eval_screenshots cache (6 concurrent processes), EasyOCR times are sequential on " + EVAL_DEVICE +
            ", union = sum. Use timing_controlled_subsample for speed. (The selection rule only uses s/image as the last "
            "tie-break.)",
            "ocr_confidence is not calibrated across engines (Tesseract: mean word confidence; EasyOCR: recogniser "
            "confidence). Nothing in the decision rule uses it.",
            info["negatives_note"],
            "Small positive sets (DEV 42 / TEST 41 images); see Wilson CIs and paired McNemar p-values.",
            "EasyOCR on the ContextRico set was run on " + EVAL_DEVICE + "; timing_controlled_subsample reports whether CPU "
            "and MPS outputs were identical on its images.",
        ],
        "runtime_seconds": round(time.perf_counter() - t_start, 1),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    print(f"[done] wrote {OUT_JSON.relative_to(ROOT)} and {FIG.relative_to(ROOT)} in {out['runtime_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
