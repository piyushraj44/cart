"""Screenshot (OCR) evaluation of the ClearCart text classifier on ContextRico-DP.

Run from the project root:  .venv/bin/python -m ml.eval_screenshots

ContextRico-DP (Zenodo 15036779, CC-BY-4.0) is used for EVALUATION ONLY. Nothing here trains or
fine-tunes a model; the text classifier was trained only on yamanalab/ec-darkpattern.

Pipeline
  1. eval set = every image with >= 1 text-observable label (positives, mapped to ClearCart
     categories via CR_TO_LABEL) + a seeded random sample of 250 images without any
     text-observable label (negatives). Split 50/50 DEV/TEST by image, stratified.
  2. OCR every image under each configuration (cached as JSON in data/processed/ocr_cache/).
  3. Compare configurations on DEV; sweep an OCR-specific p_dark threshold on DEV.
  4. Pick config + threshold on DEV with a fixed rule, then report v1_naive vs chosen on TEST.
  5. Error analysis on TEST, figures, demo samples.
"""
from __future__ import annotations

import ml  # noqa: F401  (sets OMP_NUM_THREADS=1 before torch/xgboost load - required on macOS)

import json
import math
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ml.common import ARTIFACTS, CR_DIR, CR_GT, CR_TO_LABEL, DATA_PROCESSED, FIGURES, ROOT, SEED

IMG_DIR = CR_DIR / "images" / "ContextRico-DP"
CACHE = DATA_PROCESSED / "ocr_cache"
OUT_JSON = ARTIFACTS / "screenshot_eval.json"
SAMPLES = ROOT / "data" / "samples" / "screenshots"
N_NEG = 250
N_WORKERS = 6
OCR_CACHE_MISSES = 0
CATS = ["urgency", "scarcity", "social_proof"]

CONFIGS = {
    "v1_naive": dict(upscale=False, merge_blocks=False, psm=3),
    "v2": dict(upscale=True, merge_blocks=False, psm=11),
    "v3": dict(upscale=True, merge_blocks=True, psm=11),
    "v4_psm3": dict(upscale=True, merge_blocks=True, psm=3),
}
CONFIG_DESC = {
    "v1_naive": "raw RGB image, Tesseract auto layout (psm 3), one candidate per OCR line",
    "v2": "grayscale + 2x upscale (if max side < 2200px), sparse-text layout (psm 11), lines only",
    "v3": "v2 + adjacent lines of the same Tesseract paragraph also merged into block candidates",
    "v4_psm3": "grayscale + 2x upscale + block merging, but auto layout (psm 3)",
}

TAU_GRID = [float(round(x, 2)) for x in np.arange(0.30, 0.901, 0.05)]
NEG_RATE_BOUND = 0.15
MIN_F1_GAIN_FOR_NEW_TAU = 0.02
CONFIG_TIE_F1 = 0.01

SELECTION_RULES = {
    "joint_config_and_threshold": (
        f"On DEV only, evaluate every (config, tau) with tau in {{text-model threshold}} U {{{TAU_GRID[0]}..{TAU_GRID[-1]} step "
        f"0.05}}. Admissible points have negative-image flag rate <= {NEG_RATE_BOUND}. Choose the admissible point with "
        f"the highest image-level F1; points within {CONFIG_TIE_F1} F1 of the best are tied and broken by higher "
        f"category-correct recall, then by the text model's own threshold, then by lower OCR s/image. If an admissible "
        f"point at the text model's own threshold is within {MIN_F1_GAIN_FOR_NEW_TAU} F1 of the best, it is preferred "
        f"(no OCR-specific threshold unless it clearly helps). If no point is admissible, the lowest neg flag rate wins."),
}
RULE_REVISION = (
    "The first run used a two-stage rule (pick the config by DEV F1 at the text model's threshold 0.68, then sweep tau "
    "for that config only). It selected v1_naive and then replaced 0.68 with 0.9 because 0.68 violated the neg-rate bound "
    "for every config - i.e. the config was chosen at a threshold that was then discarded. That run printed TEST "
    "metrics for v1_naive@0.68 and v1_naive@0.9 (cat-correct recall 0.634/0.512, F1 0.561/0.597) before the rule "
    "was replaced by the joint rule above, which was fixed after looking only at DEV sweeps. No other TEST numbers were "
    "seen before the revision. Both are reported here; test.all_configs_at_tau_0.9 lets a reader check every config.")

# Cue words used ONLY to auto-describe why a positive was missed (error analysis), never for prediction.
CUES = {
    "urgency": r"\d{1,2}\s*[:：]\s*\d{2}|\b\d{1,2}\s*(?:h|m|s|d)\b|\bends?\b|\bending\b|hurry|limited|today|tonight|expire|hours?\b|\bhrs?\b|"
               r"\bmins?\b|minutes?|\bsecs?\b|seconds?|\bdays?\b|countdown|\bleft\b|last chance|deal|offer|only",
    "scarcity": r"\bleft\b|stock|only \d|remaining|demand|selling fast|sold|almost gone|few|popular|limited",
    "social_proof": r"bought|purchased|viewing|people|customers|someone|just|\bago\b|watching|orders?|sold",
}


# Notes from looking at a few missed TEST images by eye (attached to the error analysis when those images appear).
MANUAL_NOTES = {
    "(305)www.bellamihair.com_5eea.jpg": ("Countdown 'SALE ENDS IN: 6D : 8H : 57M : 20S' is clearly visible as red text on a "
                                          "black bar, but OCR returned only the white part of that bar - red-on-black loses "
                                          "contrast after grayscale conversion."),
    "(360)www.jjshouse.com_7293.jpg": ("Two countdowns are visible ('1 DAY 11:44:21' on a yellow gradient banner and boxed "
                                       "digits '3 DAYS 11:44:21'); OCR recovered neither."),
    "25615.jpg": "Ad countdown 'Click To Skip AD (5s)' was read correctly; the text model gives it p_dark ~0 (ad-skip timers "
                 "do not occur in the e-commerce training text).",
    "13546.jpg": "Ad countdown 'Skip AD (9s)' read correctly but scored low by the text model (same domain gap).",
}


# --------------------------------------------------------------------------- data
def build_eval_set() -> tuple[list[dict], dict]:
    gt = json.loads(CR_GT.read_text())
    files = {p.name: p for p in IMG_DIR.rglob("*") if p.is_file() and not p.name.startswith(".")}
    missing_files = sorted(k for k in gt if k not in files)
    files_without_gt = sorted(f for f in files if f not in gt)

    pos, neg_pool = [], []
    for name in sorted(gt):
        if name not in files:
            continue
        labels = gt[name]["labels"]
        cats = sorted({CR_TO_LABEL[l] for l in labels if l in CR_TO_LABEL})
        item = {"image": name, "path": str(files[name].relative_to(ROOT)), "subset": files[name].parent.name,
                "gt_labels": labels, "gt_cats": cats}
        (pos if cats else neg_pool).append(item)

    rng = np.random.default_rng(SEED)
    neg_idx = sorted(rng.choice(len(neg_pool), size=N_NEG, replace=False).tolist())
    neg = [neg_pool[i] for i in neg_idx]

    # stratified 50/50 split. Positives are additionally stratified by their rarest category so that the
    # 11 social-proof images do not all land on one side.
    rarity = {c: n for c, n in zip(CATS, [sum(c in p["gt_cats"] for p in pos) for c in CATS])}
    strata: dict[str, list[dict]] = {}
    for p in pos:
        strata.setdefault("pos_" + min(p["gt_cats"], key=lambda c: rarity[c]), []).append(p)
    strata["neg"] = neg
    rng = np.random.default_rng(SEED)
    carry = 0  # alternate which side gets the odd item so the halves stay balanced
    for key in sorted(strata):
        items = strata[key]
        order = rng.permutation(len(items))
        n_dev = len(items) // 2 + (1 if (len(items) % 2 and carry == 0) else 0)
        if len(items) % 2:
            carry ^= 1
        for rank, i in enumerate(order):
            items[i]["split"] = "dev" if rank < n_dev else "test"
            items[i]["stratum"] = key
    for n in neg:
        n["is_pos"] = False
    for p in pos:
        p["is_pos"] = True
    items = pos + neg
    info = {
        "gt_entries": len(gt), "image_files": len(files),
        "gt_keys_without_file": [{"image": k, "labels": gt[k]["labels"]} for k in missing_files],
        "files_without_gt_key": files_without_gt,
        "filename_mapping": "GT keys are bare filenames; each matches exactly one file under Mobile/ or Web/ (no duplicate basenames).",
        "n_positive_images": len(pos), "n_negative_pool": len(neg_pool), "n_negatives_sampled": len(neg),
        "positive_images_per_category": {c: sum(c in p["gt_cats"] for p in pos) for c in CATS},
        "multi_category_positives": sum(len(p["gt_cats"]) > 1 for p in pos),
        "split_sizes": {s: {"positives": sum(i["is_pos"] and i["split"] == s for i in items),
                            "negatives": sum((not i["is_pos"]) and i["split"] == s for i in items),
                            **{c: sum(c in i["gt_cats"] and i["split"] == s for i in items) for c in CATS}}
                        for s in ("dev", "test")},
        "negatives_note": ("Negatives are ContextRico images with NO text-observable label. They are still dark-pattern "
                           "UIs of visual types (Hard to Close, Nagging, Default Choice, False Hierarchy, ...) and some may "
                           "contain unlabelled urgency/scarcity/social-proof or confirmshaming text, so the 'flag rate on "
                           "negatives' (and hence image-level false positives / precision) is an UPPER bound on true false alarms."),
    }
    return items, info


# --------------------------------------------------------------------------- OCR
def _cache_path(image: str, cfg: str) -> Path:
    return CACHE / cfg / (image + ".json")


def _ocr_one(item: dict, cfg: str) -> dict:
    from PIL import Image
    from backend.extractors import ocr_lines

    cp = _cache_path(item["image"], cfg)
    if cp.exists():
        return json.loads(cp.read_text())
    img = Image.open(ROOT / item["path"])
    img.load()
    t0 = time.perf_counter()
    cands = ocr_lines(img, **CONFIGS[cfg])
    secs = time.perf_counter() - t0
    rec = {"image": item["image"], "config": cfg, "params": CONFIGS[cfg], "seconds": round(secs, 4),
           "image_size": list(img.size), "image_mode": img.mode,
           "candidates": [{"text": c.text, "bbox": c.bbox, "ocr_confidence": c.ocr_confidence,
                           "unit": c.extra.get("unit")} for c in cands]}
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps(rec, ensure_ascii=False))
    return rec


def run_ocr(items: list[dict]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {c: {} for c in CONFIGS}
    jobs = [(it, cfg) for cfg in CONFIGS for it in items]
    todo = sum(not _cache_path(it["image"], cfg).exists() for it, cfg in jobs)
    global OCR_CACHE_MISSES
    OCR_CACHE_MISSES = todo
    print(f"[ocr] {len(jobs)} (image, config) jobs, {todo} not cached; {N_WORKERS} workers", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_ocr_one, it, cfg): (it["image"], cfg) for it, cfg in jobs}
        for f in as_completed(futs):
            img, cfg = futs[f]
            out[cfg][img] = f.result()
            done += 1
            if done % 200 == 0:
                print(f"[ocr] {done}/{len(jobs)}", flush=True)
    return out


# --------------------------------------------------------------------------- scoring
def score_texts(clf, ocr: dict) -> dict[str, dict]:
    texts = sorted({c["text"] for cfg in ocr.values() for rec in cfg.values() for c in rec["candidates"]})
    print(f"[model] scoring {len(texts)} unique OCR texts", flush=True)
    scores = {}
    for i in range(0, len(texts), 512):
        for p in clf.predict(texts[i:i + 512]):
            scores[p["text"]] = {"p_dark": p["p_dark"], "category": p["category"]}
    return scores


def image_preds(items, ocr_cfg, scores) -> dict[str, list[dict]]:
    """Per image: list of (p_dark, category, text, unit) for every OCR candidate."""
    return {it["image"]: [{"text": c["text"], "unit": c["unit"], "ocr_confidence": c["ocr_confidence"], "bbox": c["bbox"],
                           "img_h": ocr_cfg[it["image"]]["image_size"][1], **scores[c["text"]]}
                          for c in ocr_cfg[it["image"]]["candidates"]] for it in items}


CLOCKISH = re.compile(r"^[\d\s:.,%|/©®@°]*(?:am|pm)?[\d\s:.,%|/©®@°]*$", re.I)


def negative_flag_diagnostic(items, preds, tau) -> dict:
    """Why are 'negative' images flagged? Counts images whose flags come only from (a) the top 5% band of the
    image (mobile status bar: clock, battery %) or (b) clock-/number-only strings such as '00:19'."""
    N = [i for i in items if not i["is_pos"]]
    out = {"flagged": 0, "only_top5pct_band": 0, "only_clock_or_number_strings": 0, "only_band_or_clockish": 0,
           "flag_category_counts": {}}
    for i in N:
        hits = [c for c in preds[i["image"]] if c["p_dark"] >= tau]
        if not hits:
            continue
        out["flagged"] += 1
        band = [c["bbox"] is not None and c["bbox"]["y"] + c["bbox"]["height"] <= 0.05 * c["img_h"] for c in hits]
        clock = [bool(CLOCKISH.match(c["text"].strip())) for c in hits]
        out["only_top5pct_band"] += all(band)
        out["only_clock_or_number_strings"] += all(clock)
        out["only_band_or_clockish"] += all(b or k for b, k in zip(band, clock))
        for cat in {c["category"] for c in hits}:
            out["flag_category_counts"][cat] = out["flag_category_counts"].get(cat, 0) + 1
    return out


def flagged_cats(cands: list[dict], tau: float) -> set[str]:
    return {c["category"] for c in cands if c["p_dark"] >= tau}


def _auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def metrics(items, preds, tau, ocr_cfg) -> dict:
    P = [i for i in items if i["is_pos"]]
    N = [i for i in items if not i["is_pos"]]
    fl = {i["image"]: flagged_cats(preds[i["image"]], tau) for i in items}
    tp = sum(bool(fl[i["image"]]) for i in P)
    fp = sum(bool(fl[i["image"]]) for i in N)
    cat_ok = sum(bool(fl[i["image"]] & set(i["gt_cats"])) for i in P)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / len(P) if P else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    per_cat = {}
    for c in CATS:
        Pc = [i for i in P if c in i["gt_cats"]]
        k = sum(c in fl[i["image"]] for i in Pc)
        per_cat[c] = {"recall": round(k / len(Pc), 4) if Pc else None, "hit": k, "n": len(Pc), "ci95": wilson(k, len(Pc))}
    neg_by_cat = {c: sum(c in fl[i["image"]] for i in N) for c in CATS + ["misdirection"]}
    maxp = lambda i: max((c["p_dark"] for c in preds[i["image"]]), default=0.0)
    secs = [ocr_cfg[i["image"]]["seconds"] for i in items]
    ncand = [len(ocr_cfg[i["image"]]["candidates"]) for i in items]
    return {
        "tau": tau, "n_pos": len(P), "n_neg": len(N),
        "detection_recall": round(rec, 4), "detection_recall_ci95": wilson(tp, len(P)), "tp": tp,
        "category_correct_recall": round(cat_ok / len(P), 4) if P else None,
        "category_correct_recall_ci95": wilson(cat_ok, len(P)), "category_correct_hits": cat_ok,
        "per_category_recall": per_cat,
        "neg_flag_rate": round(fp / len(N), 4) if N else None, "neg_flag_rate_ci95": wilson(fp, len(N)), "fp": fp,
        "neg_flagged_by_category": neg_by_cat,
        "precision": round(prec, 4), "f1": round(f1, 4),
        "auroc_max_p_dark": None if _auc([maxp(i) for i in P], [maxp(i) for i in N]) is None
        else round(_auc([maxp(i) for i in P], [maxp(i) for i in N]), 4),
        "mean_ocr_seconds_per_image": round(float(np.mean(secs)), 3),
        "mean_candidates_per_image": round(float(np.mean(ncand)), 2),
        "images_with_no_ocr_text": sum(n == 0 for n in ncand),
    }


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return round(min(1.0, 2 * p), 4)


def paired(items, preds_a, tau_a, preds_b, tau_b) -> dict:
    """Paired comparison A (before) -> B (after) on the same TEST images."""
    out = {}
    P = [i for i in items if i["is_pos"]]
    N = [i for i in items if not i["is_pos"]]
    for name, group, ok in [
        ("positives_category_correct", P, lambda fl, i: bool(fl & set(i["gt_cats"]))),
        ("positives_detected", P, lambda fl, i: bool(fl)),
        ("negatives_flagged", N, lambda fl, i: bool(fl)),
    ]:
        a = [ok(flagged_cats(preds_a[i["image"]], tau_a), i) for i in group]
        b = [ok(flagged_cats(preds_b[i["image"]], tau_b), i) for i in group]
        only_b = sum((not x) and y for x, y in zip(a, b))
        only_a = sum(x and (not y) for x, y in zip(a, b))
        out[name] = {"before_only": only_a, "after_only": only_b, "both": sum(x and y for x, y in zip(a, b)),
                     "neither": sum((not x) and (not y) for x, y in zip(a, b)),
                     "mcnemar_exact_p": mcnemar_exact(only_a, only_b)}
    return out


# --------------------------------------------------------------------------- selection
def choose(dev_default: dict[str, dict], dev_sweeps: dict[str, list[dict]], default_tau: float) -> dict:
    pts = [(c, m) for c, m in dev_default.items()] + [(c, m) for c, sw in dev_sweeps.items() for m in sw]
    adm = [(c, m) for c, m in pts if m["neg_flag_rate"] <= NEG_RATE_BOUND]
    if not adm:
        c, m = min(pts, key=lambda cm: (cm[1]["neg_flag_rate"], -cm[1]["f1"]))
        return {"config": c, "params": CONFIGS[c], "tau": m["tau"], "tau_is_default": bool(m["tau"] == default_tau),
                "dev_point": m, "reason": f"no point meets neg flag rate <= {NEG_RATE_BOUND}; lowest neg rate chosen"}
    best = max(m["f1"] for _, m in adm)
    adm_default = [(c, m) for c, m in adm if m["tau"] == default_tau and m["f1"] >= best - MIN_F1_GAIN_FOR_NEW_TAU]
    pool = adm_default or [(c, m) for c, m in adm if m["f1"] >= best - CONFIG_TIE_F1]
    pool.sort(key=lambda cm: (-cm[1]["category_correct_recall"], cm[1]["tau"] != default_tau,
                              cm[1]["mean_ocr_seconds_per_image"]))
    c, m = pool[0]
    reason = (f"{len(adm)} of {len(pts)} DEV (config, tau) points meet neg flag rate <= {NEG_RATE_BOUND}; best admissible "
              f"F1 {best}. Chosen {c}@{m['tau']}: F1 {m['f1']}, cat-correct recall {m['category_correct_recall']}, "
              f"neg flag rate {m['neg_flag_rate']}. Tie set: {[f'{cc}@{mm['tau']}' for cc, mm in pool]}. "
              + ("No admissible point at the text model's threshold." if not any(mm["tau"] == default_tau for _, mm in adm)
                 else ""))
    ranking = sorted(({"config": cc, "tau": mm["tau"], "f1": mm["f1"], "neg_flag_rate": mm["neg_flag_rate"],
                       "category_correct_recall": mm["category_correct_recall"]} for cc, mm in adm),
                     key=lambda r: -r["f1"])
    return {"config": c, "params": CONFIGS[c], "tau": m["tau"], "tau_is_default": bool(m["tau"] == default_tau),
            "dev_point": m, "reason": reason, "admissible_ranking": ranking}


# --------------------------------------------------------------------------- error analysis
def _snippet(cands: list[dict], n: int = 200) -> str:
    lines = [c["text"] for c in cands if c["unit"] == "line"]
    return " | ".join(lines)[:n]


def miss_reason(item: dict, cands: list[dict], tau: float) -> str:
    if not cands:
        return "OCR returned no text for this image"
    fl = flagged_cats(cands, tau)
    if fl:
        return (f"category confusion: flagged as {sorted(fl)} but GT is {item['gt_cats']}")
    full = " ".join(c["text"] for c in cands if c["unit"] == "line").lower()
    cue_hits = sorted({m.group(0) for c in item["gt_cats"] for m in re.finditer(CUES[c], full)})
    top = max(cands, key=lambda c: c["p_dark"])
    mean_conf = np.mean([c["ocr_confidence"] for c in cands if c["ocr_confidence"] is not None] or [0])
    if cue_hits:
        return (f"cue words for {item['gt_cats']} were read by OCR ({cue_hits[:6]}) but no candidate reached tau={tau} "
                f"(max p_dark={top['p_dark']:.2f} on '{top['text'][:60]}')")
    if mean_conf < 0.6:
        return (f"OCR garbled (mean word confidence {mean_conf:.2f}); no {item['gt_cats']} cue words recovered "
                f"(max p_dark={top['p_dark']:.2f})")
    return (f"no {item['gt_cats']} cue words in the OCR text (mean word confidence {mean_conf:.2f}): the cue text was "
            f"dropped by OCR (stylised / low-contrast / boxed digits) or is not present as text (max p_dark={top['p_dark']:.2f})")


def error_analysis(test_items, preds, tau) -> dict:
    P = [i for i in test_items if i["is_pos"]]
    N = [i for i in test_items if not i["is_pos"]]
    missed = [i for i in P if not (flagged_cats(preds[i["image"]], tau) & set(i["gt_cats"]))]
    flagged_neg = [i for i in N if flagged_cats(preds[i["image"]], tau)]
    miss_rows = [{"image": i["image"], "gt_labels": i["gt_labels"], "gt_cats": i["gt_cats"],
                  "ocr_text_snippet": _snippet(preds[i["image"]]), "auto_reason": miss_reason(i, preds[i["image"]], tau),
                  **({"manual_note": MANUAL_NOTES[i["image"]]} if i["image"] in MANUAL_NOTES else {})}
                 for i in missed]
    fneg_rows = []
    for i in flagged_neg:
        hits = sorted([c for c in preds[i["image"]] if c["p_dark"] >= tau], key=lambda c: -c["p_dark"])
        fneg_rows.append({"image": i["image"], "gt_labels": i["gt_labels"],
                          "flagged": [{"text": c["text"][:200], "category": c["category"], "p_dark": round(c["p_dark"], 3),
                                       "unit": c["unit"]} for c in hits[:3]]})
    fneg_rows.sort(key=lambda r: -r["flagged"][0]["p_dark"])
    reason_kind = {}
    for r in miss_rows:
        ar = r["auto_reason"]
        k = ("no OCR text" if ar.startswith("OCR returned") else "category confusion" if ar.startswith("category")
             else "cue words read by OCR but scored below tau" if ar.startswith("cue words")
             else "OCR garbled, no cue words recovered" if ar.startswith("OCR garbled")
             else "no cue words in OCR text (cue text dropped by OCR, or absent)")
        reason_kind[k] = reason_kind.get(k, 0) + 1
    return {"n_missed_positives": len(miss_rows), "n_flagged_negatives": len(fneg_rows),
            "missed_reason_counts": reason_kind,
            "reason_method": ("auto_reason is a heuristic: category confusion if any other category was flagged; otherwise "
                              "it checks whether category cue words (regex, used only for this description) appear in the OCR "
                              "text and the mean Tesseract word confidence (< 0.6 => 'garbled')."),
            "missed_positives": miss_rows[:10], "flagged_negatives": fneg_rows[:10]}


# --------------------------------------------------------------------------- figures
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
                         "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 10,
                         "legend.frameon": False})
    return plt


def fig_config_comparison(panels: list[tuple[dict, float]], n_pos: int, n_neg: int, chosen: str, chosen_tau: float):
    """panels: [(metrics per config at tau, tau), ...] - one row per threshold."""
    plt = _style()
    keys = [("detection_recall", "Detection\nrecall"), ("category_correct_recall", "Category-correct\nrecall"),
            ("urgency", "Urgency\nrecall"), ("scarcity", "Scarcity\nrecall"), ("social_proof", "Social-proof\nrecall"),
            ("neg_flag_rate", "Flag rate on\nnegatives (lower=better)"), ("precision", "Image-level\nprecision"),
            ("f1", "Image-level\nF1")]
    cfgs = list(panels[0][0])
    fig, axes = plt.subplots(len(panels), 2, figsize=(13, 4.0 * len(panels)), gridspec_kw={"width_ratios": [5, 1]},
                             squeeze=False)
    w = 0.8 / len(cfgs)
    x = np.arange(len(keys))
    for r, (dev_m, tau) in enumerate(panels):
        ax, ax2 = axes[r]
        for j, c in enumerate(cfgs):
            m = dev_m[c]
            vals = [m["per_category_recall"][k]["recall"] if k in CATS else m[k] for k, _ in keys]
            lab = f"{c}{'  (chosen)' if (c == chosen and tau == chosen_tau) else ''}"
            ax.bar(x + (j - (len(cfgs) - 1) / 2) * w, vals, w * 0.92, color=COLORS[j], label=lab, edgecolor=SURF,
                   linewidth=1)
        ax.axvspan(4.5, 5.5, color=GRID, alpha=0.35, zorder=0)
        ax.set_xticks(x, [k[1] for k in keys], fontsize=8.5)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("score")
        ax.set_title(f"DEV (n={n_pos} positive / {n_neg} negative images), flag if p_dark >= {tau}", loc="left", fontsize=11)
        ax.legend(ncol=len(cfgs), loc="upper left", fontsize=8.5)
        secs = [dev_m[c]["mean_ocr_seconds_per_image"] for c in cfgs]
        ax2.bar(range(len(cfgs)), secs, 0.7, color=COLORS[:len(cfgs)], edgecolor=SURF)
        for i, s_ in enumerate(secs):
            ax2.text(i, s_, f"{s_:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
        ax2.set_xticks(range(len(cfgs)), cfgs, rotation=30, fontsize=8)
        ax2.set_title("OCR s / image", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "screenshot_config_comparison.png", dpi=200)
    plt.close(fig)


def fig_before_after(series: list[tuple[dict, str]]):
    plt = _style()
    keys = [("detection_recall", "Detection\nrecall"), ("category_correct_recall", "Category-correct\nrecall"),
            ("urgency", "Urgency\nrecall"), ("scarcity", "Scarcity\nrecall"), ("social_proof", "Social-proof\nrecall"),
            ("neg_flag_rate", "Flag rate on\nnegatives (lower=better)"), ("precision", "Image-level\nprecision"),
            ("f1", "Image-level\nF1")]
    ci_key = {"detection_recall": "detection_recall_ci95", "category_correct_recall": "category_correct_recall_ci95",
              "neg_flag_rate": "neg_flag_rate_ci95"}

    def vals_err(m):
        v, lo, hi = [], [], []
        for k, _ in keys:
            if k in CATS:
                val, ci = m["per_category_recall"][k]["recall"], m["per_category_recall"][k]["ci95"]
            else:
                val, ci = m[k], m.get(ci_key.get(k, ""), None)
            v.append(val)
            lo.append(val - ci[0] if ci else 0)
            hi.append(ci[1] - val if ci else 0)
        return v, [lo, hi]

    fig, ax = plt.subplots(figsize=(13, 5.2))
    x = np.arange(len(keys))
    k = len(series)
    w = 0.84 / k
    # before = slot 1 (blue), after = slot 2 (orange); ablations use muted greys so the before/after pair stands out
    colors = {0: COLORS[0], k - 1: COLORS[1]}
    greys = ["#b9b8b2", "#8f8e88"]
    for j, (m, lab) in enumerate(series):
        v, err = vals_err(m)
        xs = x + (j - (k - 1) / 2) * w
        col = colors.get(j, greys[(j - 1) % 2])
        ax.bar(xs, v, w * 0.92, color=col, label=lab, edgecolor=SURF, linewidth=1, yerr=err,
               error_kw={"ecolor": INK2, "elinewidth": 0.7, "capsize": 1.5})
        for xi, vi in zip(xs, v):
            ax.text(xi, vi + 0.012, f"{vi:.2f}", ha="center", va="bottom", fontsize=5.5, color=INK2, rotation=90)
    test_after = series[-1][0]
    ax.set_xticks(x, [k[1] for k in keys], fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score (95% Wilson CI where shown)")
    ax.set_title(f"Screenshots, TEST split (n={test_after['n_pos']} positive / {test_after['n_neg']} negative images): "
                 f"before vs after", loc="left", fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.legend(loc="upper center", fontsize=8, ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(FIGURES / "screenshot_before_after.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- samples
def write_samples(test_items, preds, tau, cfg_name):
    P = [i for i in test_items if i["is_pos"]]
    ok = [i for i in P if flagged_cats(preds[i["image"]], tau) & set(i["gt_cats"])]
    miss = [i for i in P if not (flagged_cats(preds[i["image"]], tau) & set(i["gt_cats"]))]

    def strength(i):
        hits = [c for c in preds[i["image"]] if c["p_dark"] >= tau and c["category"] in i["gt_cats"]]
        return max(c["p_dark"] for c in hits)

    picks = []
    for c in CATS:  # one correct example per category when possible, most confident first
        pool = sorted([i for i in ok if c in (flagged_cats(preds[i["image"]], tau) & set(i["gt_cats"])) and i not in picks],
                      key=lambda i: -strength(i))
        if pool:
            picks.append(pool[0])
    for i in sorted(ok, key=lambda i: -strength(i)):
        if len(picks) >= 3:
            break
        if i not in picks:
            picks.append(i)
    picks = picks[:3]
    missed_pick = sorted(miss, key=lambda i: i["image"])[:1]

    SAMPLES.mkdir(parents=True, exist_ok=True)
    keep = {i["image"] for i in picks + missed_pick}
    for old in SAMPLES.iterdir():  # drop demo images left over from an earlier run of this script
        if old.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp") and old.name not in keep:
            old.unlink()
    rows = []
    for i, kind in [(i, "handled correctly") for i in picks] + [(i, "MISSED") for i in missed_pick]:
        shutil.copy2(ROOT / i["path"], SAMPLES / i["image"])
        hits = sorted([c for c in preds[i["image"]] if c["p_dark"] >= tau], key=lambda c: -c["p_dark"])
        rows.append({"image": i["image"], "kind": kind, "gt_labels": i["gt_labels"], "gt_cats": i["gt_cats"],
                     "split": i["split"], "subset": i["subset"],
                     "predicted": [{"text": c["text"][:160], "category": c["category"], "p_dark": round(c["p_dark"], 3),
                                    "unit": c["unit"]} for c in hits[:4]]})
    md = ["# Screenshot demo inputs", "",
          "Images from **ContextRico-DP** (Zenodo record 15036779, licensed CC-BY-4.0; original filenames kept). They are",
          "used only as evaluation / demo inputs - ClearCart was never trained on them. All come from the **TEST** split of",
          "`ml/eval_screenshots.py`.", "",
          f"Predictions below were produced by that script with OCR config `{cfg_name}` ({CONFIG_DESC[cfg_name]}) and",
          f"flagging threshold p_dark >= {tau}. ClearCart reads text only; visual patterns in these images are not analysed.", ""]
    for r in rows:
        md += [f"## `{r['image']}` - {r['kind']}", "",
               f"- Ground-truth labels: {', '.join(r['gt_labels'])}",
               f"- Text-observable ClearCart categories expected: {', '.join(r['gt_cats'])}",
               f"- Split: {r['split']} ({r['subset']})"]
        if r["predicted"]:
            md.append("- ClearCart flags (top by p_dark):")
            for p in r["predicted"]:
                txt = p["text"].replace("|", "/").replace("`", "'")
                md.append(f"  - `{txt}` -> **{p['category']}**, p_dark={p['p_dark']} ({p['unit']})")
        else:
            md.append("- ClearCart flags: nothing (no OCR candidate reached the threshold)")
        md.append("")
    (SAMPLES / "README.md").write_text("\n".join(md))
    return rows


# --------------------------------------------------------------------------- main
def main() -> None:
    t_start = time.perf_counter()
    items, info = build_eval_set()
    print(f"[data] {info['n_positive_images']} positives, {info['n_negatives_sampled']} negatives; "
          f"split {info['split_sizes']}", flush=True)

    t0 = time.perf_counter()
    ocr = run_ocr(items)
    t_ocr = time.perf_counter() - t0

    from ml.model import TextClassifier
    t0 = time.perf_counter()
    clf = TextClassifier.load(ARTIFACTS)
    default_tau = float(clf.b.threshold)
    scores = score_texts(clf, ocr)
    t_model = time.perf_counter() - t0

    dev = [i for i in items if i["split"] == "dev"]
    test = [i for i in items if i["split"] == "test"]
    preds = {cfg: image_preds(items, ocr[cfg], scores) for cfg in CONFIGS}

    dev_default = {cfg: metrics(dev, preds[cfg], default_tau, ocr[cfg]) for cfg in CONFIGS}
    dev_sweeps = {cfg: [metrics(dev, preds[cfg], t, ocr[cfg]) for t in TAU_GRID] for cfg in CONFIGS}
    chosen = choose(dev_default, dev_sweeps, default_tau)
    cfg, tau = chosen["config"], chosen["tau"]
    print(f"[select] config={cfg} tau={tau} ({chosen['reason']})", flush=True)
    for c, m in dev_default.items():
        print(f"  DEV {c:9s} det={m['detection_recall']:.3f} cat={m['category_correct_recall']:.3f} "
              f"neg={m['neg_flag_rate']:.3f} P={m['precision']:.3f} F1={m['f1']:.3f} "
              f"AUROC={m['auroc_max_p_dark']} s/img={m['mean_ocr_seconds_per_image']}", flush=True)

    # ---- TEST (touched once, after selection) ----
    test_v1 = metrics(test, preds["v1_naive"], default_tau, ocr["v1_naive"])
    test_chosen = metrics(test, preds[cfg], tau, ocr[cfg])
    test_chosen_default_tau = metrics(test, preds[cfg], default_tau, ocr[cfg])
    test_all_configs = {c: metrics(test, preds[c], default_tau, ocr[c]) for c in CONFIGS}
    test_all_configs_09 = {c: metrics(test, preds[c], 0.9, ocr[c]) for c in CONFIGS}
    test_v1_chosen_tau = metrics(test, preds["v1_naive"], tau, ocr["v1_naive"])
    dev_diag = negative_flag_diagnostic(dev, preds[cfg], tau)
    dev_diag_default = negative_flag_diagnostic(dev, preds[cfg], default_tau)
    pair = paired(test, preds["v1_naive"], default_tau, preds[cfg], tau)
    for name, m in [("v1_naive", test_v1), (f"{cfg}@{tau}", test_chosen)]:
        print(f"  TEST {name:12s} det={m['detection_recall']:.3f} cat={m['category_correct_recall']:.3f} "
              f"neg={m['neg_flag_rate']:.3f} P={m['precision']:.3f} F1={m['f1']:.3f}", flush=True)
    errors = error_analysis(test, preds[cfg], tau)
    errors_v1 = error_analysis(test, preds["v1_naive"], default_tau)

    FIGURES.mkdir(parents=True, exist_ok=True)
    n_dev_pos = sum(i["is_pos"] for i in dev)
    dev_at_tau = {c: next(m for m in dev_sweeps[c] if m["tau"] == tau) for c in CONFIGS} if tau != default_tau else None
    panels = [(dev_default, default_tau)] + ([(dev_at_tau, tau)] if dev_at_tau else [])
    fig_config_comparison(panels, n_dev_pos, len(dev) - n_dev_pos, cfg, tau)
    fig_before_after([(test_v1, f"before: v1_naive @ tau={default_tau}"),
                      (test_v1_chosen_tau, f"ablation: v1_naive @ tau={tau} (threshold only)"),
                      (test_chosen_default_tau, f"ablation: {cfg} @ tau={default_tau} (OCR only)"),
                      (test_chosen, f"after: {cfg} @ tau={tau}")])
    samples = write_samples(test, preds[cfg], tau, cfg)

    d_cat = test_chosen["category_correct_recall"] - test_v1["category_correct_recall"]
    d_f1 = test_chosen["f1"] - test_v1["f1"]
    out = {
        "description": ("Screenshot/OCR evaluation of the ClearCart text classifier on ContextRico-DP. ContextRico is used "
                        "for evaluation only; no model is trained or tuned on it except the choice of OCR config and "
                        "flagging threshold, which uses the DEV split only."),
        "dataset": {"name": "ContextRico-DP", "source": "Zenodo 15036779", "license": "CC-BY-4.0", **info},
        "text_observable_label_map": CR_TO_LABEL,
        "split": {"rule": ("50/50 DEV/TEST by image, seed 42; negatives and positives split separately, positives "
                           "further stratified by their rarest ClearCart category"),
                  "dev_images": sorted(i["image"] for i in dev), "test_images": sorted(i["image"] for i in test)},
        "decision_rule": ("image is flagged for category c if any OCR candidate (line or merged block) has p_dark >= tau "
                          "and top dark category c; flagged overall if any candidate has p_dark >= tau. Positive = image "
                          "with >= 1 text-observable GT label. detection_recall = positives flagged with any category; "
                          "category_correct_recall = positives flagged with >= 1 of their GT categories; precision/F1 use "
                          "detection (any flag) on positives vs negatives, so they depend on the ~1:3 pos:neg ratio here."),
        "configs": {c: {"params": CONFIGS[c], "description": CONFIG_DESC[c]} for c in CONFIGS},
        "text_model_threshold": default_tau,
        "selection_rules": SELECTION_RULES,
        "selection_rule_revision": RULE_REVISION,
        "dev": {"at_default_threshold": dev_default, "threshold_sweep": dev_sweeps},
        "chosen": chosen,
        "test": {"before_v1_naive": test_v1, "after_chosen": test_chosen,
                 "chosen_config_at_default_threshold": test_chosen_default_tau,
                 "v1_naive_at_chosen_threshold": test_v1_chosen_tau,
                 "all_configs_at_default_threshold_for_reference_only": test_all_configs,
                 "all_configs_at_tau_0.9_for_reference_only": test_all_configs_09,
                 "paired_before_vs_after": pair,
                 "delta_after_minus_before": {"category_correct_recall": round(d_cat, 4), "f1": round(d_f1, 4),
                                              "detection_recall": round(test_chosen["detection_recall"] - test_v1["detection_recall"], 4),
                                              "neg_flag_rate": round(test_chosen["neg_flag_rate"] - test_v1["neg_flag_rate"], 4)}},
        "error_analysis_test_chosen": errors,
        "dev_negative_flag_diagnostic": {"chosen_config_at_chosen_tau": dev_diag,
                                         "chosen_config_at_default_tau": dev_diag_default,
                                         "note": ("DEV only. Counts flagged negative images whose flags all come from the top "
                                                  "5% of the image (mobile status bar) and/or clock/number-only strings - "
                                                  "the text model reads '00:19' as a countdown.")},
        "error_analysis_test_v1_naive": {k: errors_v1[k] for k in ("n_missed_positives", "n_flagged_negatives", "missed_reason_counts")},
        "samples": samples,
        "caveats": [info["negatives_note"],
                    f"Small positive sets (DEV {n_dev_pos}, TEST {sum(i['is_pos'] for i in test)} images; social proof has "
                    f"only {info['positive_images_per_category']['social_proof']} images in total) - see the Wilson CIs "
                    "and paired McNemar p-values before reading much into differences.",
                    "OCR timings are wall-clock per call with 6 concurrent Tesseract processes (cached on first run); "
                    "compare them relative to each other.",
                    "The backend Auditor suppresses a flagged block that lies inside an already-flagged line; for image-level "
                    "'any flag' this is equivalent to the rule used here, but per-category it can differ slightly."],
        "runtime_seconds": {"ocr_jobs_not_cached_this_run": OCR_CACHE_MISSES,
                            "note": "a cold run (empty OCR cache, 1332 OCR jobs) took 174.6 s total / 156 s OCR; see screenshot_eval.log",
                            "ocr": round(t_ocr, 1), "model_load_and_scoring": round(t_model, 1),
                            "total": round(time.perf_counter() - t_start, 1)},
        "n_unique_ocr_texts_scored": len(scores),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[done] wrote {OUT_JSON.relative_to(ROOT)} in {out['runtime_seconds']['total']}s", flush=True)


if __name__ == "__main__":
    main()
