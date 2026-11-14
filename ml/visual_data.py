"""ContextRico-DP data for the experimental VISUAL pattern-type model: labels, corrections, groups, split, image cache.

Run: .venv/bin/python -m ml.visual_data

ContextRico has no benign screenshots: every image shows at least one dark pattern. The visual model therefore learns
"which visual pattern type", never "deceptive or not". Images whose labels are all non-target (e.g. only Limited Time
Message) are kept as all-negative rows for the target labels.

Labels: groundtrue.json as published. It already reflects the dataset's correction.xlsx (checked: for 94 of the 97
matching rows, (gt - original) | new == gt). The 3 rows where correction.xlsx would add a label are reported as
discrepancies and NOT applied, because the published ground truth is the authoritative label set.

Groups (no group spans two splits): web files by domain; named mobile files by app name; numeric Rico ids are their own
group; then any two images whose 64-bit dHash differs in <= 4 bits are merged (union-find), so near-duplicate screens
never sit on both sides of a split. Split 70/15/15 by greedy iterative stratification over the target labels and the
Mobile/Web subset, at group level, seed 42.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ml.common import CR_DIR, DATA_PROCESSED, SEED

IMG_DIR = CR_DIR / "images" / "ContextRico-DP"
GT = CR_DIR / "groundtrue.json"
CORR = CR_DIR / "correction.xlsx"
SPLIT_CSV = DATA_PROCESSED / "visual_split.csv"
STATS_JSON = DATA_PROCESSED / "visual_split_stats.json"
ART = Path(__file__).parent / "artifacts_visual"
CACHE = ART / "cache"
SIZE = 384

REQUIRED = ["Hard to Close", "Default Choice", "False Hierarchy", "Nagging"]
OPTIONAL = ["Privacy Zuckering", "Pay to Avoid Ad"]
TARGETS = REQUIRED + OPTIONAL
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
HAMMING_MAX = 4

_CORR_NAMES = {"nagging-pop up ad": "Nagging", "limited time messag": "Limited Time Message", "countdown on ad": "Countdown on Ad",
               "watch ad to unlock features": "Watch Ad to Unlock Features", "no dp": None}


def _canon(name: str, vocab: dict[str, str]) -> str | None:
    n = name.strip().lower()
    if n in _CORR_NAMES:
        return _CORR_NAMES[n]
    if n in vocab:
        return vocab[n]
    raise ValueError(f"unknown label in correction.xlsx: {name!r}")


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def load_labels() -> tuple[dict[str, list[str]], dict]:
    gt = {k: list(v["labels"]) for k, v in json.loads(GT.read_text()).items()}
    vocab = {l.lower(): l for v in gt.values() for l in v}
    corr = pd.read_excel(CORR)
    corr.columns = [c.strip() for c in corr.columns]
    applied, missing, disc = 0, [], []
    for _, r in corr.iterrows():
        img = str(r["Image"]).strip()
        if img not in gt:
            missing.append(img)
            continue
        orig = {_canon(x, vocab) for x in str(r["Original Label"]).split(",") if x.strip()} - {None}
        new = {_canon(x, vocab) for x in str(r["New Label"]).split(",") if x.strip()} - {None}
        before = set(gt[img])
        after = (before - orig) | new
        applied += 1
        if after != before:
            disc.append({"image": img, "groundtruth": sorted(before), "correction_would_give": sorted(after)})
    return gt, {"rows": len(corr), "matched": applied, "already_reflected_in_groundtruth": applied - len(disc),
                "discrepancies_not_applied": disc, "image_not_in_groundtruth": missing}


def group_key(name: str, subset: str) -> str:
    stem = Path(name).stem
    if subset == "Web":
        m = re.search(r"((?:[a-z0-9-]+\.)+[a-z]{2,})", stem.lower())
        return "web:" + (m.group(1).removeprefix("www.") if m else stem)
    m = re.match(r"[a-z]+_\d+--(.+?)-\d+-\d+_", stem)
    if m:
        return "app:" + m.group(1).lower()
    return "rico:" + stem


def dhash(path: Path) -> int:
    g = Image.open(path).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def build_table() -> tuple[pd.DataFrame, dict]:
    labels, corr_info = load_labels()
    files = {p.name: p for p in IMG_DIR.rglob("*") if p.is_file() and not p.name.startswith(".")}
    rows = []
    for name in sorted(labels):
        if name not in files:
            continue
        p = files[name]
        rows.append({"image": name, "path": str(p.relative_to(CR_DIR.parents[2])), "subset": p.parent.name,
                     "labels": "|".join(labels[name]), "dhash": dhash(p)})
    df = pd.DataFrame(rows)
    keys = [group_key(r.image, r.subset) for r in df.itertuples()]
    uf = _UF(len(df))
    first = {}
    for i, k in enumerate(keys):
        if k in first:
            uf.union(i, first[k])
        first.setdefault(k, i)
    h = np.array([int(x) for x in df.dhash], dtype=np.uint64)
    near = 0
    for i in range(len(h)):
        x = np.bitwise_xor(h[i + 1:], h[i])
        d = np.array([bin(int(v)).count("1") for v in x]) if len(x) else np.array([])
        for j in np.nonzero(d <= HAMMING_MAX)[0]:
            near += 1
            uf.union(i, i + 1 + int(j))
    df["group"] = [f"g{uf.find(i)}" for i in range(len(df))]
    for t in TARGETS:
        df[t] = df.labels.apply(lambda s, t=t: int(t in s.split("|")))
    info = {"corrections": corr_info, "images": len(df), "groundtruth_entries": len(labels),
            "files_without_label": sorted(set(files) - set(labels)), "near_duplicate_pairs_merged": near,
            "groups": int(df.group.nunique()), "largest_group": int(df.group.value_counts().max())}
    return df, info


def iterative_split(df: pd.DataFrame, seed: int = SEED, ratios: dict | None = None) -> pd.Series:
    """Greedy iterative stratification (Sechidis et al. 2011) over groups; columns = TARGETS + subset."""
    ratios = ratios or RATIOS
    rng = np.random.default_rng(seed)
    cols = TARGETS + ["is_web"]
    d = df.assign(is_web=(df.subset == "Web").astype(int))
    g = d.groupby("group")[cols].sum()
    sizes = d.groupby("group").size()
    order = rng.permutation(len(g))
    g, sizes = g.iloc[order], sizes.iloc[order]
    splits = list(ratios)
    want = {s: {c: ratios[s] * g[c].sum() for c in cols} for s in splits}
    want_n = {s: ratios[s] * len(d) for s in splits}
    assign: dict[str, str] = {}
    remaining = set(g.index)
    while remaining:
        rem = g.loc[sorted(remaining, key=list(g.index).index)]
        totals = {c: rem[c].sum() for c in cols}
        live = [c for c in cols if totals[c] > 0]
        if live:
            c = min(live, key=lambda c: totals[c])
            cand = rem[rem[c] > 0]
        else:
            c, cand = None, rem
        for grp in cand.index:
            best = max(splits, key=lambda s: ((want[s][c] if c else 0), want_n[s], rng.random()))
            assign[grp] = best
            for cc in cols:
                want[best][cc] -= g.at[grp, cc]
            want_n[best] -= sizes[grp]
            remaining.discard(grp)
    return df.group.map(assign)


def cache_images(df: pd.DataFrame) -> Path:
    """Letterbox every image to SIZE x SIZE (pad with white, never crop) and store uint8 arrays once."""
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"letterbox_{SIZE}.npz"
    if out.exists():
        z = np.load(out, allow_pickle=False)
        if list(z["names"]) == list(df.image):
            return out
    root = CR_DIR.parents[2]
    arr = np.zeros((len(df), SIZE, SIZE, 3), dtype=np.uint8)
    for i, p in enumerate(df.path):
        arr[i] = letterbox(Image.open(root / p))
    np.savez(out, images=arr, names=np.array(df.image, dtype=str))
    return out


def letterbox(im: Image.Image, size: int = SIZE) -> np.ndarray:
    im = im.convert("RGB")
    s = size / max(im.size)
    w, h = max(1, round(im.width * s)), max(1, round(im.height * s))
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(im.resize((w, h), Image.Resampling.BICUBIC), ((size - w) // 2, (size - h) // 2))
    return np.asarray(canvas)


def letterbox_box(img_w: int, img_h: int, size: int = SIZE) -> tuple[float, float, float]:
    """(scale, pad_x, pad_y) to map letterboxed coordinates back to the original image."""
    s = size / max(img_w, img_h)
    return s, (size - round(img_w * s)) / 2, (size - round(img_h * s)) / 2


def main() -> None:
    df, info = build_table()
    df["split"] = iterative_split(df)
    ART.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["dhash"]).to_csv(SPLIT_CSV, index=False)
    per = {s: {"images": int((df.split == s).sum()), "groups": int(df[df.split == s].group.nunique()),
               "web": int(((df.split == s) & (df.subset == "Web")).sum()),
               **{t: int(df.loc[df.split == s, t].sum()) for t in TARGETS},
               "no_target_label": int((df.loc[df.split == s, TARGETS].sum(axis=1) == 0).sum())} for s in RATIOS}
    overlap = {f"{a}-{b}": len(set(df[df.split == a].group) & set(df[df.split == b].group))
               for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    from ml.eval_screenshots import build_eval_set
    ev, _ = build_eval_set()
    ev_names = {x["image"] for x in ev}
    info.update(per_split=per, group_overlap=overlap,
                source_md5={p.name: md5(p) for p in (GT, CORR)},
                ocr_eval_images_also_in_visual_split={s: int(df[(df.split == s) & df.image.isin(ev_names)].shape[0]) for s in RATIOS},
                ocr_eval_note=("ml/eval_screenshots DEV/TEST images are ContextRico files and also appear here. The OCR/text "
                               "evaluation is unaffected because the text model is not trained on them and does not change."))
    STATS_JSON.write_text(json.dumps(info, indent=2))
    cache_images(df)
    print(json.dumps({k: info[k] for k in ("corrections", "images", "groups", "near_duplicate_pairs_merged", "per_split", "group_overlap")}, indent=1))


if __name__ == "__main__":
    main()
