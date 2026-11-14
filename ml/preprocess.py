"""Step 2 - Preprocessing + leakage-safe train/val/test split for the text dataset.

Outputs
  data/processed/text_dataset.csv   - one row per usable example with split assignment
  data/processed/probe_excluded.csv - Obstruction / Sneaking / Forced Action rows (never trained on)
  data/processed/split_stats.json   - class counts per split + leakage checks

Split design (fixes the leakage we found in the first naive split, see train.py experiment E1):
  * group = union of (page_id) and (template_key). Rows from the same page OR sharing a
    digit/case/punctuation-normalised template ("Only 2 left" ~ "Only 7 left") always land
    in the same split.
  * StratifiedGroupKFold(7) -> fold 0 = test, fold 1 = validation, folds 2-6 = train.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


from ml.common import (DATA_PROCESSED, EC_TSV, EXCLUDED_SOURCE_CATEGORIES, LABELS,  # noqa: E402
                    SEED, SOURCE_TO_LABEL, normalize_text, template_key)


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def load_clean() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = pd.read_csv(EC_TSV, sep="\t", encoding="utf-8", dtype={"page_id": str})
    raw = raw.reset_index().rename(columns={"index": "row_id", "text": "text_raw", "Pattern Category": "source_category"})
    raw["text"] = raw["text_raw"].map(normalize_text)
    log = {"raw_rows": int(len(raw))}

    empty = raw["text"] == ""
    log["dropped_empty"] = int(empty.sum())
    raw = raw[~empty]

    # exact duplicates (after normalisation): keep first occurrence
    dup = raw.duplicated(subset=["text"], keep="first")
    log["dropped_exact_duplicates"] = int(dup.sum())
    log["dropped_exact_duplicate_texts"] = raw.loc[dup, "text"].tolist()
    raw = raw[~dup]

    raw["template_key"] = raw["text"].map(template_key)
    probe = raw[raw["source_category"].isin(EXCLUDED_SOURCE_CATEGORIES)].copy()
    df = raw[raw["source_category"].isin(SOURCE_TO_LABEL)].copy()
    df["label"] = df["source_category"].map(SOURCE_TO_LABEL)
    log["excluded_probe_rows"] = int(len(probe))

    # template groups with conflicting labels -> ambiguous, drop from training
    n_lab = df.groupby("template_key")["label"].nunique()
    conflict_keys = set(n_lab[n_lab > 1].index)
    conflict = df["template_key"].isin(conflict_keys)
    log["dropped_conflicting_template_rows"] = df.loc[conflict, ["text", "label"]].values.tolist()
    df = df[~conflict]
    return df, probe, log


def assign_groups(df: pd.DataFrame) -> pd.Series:
    uf = UnionFind()
    for pid, tk in zip(df["page_id"], df["template_key"]):
        uf.union(("p", pid), ("t", tk))
    return pd.Series([str(uf.find(("p", p))) for p in df["page_id"]], index=df.index)


def split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["group_id"] = assign_groups(df)
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
    fold = np.full(len(df), -1)
    for i, (_, idx) in enumerate(sgkf.split(df, df["label"], groups=df["group_id"])):
        fold[idx] = i
    df["split"] = np.where(fold == 0, "test", np.where(fold == 1, "val", "train"))
    return df


def leakage_checks(df: pd.DataFrame) -> dict:
    out = {}
    for col in ["text", "template_key", "page_id", "group_id"]:
        sets = {s: set(df.loc[df["split"] == s, col]) for s in ["train", "val", "test"]}
        out[f"{col}_overlap_train_test"] = len(sets["train"] & sets["test"])
        out[f"{col}_overlap_train_val"] = len(sets["train"] & sets["val"])
        out[f"{col}_overlap_val_test"] = len(sets["val"] & sets["test"])
    return out


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df, probe, log = load_clean()
    df = split(df)
    stats = {
        "seed": SEED,
        "cleaning_log": log,
        "rows_used": int(len(df)),
        "n_groups": int(df["group_id"].nunique()),
        "split_sizes": {k: int(v) for k, v in df["split"].value_counts().items()},
        "class_counts_by_split": {
            s: {lab: int(((df["split"] == s) & (df["label"] == lab)).sum()) for lab in LABELS}
            for s in ["train", "val", "test"]
        },
        "leakage_checks": leakage_checks(df),
        "probe_counts": {k: int(v) for k, v in probe["source_category"].value_counts().items()},
    }
    cols = ["row_id", "page_id", "group_id", "split", "source_category", "label", "text", "text_raw", "template_key"]
    df[cols].to_csv(DATA_PROCESSED / "text_dataset.csv", index=False)
    probe[["row_id", "page_id", "source_category", "text"]].to_csv(DATA_PROCESSED / "probe_excluded.csv", index=False)
    (DATA_PROCESSED / "split_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps({k: stats[k] for k in ["rows_used", "n_groups", "split_sizes", "class_counts_by_split", "leakage_checks", "probe_counts"]}, indent=1))
    print("cleaning:", {k: (v if not isinstance(v, list) else len(v)) for k, v in log.items()})


if __name__ == "__main__":
    main()
