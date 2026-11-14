"""Mathur et al. (2019) "Dark Patterns at Scale" dataset: inspection, matching to ec-darkpattern, and
leakage-safe eligibility for training.

Run from the project root:  .venv/bin/python -m ml.mathur

Inputs
  data/raw/mathur-dark-patterns/data/final-dark-patterns/dark-patterns.csv   (git clone of
      https://github.com/aruneshmathur/dark-patterns, licence GPL-3.0, commit recorded below)
  data/processed/text_dataset.csv   (frozen ClearCart split; READ ONLY)

Outputs (new files; nothing existing is modified)
  data/processed/mathur_rows.csv        one row per Mathur record with domain, label mapping and eligibility flags
  data/processed/mathur_inspection.md   inspection + relationship to ec-darkpattern + exclusion counts + provenance
  data/processed/mathur_stats.json      the numbers behind the markdown

Key fact established here (not assumed): ec-darkpattern `page_id` k is the k-th data row (1-based) of Mathur's
dark-patterns.csv. Every ec-darkpattern dark row has exactly Mathur's Pattern String and Pattern Category for
that row, and ec-darkpattern's benign rows were scraped from that row's `Website Page`. That gives every
ClearCart row a website domain.
"""
from __future__ import annotations

import json
import subprocess
from urllib.parse import urlparse

import pandas as pd

from ml.common import (DATA_PROCESSED, DATA_RAW, EC_TSV, EXCLUDED_SOURCE_CATEGORIES, SOURCE_TO_LABEL,
                       normalize_text, template_key)

MATHUR_DIR = DATA_RAW / "mathur-dark-patterns"
MATHUR_CSV = MATHUR_DIR / "data" / "final-dark-patterns" / "dark-patterns.csv"
MATHUR_URL = "https://github.com/aruneshmathur/dark-patterns"
TEXT_DATASET = DATA_PROCESSED / "text_dataset.csv"

_TWO_LEVEL = {"co.uk", "org.uk", "ac.uk", "com.au", "net.au", "co.nz", "co.in", "com.br", "co.jp", "com.sg",
              "com.mx", "co.za", "com.tr", "com.my", "com.hk", "com.cn", "co.kr", "com.ar", "com.pk", "co.il"}


def site_of(url: str) -> str:
    """Registered-domain heuristic: lower-cased host, 'www.' dropped, last 2 labels (3 for co.uk-style suffixes)."""
    u = str(url).strip()
    host = urlparse(u if "://" in u else "http://" + u).netloc.lower().split(":")[0]
    host = host.removeprefix("www.")
    parts = host.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in _TWO_LEVEL:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def load_mathur() -> pd.DataFrame:
    m = pd.read_csv(MATHUR_CSV, dtype=str, keep_default_na=True)
    m.insert(0, "mathur_row", range(1, len(m) + 1))       # == ec-darkpattern page_id (verified in inspect())
    m["text"] = m["Pattern String"].map(lambda s: normalize_text(s) if isinstance(s, str) else "")
    m["template_key"] = m["text"].map(template_key)
    m["domain"] = m["Website Page"].map(site_of)
    m["label"] = m["Pattern Category"].map(SOURCE_TO_LABEL)  # NaN for out-of-taxonomy categories
    return m


def ec_with_domains() -> pd.DataFrame:
    """The frozen ClearCart dataset with a website domain per row (via page_id -> Mathur row)."""
    df = pd.read_csv(TEXT_DATASET, dtype={"page_id": str, "group_id": str})
    m = load_mathur()
    dom = dict(zip(m["mathur_row"].astype(str), m["domain"]))
    df["domain"] = df["page_id"].map(dom)
    return df


def conflict_template_keys() -> set[str]:
    """Template keys ClearCart dropped as label-conflicting (recomputed exactly as ml/preprocess.py does)."""
    from ml.preprocess import load_clean
    _, _, log = load_clean()
    return {template_key(t) for t, _ in log["dropped_conflicting_template_rows"]}


def eligibility(m: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Flag every Mathur row. A row is eligible for TRAINING only if no flag is set."""
    m = m.copy()
    in_cc = set(df.loc[df["label"] != "benign", "page_id"].astype(int))      # Mathur rows already in ClearCart
    held = df[df["split"].isin(["val", "test"])]
    held_text, held_tk = set(held["text"]), set(held["template_key"])
    held_page, held_dom = set(held["page_id"].astype(int)), set(held["domain"])
    tk_label = df.groupby("template_key")["label"].agg(lambda s: set(s)).to_dict()
    conflicts = conflict_template_keys()

    m["flag_empty_text"] = m["text"] == ""
    m["flag_out_of_taxonomy"] = m["Pattern Category"].isin(EXCLUDED_SOURCE_CATEGORIES)
    m["flag_already_in_clearcart"] = m["mathur_row"].isin(in_cc)
    m["flag_template_conflict"] = [
        (tk in conflicts) or (tk in tk_label and isinstance(lab, str) and tk_label[tk] != {lab})
        for tk, lab in zip(m["template_key"], m["label"])]
    m["flag_valtest_text"] = m["text"].isin(held_text)
    m["flag_valtest_template"] = m["template_key"].isin(held_tk)
    m["flag_valtest_page"] = m["mathur_row"].isin(held_page)
    m["flag_valtest_domain"] = m["domain"].isin(held_dom)
    flags = [c for c in m.columns if c.startswith("flag_")]
    m["eligible"] = ~m[flags].any(axis=1)
    m["first_reason"] = [next((f[5:] for f in flags if r[f]), "eligible") for _, r in m[flags].iterrows()]
    return m


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(MATHUR_DIR), "log", "-1", "--format=%H %ci"], text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def licence_header() -> str:
    lic = (MATHUR_DIR / "LICENSE").read_text(errors="ignore").splitlines()
    return " ".join(l.strip() for l in lic[:2] if l.strip())


def md_table(d: pd.DataFrame) -> str:
    cols = list(d.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def main() -> None:
    m = load_mathur()
    df = ec_with_domains()
    raw = pd.read_csv(EC_TSV, sep="\t", dtype={"page_id": int})

    # ---- verify the page_id <-> Mathur row relationship instead of assuming it
    dark = raw[raw["label"] == 1].merge(m[["mathur_row", "Pattern String", "Pattern Category"]], left_on="page_id",
                                         right_on="mathur_row", how="left", suffixes=("_ec", "_m"))
    rel = {
        "ec_dark_rows": int(len(dark)),
        "ec_dark_rows_text_equal_mathur_pattern_string": int((dark["text"] == dark["Pattern String"]).sum()),
        "ec_dark_rows_category_equal": int((dark["Pattern Category_ec"] == dark["Pattern Category_m"]).sum()),
        "ec_benign_rows": int((raw["label"] == 0).sum()),
        "ec_benign_rows_page_id_is_a_mathur_row": int(raw.loc[raw["label"] == 0, "page_id"].between(1, len(m)).sum()),
        "clearcart_rows_without_domain": int(df["domain"].isna().sum()),
    }

    me = eligibility(m, df)
    me.drop(columns=["Pattern String"]).to_csv(DATA_PROCESSED / "mathur_rows.csv", index=False)

    # ---- exclusion accounting
    flags = [c for c in me.columns if c.startswith("flag_")]
    cand = me[~me["flag_empty_text"] & ~me["flag_already_in_clearcart"]]
    elig = me[me["eligible"]]
    train_text = set(df.loc[df["split"] == "train", "text"])
    excl = {
        "mathur_rows": int(len(me)),
        "flag_counts_all_rows": {f[5:]: int(me[f].sum()) for f in flags},
        "first_reason_counts": {k: int(v) for k, v in me["first_reason"].value_counts().items()},
        "candidates_not_already_in_clearcart_nonempty": int(len(cand)),
        "candidates_excluded_by_valtest_rule": int(cand[["flag_valtest_text", "flag_valtest_template", "flag_valtest_page",
                                                         "flag_valtest_domain"]].any(axis=1).sum()),
        "candidates_excluded_by_valtest_rule_by_kind": {k: int(cand[f"flag_valtest_{k}"].sum()) for k in ["text", "template", "page", "domain"]},
        "in_taxonomy_nonconflict_candidates": int((~cand["flag_out_of_taxonomy"] & ~cand["flag_template_conflict"]).sum()),
        "in_taxonomy_nonconflict_candidates_excluded_by_valtest_rule": int((~cand["flag_out_of_taxonomy"] & ~cand["flag_template_conflict"] & ~cand["eligible"]).sum()),
        "eligible_rows": int(len(elig)),
        "eligible_by_label": {k: int(v) for k, v in elig["label"].value_counts().items()},
        "eligible_unique_texts": int(elig["text"].nunique()),
        "eligible_texts_already_in_clearcart_train": int(elig["text"].isin(train_text).sum()),
        "eligible_unique_texts_not_in_clearcart_train": int(elig.loc[~elig["text"].isin(train_text), "text"].nunique()),
        "eligible_templates_not_in_clearcart_train": int((~elig["template_key"].isin(set(df.loc[df["split"] == "train", "template_key"]))).sum()),
        "candidate_texts_present_in_raw_ec_tsv": int(cand["text"].isin(set(raw["text"].map(normalize_text))).sum()),
    }

    # ---- domain overlap in the frozen ClearCart split (page_id grouping does not group by website)
    dom_overlap = {}
    for a, b in [("train", "test"), ("train", "val"), ("val", "test")]:
        s = set(df.loc[df["split"] == a, "domain"])
        t = df[df["split"] == b]
        dom_overlap[f"{b}_rows_sharing_domain_with_{a}"] = int(t["domain"].isin(s).sum())
        dom_overlap[f"{b}_rows"] = int(len(t))

    cat_counts = m["Pattern Category"].value_counts().to_dict()
    type_tab = (m.groupby(["Pattern Category", "Pattern Type"]).size().rename("rows").reset_index()
                .sort_values(["Pattern Category", "rows"], ascending=[True, False]))
    dec_tab = pd.crosstab(m["Pattern Type"], m["Deceptive?"]).reset_index()
    dec_cat = pd.crosstab(m["Pattern Category"], m["Deceptive?"]).reset_index()
    missing = m[["Pattern String", "Comment", "Pattern Category", "Pattern Type", "Where in website?", "Deceptive?", "Website Page"]].isna().sum().to_dict()
    nonempty = m[m["text"] != ""]
    dec_nonempty = nonempty["Deceptive?"].value_counts().to_dict()

    stats = {"source": MATHUR_URL, "commit": git_commit(), "licence": licence_header(), "columns": list(m.columns[1:8]),
             "rows": int(len(m)), "missing_values": {k: int(v) for k, v in missing.items()},
             "pattern_category_counts": {k: int(v) for k, v in cat_counts.items()},
             "deceptive_counts": {k: int(v) for k, v in m["Deceptive?"].value_counts().items()},
             "deceptive_counts_nonempty_text": {k: int(v) for k, v in dec_nonempty.items()},
             "unique_domains": int(m["domain"].nunique()), "relationship_to_ec_darkpattern": rel,
             "exclusions": excl, "clearcart_split_domain_overlap": dom_overlap}
    (DATA_PROCESSED / "mathur_stats.json").write_text(json.dumps(stats, indent=2))

    reason_tab = pd.DataFrame([{"first reason (flags checked in this order)": k, "rows": v} for k, v in excl["first_reason_counts"].items()])
    md = f"""# Mathur et al. (2019) dataset - inspection and relationship to ec-darkpattern

Generated by `ml/mathur.py` from `{MATHUR_CSV.relative_to(DATA_RAW.parents[1])}`. Every number below is computed by that script.

## Provenance and licence

| Dataset | Source | Version | Licence (verified from file) | Use in ClearCart |
|---|---|---|---|---|
| Mathur et al. 2019, "Dark Patterns at Scale" (`data/final-dark-patterns/dark-patterns.csv`) | {MATHUR_URL} | git commit `{stats['commit']}` | `LICENSE` file at repo root: **{stats['licence']}** (README: "Please see the license file") | v2 training rows (eligible only), `Deceptive?` research model, website-domain grouping |
| ec-darkpattern (Yada et al. 2022) `dataset/dataset.tsv` | https://github.com/yamanalab/ec-darkpattern | as already used | Apache-2.0 (`LICENSE`) | v1 + v2 train / frozen val / frozen test |
| distilroberta-base | https://huggingface.co/distilroberta-base | HF snapshot `fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b` | Apache-2.0 (model card) | fine-tuned for transformer_v1data / transformer_v2 |
| all-MiniLM-L6-v2 | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | as already used | Apache-2.0 | frozen embedder in the ensemble |

Licence note: GPL-3.0 is an open-source licence that permits use, study and modification, so using the data for
training and evaluation is allowed. It is a copyleft licence: if Mathur-derived files (this processed CSV, or a model
trained on it) are redistributed, they should be distributed under GPL-3.0-compatible terms with attribution. ec-darkpattern
itself states its dark rows were taken from this file (its README "Changes from Mathur et al.'s Dataset").
Citation: Mathur, Acar, Friedman, Lucherini, Mayer, Chetty, Narayanan. Dark Patterns at Scale. Proc. ACM HCI (CSCW) 2019.

## Columns and missing values

Columns: {', '.join('`'+c+'`' for c in stats['columns'])}. Rows: {len(m)}. Unique website domains: {stats['unique_domains']}.

{md_table(pd.DataFrame([{'column': k, 'missing': v} for k, v in missing.items()]))}

`Pattern String` is missing for {missing['Pattern String']} rows (patterns recorded only as screenshots / interactions); those rows
carry no text and cannot be used by a text model.

## Pattern Category and Pattern Type

{md_table(pd.DataFrame([{'Pattern Category': k, 'rows': v} for k, v in cat_counts.items()]))}

{md_table(type_tab)}

## `Deceptive?`

All Mathur rows are dark patterns. `Deceptive? = Yes` means the authors verified the claim was false (e.g. a countdown
timer that resets, a random "low stock" number); `No` means a real pressure tactic that was **not shown to be fake** - it is
NOT a benign label. `Depends` is used for Sneaking rows.

All rows: {stats['deceptive_counts']}. Rows with non-empty text: {dec_nonempty}.

{md_table(dec_tab)}

{md_table(dec_cat)}

`Yes` is concentrated in a few types (Countdown Timer, Activity Notification, Low-stock Message); several types are never `Yes`.

## Relationship to ec-darkpattern (verified)

* ec-darkpattern `page_id` k = data row k (1-based) of `dark-patterns.csv`: {rel['ec_dark_rows_text_equal_mathur_pattern_string']} / {rel['ec_dark_rows']} ec dark rows have
  exactly that row's Pattern String, and {rel['ec_dark_rows_category_equal']} / {rel['ec_dark_rows']} its Pattern Category. All {rel['ec_benign_rows_page_id_is_a_mathur_row']} / {rel['ec_benign_rows']} benign
  rows have a page_id that is a Mathur row (they were scraped from that row's Website Page).
* So Mathur contributes **no new benign text** and, for dark text, only the rows ec-darkpattern dropped as missing or
  duplicate. Of the {excl['candidates_not_already_in_clearcart_nonempty']} non-empty Mathur rows that are not already ClearCart rows, {excl['candidate_texts_present_in_raw_ec_tsv']} have a text that is already present in
  ec-darkpattern's `dataset.tsv` (i.e. they are exact duplicates recorded on another website).
* What Mathur adds is therefore (a) the `Deceptive?` flag, (b) the finer `Pattern Type`, (c) website URL/domain per
  row, and (d) duplicate occurrences of existing strings on other websites.

## Leakage rule and exclusions

Frozen: ClearCart VAL and TEST rows (`data/processed/text_dataset.csv`) are untouched. A Mathur row may be used for
training only if it has text, is in ClearCart's taxonomy, is not already a ClearCart row, is not in a label-conflicting
template, and its normalised text, template key, page (Mathur row = ec page_id) and website domain all differ from
every VAL/TEST row.

{md_table(reason_tab)}

All flags (a row can have several): {excl['flag_counts_all_rows']}.

* Candidates (non-empty, not already a ClearCart row): **{excl['candidates_not_already_in_clearcart_nonempty']}**; excluded by the VAL/TEST rule: **{excl['candidates_excluded_by_valtest_rule']}**
  (text {excl['candidates_excluded_by_valtest_rule_by_kind']['text']}, template {excl['candidates_excluded_by_valtest_rule_by_kind']['template']}, page {excl['candidates_excluded_by_valtest_rule_by_kind']['page']}, domain {excl['candidates_excluded_by_valtest_rule_by_kind']['domain']}; overlapping).
  Restricted to in-taxonomy, non-conflicting candidates: {excl['in_taxonomy_nonconflict_candidates']}, of which {excl['in_taxonomy_nonconflict_candidates_excluded_by_valtest_rule']} excluded by the VAL/TEST rule.
* **Eligible for training: {excl['eligible_rows']} rows** ({excl['eligible_by_label']}), {excl['eligible_unique_texts']} unique texts.
  {excl['eligible_texts_already_in_clearcart_train']} of them have a text already in ClearCart TRAIN; unique eligible texts not in TRAIN: **{excl['eligible_unique_texts_not_in_clearcart_train']}**;
  eligible rows whose template is not in TRAIN: {excl['eligible_templates_not_in_clearcart_train']}.
  In other words the v2 data mostly re-weights strings TRAIN already contains; it adds little new language.

## Finding: website-level overlap in the frozen split

ClearCart groups by page_id (= one Mathur row) and template, not by website. With domains recovered from Mathur:
{dom_overlap['test_rows_sharing_domain_with_train']} / {dom_overlap['test_rows']} TEST rows and {dom_overlap['val_rows_sharing_domain_with_train']} / {dom_overlap['val_rows']} VAL rows come from a website that also has TRAIN rows. The split is kept
frozen as required; `ml/train_v2.py` additionally reports TEST metrics on the subset of TEST rows whose domain never
appears in training.
"""
    (DATA_PROCESSED / "mathur_inspection.md").write_text(md)
    print(json.dumps({"relationship": rel, "exclusions": excl, "domain_overlap": dom_overlap}, indent=1))


if __name__ == "__main__":
    main()
