import json
import pandas as pd
import pytest

from ml.common import CR_GT, DATA_PROCESSED, EC_TSV, LABELS, normalize_text, template_key


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA_PROCESSED / "text_dataset.csv", dtype={"page_id": str}, keep_default_na=False)


import pytest  # noqa: E402

needs_raw = pytest.mark.skipif(not EC_TSV.exists() or not CR_GT.exists(),
                               reason="raw datasets not downloaded: run `python -m ml.download_data`")


@needs_raw
def test_raw_tsv():
    raw = pd.read_csv(EC_TSV, sep="\t", dtype={"page_id": str})
    assert len(raw) == 2356
    assert list(raw.columns) == ["page_id", "text", "label", "Pattern Category"]


@needs_raw
def test_contextrico_groundtruth():
    g = json.loads(CR_GT.read_text())
    assert len(g) == 1608
    assert all("labels" in v for v in g.values())


def test_labels_and_text(df):
    assert set(df["label"]) <= set(LABELS)
    assert df["text"].notna().all()
    assert (df["text"].astype(str).str.strip() != "").all()
    assert set(df["split"]) == {"train", "val", "test"}


@pytest.mark.parametrize("col", ["text", "template_key", "page_id", "group_id"])
def test_no_split_overlap(df, col):
    sets = {s: set(df.loc[df["split"] == s, col]) for s in ["train", "val", "test"]}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not (sets[a] & sets[b]), f"{col} overlap {a}/{b}: {list(sets[a] & sets[b])[:5]}"


def test_template_key_recomputed(df):
    assert (df["text"].map(template_key) == df["template_key"]).all()


def test_split_stats_match(df):
    st = json.loads((DATA_PROCESSED / "split_stats.json").read_text())
    for s in ["train", "val", "test"]:
        for lab in LABELS:
            assert st["class_counts_by_split"][s][lab] == int(((df.split == s) & (df.label == lab)).sum()), (s, lab)
        assert st["split_sizes"][s] == int((df.split == s).sum())
    assert st["rows_used"] == len(df)


def test_normalize_preserves_case_punct():
    assert normalize_text("  HURRY!!   Only  2 left. \n") == "HURRY!! Only 2 left."
    assert normalize_text(None) == ""


def test_template_key_collisions():
    assert template_key("Only 2 left!") == template_key("only 7 left")
    assert template_key("Only 2 left!") != template_key("Only 2 right")
