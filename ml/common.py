"""Shared paths, taxonomy and text utilities for the ClearCart ML pipeline."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "ml" / "artifacts"
FIGURES = ROOT / "report" / "figures"

EC_TSV = DATA_RAW / "ec-darkpattern" / "dataset" / "dataset.tsv"
CR_DIR = DATA_RAW / "contextrico"
CR_GT = CR_DIR / "groundtrue.json"

SEED = 42
EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Taxonomy chosen AFTER inspecting dataset.tsv (see data/processed/inspection_report.md).
# Categories with < 30 examples (Obstruction 27, Sneaking 12, Forced Action 4) cannot
# support a defensible classifier and are excluded from training; they are kept as an
# out-of-taxonomy probe set.
LABELS = ["benign", "urgency", "scarcity", "social_proof", "misdirection"]
SOURCE_TO_LABEL = {
    "Not Dark Pattern": "benign",
    "Urgency": "urgency",
    "Scarcity": "scarcity",
    "Social Proof": "social_proof",
    "Misdirection": "misdirection",
}
EXCLUDED_SOURCE_CATEGORIES = ["Obstruction", "Sneaking", "Forced Action"]

DISPLAY_NAMES = {
    "benign": "No dark pattern",
    "urgency": "Misleading urgency",
    "scarcity": "Misleading scarcity",
    "social_proof": "Social proof pressure",
    "misdirection": "Misdirection / confirmshaming",
}

# ContextRico-DP image labels that correspond to text-observable categories.
# Used ONLY for cross-domain (screenshot OCR) evaluation and threshold dev, never mixed
# into the text test metrics.
CR_TO_LABEL = {
    "Limited Time Message": "urgency",
    "Countdown Timer": "urgency",
    "Countdown on Ad": "urgency",
    "Low Stock Message": "scarcity",
    "High Demand Message": "scarcity",
    "Activity Message": "social_proof",
}


def normalize_text(text: str) -> str:
    """Light normalisation: unicode NFKC + whitespace collapse.

    Case, punctuation and digits are preserved on purpose: "HURRY!" and "Only 2 left"
    carry signal that aggressive cleaning would remove.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace(" ", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def template_key(text: str) -> str:
    """Key used for near-duplicate / template leakage detection.

    Lower-cases, maps every digit run to '0' and strips punctuation so that
    "Hurry! Only 2 left" and "Hurry, only 3 left" collide.
    """
    t = normalize_text(text).lower()
    t = re.sub(r"\d+", "0", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()
