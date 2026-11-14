"""Pressure aggregation shared by FSPM and CDI.

Content pressure of a page/stage combines the classifier's calibrated per-category scores:

    c_k  = max calibrated P(dark) over FLAGGED findings of category k      (0 if none flagged)
    c'_urgency = max(c_urgency, strength of observed pressure facts)       (e.g. countdown reset)
    p    = lam * max_k c'_k + (1 - lam) * (1/4) * sum_k c'_k                (lam = 0.5)

The first term is intensity (the single strongest tactic), the second breadth (how much of the
four-category repertoire is used). Below-threshold elements contribute 0: at the validated operating
point they are screened as benign, so 0 is a measured value, not a missing one. Within a category
only the strongest finding counts, so ten copies of the same banner cannot inflate the score
(de-duplication of correlated signals).

Design note: an earlier noisy-OR version (1 - prod_k(1 - c_k)) saturated at ~1.0 for any page with a
single flagged element because the calibrated classifier is near-binary on in-distribution text,
which made FSPM weighting schemes indistinguishable on real journeys. noisy_or() is kept for reference.
"""
from __future__ import annotations

from backend.intelligence.config import OBSERVED_FACT_STRENGTH, OBSERVED_WEAK_FACT_STRENGTH, PRESSURE_INTENSITY_WEIGHT
from backend.intelligence.schemas import EvidenceRef

ML_CATEGORIES = ("urgency", "scarcity", "social_proof", "misdirection")


def noisy_or(values: list[float]) -> float:
    prod = 1.0
    for v in values:
        prod *= 1.0 - min(1.0, max(0.0, float(v)))
    return 1.0 - prod


def category_maxima(findings, stage_index: int | None = None) -> tuple[dict[str, float], dict[str, EvidenceRef]]:
    """Per-category max calibrated score over ML findings, with the finding that set it."""
    best: dict[str, float] = {}
    src: dict[str, EvidenceRef] = {}
    for f in findings:
        if f.detection_method != "ml_classifier" or f.category not in ML_CATEGORIES or f.model_score is None:
            continue
        if f.model_score > best.get(f.category, -1):
            best[f.category] = float(f.model_score)
            src[f.category] = EvidenceRef(stage_index=stage_index, finding_id=f.id, text=f.evidence)
    return best, src


def blend(channels: dict[str, float], lam: float = PRESSURE_INTENSITY_WEIGHT) -> float:
    """Intensity-breadth pressure over the four categories (missing categories = 0, i.e. not flagged)."""
    vals = [min(1.0, max(0.0, float(channels.get(k, 0.0)))) for k in ML_CATEGORIES]
    return lam * max(vals) + (1.0 - lam) * sum(vals) / len(ML_CATEGORIES)


def content_pressure(findings, stage_index: int | None = None, extra_facts: list[tuple[float, EvidenceRef]] | None = None):
    """Intensity-breadth pressure of a page/stage; observed pressure facts raise the urgency channel.

    Returns (value in [0,1], evidence sources ordered by strength).
    """
    best, src = category_maxima(findings, stage_index)
    channels = dict(best)
    sources = [src[k] for k in sorted(src, key=lambda k: -best[k])]
    for strength, ref in extra_facts or []:
        channels["urgency"] = max(channels.get("urgency", 0.0), float(strength))
        sources.append(ref)
    return blend(channels), sources


def obstruction_facts(findings, stage_index: int | None = None) -> list[tuple[float, EvidenceRef]]:
    """Directly observed obstruction facts from DOM observations (pre-selected options)."""
    out = []
    for f in findings:
        if f.detection_method == "dom_observation" and f.category == "preselected_option":
            strong = "cost or consent" in (f.short_reason or "")
            out.append((OBSERVED_FACT_STRENGTH if strong else OBSERVED_WEAK_FACT_STRENGTH,
                        EvidenceRef(stage_index=stage_index, finding_id=f.id, text=f.evidence)))
    return out
