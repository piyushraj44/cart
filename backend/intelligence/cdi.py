"""CDI - Composite Deception Intelligence index (0-100 screening index).

    CDI = 100 * sum_{j in A} w_j x_j / sum_{j in A} w_j          A = available components

Components x_j in [0,1] (config.CDI_WEIGHTS, version-tagged):
  content_pressure     single page: noisy-OR over per-category maxima of calibrated classifier scores
                       (urgency, scarcity, social proof, misdirection); journey: FSPM/100, which is built
                       from the same per-stage quantity - so the two are never added together.
  choice_asymmetry     worst choice set's MCAI over the VISUAL + INTERACTION dimensions only. The semantic
                       MCAI dimension re-uses the classifier's misdirection probability, already inside
                       content_pressure, so it is excluded here to avoid double counting.
  observed_obstruction strongest directly observed obstruction fact (pre-selected options in the DOM,
                       newly introduced pre-selections across a journey).

Missing components are removed from both sums (never treated as 0); `coverage` reports the share
of configured weight that was available, separately from the score. The CDI is NOT a probability
that a merchant is fraudulent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from backend.intelligence.config import (CDI_BANDS, CDI_WEIGHTS, CDI_WEIGHTS_VERSION, SENSITIVITY_DIRICHLET_CONCENTRATION,
                                         SENSITIVITY_DIRICHLET_SAMPLES, SENSITIVITY_ONE_AT_A_TIME, SENSITIVITY_SEED)
from backend.intelligence.aggregation import alternatives as _alternatives
from backend.intelligence.schemas import CDIComponent, CDIConfidence, CDIResult, CDISensitivity, EvidenceRef

LABELS = {"content_pressure": "Content pressure (classifier)", "choice_asymmetry": "Choice asymmetry (visual + interaction)",
          "observed_obstruction": "Observed obstruction (DOM / journey facts)"}


@dataclass
class ComponentInput:
    key: str
    available: bool
    value: Optional[float] = None
    provenance: str = "unavailable"
    explanation: str = ""
    evidence: list[EvidenceRef] = field(default_factory=list)


def band(score: Optional[float]) -> Optional[str]:
    if score is None:
        return None
    for lo, name in CDI_BANDS:
        if score >= lo:
            return name
    return "low"


def _score(values: dict[str, float], weights: dict[str, float]) -> float:
    wsum = sum(weights[k] for k in values)
    return 100.0 * sum(weights[k] * v for k, v in values.items()) / wsum


def sensitivity(values: dict[str, float], weights: dict[str, float]) -> CDISensitivity:
    oat = []
    for k in values:
        res = []
        for f in (1 - SENSITIVITY_ONE_AT_A_TIME, 1 + SENSITIVITY_ONE_AT_A_TIME):
            w = dict(weights); w[k] = weights[k] * f
            res.append(_score(values, w))
        oat.append({"component": k, "low": round(min(res), 2), "high": round(max(res), 2)})
    keys = list(values)
    if len(keys) >= 2:
        rng = np.random.default_rng(SENSITIVITY_SEED)
        base = np.array([weights[k] for k in keys]); base = base / base.sum()
        draws = rng.dirichlet(SENSITIVITY_DIRICHLET_CONCENTRATION * base, SENSITIVITY_DIRICHLET_SAMPLES)
        vals = 100.0 * draws @ np.array([values[k] for k in keys])
        p05, p95 = float(np.percentile(vals, 5)), float(np.percentile(vals, 95))
    else:
        p05 = p95 = _score(values, weights)
    return CDISensitivity(one_at_a_time=oat, dirichlet_p05=round(p05, 2), dirichlet_p95=round(p95, 2),
                          samples=SENSITIVITY_DIRICHLET_SAMPLES if len(keys) >= 2 else 0)


def compute_cdi(inputs: list[ComponentInput], weights: dict[str, float] = CDI_WEIGHTS,
                with_sensitivity: bool = True) -> CDIResult:
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("CDI weights must sum to 1")
    by_key = {c.key: c for c in inputs}
    unknown = set(by_key) - set(weights)
    if unknown:
        raise ValueError(f"unknown CDI components: {unknown}")
    avail = {k: float(min(1.0, max(0.0, c.value))) for k, c in by_key.items() if c.available and c.value is not None}
    wsum = sum(weights[k] for k in avail)
    comps = []
    for k in weights:
        c = by_key.get(k) or ComponentInput(key=k, available=False, explanation="Not computed for this input.")
        ok = k in avail
        comps.append(CDIComponent(
            key=k, label=LABELS[k], available=ok, value=round(avail[k], 4) if ok else None, weight=weights[k],
            effective_weight=round(weights[k] / wsum, 4) if ok and wsum else None,
            contribution=round(100.0 * weights[k] * avail[k] / wsum, 2) if ok and wsum else None,
            provenance=c.provenance if ok else "unavailable", explanation=c.explanation, evidence=c.evidence))
    if not avail:
        return CDIResult(cdi=None, band=None, weights_version=CDI_WEIGHTS_VERSION, components=comps, coverage=0.0,
                         evidence_items=0, sensitivity=None)
    score = _score(avail, weights)
    return CDIResult(cdi=round(score, 2), band=band(score), weights_version=CDI_WEIGHTS_VERSION, components=comps,
                     coverage=round(wsum, 4), evidence_items=sum(len(c.evidence) for c in comps if c.available),
                     sensitivity=sensitivity(avail, weights) if with_sensitivity else None,
                     alternatives=_alternatives(avail, weights))


def confidence(result: CDIResult, model_scores: dict) -> CDIConfidence:
    """Evidence confidence, reported separately from severity (CDI) and coverage.

    model_scores maps (stage_index, finding_id) -> calibrated P(dark) of ML findings.
    Counting rule: content-pressure evidence that is an ML finding counts as 'model', otherwise 'observed'
    (directly observed facts such as a countdown reset); choice-asymmetry counts one item per provenance
    token of its dimensions (measured / inferred / declared / default); obstruction evidence is 'observed'.
    Level rule: none if no evidence; high if every model item has calibrated P >= 0.90 (or there are only
    observed facts); medium if the weakest model item is >= 0.75; otherwise low.
    """
    counts: dict[str, int] = {}
    scores: list[float] = []

    def bump(k, n=1):
        counts[k] = counts.get(k, 0) + n
    for c in result.components:
        if not c.available:
            continue
        if c.key == "content_pressure":
            for ref in c.evidence:
                sc = model_scores.get((ref.stage_index, ref.finding_id))
                if sc is not None:
                    bump("model"); scores.append(sc)
                else:
                    bump("observed")
        elif c.key == "choice_asymmetry":
            for tok in (c.provenance or "").split("+"):
                if tok and tok != "unavailable":
                    bump(tok)
        else:
            bump("observed", max(1, len(c.evidence)) if c.value else 1)
    if not counts:
        level = "none"
    elif not scores:
        level = "high"
    else:
        lo = min(scores)
        level = "high" if lo >= 0.90 else "medium" if lo >= 0.75 else "low"
    return CDIConfidence(evidence_by_provenance=counts, model_evidence=len(scores),
                         mean_model_confidence=round(sum(scores) / len(scores), 4) if scores else None,
                         min_model_confidence=round(min(scores), 4) if scores else None, level=level)
