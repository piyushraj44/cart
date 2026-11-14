"""Aggregation functions shared by MCAI and CDI, plus the alternatives we compared them against.

Selected: renormalised weighted arithmetic mean over AVAILABLE components
    A(x) = sum_{j in A} w_j x_j / sum_{j in A} w_j
Properties: bounded in [min x, max x] subset of [0,1]; monotone non-decreasing in every x_j; idempotent
(all x_j = c gives c); linear, so each component's contribution w_j x_j / sum w is exact and additive
(explainable attribution); missing components are excluded, never zero-filled.

Alternatives (reported for transparency in each response, evaluated in ml/benchmark_intelligence.py):
    geometric  prod x_j^(w_j/W)          - non-compensatory; ANY zero component forces 0, which turns
                                            "no measurable asymmetry" into "no deception" (rejected).
    max        max_j x_j                 - ignores weights; one weak-evidence component dominates.
    power2     (sum w_j x_j^2 / W)^(1/2) - emphasises the strongest component, still bounded/monotone.
    noisy_or   1 - prod (1 - x_j)^(w_j/W) - probabilistic-OR reading; assumes independence the
                                            components do not have (they share evidence).
"""
from __future__ import annotations

import math

import numpy as np

from backend.intelligence.config import (SENSITIVITY_DIRICHLET_CONCENTRATION, SENSITIVITY_DIRICHLET_SAMPLES,
                                         SENSITIVITY_ONE_AT_A_TIME, SENSITIVITY_SEED)

EPS = 1e-12


def weighted_mean(values: dict[str, float], weights: dict[str, float]) -> float:
    W = sum(weights[k] for k in values)
    if W <= 0:
        raise ValueError("no positive weight among available components")
    return sum(weights[k] * values[k] for k in values) / W


def alternatives(values: dict[str, float], weights: dict[str, float]) -> dict[str, float]:
    """All aggregators on the same available inputs, scaled to 0-100 (rounded)."""
    if not values:
        return {}
    W = sum(weights[k] for k in values)
    x = {k: min(1.0, max(0.0, float(v))) for k, v in values.items()}
    geo = 0.0 if any(v <= 0 for v in x.values()) else math.exp(sum(weights[k] / W * math.log(x[k]) for k in x))
    out = {
        "weighted_mean (selected)": weighted_mean(x, weights),
        "geometric": geo,
        "max": max(x.values()),
        "power2": math.sqrt(sum(weights[k] * x[k] ** 2 for k in x) / W),
        "noisy_or": 1.0 - math.prod((1.0 - x[k]) ** (weights[k] / W) for k in x),
    }
    return {k: round(100.0 * v, 2) for k, v in out.items()}


def contributions(values: dict[str, float], weights: dict[str, float]) -> dict[str, float]:
    """Exact additive attribution of the weighted mean (points of 0-100)."""
    W = sum(weights[k] for k in values)
    return {k: round(100.0 * weights[k] * values[k] / W, 2) for k in values}


def weight_sensitivity(values: dict[str, float], weights: dict[str, float]) -> dict:
    """Deterministic sensitivity of the weighted mean to its weights.

    one_at_a_time: each available weight scaled by (1 -/+ 0.5), others fixed, then renormalised.
    dirichlet: 1000 weight vectors ~ Dirichlet(20 * w_normalised) with a fixed seed; 5th/95th percentile.
    """
    def score(w):
        return 100.0 * weighted_mean(values, w)
    oat = []
    for k in values:
        res = []
        for f in (1 - SENSITIVITY_ONE_AT_A_TIME, 1 + SENSITIVITY_ONE_AT_A_TIME):
            w = dict(weights); w[k] = weights[k] * f
            res.append(score(w))
        oat.append({"component": k, "low": round(min(res), 2), "high": round(max(res), 2)})
    keys = list(values)
    if len(keys) >= 2:
        rng = np.random.default_rng(SENSITIVITY_SEED)
        base = np.array([weights[k] for k in keys], dtype=float); base = base / base.sum()
        draws = rng.dirichlet(SENSITIVITY_DIRICHLET_CONCENTRATION * base, SENSITIVITY_DIRICHLET_SAMPLES)
        vals = 100.0 * draws @ np.array([values[k] for k in keys])
        p05, p95, n = float(np.percentile(vals, 5)), float(np.percentile(vals, 95)), SENSITIVITY_DIRICHLET_SAMPLES
    else:
        p05 = p95 = score(weights); n = 0
    return {"one_at_a_time": oat, "dirichlet_p05": round(p05, 2), "dirichlet_p95": round(p95, 2), "samples": n}
