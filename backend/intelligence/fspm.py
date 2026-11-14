"""FSPM - Fibonacci-weighted Sequential Pressure Model.

    F(1) = F(2) = 1,  F(n) = F(n-1) + F(n-2)
    FSPM = 100 * sum_i F(pos_i) * p_i / sum_i F(pos_i)

p_i is the normalised manipulation pressure of journey stage i (see pressure.py), pos_i its
1-based position. Positions are canonical (product=2, cart=3, ...) when every stage declares a
known `stage_type` in increasing order, otherwise the submission order. Only stages that were
actually submitted enter the sums: missing stages are never invented or zero-filled.

Fibonacci weighting is a PROPOSED weighting that emphasises later stages (where decisions become
binding). It is not an established psychological law; it is always reported next to equal and
linear weighting so its effect is visible.
"""
from __future__ import annotations

from backend.intelligence.config import CANONICAL_STAGES
from backend.intelligence.schemas import EvidenceRef, FSPMResult, FSPMStage


def fibonacci(n: int) -> list[int]:
    """First n Fibonacci numbers starting F(1)=F(2)=1."""
    if n < 0:
        raise ValueError("n must be >= 0")
    seq: list[int] = []
    a, b = 1, 1
    for _ in range(n):
        seq.append(a)
        a, b = b, a + b
    return seq


def raw_weight(scheme: str, position: int) -> float:
    if position < 1:
        raise ValueError("positions are 1-based")
    if scheme == "fibonacci":
        return float(fibonacci(position)[-1])
    if scheme == "linear":
        return float(position)
    if scheme == "equal":
        return 1.0
    raise ValueError(f"unknown scheme {scheme}")


def normalised_weights(scheme: str, positions: list[int]) -> list[float]:
    raw = [raw_weight(scheme, p) for p in positions]
    total = sum(raw)
    return [r / total for r in raw]


def weighted_score(pressures: list[float], positions: list[int], scheme: str) -> float:
    if len(pressures) != len(positions) or not pressures:
        raise ValueError("need one position per pressure")
    for p in pressures:
        if not 0.0 <= p <= 1.0:
            raise ValueError("pressures must be in [0, 1]")
    w = normalised_weights(scheme, positions)
    return 100.0 * sum(wi * pi for wi, pi in zip(w, pressures))


def resolve_positions(stage_types: list[str | None]) -> tuple[list[int], str]:
    """Canonical positions if all stages declare known, strictly increasing stage types."""
    idx = []
    for t in stage_types:
        if t is None or t not in CANONICAL_STAGES:
            return list(range(1, len(stage_types) + 1)), "submission_order"
        idx.append(CANONICAL_STAGES.index(t) + 1)
    if any(b <= a for a, b in zip(idx, idx[1:])):
        return list(range(1, len(stage_types) + 1)), "submission_order"
    return idx, "canonical"


def compute_fspm(names: list[str], stage_types: list[str | None], pressures: list[float],
                 sources: list[list[EvidenceRef]]) -> FSPMResult:
    n = len(pressures)
    if n == 0:
        return FSPMResult(available=False, reason="No journey stages were provided.")
    positions, mode = resolve_positions(stage_types)
    schemes = ("fibonacci", "linear", "equal")
    weights = {s: normalised_weights(s, positions) for s in schemes}
    stages = []
    for i in range(n):
        stages.append(FSPMStage(
            stage_index=i, name=names[i], position=positions[i], pressure=round(pressures[i], 4),
            pressure_sources=sources[i],
            weights={s: round(weights[s][i], 4) for s in schemes},
            contributions={s: round(100.0 * weights[s][i] * pressures[i], 2) for s in schemes},
        ))
    scores = {s: round(weighted_score(pressures, positions, s), 2) for s in schemes}
    top = max(range(n), key=lambda i: (stages[i].contributions["fibonacci"], i))
    return FSPMResult(
        available=True,
        reason=None if n >= 2 else "Only one stage: FSPM equals that stage's pressure; submit 2+ stages for a sequential view.",
        stages=stages, scores=scores, top_stage_index=top if pressures[top] > 0 else None,
        fibonacci_numbers=fibonacci(max(positions)), positions_mode=mode,
    )
