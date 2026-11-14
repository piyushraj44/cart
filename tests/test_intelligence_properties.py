"""Seeded randomized property tests for the ClearCart intelligence layer (intel-v1.1).

Hypothesis is not installed, so every property is checked on >= 200 cases drawn from a seeded numpy Generator
(np.random.default_rng). No model is loaded: wherever a classifier is needed a deterministic stub is used.
Failures print the offending input so they can be reproduced exactly.

Run from the project root:  .venv/bin/python -m pytest tests/test_intelligence_properties.py -q
"""
import ml  # noqa: F401  (must be first: sets OMP_NUM_THREADS before torch/xgboost - macOS segfault otherwise)

import hashlib
import itertools
import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backend.intelligence import aggregation as A
from backend.intelligence import fspm as F
from backend.intelligence import mcai as M
from backend.intelligence.cdi import ComponentInput, compute_cdi, confidence as cdi_confidence
from backend.intelligence.config import CDI_WEIGHTS, FSPM_SCHEMES, MCAI_WEIGHTS
from backend.intelligence.counterfactual import facts_preserved, plan_edits
from backend.intelligence.optimiser import TIE, optimise
from backend.intelligence.schemas import EvidenceRef
from backend.intelligence.temporal import StageObservation, analyse_journey
from ml.common import DATA_PROCESSED

N = 200                      # cases per property (minimum required)
SEED = 20271
CATS = ("urgency", "scarcity", "social_proof", "misdirection")


def rng_for(name: str) -> np.random.Generator:
    """One independent, reproducible stream per property."""
    return np.random.default_rng([SEED, int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)])


def rand_positions(rng, n):
    """n strictly increasing 1-based positions (canonical-style gaps allowed)."""
    return sorted(int(x) for x in rng.choice(np.arange(1, 13), n, replace=False))


def rand_pressures(rng, n):
    p = rng.uniform(0, 1, n)
    p[rng.random(n) < 0.2] = 0.0          # exact zeros are common (stage with nothing flagged)
    p[rng.random(n) < 0.05] = 1.0
    return [float(x) for x in p]


def h01(text: str, salt: str = "") -> float:
    return int(hashlib.sha256((salt + text).encode()).hexdigest()[:8], 16) / 2 ** 32


def stub_score_fn(texts):
    """Deterministic classifier stand-in: hash-based scores, same output keys as TextClassifier.predict."""
    out = []
    for t in texts:
        p = h01(t)
        mis = p * h01(t, "mis")
        out.append({"text": t, "p_dark": p, "probabilities": {"benign": 1 - p, "urgency": p - mis, "scarcity": 0.0,
                                                              "social_proof": 0.0, "misdirection": mis}})
    return out


def ml_finding(fid, category, score, evidence):
    return SimpleNamespace(id=fid, category=category, model_score=score, detection_method="ml_classifier",
                           evidence=evidence, short_reason="")


@pytest.fixture(scope="module")
def test_strings():
    df = pd.read_csv(DATA_PROCESSED / "text_dataset.csv", keep_default_na=False)
    t = df[df["split"] == "test"]
    return t[["label", "text"]].values.tolist()


# ============================================================================ FSPM
def test_fspm_bounded_0_100():
    rng = rng_for("fspm_bounded")
    for _ in range(N):
        n = int(rng.integers(1, 8))
        p, pos, s = rand_pressures(rng, n), rand_positions(rng, n), FSPM_SCHEMES[rng.integers(3)]
        v = F.weighted_score(p, pos, s)
        assert -1e-9 <= v <= 100 + 1e-9, (p, pos, s, v)


def test_fspm_idempotent_equals_max_when_all_equal():
    rng = rng_for("fspm_idem")
    for _ in range(N):
        n = int(rng.integers(1, 8))
        c = float(rng.choice([0.0, 1.0, rng.uniform()]))
        pos = rand_positions(rng, n)
        for s in FSPM_SCHEMES:
            v = F.weighted_score([c] * n, pos, s)
            assert v == pytest.approx(100 * c, abs=1e-9), (c, pos, s, v)
            assert v == pytest.approx(100 * max([c] * n), abs=1e-9)


def test_fspm_monotone_in_every_stage_pressure():
    rng = rng_for("fspm_mono")
    for _ in range(N):
        n = int(rng.integers(1, 8))
        p, pos = rand_pressures(rng, n), rand_positions(rng, n)
        i = int(rng.integers(n))
        q = list(p); q[i] = float(rng.uniform(p[i], 1.0))
        for s in FSPM_SCHEMES:
            assert F.weighted_score(q, pos, s) >= F.weighted_score(p, pos, s) - 1e-9, (p, q, pos, s)


def test_fspm_weights_sum_to_one_for_any_positions():
    rng = rng_for("fspm_wsum")
    for _ in range(N):
        n = int(rng.integers(1, 10))
        pos = [int(x) for x in rng.integers(1, 40, n)]          # any positions, repeats allowed
        for s in FSPM_SCHEMES:
            w = F.normalised_weights(s, pos)
            assert math.fsum(w) == pytest.approx(1.0, abs=1e-12), (pos, s)
            assert all(x > 0 for x in w)


def test_fspm_equal_weighting_is_order_invariant():
    rng = rng_for("fspm_equal_perm")
    for _ in range(N):
        n = int(rng.integers(2, 8))
        p, pos = rand_pressures(rng, n), rand_positions(rng, n)
        perm = [p[i] for i in rng.permutation(n)]
        assert F.weighted_score(perm, pos, "equal") == pytest.approx(F.weighted_score(p, pos, "equal"), abs=1e-9)


def test_fspm_fibonacci_vs_linear_for_positions_1_2():
    """Requested property 'Fibonacci = linear for positions {1,2}' is FALSE by the documented definitions:
    F(1) = F(2) = 1 gives weights (1/2, 1/2), linear gives (1/3, 2/3). What does hold is Fibonacci = EQUAL weighting
    for positions {1,2}; both facts are asserted here (see the research report)."""
    assert F.normalised_weights("fibonacci", [1, 2]) == [0.5, 0.5]
    assert F.normalised_weights("linear", [1, 2]) == pytest.approx([1 / 3, 2 / 3])
    rng = rng_for("fib_12")
    for _ in range(N):
        p = rand_pressures(rng, 2)
        fib = F.weighted_score(p, [1, 2], "fibonacci")
        assert fib == pytest.approx(F.weighted_score(p, [1, 2], "equal"), abs=1e-9), p
        # linear differs exactly when the two pressures differ: lin - fib = 100 * (p2 - p1) / 6
        lin = F.weighted_score(p, [1, 2], "linear")
        assert lin - fib == pytest.approx(100 * (p[1] - p[0]) / 6, abs=1e-9), p


# ============================================================================ aggregation
def rand_components(rng, keys=("a", "b", "c", "d")):
    k = int(rng.integers(1, len(keys) + 1))
    use = list(rng.choice(keys, k, replace=False))
    vals = {key: float(rng.choice([0.0, 1.0, rng.uniform()], p=[0.15, 0.05, 0.8])) for key in use}
    weights = {key: float(rng.uniform(0.01, 1.0)) for key in keys}
    return vals, weights


def test_weighted_mean_bounded_between_min_and_max():
    rng = rng_for("wm_bounds")
    for _ in range(N):
        v, w = rand_components(rng)
        m = A.weighted_mean(v, w)
        assert min(v.values()) - 1e-12 <= m <= max(v.values()) + 1e-12, (v, w, m)


def test_weighted_mean_monotone():
    rng = rng_for("wm_mono")
    for _ in range(N):
        v, w = rand_components(rng)
        k = list(v)[int(rng.integers(len(v)))]
        v2 = dict(v); v2[k] = float(rng.uniform(v[k], 1.0))
        assert A.weighted_mean(v2, w) >= A.weighted_mean(v, w) - 1e-12, (v, v2, w)


def test_weighted_mean_idempotent():
    rng = rng_for("wm_idem")
    for _ in range(N):
        v, w = rand_components(rng)
        c = float(rng.uniform())
        assert A.weighted_mean({k: c for k in v}, w) == pytest.approx(c, abs=1e-12)


def test_weighted_mean_invariant_to_weight_scaling():
    rng = rng_for("wm_scale")
    for _ in range(N):
        v, w = rand_components(rng)
        s = float(10 ** rng.uniform(-3, 3))
        assert A.weighted_mean(v, {k: s * x for k, x in w.items()}) == pytest.approx(A.weighted_mean(v, w), rel=1e-9, abs=1e-12)


def test_contributions_sum_to_score():
    rng = rng_for("contrib")
    for _ in range(N):
        v, w = rand_components(rng)
        c = A.contributions(v, w)
        # each contribution is rounded to 2 dp, so the sum may differ by at most 0.005 per component
        assert sum(c.values()) == pytest.approx(100 * A.weighted_mean(v, w), abs=0.005 * len(c) + 1e-9), (v, w, c)


def test_alternatives_bounded_and_geometric_collapse_iff_zero():
    rng = rng_for("alts")
    for _ in range(N):
        v, w = rand_components(rng)
        alts = A.alternatives(v, w)
        assert set(alts) == {"weighted_mean (selected)", "geometric", "max", "power2", "noisy_or"}
        for name, x in alts.items():
            assert 0.0 <= x <= 100.0, (name, v, w, x)
        has_zero = any(x <= 0 for x in v.values())
        assert (alts["geometric"] == 0.0) == has_zero, (v, w, alts)


# ============================================================================ CDI
def rand_cdi_inputs(rng):
    inputs, avail = [], {}
    for k in CDI_WEIGHTS:
        ok = bool(rng.random() < 0.7)
        val = float(rng.choice([0.0, rng.uniform()], p=[0.2, 0.8])) if ok else None
        inputs.append(ComponentInput(key=k, available=ok, value=val, provenance="model" if ok else "unavailable"))
        if ok:
            avail[k] = val
    return inputs, avail


def test_cdi_missing_component_is_renormalised_not_zero_filled():
    rng = rng_for("cdi_renorm")
    checked_missing = 0
    for _ in range(N):
        inputs, avail = rand_cdi_inputs(rng)
        res = compute_cdi(inputs, with_sensitivity=False)
        if not avail:
            assert res.cdi is None and res.coverage == 0.0
            continue
        wsum = sum(CDI_WEIGHTS[k] for k in avail)
        renorm = 100 * sum(CDI_WEIGHTS[k] * x for k, x in avail.items()) / wsum
        zero_filled = 100 * sum(CDI_WEIGHTS[k] * x for k, x in avail.items())
        assert res.cdi == pytest.approx(renorm, abs=0.005 + 1e-9), (avail, res.cdi, renorm)
        assert res.coverage == pytest.approx(wsum, abs=1e-4)
        if wsum < 1 - 1e-9 and abs(renorm - zero_filled) > 0.01:
            checked_missing += 1
            assert abs(res.cdi - zero_filled) > 0.005, (avail, res.cdi, zero_filled)
        for c in res.components:
            if c.key not in avail:
                assert not c.available and c.value is None and c.contribution is None
    assert checked_missing >= 50


def expected_level(content_scores, any_item):
    if not any_item:
        return "none"
    if not content_scores:
        return "high"
    lo = min(content_scores)
    return "high" if lo >= 0.90 else "medium" if lo >= 0.75 else "low"


def test_cdi_confidence_level_rules():
    rng = rng_for("cdi_conf")
    for case in range(N):
        n_ev = int(rng.integers(0, 5))
        refs = [EvidenceRef(stage_index=None, finding_id=f"f{i}", text=f"e{i}") for i in range(n_ev)]
        model_scores, content_scores = {}, []
        for ref in refs:
            if rng.random() < 0.75:                       # ML evidence with a calibrated score; else an observed fact
                s = float(rng.choice([rng.uniform(0.5, 1.0), 0.75, 0.90]))
                model_scores[(None, ref.finding_id)] = s
        c_ok, a_ok, o_ok = (bool(rng.random() < p) for p in (0.8, 0.4, 0.4))
        prov = "+".join(sorted(set(rng.choice(["measured", "inferred", "declared", "default"], int(rng.integers(1, 3))))))
        inputs = [ComponentInput(key="content_pressure", available=c_ok, value=float(rng.uniform()) if c_ok else None,
                                 provenance="model", evidence=refs),
                  ComponentInput(key="choice_asymmetry", available=a_ok, value=float(rng.uniform()) if a_ok else None,
                                 provenance=prov),
                  ComponentInput(key="observed_obstruction", available=o_ok, value=float(rng.choice([0.0, 0.5, 1.0])) if o_ok else None,
                                 provenance="observed")]
        res = compute_cdi(inputs, with_sensitivity=False)
        conf = cdi_confidence(res, model_scores)
        if c_ok:
            content_scores = [model_scores[(None, r.finding_id)] for r in refs if (None, r.finding_id) in model_scores]
        any_item = (c_ok and n_ev > 0) or a_ok or o_ok
        exp = expected_level(content_scores, any_item)
        assert conf.level == exp, (case, c_ok, a_ok, o_ok, content_scores, conf)
        assert conf.model_evidence == len(content_scores)
        if content_scores:
            assert conf.min_model_confidence == pytest.approx(min(content_scores), abs=1e-4)


# ============================================================================ MCAI
def test_rel_asym_bounded_and_zero_when_decline_at_least_accept():
    rng = rng_for("rel_asym")
    for _ in range(N):
        a = float(rng.choice([0.0, rng.uniform(0, 100)]))
        d = float(rng.choice([0.0, a, rng.uniform(0, 100), a + rng.uniform(0, 10)]))
        v = M.rel_asym(a, d)
        assert 0.0 <= v <= 1.0, (a, d, v)
        if d >= a:
            assert v == 0.0, (a, d, v)
    assert M.rel_asym(None, 1.0) is None and M.rel_asym(1.0, None) is None


def test_interaction_dimension_bounds_and_monotone_in_decline_steps():
    rng = rng_for("interaction")
    for _ in range(N):
        acc = int(rng.integers(1, 51))
        d1 = int(rng.integers(1, 51)); d2 = int(rng.integers(d1, 51))
        s1 = M.interaction_dimension(acc, d1).score
        s2 = M.interaction_dimension(acc, d2).score
        assert 0.0 <= s1 <= 1.0 and 0.0 <= s2 <= 1.0, (acc, d1, d2)
        assert s2 >= s1 - 1e-12, (acc, d1, d2, s1, s2)
        if d1 <= acc:
            assert s1 == 0.0
    assert not M.interaction_dimension(None, 3).available


# ============================================================================ optimiser
def chosen_candidate(edit):
    if edit.action == "no_safe_rewrite":
        return next(c for c in edit.candidates if c.variant == "original")
    if edit.action == "remove_unverifiable_claim":
        return next(c for c in edit.candidates if c.variant == "remove_unverifiable_claim")
    return next(c for c in edit.candidates if c.variant != "original" and c.text == edit.redesigned)


def test_optimiser_picks_feasible_minimal_candidate(test_strings):
    """Every re-selected edit uses a feasible candidate whose score is the minimum over feasible candidates.

    The optimiser's documented tie band (TIE = 0.005: prefer a real change, then the smaller edit) means exact
    minimality is only guaranteed when no other feasible candidate scores within TIE of the minimum; the hash stub
    gives near-continuous scores, so the strict check below is the property as requested, and the TIE-bounded check
    is the documented contract."""
    rng = rng_for("optimiser")
    dark = [t for lab, t in test_strings if lab != "benign"]
    n_edits = 0
    for case in range(N):
        k = int(rng.integers(1, 5))
        texts = [dark[i] for i in rng.choice(len(dark), k, replace=False)]
        findings = [ml_finding(f"f{i}", CATS[int(rng.integers(4))], float(rng.uniform(0.68, 1.0)), t) for i, t in enumerate(texts)]
        edits = optimise(plan_edits(findings, stub_score_fn), stub_score_fn)
        for e in edits:
            if e.action not in ("rewrite", "no_safe_rewrite", "remove_unverifiable_claim"):
                continue
            n_edits += 1
            feas = [c for c in e.candidates if c.feasible]
            assert feas, (case, e)
            best = min(c.score for c in feas)
            ch = chosen_candidate(e)
            assert ch.feasible and not ch.violated, (case, e.original, ch)
            assert ch.score <= best + TIE + 1e-12, (case, e.original, ch, best)
            assert ch.score == pytest.approx(best, abs=1e-12), (case, e.original, [(c.variant, c.score) for c in feas], ch)
    assert n_edits >= N


def test_optimiser_never_selects_infeasible_even_if_it_scores_lowest(test_strings):
    """Adversarial stub: neutral declines score 0 (and removal always scores 0 inside the optimiser), so infeasible
    declines/removals (dropped disclosure terms, prices or numbers) would win without the constraints."""
    from backend.intelligence.optimiser import NEUTRAL_DECLINES, violations
    rng = rng_for("optimiser_adv")
    pool = [t for _, t in test_strings]
    # texts that make declines/removals infeasible are rare in the data, so half the cases use authored variants
    authored = ["No thanks, I don't want free shipping on 2 items", "No, I'll pay the $5 fee later",
                "1,204 people bought this - $19.99 with free returns", "No thanks, cancel my subscription renewal",
                "Join 3,000 members who renew every month for $9"]
    seen_would_win = 0
    for case in range(N):
        texts = [pool[i] for i in rng.choice(len(pool), int(rng.integers(1, 4)), replace=False)]
        if rng.random() < 0.5:
            texts.append(authored[int(rng.integers(len(authored)))])
        findings = [ml_finding(f"f{i}", CATS[int(rng.integers(4))], 0.9, t) for i, t in enumerate(texts)]

        def adv(ts):
            out = []
            for t in ts:
                p = 0.0 if t in NEUTRAL_DECLINES else 0.2 + 0.8 * h01(t)
                out.append({"text": t, "p_dark": p, "probabilities": {"benign": 1 - p, "misdirection": p / 2}})
            return out
        for e in optimise(plan_edits(findings, adv), adv):
            if e.action not in ("rewrite", "no_safe_rewrite", "remove_unverifiable_claim"):
                continue
            ch = chosen_candidate(e)
            seen_would_win += sum(1 for c in e.candidates if not c.feasible and c.score < ch.score)
            assert ch.feasible, (case, e.original, ch)
            assert violations(e.category, e.original, ch.variant, ch.text) == []
    assert seen_would_win > 0          # the constraint actually blocked lower-scoring infeasible candidates


def test_facts_preserved_never_accepts_an_added_number(test_strings):
    rng = rng_for("facts")
    words = ["seven", "twelve", "three", "forty"]
    for case in range(N):
        orig = test_strings[int(rng.integers(len(test_strings)))][1]
        toks = orig.split()
        n = int(rng.integers(0, 1000))
        form = int(rng.integers(5))
        new = [str(n), f"${n}", f"{n}%", f"{n} hours", words[int(rng.integers(len(words)))]][form]
        pos = int(rng.integers(len(toks) + 1))
        red = " ".join(toks[:pos] + [new] + toks[pos:])
        assert not facts_preserved(orig, red), (case, orig, red)
        assert facts_preserved(orig, orig)


# ============================================================================ temporal
BASE = datetime(2026, 11, 12, 10, 0, 0, tzinfo=timezone.utc)


def hms(sec):
    sec = int(sec)
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def stage(i, texts, ts=None):
    return StageObservation(index=i, name=f"stage {i + 1}", captured_at=ts, findings=[], texts=texts)


def resets(stages):
    res, pf, _ = analyse_journey(stages)
    ev = [e for e in res.events if e.category == "countdown_reset"]
    return ev, pf


def test_no_countdown_reset_from_a_single_stage():
    rng = rng_for("temporal_single")
    for case in range(N):
        k = int(rng.integers(1, 4))
        readings = [f"OFFER ENDS {hms(rng.integers(60, 86000))}" for _ in range(k)]   # several readings, any order
        res, pf, of = analyse_journey([stage(0, readings)])
        assert not res.available and not res.events and not pf, (case, readings)
        # in a longer journey where only ONE stage shows countdowns there is nothing to compare either
        n = int(rng.integers(2, 5))
        j = int(rng.integers(n))
        stages = [stage(i, readings if i == j else ["Free delivery on all orders"]) for i in range(n)]
        ev, pf = resets(stages)
        assert not ev and not any(pf.values()), (case, readings, j)


def test_no_countdown_reset_when_later_reading_is_lower_and_consistent():
    rng = rng_for("temporal_consistent")
    for case in range(N):
        a = int(rng.integers(3600, 80000))
        elapsed = int(rng.integers(30, 3000))
        timed = bool(rng.random() < 0.6)
        if timed:
            b = a - elapsed + int(rng.integers(-60, 61))         # within the 60 s tolerance of real-time running
            b = min(b, a - 1)
            ts = [BASE.isoformat(), (BASE + timedelta(seconds=elapsed)).isoformat()]
        else:
            b = a - int(rng.integers(1, a))                       # any lower later reading, no timestamps
            ts = [None, None]
        stages = [stage(0, [f"OFFER ENDS {hms(a)}"], ts[0]), stage(1, [f"OFFER ENDS {hms(b)}"], ts[1])]
        ev, pf = resets(stages)
        assert not ev and not any(pf.values()), (case, a, b, elapsed, timed)


# ============================================================================ determinism
def rand_choice_html(rng):
    acc_w = rng.choice(["400", "700"]); size = rng.choice(["13px", "16px", "20px"])
    dec_col = rng.choice(["#111111", "#b0b0b0"]); fill = rng.choice(["#0f766e", "#ffffff"])
    dec = rng.choice(["No thanks", "No, I'd rather pay full price", "Skip this offer"])
    tag = rng.choice(["button", "a"])
    pre = '<label><input type="checkbox" checked> Add warranty for $9.99</label>' if rng.random() < 0.5 else ""
    dec_el = f'<a href="/cart" class="dec">{dec}</a>' if tag == "a" else f'<button class="dec">{dec}</button>'
    if rng.random() < 0.3:
        dec_el = f"<details><summary>More</summary>{dec_el}</details>"
    return (f"<html><head><style>.acc{{font-weight:{acc_w};font-size:{size};background-color:{fill}}}"
            f".dec{{color:{dec_col}}}</style></head><body><main><div class='offer'>{pre}"
            f"<button class='acc'>Yes, add protection</button>{dec_el}</div></main></body></html>")


def test_deterministic_outputs_fspm_cdi_mcai():
    rng = rng_for("determinism")
    for case in range(N):
        n = int(rng.integers(1, 6))
        p = rand_pressures(rng, n)
        types = [None] * n if rng.random() < 0.5 else [t for t in ("landing", "product", "cart", "checkout", "payment", "confirmation", "exit")][:n]
        names = [f"s{i}" for i in range(n)]
        src = [[EvidenceRef(stage_index=i, finding_id="f1", text="x")] for i in range(n)]
        a1 = F.compute_fspm(names, types, p, src).model_dump_json()
        a2 = F.compute_fspm(list(names), list(types), list(p), src).model_dump_json()
        assert a1 == a2, case
        inputs, _ = rand_cdi_inputs(rng)
        assert compute_cdi(inputs).model_dump_json() == compute_cdi(inputs).model_dump_json(), case
        html = rand_choice_html(rng)
        m1 = M.compute_mcai("html", stub_score_fn, html=html).model_dump_json()
        m2 = M.compute_mcai("html", stub_score_fn, html=html).model_dump_json()
        assert m1 == m2, (case, html)
