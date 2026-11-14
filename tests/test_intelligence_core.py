"""Fast unit tests for the ClearCart intelligence layer (FSPM, pressure, CDI, MCAI).

No model is loaded: wherever a classifier is needed a deterministic stub `score_fn` is used.
Hand-computed expectations show their arithmetic in comments.

Run from the project root:  .venv/bin/python -m pytest tests/test_intelligence_core.py -q
"""
import ml  # noqa: F401  (must be first: sets OMP_NUM_THREADS before torch/xgboost - macOS segfault otherwise)

from types import SimpleNamespace

import pytest
from bs4 import BeautifulSoup

from backend.extractors import segment_text
from backend.intelligence import cdi as cdi_mod
from backend.intelligence import fspm as F
from backend.intelligence import mcai as M
from backend.intelligence import pressure as P
from backend.intelligence.cdi import ComponentInput, band, compute_cdi
from backend.intelligence.config import CDI_WEIGHTS, FSPM_SCHEMES
from backend.intelligence.schemas import DimensionScore, EvidenceRef


# ============================================================================ helpers
def stub_score_fn(texts):
    """Deterministic stand-in for TextClassifier.predict (same output keys the MCAI code reads).

    P(misdirection)=0.8 for guilt-trip wording ('rather' / "don't like" / 'full price'), else 0.05.
    p_dark = 0.7 if the text contains '!' else 0.1.
    """
    out = []
    for t in texts:
        low = t.lower()
        mis = 0.8 if ("rather" in low or "don't like" in low or "full price" in low) else 0.05
        p_dark = max(mis, 0.7 if "!" in t else 0.1)
        rest = max(0.0, p_dark - mis)
        probs = {"benign": 1 - p_dark, "urgency": rest, "scarcity": 0.0, "social_proof": 0.0, "misdirection": mis}
        out.append({"text": t, "probabilities": probs, "p_dark": p_dark})
    return out


def ml_finding(fid, category, score, evidence="x"):
    return SimpleNamespace(id=fid, category=category, model_score=score, detection_method="ml_classifier",
                           evidence=evidence, short_reason="")


def dom_finding(fid, short_reason, evidence="Add insurance for $5"):
    return SimpleNamespace(id=fid, category="preselected_option", model_score=None, detection_method="dom_observation",
                           evidence=evidence, short_reason=short_reason)


def el(html, selector):
    return BeautifulSoup(html, "lxml").select_one(selector)


# ============================================================================ FSPM
class TestFibonacci:
    def test_values(self):
        assert F.fibonacci(1) == [1]
        assert F.fibonacci(2) == [1, 1]
        assert F.fibonacci(10) == [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]

    def test_zero_is_empty(self):
        assert F.fibonacci(0) == []

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            F.fibonacci(-1)


class TestWeights:
    def test_raw_weight(self):
        assert F.raw_weight("fibonacci", 1) == 1.0
        assert F.raw_weight("fibonacci", 6) == 8.0       # 1,1,2,3,5,8
        assert F.raw_weight("linear", 6) == 6.0
        assert F.raw_weight("equal", 6) == 1.0

    def test_raw_weight_invalid(self):
        with pytest.raises(ValueError):
            F.raw_weight("fibonacci", 0)                 # positions are 1-based
        with pytest.raises(ValueError):
            F.raw_weight("quadratic", 2)

    @pytest.mark.parametrize("scheme", ["fibonacci", "linear", "equal"])
    @pytest.mark.parametrize("positions", [[1], [3], [2, 4], [1, 2, 3, 4, 5, 6, 7], [1, 3, 7]])
    def test_normalised_weights_sum_to_one(self, scheme, positions):
        w = F.normalised_weights(scheme, positions)
        assert len(w) == len(positions)
        assert sum(w) == pytest.approx(1.0, abs=1e-12)
        assert all(x > 0 for x in w)

    def test_fibonacci_weights_canonical_gap(self):
        # positions [2,4] -> F(2)=1, F(4)=3, total 4 -> [0.25, 0.75]
        assert F.normalised_weights("fibonacci", [2, 4]) == pytest.approx([0.25, 0.75])
        # linear: 2/6, 4/6
        assert F.normalised_weights("linear", [2, 4]) == pytest.approx([1 / 3, 2 / 3])


class TestWeightedScore:
    P4 = [1.0, 0.0, 0.0, 0.5]
    POS = [1, 2, 3, 4]

    def test_fibonacci(self):
        # F = 1,1,2,3 (sum 7): 100 * (1*1.0 + 1*0 + 2*0 + 3*0.5) / 7 = 100 * 2.5/7 = 35.714285...
        assert F.weighted_score(self.P4, self.POS, "fibonacci") == pytest.approx(250 / 7)

    def test_linear(self):
        # w = 1,2,3,4 (sum 10): 100 * (1*1.0 + 4*0.5) / 10 = 30.0
        assert F.weighted_score(self.P4, self.POS, "linear") == pytest.approx(30.0)

    def test_equal(self):
        # 100 * (1.0 + 0 + 0 + 0.5) / 4 = 37.5
        assert F.weighted_score(self.P4, self.POS, "equal") == pytest.approx(37.5)

    def test_constant_pressure_is_scheme_invariant(self):
        for s in FSPM_SCHEMES:
            assert F.weighted_score([0.4] * 5, [1, 2, 3, 4, 5], s) == pytest.approx(40.0)

    @pytest.mark.parametrize("bad", [[1.01], [-0.01], [0.5, 2.0]])
    def test_invalid_pressure_raises(self, bad):
        with pytest.raises(ValueError):
            F.weighted_score(bad, list(range(1, len(bad) + 1)), "fibonacci")

    def test_length_mismatch_and_empty_raise(self):
        with pytest.raises(ValueError):
            F.weighted_score([0.1, 0.2], [1], "equal")
        with pytest.raises(ValueError):
            F.weighted_score([], [], "equal")


class TestResolvePositions:
    def test_canonical(self):
        assert F.resolve_positions(["product", "checkout"]) == ([2, 4], "canonical")
        assert F.resolve_positions(["landing", "cart", "payment", "exit"]) == ([1, 3, 5, 7], "canonical")

    def test_unknown_type_falls_back(self):
        assert F.resolve_positions(["product", "banana"]) == ([1, 2], "submission_order")
        assert F.resolve_positions([None, "cart"]) == ([1, 2], "submission_order")

    def test_non_increasing_falls_back(self):
        assert F.resolve_positions(["checkout", "cart"]) == ([1, 2], "submission_order")
        assert F.resolve_positions(["cart", "cart"]) == ([1, 2], "submission_order")   # strictly increasing required


class TestComputeFSPM:
    def test_missing_stages_not_invented(self):
        r = F.compute_fspm(["P", "C"], ["product", "checkout"], [0.4, 0.8], [[], []])
        assert r.available and r.positions_mode == "canonical"
        assert [s.position for s in r.stages] == [2, 4]
        assert len(r.stages) == 2                                   # no zero-filled cart stage
        # weights normalised over the PRESENT stages only
        assert r.stages[0].weights["fibonacci"] == pytest.approx(0.25)
        assert r.stages[1].weights["fibonacci"] == pytest.approx(0.75)
        for s in ("fibonacci", "linear", "equal"):
            assert sum(st.weights[s] for st in r.stages) == pytest.approx(1.0, abs=1e-3)
        # fibonacci: 100*(0.25*0.4 + 0.75*0.8) = 70.0 ; linear: 100*(0.4/3 + 1.6/3) = 66.67 ; equal: 60.0
        assert r.scores == pytest.approx({"fibonacci": 70.0, "linear": 66.67, "equal": 60.0})
        assert r.fibonacci_numbers == [1, 1, 2, 3]
        # contributions: fib 100*0.25*0.4 = 10.0 and 100*0.75*0.8 = 60.0 -> they sum to the score
        assert [st.contributions["fibonacci"] for st in r.stages] == pytest.approx([10.0, 60.0])
        assert r.top_stage_index == 1

    def test_top_stage_uses_fibonacci_contribution(self):
        # submission order, positions 1,2,3 -> F = 1,1,2 -> w = .25,.25,.5
        # contributions: 100*.25*.9 = 22.5 ; 0 ; 100*.5*.5 = 25.0  -> top is stage 2 though stage 0 has the larger pressure
        r = F.compute_fspm(["a", "b", "c"], [None, None, None], [0.9, 0.0, 0.5], [[], [], []])
        assert r.top_stage_index == 2

    def test_top_stage_tie_goes_to_later_stage(self):
        r = F.compute_fspm(["a", "b"], [None, None], [0.5, 0.5], [[], []])     # F(1)=F(2)=1: equal contributions
        assert r.top_stage_index == 1

    def test_single_stage(self):
        r = F.compute_fspm(["only"], [None], [0.3], [[]])
        assert r.available
        assert r.reason and "one stage" in r.reason.lower()
        assert r.scores == pytest.approx({"fibonacci": 30.0, "linear": 30.0, "equal": 30.0})

    def test_zero_pressure_has_no_top_stage(self):
        r = F.compute_fspm(["a", "b", "c"], ["product", "cart", "checkout"], [0.0, 0.0, 0.0], [[], [], []])
        assert r.top_stage_index is None
        assert r.scores["fibonacci"] == 0.0

    def test_no_stages_unavailable(self):
        r = F.compute_fspm([], [], [], [])
        assert not r.available and r.reason


# ============================================================================ pressure
class TestPressure:
    def test_noisy_or_empty(self):
        assert P.noisy_or([]) == 0.0

    def test_noisy_or_values(self):
        assert P.noisy_or([0.5, 0.5]) == pytest.approx(0.75)        # 1 - 0.5*0.5
        assert P.noisy_or([0.9, 0.5]) == pytest.approx(0.95)        # 1 - 0.1*0.5

    def test_noisy_or_monotone_and_bounded(self):
        import random
        rnd = random.Random(0)
        for _ in range(200):
            vals = [rnd.random() for _ in range(rnd.randint(0, 6))]
            base = P.noisy_or(vals)
            assert 0.0 <= base <= 1.0
            assert P.noisy_or(vals + [rnd.random()]) >= base - 1e-12
            assert base >= max(vals, default=0.0) - 1e-12           # never below the strongest signal
        assert P.noisy_or([2.0]) == 1.0                             # clipped
        assert P.noisy_or([-1.0]) == 0.0

    # ---- intensity-breadth blend: p = lam*max_k c_k + (1-lam)*sum_k c_k/4, lam = 0.5
    def test_lambda_config(self):
        from backend.intelligence.config import PRESSURE_INTENSITY_WEIGHT
        assert PRESSURE_INTENSITY_WEIGHT == 0.5

    def test_blend_empty_and_full(self):
        assert P.blend({}) == 0.0
        assert P.blend({k: 1.0 for k in P.ML_CATEGORIES}) == pytest.approx(1.0)

    def test_blend_single_channel(self):
        # 0.5*0.9 + 0.5*(0.9/4) = 0.45 + 0.1125 = 0.5625
        assert P.blend({"urgency": 0.9}) == pytest.approx(0.5625)

    def test_blend_lambda_extremes(self):
        ch = {"urgency": 0.9, "scarcity": 0.5}
        assert P.blend(ch, lam=1.0) == pytest.approx(0.9)            # pure intensity = max
        assert P.blend(ch, lam=0.0) == pytest.approx(1.4 / 4)        # pure breadth = mean over the 4 categories

    def test_blend_breadth_not_saturated(self):
        # the case that made noisy-OR useless: near-binary calibrated scores
        one = P.blend({"urgency": 0.99})                                   # 0.495 + 0.5*0.99/4 = 0.61875
        two = P.blend({"urgency": 0.99, "scarcity": 0.99})                 # 0.495 + 0.5*1.98/4 = 0.7425
        four = P.blend({k: 0.99 for k in P.ML_CATEGORIES})                 # 0.495 + 0.495 = 0.99
        assert (one, two, four) == pytest.approx((0.61875, 0.7425, 0.99))
        assert one < two < four
        assert P.noisy_or([0.99]) == pytest.approx(0.99) and P.noisy_or([0.99, 0.99]) == pytest.approx(0.9999)

    def test_blend_bounded_monotone_clipped(self):
        import random
        rnd = random.Random(1)
        for _ in range(300):
            ch = {k: rnd.random() for k in P.ML_CATEGORIES if rnd.random() < 0.6}
            v = P.blend(ch)
            assert 0.0 <= v <= 1.0
            assert v >= 0.5 * max(ch.values(), default=0.0) - 1e-12
            k = rnd.choice(P.ML_CATEGORIES)
            up = dict(ch); up[k] = min(1.0, ch.get(k, 0.0) + rnd.random() * 0.5)
            assert P.blend(up) >= v - 1e-12                             # monotone in each channel
        assert P.blend({"urgency": 2.0}) == pytest.approx(P.blend({"urgency": 1.0}))
        assert P.blend({"urgency": -1.0}) == 0.0
        assert P.blend({"unknown_category": 1.0}) == 0.0               # only the four ML categories count

    def test_content_pressure_dedups_within_category(self):
        fs = [ml_finding("f1", "urgency", 0.9, "Hurry"), ml_finding("f2", "urgency", 0.8, "Ends soon")]
        v, src = P.content_pressure(fs)
        # only the 0.9 counts: 0.5*0.9 + 0.5*0.9/4 = 0.5625
        # (if 0.8 were a separate channel it would be 0.45 + 0.5*1.7/4 = 0.6625)
        assert v == pytest.approx(0.5625)
        assert len(src) == 1 and src[0].finding_id == "f1"
        many = [ml_finding(f"f{i}", "urgency", 0.9) for i in range(10)]
        assert P.content_pressure(many)[0] == pytest.approx(0.5625)       # ten copies do not inflate

    def test_content_pressure_across_categories(self):
        fs = [ml_finding("f1", "urgency", 0.9), ml_finding("f2", "urgency", 0.8), ml_finding("f3", "scarcity", 0.5)]
        v, src = P.content_pressure(fs, stage_index=2)
        # channels urgency 0.9, scarcity 0.5: 0.5*0.9 + 0.5*(0.9+0.5)/4 = 0.45 + 0.175 = 0.625
        assert v == pytest.approx(0.625)
        assert [s.finding_id for s in src] == ["f1", "f3"]          # sorted by strength
        assert all(s.stage_index == 2 for s in src)

    def test_content_pressure_ignores_non_model_findings(self):
        fs = [dom_finding("f1", "pre-ticked (mentions a cost or consent)"),
              SimpleNamespace(id="f2", category="urgency", model_score=None, detection_method="ml_classifier", evidence="x")]
        v, src = P.content_pressure(fs)
        assert v == 0.0 and src == []

    def test_content_pressure_extra_facts_raise_urgency_channel(self):
        ref = EvidenceRef(text="countdown reset")
        v, src = P.content_pressure([ml_finding("f1", "scarcity", 0.5)], extra_facts=[(1.0, ref)])
        # channels urgency 1.0 (fact), scarcity 0.5: 0.5*1.0 + 0.5*1.5/4 = 0.6875
        assert v == pytest.approx(0.6875)
        assert src[-1].text == "countdown reset"
        # a weaker fact never lowers an existing urgency score: max(0.9, 0.5) = 0.9 -> 0.5625
        v2, _ = P.content_pressure([ml_finding("f1", "urgency", 0.9)], extra_facts=[(0.5, ref)])
        assert v2 == pytest.approx(0.5625)
        # fact alone: 0.5*1.0 + 0.5*1.0/4 = 0.625
        assert P.content_pressure([], extra_facts=[(1.0, ref)])[0] == pytest.approx(0.625)

    def test_obstruction_facts_strengths(self):
        fs = [dom_finding("f1", "Pre-selected option (mentions a cost or consent)"),
              dom_finding("f2", "Pre-selected option"),
              ml_finding("f3", "urgency", 0.99)]
        facts = P.obstruction_facts(fs, stage_index=0)
        assert [s for s, _ in facts] == [1.0, 0.5]
        assert [r.finding_id for _, r in facts] == ["f1", "f2"]


# ============================================================================ CDI
def comp(key, value=None, available=True):
    return ComponentInput(key=key, available=available, value=value, provenance="model" if available else "unavailable",
                          evidence=[EvidenceRef(text=key)] if available else [])


class TestCDI:
    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError):
            compute_cdi([comp("content_pressure", 0.5)],
                        weights={"content_pressure": 0.5, "choice_asymmetry": 0.25, "observed_obstruction": 0.2})

    def test_unknown_component_raises(self):
        with pytest.raises(ValueError):
            compute_cdi([comp("content_pressure", 0.5), comp("vibes", 0.9)])

    def test_unavailable_excluded_not_zero(self):
        r = compute_cdi([comp("content_pressure", 0.6), comp("choice_asymmetry", available=False),
                         comp("observed_obstruction", 0.0)])
        # renormalised: 100 * (0.55*0.6 + 0.20*0.0) / (0.55 + 0.20) = 100 * 0.33/0.75 = 44.0
        manual = 100 * (0.55 * 0.6 + 0.20 * 0.0) / (0.55 + 0.20)
        zero_filled = 100 * (0.55 * 0.6 + 0.25 * 0.0 + 0.20 * 0.0)          # 33.0 - what we must NOT get
        assert r.cdi == pytest.approx(manual, abs=0.01) == 44.0
        assert r.cdi != pytest.approx(zero_filled, abs=0.5)
        assert r.coverage == pytest.approx(0.75)
        asym = next(c for c in r.components if c.key == "choice_asymmetry")
        assert not asym.available and asym.value is None and asym.contribution is None and asym.effective_weight is None
        assert r.evidence_items == 2

    def test_all_available(self):
        r = compute_cdi([comp("content_pressure", 0.6), comp("choice_asymmetry", 0.8), comp("observed_obstruction", 1.0)])
        # 100 * (0.55*0.6 + 0.25*0.8 + 0.20*1.0) = 100 * (0.33 + 0.20 + 0.20) = 73.0
        assert r.cdi == pytest.approx(73.0)
        assert r.coverage == pytest.approx(1.0)
        assert r.band == "elevated"

    def test_nothing_available(self):
        r = compute_cdi([comp("content_pressure", available=False)])
        assert r.cdi is None and r.band is None and r.coverage == 0.0 and r.sensitivity is None

    def test_values_clipped(self):
        r = compute_cdi([comp("content_pressure", 1.7)])
        assert r.cdi == pytest.approx(100.0)

    @pytest.mark.parametrize("score,expected", [(0, "low"), (24.99, "low"), (25, "moderate"), (49.99, "moderate"),
                                                (50, "elevated"), (74.99, "elevated"), (75, "high"), (100, "high")])
    def test_band_boundaries(self, score, expected):
        assert band(score) == expected

    def test_band_none(self):
        assert band(None) is None

    @pytest.mark.parametrize("vals", [(0.6, 0.8, 0.0), (0.1, 0.9, 1.0), (0.95, 0.2, 0.5), (0.3, None, 1.0)])
    def test_contributions_sum_to_cdi(self, vals):
        keys = ("content_pressure", "choice_asymmetry", "observed_obstruction")
        r = compute_cdi([comp(k, v, available=v is not None) for k, v in zip(keys, vals)])
        total = sum(c.contribution for c in r.components if c.available)
        assert total == pytest.approx(r.cdi, abs=0.02)                         # per-term rounding to 0.01
        assert sum(c.effective_weight for c in r.components if c.available) == pytest.approx(1.0, abs=1e-3)

    def test_sensitivity_one_at_a_time_hand_computed(self):
        vals = {"content_pressure": 0.6, "choice_asymmetry": 0.8, "observed_obstruction": 0.0}
        r = compute_cdi([comp(k, v) for k, v in vals.items()])
        w = CDI_WEIGHTS
        for row in r.sensitivity.one_at_a_time:
            k = row["component"]
            res = []
            for f in (0.5, 1.5):                                               # SENSITIVITY_ONE_AT_A_TIME = 0.5
                ww = dict(w); ww[k] = w[k] * f
                res.append(100 * sum(ww[j] * vals[j] for j in vals) / sum(ww.values()))
            assert row["low"] == pytest.approx(min(res), abs=0.01)
            assert row["high"] == pytest.approx(max(res), abs=0.01)
        # e.g. content_pressure x0.5: 100*(0.275*0.6 + 0.25*0.8)/0.725 = 50.34 ; x1.5: 100*(0.825*0.6+0.2)/1.275 = 54.51
        cp = next(x for x in r.sensitivity.one_at_a_time if x["component"] == "content_pressure")
        assert (cp["low"], cp["high"]) == pytest.approx((50.34, 54.51), abs=0.01)

    def test_sensitivity_deterministic_and_brackets_cdi(self):
        for vals in [(0.6, 0.8, 0.0), (0.1, 0.9, 1.0), (0.95, 0.2, 0.5), (0.5, 0.5, 0.5)]:
            inputs = [comp(k, v) for k, v in zip(("content_pressure", "choice_asymmetry", "observed_obstruction"), vals)]
            a, b = compute_cdi(inputs), compute_cdi(inputs)
            assert a.sensitivity.model_dump() == b.sensitivity.model_dump()
            assert a.sensitivity.samples == 1000
            assert a.sensitivity.dirichlet_p05 <= a.cdi + 1e-9 <= a.sensitivity.dirichlet_p95 + 1e-9

    def test_equal_components_have_no_weight_sensitivity(self):
        r = compute_cdi([comp(k, 0.5) for k in CDI_WEIGHTS])
        assert r.cdi == pytest.approx(50.0)
        assert r.sensitivity.dirichlet_p05 == pytest.approx(50.0) == r.sensitivity.dirichlet_p95

    def test_single_component_sensitivity(self):
        r = compute_cdi([comp("content_pressure", 0.42)])
        assert r.cdi == pytest.approx(42.0)
        s = r.sensitivity
        assert s.samples == 0
        assert s.dirichlet_p05 == s.dirichlet_p95 == pytest.approx(42.0)
        assert s.one_at_a_time == [{"component": "content_pressure", "low": 42.0, "high": 42.0}]  # renormalisation cancels

    def test_cdi_score_helper_matches(self):
        vals = {"content_pressure": 0.2, "observed_obstruction": 0.9}
        assert cdi_mod._score(vals, CDI_WEIGHTS) == pytest.approx(100 * (0.55 * 0.2 + 0.2 * 0.9) / 0.75)


# ============================================================================ MCAI: parsing
class TestColourAndUnits:
    def test_parse_color(self):
        assert M.parse_color("#fff") == (255.0, 255.0, 255.0)
        assert M.parse_color("#0f766e") == (15.0, 118.0, 110.0)
        assert M.parse_color("White") == (255.0, 255.0, 255.0)
        assert M.parse_color("rgb(1, 2, 3)") == (1.0, 2.0, 3.0)
        # alpha composited over white: 0.5*0 + 0.5*255 = 127.5
        assert M.parse_color("rgba(0,0,0,0.5)") == pytest.approx((127.5, 127.5, 127.5))
        for v in ("transparent", "rgba(0,0,0,0)", "", None, "inherit", "hsl(0,0%,0%)"):
            assert M.parse_color(v) is None

    def test_contrast_ratio(self):
        white, black = (255.0, 255.0, 255.0), (0.0, 0.0, 0.0)
        assert M.contrast_ratio(white, black) == pytest.approx(21.0)     # (1.0+0.05)/(0+0.05)
        assert M.contrast_ratio(black, white) == pytest.approx(21.0)     # symmetric
        assert M.contrast_ratio((10.0, 120.0, 200.0), (10.0, 120.0, 200.0)) == pytest.approx(1.0)
        assert M.contrast_ratio((156.0, 163.0, 175.0), white) < 4.5      # #9ca3af on white fails WCAG AA

    @pytest.mark.parametrize("v,px", [("12px", 12), ("12", 12), ("12pt", 16), ("1.5em", 24), ("2rem", 32), ("50%", 8),
                                      ("large", 18), ("x-small", 10)])
    def test_parse_px(self, v, px):
        assert M.parse_px(v) == pytest.approx(px)

    @pytest.mark.parametrize("v", ["", "calc(1px + 2px)", "big", None])
    def test_parse_px_invalid(self, v):
        assert M.parse_px(v) is None

    def test_parse_weight(self):
        assert M.parse_weight("bold") == 700.0
        assert M.parse_weight("600") == 600.0
        assert M.parse_weight("normal") == 400.0
        assert M.parse_weight("heavy") is None


class TestCssRules:
    def test_specificity_id_beats_class_beats_tag(self):
        css = "#x{font-size:10px} .big{font-size:30px} button{font-size:20px; color:red}"
        e = el('<button id="x" class="big">Go</button>', "button")
        d = M.CssRules(css).declared(e)
        assert d["font-size"] == "10px"          # id (1,0,0) > class (0,1,0) > tag (0,0,1), regardless of order
        assert d["color"] == "red"               # only the tag rule sets colour

    def test_class_beats_tag_even_if_tag_later(self):
        e = el('<button class="big">Go</button>', "button")
        assert M.CssRules(".big{font-size:30px} button{font-size:20px}").declared(e)["font-size"] == "30px"

    def test_compound_tag_class_beats_class(self):
        e = el('<button class="big">Go</button>', "button")
        assert M.CssRules("button.big{font-size:22px} .big{font-size:30px}").declared(e)["font-size"] == "22px"

    def test_later_wins_at_equal_specificity(self):
        e = el('<a class="a b" href="#">x</a>', "a")
        assert M.CssRules(".a{color:red} .b{color:blue}").declared(e)["color"] == "blue"
        assert M.CssRules(".b{color:blue} .a{color:red}").declared(e)["color"] == "red"

    def test_inline_wins(self):
        e = el('<a id="i" class="a" href="#" style="color: green !important">x</a>', "a")
        assert M.CssRules("#i{color:red} .a{color:blue}").declared(e)["color"] == "green"

    def test_combinator_pseudo_attribute_selectors_skipped(self):
        r = M.CssRules("div .x{color:red} div>.x{color:red} .x:hover{color:red} a[href]{color:red} .x{font-size:9px}")
        assert r.skipped == 4
        assert len(r.rules) == 1
        e = el('<div><a class="x" href="#">x</a></div>', "a")
        assert r.declared(e) == {"font-size": "9px"}

    def test_comments_ignored_and_grouped_selectors(self):
        r = M.CssRules("/* .x{color:red} */ .y, .z{color:blue}")
        assert len(r.rules) == 2 and r.skipped == 0
        assert r.declared(el('<a class="z" href="#">x</a>', "a"))["color"] == "blue"


class TestVisualFeatures:
    def test_defaults_have_default_provenance(self):
        soup = BeautifulSoup('<div><button>Yes</button><a href="#">No thanks</a></div>', "lxml")
        rules = M.CssRules("")
        b = M.html_visual_features(soup.button, rules)
        a = M.html_visual_features(soup.a, rules)
        assert b["font_size_px"] == (13.33, "default", None)
        assert a["font_size_px"] == (16.0, "default", None)
        assert b["font_weight"][1] == "default" and a["font_weight"][1] == "default"
        assert b["background_fill"][:2] == (1.0, "default")       # browser-default button face
        assert a["background_fill"][:2] == (0.0, "default")
        assert b["opacity"] == (1.0, "default", None)
        assert b["control_kind"][0] == 1.0 and a["control_kind"][0] == 0.35
        assert b["contrast_ratio"][1] == "default"

    def test_measured_and_inherited(self):
        html = ('<div class="wrap" style="font-size:20px; opacity:0.5"><a class="d" href="#">No thanks</a></div>')
        soup = BeautifulSoup(html, "lxml")
        rules = M.CssRules(".d{color:#999999; font-weight:300; opacity:0.5} .wrap{background-color:#ffffff}")
        f = M.html_visual_features(soup.a, rules)
        assert f["font_size_px"] == (20.0, "measured", None)     # inherited from parent inline style
        assert f["font_weight"] == (300.0, "measured", None)
        assert f["opacity"] == (0.25, "measured", None)           # 0.5 own * 0.5 ancestor
        assert f["contrast_ratio"][1] == "measured"               # colour + ancestor background both declared
        assert f["contrast_ratio"][0] == pytest.approx(M.contrast_ratio((153.0, 153.0, 153.0), (255.0, 255.0, 255.0)), abs=1e-3)

    def test_rgb_background_with_spaces_counts_as_fill(self):
        soup = BeautifulSoup('<button class="a">Yes</button>', "lxml")
        f = M.html_visual_features(soup.button, M.CssRules(".a{background-color: rgb(15, 118, 110); color:#ffffff}"))
        assert f["background_fill"][0] == 1.0
        assert f["contrast_ratio"][0] > 4.5      # white on teal, not white on (assumed) white page


# ============================================================================ MCAI: roles + dimensions
class TestLexicon:
    @pytest.mark.parametrize("t", ["No thanks", "no thank you", "NO, I DON'T FEEL LUCKY", "No, I'll rather pay full price.",
                                   "I do not want to protect my item", "Not now", "Maybe later", "Skip", "Reject all",
                                   "Continue without discount", "I'll pass", "Opt-out", "Cancel"])
    def test_decline(self, t):
        assert M.is_decline(t)
        assert not M.is_accept(t)            # a decline is never an accept, even when it contains 'continue'/'protect'

    @pytest.mark.parametrize("t", ["Yes, add protection", "Accept all", "Add to cart", "Continue", "Claim discount",
                                   "Place order", "Sign up", "Keep my membership"])
    def test_accept(self, t):
        assert M.is_accept(t) and not M.is_decline(t)

    @pytest.mark.parametrize("t", ["Notify me", "Nothing to see", "Shipping and Duties", "", None])
    def test_neither(self, t):
        assert not M.is_decline(t)
        assert not M.is_accept(t)


class TestDimensions:
    def test_rel_asym_direction(self):
        assert M.rel_asym(2.0, 1.0) == pytest.approx(0.5)        # (2-1)/2: accept more prominent
        assert M.rel_asym(1.0, 2.0) == 0.0                       # decline more prominent -> consumer-friendly -> 0
        assert M.rel_asym(3.0, 3.0) == 0.0
        assert M.rel_asym(0.0, 0.0) == 0.0
        assert M.rel_asym(None, 1.0) is None and M.rel_asym(1.0, None) is None

    def test_interaction_dimension(self):
        d = M.interaction_dimension(1, 5, "cancel")
        assert d.available and d.score == pytest.approx(0.8)     # (5-1)/5
        assert d.provenance == "declared"
        assert M.interaction_dimension(5, 1).score == 0.0         # decline easier: no asymmetry
        assert M.interaction_dimension(3, 3).score == 0.0
        for a, b in ((None, 5), (1, None), (None, None)):
            u = M.interaction_dimension(a, b)
            assert not u.available and u.score is None

    def _dim(self, name, score):
        return DimensionScore(name=name, available=score is not None, score=score,
                              provenance="measured" if score is not None else "unavailable")

    def test_combine_all(self):
        v, cov = M.combine([self._dim("visual", 0.5), self._dim("interaction", 0.8), self._dim("semantic", 0.2)])
        # 100 * (0.4*0.5 + 0.3*0.8 + 0.3*0.2) = 100 * (0.2 + 0.24 + 0.06) = 50.0
        assert v == pytest.approx(50.0) and cov == pytest.approx(1.0)

    def test_combine_renormalises(self):
        v, cov = M.combine([self._dim("visual", None), self._dim("interaction", 0.8), self._dim("semantic", 0.5)])
        # 100 * (0.3*0.8 + 0.3*0.5) / 0.6 = 65.0 ; coverage 0.6 / 1.0
        assert v == pytest.approx(65.0) and cov == pytest.approx(0.6)

    def test_combine_only(self):
        dims = [self._dim("visual", 0.5), self._dim("interaction", None), self._dim("semantic", 0.9)]
        v, cov = M.combine(dims, only=("visual", "interaction"))
        # semantic ignored; interaction unavailable -> 100*0.4*0.5/0.4 = 50.0 ; coverage 0.4/(0.4+0.3)
        assert v == pytest.approx(50.0) and cov == pytest.approx(0.4 / 0.7, abs=1e-4)

    def test_combine_none_available(self):
        assert M.combine([self._dim("visual", None)]) == (None, 0.0)

    def test_semantic_dimension_with_stub(self):
        d = M._semantic_dimension("Yes, save 20%!", "No thanks, I'd rather pay full price", stub_score_fn)
        # guilt 0.8 ; accept_pressure 0.7 ; wording |3 - 7|/7 = 4/7
        # 0.5*0.8 + 0.25*0.7 + 0.25*(4/7) = 0.4 + 0.175 + 0.142857 = 0.717857
        assert d.score == pytest.approx(0.7179, abs=1e-4)
        only_decline = M._semantic_dimension(None, "No thanks", stub_score_fn)
        assert only_decline.score == pytest.approx(0.05) and only_decline.reason


# ============================================================================ MCAI: end to end (stub classifier)
ASYM_HTML = """<html><head><style>
.btn-primary{background:#0f766e;color:#ffffff;font-size:20px;font-weight:700}
.decline{color:#9ca3af;font-size:11px}
</style></head><body><main><div class="offer">
<p>Protect your order</p>
<button class="btn-primary">Yes, add protection</button>
<a href="#" class="decline">No thanks</a>
</div></main></body></html>"""

SYM_HTML = """<html><head><style>.c{background:#ffffff;color:#111111;font-size:16px;font-weight:600;border:1px solid #111}</style>
</head><body><main><div><button class="c">Accept all</button><button class="c">Reject all</button></div></main></body></html>"""


class TestComputeMCAI:
    def test_asymmetric_html(self):
        r = M.compute_mcai("html", stub_score_fn, html=ASYM_HTML)
        assert r.available and len(r.choice_sets) == 1
        cs = r.choice_sets[0]
        assert cs.accept.text == "Yes, add protection" and cs.decline.text == "No thanks"
        vis = next(d for d in cs.dimensions if d.name == "visual")
        assert vis.available and vis.score > 0.4
        assert vis.provenance == "measured"
        feats = {f.feature: f for f in vis.features}
        assert feats["font_size_px"].asymmetry == pytest.approx((20 - 11) / 20)
        assert feats["font_weight"].asymmetry == pytest.approx((700 - 400) / 700, abs=1e-4)
        assert feats["control_kind"].asymmetry == pytest.approx(0.65)
        assert feats["background_fill"].asymmetry == 1.0
        assert feats["rendered_area"].asymmetry is None          # reported as not computable, not 0
        inter = next(d for d in cs.dimensions if d.name == "interaction")
        # intel-v1.1: without declared step counts, HTML gets a STRUCTURAL (inferred) estimate; this page has
        # no collapsed/navigating decline and no pre-ticked opt-outs, so inferred extra steps = 0
        assert inter.available and inter.provenance == "inferred" and inter.score == 0.0
        assert cs.coverage == pytest.approx(1.0)                 # visual 0.4 + interaction 0.3 + semantic 0.3

    def test_symmetric_html_visual_zero(self):
        r = M.compute_mcai("html", stub_score_fn, html=SYM_HTML)
        cs = r.choice_sets[0]
        assert cs.accept.text == "Accept all" and cs.decline.text == "Reject all"
        vis = next(d for d in cs.dimensions if d.name == "visual")
        assert vis.available and vis.score == 0.0

    def test_declared_paths_enter_interaction(self):
        r = M.compute_mcai("html", stub_score_fn, html=SYM_HTML, declared_paths=[(1, 5, "cancel flow")])
        inter = next(d for d in r.choice_sets[0].dimensions if d.name == "interaction")
        assert inter.available and inter.score == pytest.approx(0.8)
        assert r.choice_sets[0].coverage == pytest.approx(1.0)

    def test_text_input_visual_unavailable_never_zero(self):
        cands = segment_text("Protect your order\nYes, add protection\nNo thanks")
        r = M.compute_mcai("text", stub_score_fn, cands=cands)
        assert r.available
        cs = r.choice_sets[0]
        vis = next(d for d in cs.dimensions if d.name == "visual")
        assert vis.available is False and vis.score is None
        assert "no visual" in (vis.reason or "").lower()
        assert cs.coverage == pytest.approx(0.3)                 # semantic only
        assert cs.mcai is not None

    def test_no_choice_pair(self):
        r = M.compute_mcai("text", stub_score_fn, cands=segment_text("Only 2 left\nFree shipping"))
        assert not r.available and r.mcai is None and r.reason
