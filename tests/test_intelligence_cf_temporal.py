"""Tests for backend/intelligence/counterfactual.py and backend/intelligence/temporal.py.

A deterministic keyword stub replaces the classifier (no model load).
"""
import ml  # noqa: F401  (must be first: sets OMP_NUM_THREADS before torch/xgboost)

import re

import pytest

from backend.intelligence import counterfactual as cf
from backend.intelligence import temporal as tp
from backend.intelligence.config import (COUNTDOWN_TOLERANCE_SECONDS, OBSERVED_FACT_STRENGTH,
                                         OBSERVED_WEAK_FACT_STRENGTH)
from backend.intelligence.counterfactual import facts_preserved, plan_edits, rewrite, word_diff
from backend.intelligence.schemas import ChoiceOption, ChoiceSetResult, CounterfactualEdit, EvidenceRef
from backend.intelligence.temporal import (StageObservation, analyse_journey, parse_countdowns, parse_prices,
                                           parse_stock_counts)
from backend.schemas import Finding, SourceLocation

# ============================================================================ helpers
_KW = re.compile(r"hurry|only|now|last chance|too late|!|don't|i don't|save money|people", re.I)


class StubScorer:
    """Deterministic stand-in for TextClassifier.predict: p_dark grows with pressure keywords."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        out = []
        for t in texts:
            k = len(_KW.findall(t))
            p = min(0.95, 0.1 + 0.2 * k) + 0.000049   # forces rounding to 4 dp to matter
            out.append({"p_dark": p, "category": "urgency", "probabilities": {"benign": 1 - p, "urgency": p},
                        "flagged": p >= 0.5})
        return out


_LABELS = {"urgency": "Misleading urgency", "scarcity": "Misleading scarcity", "social_proof": "Social proof pressure",
           "misdirection": "Misdirection / confirmshaming", "preselected_option": "Preselected option"}


def ml_finding(fid, category, evidence, score=0.87):
    return Finding(id=fid, category=category, category_label=_LABELS[category], evidence=evidence, explanation="x",
                   detection_method="ml_classifier", model_score=score,
                   confidence_pct=None if score is None else round(score * 100),
                   confidence_statement="x", risk_level="high", source_location=SourceLocation(source_type="text"),
                   ethical_recommendation="x")


def dom_finding(fid, evidence, cost=False):
    return Finding(id=fid, category="preselected_option", category_label=_LABELS["preselected_option"], evidence=evidence,
                   explanation="x", detection_method="dom_observation", model_score=None, confidence_pct=None,
                   confidence_statement="x", risk_level="medium" if cost else "low",
                   source_location=SourceLocation(source_type="html"), ethical_recommendation="x",
                   short_reason="Option is pre-ticked, so shoppers opt in by default" + (" (mentions a cost or consent)" if cost else ""))


def choice_set(cid, decline, accept="Yes, add protection", decline_fid=None, mcai=42.0):
    return ChoiceSetResult(id=cid, accept=ChoiceOption(role="accept", text=accept),
                           decline=ChoiceOption(role="decline", text=decline, finding_id=decline_fid),
                           dimensions=[], mcai=mcai, coverage=1.0)


def stage(i, texts=(), findings=(), ts=None, name=None):
    return StageObservation(index=i, name=name or f"stage{i}", captured_at=ts, findings=list(findings), texts=list(texts))


def kinds(result, category):
    return [e for e in result.events if e.category == category]


# ============================================================================ rewrite
@pytest.mark.parametrize("category", ["scarcity", "urgency"])
def test_rewrite_caps_example(category):
    out, expl = rewrite(category, "ONLY 2 LEFT! BUY NOW BEFORE IT'S TOO LATE!")
    assert out == "2 left in stock."
    assert facts_preserved("ONLY 2 LEFT! BUY NOW BEFORE IT'S TOO LATE!", out)
    assert "BUY NOW" in expl and "No figures" in expl


@pytest.mark.parametrize("text,expected", [
    ("Hurry, only a few left!", "A few left in stock."),
    ("Sale ends in 02:14:33 - hurry!", "Sale ends in 02:14:33."),
    ("Limited time only! 50% off", "50% off"),
    ("Today only: 20% off everything", "20% off everything"),
    ("Only 3 left in stock - order now.", "3 left in stock."),
    ("Only 2 left so buy now!!!", "2 left in stock."),
    ("Offer ends in 3 days - shop now and don't miss out", "Offer ends in 3 days"),
    ("Only 1,299 units left in stock, grab yours now", "1,299 units left in stock"),
    ("Don’t miss out on our summer sale!", "Our summer sale."),
    ("Only 2 days left to save $10", "2 days left to save $10"),
])
def test_rewrite_pressure_rules(text, expected):
    out, _ = rewrite("urgency", text)
    assert out == expected
    assert facts_preserved(text, out)


def test_rewrite_keeps_quantity_words_and_never_adds_numbers():
    texts = ["ONLY 2 LEFT! BUY NOW BEFORE IT'S TOO LATE!", "Hurry! Only 7 left in stock", "Only two left - act fast!",
             "Sale ends in 01:59:59! Hurry up", "While stocks last: 30% off, today only!", "Just 4 items left. Hurry!",
             "Ends in 20 HRS 28 MINS. Don't miss out!", "Rs. 499 only today only! Grab it now"]
    for t in texts:
        for cat in ("urgency", "scarcity"):
            out, _ = rewrite(cat, t)
            assert out is not None, t
            assert facts_preserved(t, out), (t, out)
            assert not re.search(r"\b(hurry|act fast|buy now|today only|grab it|don't miss)\b", out, re.I), out
            assert "!" not in out


def test_rewrite_number_words_kept():
    out, _ = rewrite("scarcity", "Only two left - act fast!")
    assert out == "Two left in stock."


@pytest.mark.parametrize("text", ["Hurry!", "Last chance to buy", "BUY NOW!!!", "Don't miss out!"])
def test_rewrite_none_when_nothing_left(text):
    out, expl = rewrite("urgency", text)
    assert out is None and "no informative content" in expl


@pytest.mark.parametrize("text", ["Limited time offer!", "LIMITED TIME OFFER", "Almost gone"])
def test_rewrite_none_when_identical_case_insensitive(text):
    out, expl = rewrite("scarcity", text)
    assert out is None and "no safe automatic rewrite" in expl


@pytest.mark.parametrize("text", ["No thanks, I don't want to save money", "No, I prefer paying full price",
                                  "I don't like saving money", "I’d rather pay full price", "Nope, I'll stay unprotected",
                                  "Not now", "I do not want free shipping", "No thank you, I hate deals"])
def test_rewrite_misdirection_decline(text):
    out, expl = rewrite("misdirection", text)
    assert out == "No thanks"
    assert "neutral" in expl.lower()


@pytest.mark.parametrize("text", ["Upgrade to premium for the best experience", "No hidden fees - upgrade now",
                                  "Most shoppers add protection"])
def test_rewrite_misdirection_upsell_none(text):
    out, expl = rewrite("misdirection", text)
    assert out is None and "no safe automatic rewrite" in expl


def test_rewrite_misdirection_already_neutral():
    assert rewrite("misdirection", "No thanks")[0] is None
    assert rewrite("misdirection", "no thanks.")[0] is None


def test_rewrite_social_proof_preselected_unknown():
    out, expl = rewrite("social_proof", "23 people are viewing this right now!")
    assert out is None and "verifiable data source" in expl and "aggregate" in expl
    out, expl = rewrite("preselected_option", "Add insurance for $4.99")
    assert out is None and "unchecked" in expl
    assert rewrite("benign", "Free returns")[0] is None
    assert rewrite("urgency", "   ")[0] is None


# ============================================================================ facts_preserved
@pytest.mark.parametrize("orig,red", [
    ("ONLY 2 LEFT!", "2 left in stock."),
    ("Only two left", "2 left"),
    ("Rs. 500 off today only", "₹500 off"),
    ("Sale: $1,299.00 now", "Sale: $1299"),
    ("Ends in 02:14:33", "Ends in 2:14:33"),
    ("Free shipping - hurry", "Free shipping"),
    ("Save 20% off now", "Save 20% off"),
    ("20 HRS 28 MINS left", "20 hours 28 minutes left"),
    ("Hurry!", "No thanks"),
])
def test_facts_preserved_accepts(orig, red):
    assert facts_preserved(orig, red)


@pytest.mark.parametrize("orig,red", [
    ("Only 2 left", "3 left"),                       # changed number
    ("Only 2 left", "2 left, 10 sold today"),        # added number
    ("Only 2 left", "2 left 2 left"),                # repeated fact (multiset)
    ("Only 2 left", "2 left - free shipping"),       # 'free' introduced
    ("Only 2 left", "2 left with money-back guarantee"),
    ("Only 2 left", "2 left, full refund"),
    ("Only 2 left", "2 left, 2-year warranty"),
    ("Save 20%", "Save 20% off"),                    # '% off' introduced
    ("Only 2 left", "2 left at a discount"),
    ("Ends in 02:14:33", "Ends in 02:14:34"),
    ("Ends in 2 days", "Ends in 2 hours"),           # time unit changed
    ("$5 off", "£5 off"),                            # currency changed
    ("Only 2 left", "2 left, 50%"),
])
def test_facts_preserved_rejects(orig, red):
    assert not facts_preserved(orig, red)


# ============================================================================ word_diff
def test_word_diff_ops_and_merging():
    d = word_diff("Only 2 left hurry", "2 left in stock")
    assert [(s.op, s.text) for s in d] == [("delete", "Only"), ("equal", "2 left"), ("delete", "hurry"),
                                           ("insert", "in stock")]


def test_word_diff_replace_is_delete_then_insert():
    d = word_diff("No, I hate saving money", "No thanks")      # 'No,' != 'No': whole span replaced
    assert [(s.op, s.text) for s in d] == [("delete", "No, I hate saving money"), ("insert", "No thanks")]
    d = word_diff("Hurry only 2 left", "Just 2 left")
    assert [(s.op, s.text) for s in d] == [("delete", "Hurry only"), ("insert", "Just"), ("equal", "2 left")]
    d = word_diff("a b c", "x y z")
    assert [(s.op, s.text) for s in d] == [("delete", "a b c"), ("insert", "x y z")]


def test_word_diff_edge_cases():
    assert [(s.op, s.text) for s in word_diff("a  b", "a b")] == [("equal", "a b")]
    assert word_diff("", "") == []
    assert [(s.op, s.text) for s in word_diff("", "new text")] == [("insert", "new text")]
    assert [(s.op, s.text) for s in word_diff("old text", "")] == [("delete", "old text")]
    # reconstructing both sides from the segments
    a, b = "ONLY 2 LEFT! BUY NOW BEFORE IT'S TOO LATE!", "2 left in stock."
    d = word_diff(a, b)
    assert " ".join(s.text for s in d if s.op != "insert") == a
    assert " ".join(s.text for s in d if s.op != "delete") == b


# ============================================================================ plan_edits
def test_plan_edits_actions_scores_and_single_batch():
    findings = [
        ml_finding("f1", "scarcity", "ONLY 2 LEFT! BUY NOW BEFORE IT'S TOO LATE!", 0.9312),
        ml_finding("f2", "social_proof", "23 people bought this today", 0.71),
        ml_finding("f3", "misdirection", "No thanks, I don't want to save money", 0.8),
        dom_finding("f4", "Add travel insurance ($4.99)", cost=True),
        ml_finding("f5", "urgency", "Hurry, sale ends in 02:00:00!", None),   # no stored model score
    ]
    sets = [choice_set("c1", "No thanks, I don't want to save money", decline_fid="f3"),  # covered by id
            choice_set("c2", "no thanks,  i don't want to save money"),                  # covered by text
            choice_set("c3", "No, I prefer paying full price", mcai=55.5),               # uncovered, reworded
            choice_set("c4", "No thanks")]                                               # uncovered, already neutral
    stub = StubScorer()
    edits = plan_edits(findings, stub, stage_index=2, choice_sets=sets)
    assert all(isinstance(e, CounterfactualEdit) for e in edits)
    assert len(stub.calls) == 1                                   # one batched call
    scored = stub.calls[0]
    assert len(scored) == len(set(scored))                        # de-duplicated
    # originals with a stored model score are not re-scored
    assert "ONLY 2 LEFT! BUY NOW BEFORE IT'S TOO LATE!" not in scored
    assert "Hurry, sale ends in 02:00:00!" in scored              # model_score None -> scored
    assert {"2 left in stock.", "No thanks", "Sale ends in 02:00:00.", "No, I prefer paying full price"} <= set(scored)

    by = {e.target: e for e in edits}
    assert [e.target for e in edits] == ["f1", "f2", "f3", "f4", "f5", "c3", "c4"]
    assert all(e.stage_index == 2 for e in edits)

    e = by["f1"]
    assert e.action == "rewrite" and e.redesigned == "2 left in stock." and e.facts_preserved
    assert e.original_score == 0.9312
    assert e.redesigned_score == round(0.1 + 0.000049, 4)
    assert [s.op for s in e.diff] and any(s.op == "delete" for s in e.diff)

    e = by["f2"]
    assert e.action == "no_safe_rewrite" and e.redesigned is None and e.redesigned_score is None and e.diff == []
    assert e.original_score == 0.71 and "verifiable" in e.explanation

    e = by["f3"]
    assert e.action == "rewrite" and e.redesigned == "No thanks"

    e = by["f4"]
    assert e.action == "uncheck_default"
    assert e.redesigned == "[unchecked by default] Add travel insurance ($4.99)"
    assert e.original_score is None and e.redesigned_score is None and e.facts_preserved

    e = by["f5"]
    assert e.action == "rewrite" and e.redesigned == "Sale ends in 02:00:00."
    assert e.original_score == round(min(0.95, 0.1 + 0.2 * 2) + 0.000049, 4)   # 'Hurry' + '!'

    e = by["c3"]
    assert e.action == "equalise_choice" and e.category == "choice_asymmetry"
    assert e.redesigned == "No thanks" and "equal visual weight" in e.explanation and "55.5" in e.explanation
    assert e.original_score is not None and e.redesigned_score is not None

    e = by["c4"]
    assert e.action == "equalise_choice" and e.redesigned == "No thanks" and e.original == "No thanks"
    assert all(s.op == "equal" for s in e.diff)


def test_plan_edits_no_scoring_needed_makes_no_call():
    stub = StubScorer()
    edits = plan_edits([ml_finding("f1", "social_proof", "12 people are looking", 0.6), dom_finding("f2", "Subscribe")],
                       stub)
    assert stub.calls == []
    assert [e.action for e in edits] == ["no_safe_rewrite", "uncheck_default"]
    assert edits[0].stage_index is None and edits[0].original_score == 0.6
    assert plan_edits([], stub) == [] and stub.calls == []


def test_plan_edits_rejects_misaligned_score_fn():
    with pytest.raises(ValueError):
        plan_edits([ml_finding("f1", "urgency", "Hurry! Ends in 00:10:00")], lambda texts: [])


def test_counterfactual_docstring_honesty():
    doc = cf.__doc__.lower()
    assert "never invent" in doc and "stock figures" in doc and "discounts" in doc and "guarantees" in doc
    assert "same classifier" in doc and "not independent validation" in doc


# ============================================================================ parsers
@pytest.mark.parametrize("text,expected", [
    ("Offer ends in 02:14:33", [8073]),
    ("02:14:33", [8073]),
    ("01:02:14:33", [86400 + 2 * 3600 + 14 * 60 + 33]),
    ("Ends in 09:59", [599]),
    ("Hurry! 05:00 remaining", [300]),
    ("20 HRS 28 MINS 30 SECS", [20 * 3600 + 28 * 60 + 30]),
    ("2 days 3 hours", [2 * 86400 + 3 * 3600]),
    ("10h 5m", [36300]),
    ("Ends in 3d 04:10:00", [3 * 86400 + 4 * 3600 + 600]),
    ("Ends in 3 hours", [10800]),
    ("Only 2 hours left", [7200]),
    ("Sale A ends in 00:10:00, sale B in 01:00:00", [600, 3600]),
])
def test_parse_countdowns_positive(text, expected):
    assert parse_countdowns(text) == expected


@pytest.mark.parametrize("text", [
    "09:59",                                # MM:SS without a cue word
    "Thu 12:00PM", "Sale ends Thu 12:00PM", "Opens 9:30 am", "Offer ends at 23:59:59", "Offer ends 23:59:59 IST",
    "Sale ends Thu 12:00", "Ends Dec 5 10:00:00", "Updated 2024-01-05 10:00:00", "2024-01-05T10:00:00Z",
    "$12.99", "Price Rs. 1,299.00", "12/05/2024", "Offer valid till 05.12.2024",
    "Free returns within 30 days", "Delivered in 2 days 3 hours", "Posted 2 hours ago", "30-day warranty",
    "Ships in 24 hours", "Only 2 left",
])
def test_parse_countdowns_negative(text):
    assert parse_countdowns(text) == []


@pytest.mark.parametrize("text,expected", [
    ("Only 3 left in stock", [3]), ("5 left in stock", [5]), ("2 remaining", [2]), ("only 7 in stock", [7]),
    ("Just 4 items left!", [4]), ("1,200 remaining", [1200]), ("Only 2 left", [2]),
    ("Only 2 days left", []), ("Only $5 left for free shipping", []), ("00:10 remaining", []), ("3 hours remaining", []),
    ("In stock", []),
])
def test_parse_stock_counts(text, expected):
    assert parse_stock_counts(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("$12.99", [("$", 12.99)]), ("£5", [("£", 5.0)]), ("€12,99", [("€", 12.99)]), ("₹1,00,000", [("₹", 100000.0)]),
    ("Rs. 499", [("₹", 499.0)]), ("Rs499", [("₹", 499.0)]), ("INR 250", [("₹", 250.0)]), ("20 USD", [("$", 20.0)]),
    ("USD 20.50", [("$", 20.5)]), ("Total $1,299.00 incl. $5 shipping", [("$", 1299.0), ("$", 5.0)]),
    ("20 HRS 28 MINS", []), ("50% off", []), ("Only 3 left", []), ("12:00PM", []), ("Order 12345", []),
])
def test_parse_prices(text, expected):
    assert parse_prices(text) == expected


# ============================================================================ analyse_journey
def test_fewer_than_two_stages_unavailable():
    for stages in ([], [stage(0, ["Offer ends in 00:10:00"])]):
        res, pf, of = analyse_journey(stages)
        assert res.available is False and "at least two" in res.reason
        assert res.events == [] and pf == {} and of == {}


def test_countdown_reset_without_timestamps():
    f = ml_finding("f1", "urgency", "Offer ends in 00:15:00", 0.9)
    res, pf, of = analyse_journey([stage(0, ["Offer ends in 00:10:00"]), stage(1, ["Offer ends in 00:15:00"], [f])])
    assert res.available and res.ordering == "submission_order"
    ev = kinds(res, "countdown_reset")
    assert len(ev) == 1
    e = ev[0]
    assert e.kind == "observed_fact" and e.deceptive_signal is True and e.stage_index == 1
    assert "00:10:00" in e.description and "00:15:00" in e.description and "MORE time remaining" in e.description
    assert "real time would show" not in e.description          # no time-based claim without timestamps
    assert [r.stage_index for r in e.evidence] == [0, 1]
    assert e.evidence[1].finding_id == "f1"
    assert pf[1] == [(OBSERVED_FACT_STRENGTH, EvidenceRef(stage_index=1, finding_id="f1", text="Offer ends in 00:15:00"))]
    assert pf[0] == [] and of == {0: [], 1: []}
    assert e.pressure is None and res.pressure_progression == []


def test_countdown_increase_within_one_second_not_claimed():
    res, pf, _ = analyse_journey([stage(0, ["Offer ends in 00:10:00"]), stage(1, ["Offer ends in 00:10:01"])])
    assert kinds(res, "countdown_reset") == [] and pf[1] == []


def test_no_reset_claimed_from_a_single_stage():
    # two countdowns on one page, nothing to compare with in the other stage
    res, pf, _ = analyse_journey([stage(0, ["Sale ends in 00:10:00", "Deal ends in 00:50:00"]), stage(1, ["Your cart"])])
    assert not [e for e in res.events if e.category.startswith("countdown")]
    assert all(v == [] for v in pf.values())
    res, pf, _ = analyse_journey([stage(0, ["Welcome"]), stage(1, ["Offer ends in 00:15:00"])])
    assert not [e for e in res.events if e.category.startswith("countdown")]


def test_countdown_consistent_without_timestamps():
    res, pf, _ = analyse_journey([stage(0, ["Offer ends in 00:10:00"]), stage(1, ["Offer ends in 00:08:00"])])
    ev = kinds(res, "countdown_consistent")
    assert len(ev) == 1 and ev[0].kind == "observation" and ev[0].deceptive_signal is False
    assert "rate was not checked" in ev[0].description
    assert not kinds(res, "countdown_reset") and pf[1] == []


def test_countdown_unchanged_without_timestamps_is_neutral():
    res, pf, _ = analyse_journey([stage(0, ["Offer ends in 00:10:00"]), stage(1, ["Offer ends in 00:10:00"])])
    ev = kinds(res, "countdown_unchanged")
    assert len(ev) == 1 and ev[0].deceptive_signal is False and pf[1] == []


def test_countdown_not_running_down_with_timestamps():
    res, pf, _ = analyse_journey([
        stage(0, ["Offer ends in 00:10:00"], ts="2026-01-01T10:00:00+00:00"),
        stage(1, ["Offer ends in 00:09:55"], ts="2026-01-01T10:05:00+00:00")])
    assert res.ordering == "timestamp"
    ev = kinds(res, "countdown_reset")
    assert len(ev) == 1 and ev[0].deceptive_signal
    d = ev[0].description
    assert "00:05:00 apart" in d and "about 00:05:00" in d and "real-time rate" in d
    assert "MORE time" not in d                                  # it decreased, just too slowly
    assert ev[0].timestamp == "2026-01-01T10:05:00+00:00"
    assert len(pf[1]) == 1


def test_countdown_consistent_with_timestamps_and_tolerance_boundary():
    t0, t1 = "2026-01-01T10:00:00Z", "2026-01-01T10:05:00Z"
    # expected 300 s; tolerance 60 s (+1 s reading resolution)
    ok = 300 + COUNTDOWN_TOLERANCE_SECONDS + 1
    res, pf, _ = analyse_journey([stage(0, ["Offer ends in 00:10:00"], ts=t0),
                                  stage(1, [f"Offer ends in 00:{ok // 60:02d}:{ok % 60:02d}"], ts=t1)])
    assert kinds(res, "countdown_consistent") and not kinds(res, "countdown_reset") and pf[1] == []
    bad = ok + 1
    res, pf, _ = analyse_journey([stage(0, ["Offer ends in 00:10:00"], ts=t0),
                                  stage(1, [f"Offer ends in 00:{bad // 60:02d}:{bad % 60:02d}"], ts=t1)])
    assert kinds(res, "countdown_reset") and len(pf[1]) == 1


def test_countdown_increase_with_timestamps_and_expired_timer():
    res, _, _ = analyse_journey([stage(0, ["Offer ends in 00:10:00"], ts="2026-01-01T10:00:00Z"),
                                 stage(1, ["Offer ends in 00:12:00"], ts="2026-01-01T10:01:00Z")])
    d = kinds(res, "countdown_reset")[0].description
    assert "MORE time remaining" in d and "about 00:09:00" in d
    res, _, _ = analyse_journey([stage(0, ["Offer ends in 00:10:00"], ts="2026-01-01T10:00:00Z"),
                                 stage(1, ["Offer ends in 00:09:00"], ts="2026-01-01T11:00:00Z")])
    d = kinds(res, "countdown_reset")[0].description
    assert "reached zero" in d


def test_partial_timestamps_use_submission_order_and_no_time_inference():
    res, pf, _ = analyse_journey([stage(0, ["Offer ends in 00:10:00"], ts="2026-01-01T10:00:00Z"),
                                  stage(1, ["Offer ends in 00:09:58"], ts=None)])
    assert res.ordering == "submission_order"
    assert kinds(res, "ordering") and "no time-based inference" in kinds(res, "ordering")[0].description
    assert not kinds(res, "countdown_reset") and pf[1] == []
    assert kinds(res, "countdown_consistent")


def test_mixed_timezone_awareness_falls_back_to_submission_order():
    res, _, _ = analyse_journey([stage(0, ["x"], ts="2026-01-01T10:00:00Z"), stage(1, ["y"], ts="2026-01-01T09:00:00")])
    assert res.ordering == "submission_order"


def test_ordering_by_timestamp_when_submitted_out_of_order():
    # submitted: cart (later capture) then product (earlier capture)
    res, pf, _ = analyse_journey([
        stage(0, ["Offer ends in 00:05:00"], ts="2026-01-01T10:05:00Z", name="cart"),
        stage(1, ["Offer ends in 00:10:00"], ts="2026-01-01T10:00:00Z", name="product")])
    assert res.ordering == "timestamp"
    order_ev = kinds(res, "ordering")
    assert len(order_ev) == 1 and "differs from the submission order" in order_ev[0].description
    assert order_ev[0].stage_index == 1                           # first chronological stage
    assert not kinds(res, "countdown_reset")                      # 10:00 -> 05:00 in 5 min is consistent
    ev = kinds(res, "countdown_consistent")
    assert len(ev) == 1 and ev[0].stage_index == 0 and "differs from submission order" in ev[0].description
    assert res.events[0].stage_index == 1                         # timeline follows chronological order
    assert all(v == [] for v in pf.values())


def test_same_wording_timers_are_paired_and_unrelated_timers_compared_conservatively():
    res, pf, _ = analyse_journey([
        stage(0, ["Sale ends in 00:10:00", "Cart reserved for 00:20:00"]),
        stage(1, ["Sale ends in 00:09:00", "Cart reserved for 00:19:00"])])
    assert len(kinds(res, "countdown_consistent")) == 2 and not kinds(res, "countdown_reset")
    # different timers: a new, longer timer next to the running one is not a reset
    res, pf, _ = analyse_journey([stage(0, ["Sale ends in 00:10:00"]),
                                  stage(1, ["Cart reserved for 00:15:00", "Deal ends: 00:05:00"])])
    assert not kinds(res, "countdown_reset") and pf[1] == []


def test_new_preselected_option():
    s0 = stage(0, findings=[dom_finding("f1", "Add gift wrap ($4.99)", cost=True)], name="product")
    s1 = stage(1, findings=[dom_finding("f1", "Add gift wrap ($4.99)", cost=True),
                            dom_finding("f2", "Add protection plan for $9.99", cost=True),
                            dom_finding("f3", "Keep me updated", cost=False)], name="checkout")
    res, pf, of = analyse_journey([s0, s1])
    present = kinds(res, "preselected_option_present")
    assert len(present) == 1 and present[0].stage_index == 0 and present[0].deceptive_signal is False
    new = kinds(res, "new_preselected_option")
    assert [e.evidence[0].finding_id for e in new] == ["f2", "f3"]
    assert all(e.kind == "observed_fact" and e.deceptive_signal and e.stage_index == 1 for e in new)
    assert of[0] == []
    assert [(s, r.finding_id) for s, r in of[1]] == [(OBSERVED_FACT_STRENGTH, "f2"), (OBSERVED_WEAK_FACT_STRENGTH, "f3")]
    assert all(v == [] for v in pf.values())


def test_repeated_scarcity_single_neutral_event():
    stages = [stage(0, findings=[ml_finding("f1", "scarcity", "Only 3 left", 0.7)], name="product"),
              stage(1, ["Your cart"], name="cart"),
              stage(2, findings=[ml_finding("f1", "scarcity", "Only 3 left", 0.8),
                                 ml_finding("f2", "scarcity", "Selling fast", 0.6)], name="checkout"),
              stage(3, findings=[ml_finding("f1", "scarcity", "Almost gone", 0.6)], name="payment")]
    res, pf, of = analyse_journey(stages)
    ev = kinds(res, "repeated_scarcity")
    assert len(ev) == 1 and ev[0].stage_index == 2 and ev[0].kind == "observation" and not ev[0].deceptive_signal
    assert "'product'" in ev[0].description and "'checkout'" in ev[0].description and "'payment'" in ev[0].description
    assert [(r.stage_index, r.finding_id) for r in ev[0].evidence] == [(0, "f1"), (2, "f1"), (3, "f1")]
    assert all(v == [] for v in pf.values()) and all(v == [] for v in of.values())


def test_stock_count_change_is_neutral():
    res, pf, _ = analyse_journey([stage(0, ["Only 5 left in stock"]), stage(1, ["Only 3 left in stock"])])
    ev = kinds(res, "stock_count_change")
    assert len(ev) == 1 and not ev[0].deceptive_signal and ev[0].kind == "observation"
    assert "5" in ev[0].description and "3" in ev[0].description and "restocks" in ev[0].description
    res, _, _ = analyse_journey([stage(0, ["Only 5 left in stock"]), stage(1, ["5 left in stock"])])
    assert not kinds(res, "stock_count_change")


def test_price_change_is_neutral():
    res, pf, of = analyse_journey([stage(0, ["Headphones $49.99"]),
                                   stage(1, ["Headphones $49.99", "Shipping $5.00", "Total $54.99"])])
    ev = kinds(res, "price_change")
    assert len(ev) == 1 and ev[0].kind == "observation" and not ev[0].deceptive_signal
    d = ev[0].description
    assert "$5" in d and "$54.99" in d and "may be legitimate" in d and "shipping/tax" in d
    assert "no longer shown" not in d
    assert all(v == [] for v in pf.values()) and all(v == [] for v in of.values())
    # a stage without prices is not compared
    res, _, _ = analyse_journey([stage(0, ["Welcome"]), stage(1, ["Total $54.99"])])
    assert not kinds(res, "price_change")
    res, _, _ = analyse_journey([stage(0, ["Total ₹1,299"]), stage(1, ["Total Rs. 1,299"])])
    assert not kinds(res, "price_change")                         # same amount, currency aliases


def test_model_inference_events_and_sort_order():
    s0 = stage(0, ["Offer ends in 00:10:00"])
    s1 = stage(1, ["Offer ends in 00:15:00", "Only 3 left in stock"],
               [ml_finding("f1", "urgency", "Offer ends in 00:15:00", 0.87),
                ml_finding("f2", "scarcity", "Only 3 left in stock", 0.62)])
    s0.texts.append("Only 4 left in stock")
    res, _, _ = analyse_journey([s0, s1])
    mi = [e for e in res.events if e.kind == "model_inference"]
    assert [e.description for e in mi] == ["Classifier flagged Misleading urgency (87% calibrated).",
                                           "Classifier flagged Misleading scarcity (62% calibrated)."]
    assert all(not e.deceptive_signal and e.evidence[0].finding_id for e in mi)
    prio = {"observed_fact": 0, "observation": 1, "model_inference": 2}
    keys = [(e.stage_index, prio[e.kind]) for e in res.events]
    assert keys == sorted(keys)
    assert [e.kind for e in res.events if e.stage_index == 1] == ["observed_fact", "observation", "model_inference",
                                                                  "model_inference"]


def test_all_events_are_valid_and_only_facts_are_deceptive():
    stages = [stage(0, ["Offer ends in 00:10:00", "$20"], [dom_finding("f1", "Add warranty $3", True)]),
              stage(1, ["Offer ends in 00:20:00", "$25", "Only 2 left"],
                    [dom_finding("f1", "Newsletter"), ml_finding("f2", "scarcity", "Only 2 left")])]
    res, pf, of = analyse_journey(stages)
    for e in res.events:
        assert e.deceptive_signal == (e.kind == "observed_fact")
        assert e.pressure is None
    assert set(pf) == set(of) == {0, 1}


def test_temporal_docstring_states_rules():
    doc = tp.__doc__.lower()
    assert "two observations" in doc and "observed_fact" in doc and "model_inference" in doc
    assert "first stage" in doc
