"""Claim specificity check and the three evidence tiers (backend/claim_check.py, backend/auditor.py)."""
from datetime import date

import pytest

from backend.auditor import Auditor
from backend.claim_check import check_claim
from backend.extractors import Candidate, extract_html
from ml.common import ARTIFACTS
from ml.model import TextClassifier

TODAY = date(2026, 11, 14)


@pytest.fixture(scope="module")
def clf():
    return TextClassifier.load(ARTIFACTS)


@pytest.mark.parametrize("text,verdict", [
    ("20% off selected shoes through 30 November", "checkable"),
    ("Was ₹1,499, now ₹999", "checkable"),
    ("Free delivery on orders over £40", "checkable"),
    ("Save $15 with code WELCOME15", "checkable"),
    ("Offer valid until 15/12/2026", "checkable"),
    ("Winter sale ends Sunday at 23:59", "checkable"),
    ("Offer ends in 02:00:00 on 30 November", "checkable"),   # countdown pinned to a calendar date
    ("Hurry! Only 2 left in stock.", "vague"),
    ("Deal ends in 00:09:59", "vague"),                         # undated countdown
    ("Up to 70% off - limited time only", "vague"),
    ("12 people are viewing this right now", "vague"),
    ("Grab it before it's gone - 20% off until 30 November", "mixed"),
    ("View all available sizes", "none"),
])
def test_verdicts(text, verdict):
    assert check_claim(text, today=TODAY).verdict == verdict


def test_past_deadline_is_not_checkable():
    c = check_claim("Offer ends 1 January 2020 - 50% off", today=TODAY)
    assert c.verdict == "mixed"
    assert any(v.startswith("deadline already passed") for v in c.vague)


def test_up_to_is_not_a_stated_discount():
    c = check_claim("Up to 40% off", today=TODAY)
    assert not any(x.startswith("stated discount") for x in c.checkable)


def _audit(clf, text, **kw):
    return Auditor(clf, **kw).audit([Candidate(text=text, source_type="text", char_start=0, char_end=len(text))], "text")


def test_genuine_sale_is_held_for_review_not_a_finding(clf):
    text = "Winter sale ends Sunday at 23:59"
    before = _audit(clf, text, claim_check_gate=False)
    assert before.findings, "fixture must cross the threshold without the gate"
    after = _audit(clf, text)
    assert not after.findings
    u = after.uncertain_elements[0]
    assert u.gate == "checkable_claim" and u.claim_check.verdict == "checkable"
    assert after.summary.tier_counts == {"observed": 0, "pressure_claim": 0, "review": 1}
    assert after.summary.no_concern
    assert any("checkable terms" in w for w in after.warnings)


def test_vague_pressure_stays_a_finding_with_its_markers(clf):
    r = _audit(clf, "Hurry! Only 2 left in stock.")
    assert len(r.findings) == 1
    f = r.findings[0]
    assert f.evidence_tier == "pressure_claim"
    assert f.claim_check.verdict == "vague"
    assert {v.split(":")[0] for v in f.claim_check.vague} >= {"hurry", "stock count"}


def test_gate_never_applies_to_social_proof_or_misdirection(clf):
    for text in ("No thanks, I prefer paying full price", "12 people are viewing this right now"):
        r = _audit(clf, text)
        assert not any(u.gate == "checkable_claim" for u in r.uncertain_elements)


def test_dom_observation_is_the_observed_tier(clf):
    cands, _ = extract_html('<label><input type="checkbox" checked>Add shipping protection for $4.99</label>')
    r = Auditor(clf).audit(cands, "html")
    dom = [f for f in r.findings if f.detection_method == "dom_observation"]
    assert dom and all(f.evidence_tier == "observed" and f.claim_check is None for f in dom)
    assert r.summary.tier_counts["observed"] == len(dom)
