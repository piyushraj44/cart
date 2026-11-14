"""Regression checks for conservative, observable context decisions."""
import pytest

from backend.auditor import Auditor
from backend.extractors import extract_html
from ml.common import ARTIFACTS
from ml.model import TextClassifier


@pytest.fixture(scope="module")
def clf():
    return TextClassifier.load(ARTIFACTS)


@pytest.mark.parametrize("html,target,gate", [
    ('<section><p>20% off selected items until 30 November.</p><button>Get 20% off</button>'
     '<button>Continue shopping</button></section>', "Get 20% off", "structured_promotion"),
    ('<div><del>$80</del><ins>$60</ins><span>25% off</span></div>', "25% off", "structured_promotion"),
    ('<div><time datetime="2026-11-30T23:59:00">Offer ends 30 November at 23:59</time></div>',
     "Offer ends 30 November at 23:59", "structured_promotion"),
    ('<div><p>Get 10% off your first order.</p><button>Sign up</button><button>No thanks</button></div>',
     "No thanks", "neutral_choice"),
])
def test_benign_html_context_is_visible_as_uncertain(clf, html, target, gate):
    cands, _ = extract_html(html)
    baseline = Auditor(clf, promotion_context_gate=False, neutral_choice_gate=False,
                       claim_check_gate=False).audit(cands, "html")
    assert any(f.evidence == target for f in baseline.findings)
    improved = Auditor(clf).audit(cands, "html")
    assert not any(f.evidence == target for f in improved.findings)
    assert any(u.text == target and u.gate == gate and u.p_dark >= clf.b.threshold
               for u in improved.uncertain_elements)
    assert improved.summary.threshold == clf.b.threshold


def test_pressure_and_preselected_cost_still_flagged(clf):
    html = ('<div data-timer-reset="on-load"><p>Your discount expires in 00:05:00</p><button>Buy now</button></div>'
            '<form><label><input type="checkbox" checked name="protection">'
            'Add shipping protection for $4.99</label></form>')
    cands, _ = extract_html(html)
    result = Auditor(clf).audit(cands, "html")
    assert any(f.evidence == "Your discount expires in 00:05:00" for f in result.findings)
    assert any(f.detection_method == "dom_observation" and f.evidence == "Add shipping protection for $4.99"
               for f in result.findings)


def test_context_requires_structured_terms(clf):
    for html in ('<div><button>Get 20% off</button></div>',
                 '<time datetime="not-a-date">Offer ends 30 November at 23:59</time>',
                 '<div><span>25% off</span></div>'):
        cands, _ = extract_html(html)
        assert not any(c.extra.get("structured_promotion") for c in cands)


def test_disabled_essential_cookie_control_is_not_a_preselection_finding(clf):
    html = ('<div role="dialog"><label><input type="checkbox" checked disabled>'
            'Strictly necessary cookies</label><label><input type="checkbox" checked>'
            'Marketing cookies</label><label><input type="checkbox" checked disabled>'
            'Essential cookies and marketing</label></div>')
    cands, _ = extract_html(html)
    assert next(c for c in cands if c.text == "Strictly necessary cookies").extra["disabled"]
    before = Auditor(clf, essential_consent_gate=False).audit(cands, "html")
    after = Auditor(clf).audit(cands, "html")
    def dom_evidence(result):
        return {f.evidence for f in result.findings if f.detection_method == "dom_observation"}
    assert dom_evidence(before) == {"Strictly necessary cookies", "Marketing cookies",
                                    "Essential cookies and marketing"}
    assert dom_evidence(after) == {"Marketing cookies", "Essential cookies and marketing"}


def test_synthetic_experiment_and_heldout_guardrail():
    from ml.eval_conservative_detection import evaluate
    result = evaluate()
    before = result["threshold_sweep"]["0.68"]
    after = result["improved_decision"]["metrics"]
    assert after["precision"] > before["precision"]
    assert after["false_positive_rate"] < before["false_positive_rate"]
    assert after["recall"] >= before["recall"] - result["max_acceptable_recall_drop"]
    assert result["heldout_improved"] == result["heldout_threshold_sweep"]["0.68"]
    assert result["thresholds_passing_both"] == []
    check = result["second_synthetic_check"]
    assert check["after"]["precision"] > check["before"]["precision"]
    assert check["after"]["false_positive_rate"] < check["before"]["false_positive_rate"]
    assert check["after"]["recall"] == check["before"]["recall"]
    consent = result["essential_consent_check"]
    assert consent["after"]["false_positive_rate"] < consent["before"]["false_positive_rate"]
    assert consent["after"]["recall"] == consent["before"]["recall"]
