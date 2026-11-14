"""The headline gauge (summary.risk_score) reflects flagged findings only."""
import pytest

from backend.auditor import Auditor
from backend.extractors import Candidate, extract_html
from ml.common import ARTIFACTS
from ml.model import TextClassifier

MRP = ("Maximum retail price (GST Incl.). The maximum retail price is applicable for imports in India effect from 1st "
       "April 2026. Retailers may have stock at different MRP based on their date of purchase.")


@pytest.fixture(scope="module")
def clf():
    return TextClassifier.load(ARTIFACTS)


def _ocr(text, y, unit=None):
    return Candidate(text=text, source_type="screenshot_ocr", bbox={"x": 0, "y": y, "width": 400, "height": 20},
                     extra={"unit": unit} if unit else {})


def test_gauge_is_zero_when_nothing_is_flagged_even_if_an_element_scored_high(clf):
    # the Omega MRP note: lines score low, the merged paragraph scores just under the 0.90 OCR threshold
    lines = ["Maximum retail price (GST Incl.). The maximum retail",
             "price is applicable for imports in India effect from 1st",
             "April 2026. Retailers may have stock at different MRP", "based on their date of purchase."]
    cands = [_ocr(t, 20 * k) for k, t in enumerate(lines)]
    block = _ocr(MRP, 0, unit="block")
    block.bbox = {"x": 0, "y": 0, "width": 400, "height": 80}
    r = Auditor(clf).audit(cands + [block], "screenshot_ocr", threshold=0.90)
    block_score = clf.predict([MRP])[0]["p_dark"]
    assert block_score < 0.90, "fixture assumes the paragraph stays below the OCR threshold"
    assert not r.findings and r.summary.no_concern
    assert r.summary.risk_score == 0
    assert r.summary.overall_score == pytest.approx(block_score, abs=1e-4)
    # the paragraph that sets the highest score is listed, so the statement's number is traceable
    assert any(a.text == MRP for a in r.benign_elements)
    assert r.summary.elements_analysed == len(lines)


def test_gauge_is_the_top_flagged_score(clf):
    r = Auditor(clf).audit([Candidate(text="Hurry! Only 2 left in stock.", source_type="text")], "text")
    assert r.summary.risk_score == r.findings[0].confidence_pct


def test_gauge_is_unreported_for_dom_only_findings(clf):
    cands, _ = extract_html('<label><input type="checkbox" checked>Add shipping protection for $4.99</label>')
    r = Auditor(clf).audit(cands, "html")
    if all(f.detection_method == "dom_observation" for f in r.findings):
        assert r.summary.risk_score is None
