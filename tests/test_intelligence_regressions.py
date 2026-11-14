"""Regression tests for bugs found by the intel-v1.0 benchmark/QA pass (each was reproduced first)."""
import ml  # noqa: F401  (OMP fix before backend imports)

from backend.intelligence.counterfactual import rewrite
from backend.intelligence.mcai import CssRules, compute_mcai, first_color


def stub(texts):
    return [{"p_dark": 0.5, "probabilities": {"misdirection": 0.5}} for _ in texts]


def test_first_color_handles_rgb_with_spaces_and_shorthand():
    assert first_color("rgb(15, 118, 110)") == (15.0, 118.0, 110.0)
    assert first_color("url(x.png) #eee no-repeat") == (238.0, 238.0, 238.0)
    assert first_color("transparent") is None


def test_media_query_rules_are_not_applied_unconditionally():
    rules = CssRules(".a{color:red} @media (max-width:600px){ .a{color:blue} .b{font-size:40px} } .c{color:green}")
    assert rules.skipped_at_rules == 1
    from bs4 import BeautifulSoup
    el = BeautifulSoup('<p class="a b c">x</p>', "lxml").p
    d = rules.declared(el)
    assert d["color"] == "green"                # .c (later, same specificity) - not the @media blue
    assert "font-size" not in d                 # only declared inside @media


def test_stock_clarification_not_duplicated():
    out, _ = rewrite("scarcity", "In stock | 44 left")
    assert out is None or out.lower().count("in stock") == 1


def test_screenshot_without_decline_explains_ocr_blind_spot():
    from backend.extractors import Candidate
    cands = [Candidate(text="Accept offer", source_type="screenshot_ocr", bbox={"x": 0, "y": 0, "width": 50, "height": 10},
                       extra={"unit": "line"})]
    res = compute_mcai("screenshot_ocr", stub, cands=cands, image=None)
    assert not res.available and "OCR" in res.reason and "decline" in res.reason


def test_text_input_mcai_never_fakes_visual_measurements():
    from backend.extractors import segment_text
    res = compute_mcai("text", stub, cands=segment_text("Yes, give me the deal\nNo thanks"))
    vis = [d for d in res.choice_sets[0].dimensions if d.name == "visual"][0]
    assert vis.available is False and vis.score is None


def test_abstention_gate_definition():
    from ml.model import has_lexical_evidence
    for t in ["₹ 3,950.00", "₹ 2,990.00 ₹ 3,950.00", "£3.950.00 £39500 +2.980.00 +3:950.00", "$49.99", "00:12:47", "Rs 499", "INR 2,999", "★ ★ ★"]:
        assert not has_lexical_evidence(t), t
    for t in ["OFFER ENDS 00:54:59", "1 day 08:15:25", "Only 2 left!", "Sale price ₹ 2,990.00", "62% OFF"]:
        assert has_lexical_evidence(t), t


def test_price_rows_are_not_flagged_end_to_end():
    """Real-world false positive: a price grid read by OCR was flagged as 93% 'urgency'."""
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as c:
        a = c.post("/api/audit/text", json={"text": "LONG HALTER SATIN DRESS\n₹ 3,950.00\n£3.950.00 £39500 +2.980.00 +3:950.00\n$49.99\nAdd to bag"}).json()
    assert a["findings"] == []
    assert a["summary"]["no_concern"] is True
    assert any("no words" in w for w in a["warnings"])


# ---------------------------------------------------------------- intel-v1.1: inferred interaction + attribution
def _inter(html):
    from backend.intelligence.mcai import compute_mcai
    res = compute_mcai("html", stub, html=html)
    cs = res.choice_sets[0]
    return cs, next(d for d in cs.dimensions if d.name == "interaction")


def test_inferred_interaction_preticked_optout_counts_one_extra_step():
    cs, inter = _inter('<form><label><input type="checkbox" checked> Add insurance for $3</label>'
                       '<button>Continue</button><a href="#">No thanks</a></form>')
    assert inter.provenance == "inferred" and inter.score == 0.5          # decline 2 steps vs accept 1
    assert {f.feature for f in inter.features} >= {"steps", "preselected_optouts"}


def test_inferred_interaction_collapsed_and_navigating_decline():
    cs, inter = _inter('<form><button>Accept all</button><details><summary>More</summary>'
                       '<a href="/settings/privacy">Reject all</a></details></form>')
    feats = {f.feature for f in inter.features}
    assert {"collapsed_decline", "decline_navigates_away"} <= feats
    assert inter.score == pytest.approx(2 / 3, abs=1e-4)                  # decline 3 steps vs accept 1 (4-dp rounding)


def test_declared_steps_override_inference():
    from backend.intelligence.mcai import compute_mcai
    res = compute_mcai("html", stub, html='<form><button>Continue</button><a href="#">No thanks</a></form>',
                       declared_paths=[(1, 5, "cancel")])
    inter = next(d for d in res.choice_sets[0].dimensions if d.name == "interaction")
    assert inter.provenance == "declared" and inter.score == 0.8


def test_choice_set_attribution_is_exact_and_consistent():
    cs, _ = _inter('<form><label><input type="checkbox" checked> Add insurance for $3</label>'
                   '<button style="font-size:20px;background:#0f766e;color:#fff">Continue</button>'
                   '<a href="#" style="color:#aaa;font-size:11px">No thanks</a></form>')
    assert sum(cs.dimension_contributions.values()) == pytest.approx(cs.mcai, abs=0.05)
    assert cs.top_dimension == max(cs.dimension_contributions, key=cs.dimension_contributions.get)
    assert cs.alternatives["weighted_mean (selected)"] == pytest.approx(cs.mcai, abs=0.05)
    assert cs.sensitivity.dirichlet_p05 <= cs.mcai + 1e-6 and cs.mcai <= cs.sensitivity.dirichlet_p95 + 1e-6


import pytest  # noqa: E402  (used by the tests above)
