"""Reproducible, deliberately small synthetic stress test for false-positive changes.

Run: .venv/bin/python -m ml.eval_conservative_detection
Each case labels ONE target element. These invented fixtures are not representative of
real websites and must never be described as an external benchmark. A positive label
means the fixture explicitly contains a deceptive claim or manipulative choice.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from backend.auditor import Auditor
from backend.extractors import Candidate, extract_html, segment_text
from ml.common import ARTIFACTS, DATA_PROCESSED
from ml.model import TextClassifier

OUT = Path(__file__).parent / "artifacts" / "conservative_detection_eval.json"
THRESHOLDS = (0.68, 0.75, 0.80, 0.85, 0.90)
# A deployment candidate must strictly improve precision and FPR, without losing
# more than five percentage points of recall on this set. Fixed before scoring.
MAX_RECALL_DROP = 0.05

# id, label (1=dark), target, optional HTML. HTML cases test the target in its
# natural DOM context; other cases represent pasted page text. Promos with a real
# stated expiry are benign by construction; a timer resetting on refresh is dark.
CASES = [
    ("b01", 0, "Sale price ₹2,990. Regular price ₹3,950.", None),
    ("b02", 0, "Free shipping on orders over ₹2,990", None),
    ("b03", 0, "Delivery in 3-5 working days", None),
    ("b04", 0, "20% off selected shoes through 30 November", None),
    ("b05", 0, "Save $15 with code WELCOME15", None),
    ("b06", 0, "Buy one, get one half price", None),
    ("b07", 0, "Limited-time offer: 15% off until Sunday at 23:59", None),
    ("b08", 0, "Offer ends in 02:00:00 on 30 November", None),
    ("b09", 0, "Only 2 left in stock at this location", None),
    ("b10", 0, "12 customers reviewed this product", None),
    ("b11", 0, "Popular this week", None),
    ("b12", 0, "Subscribe to get 10% off your first order", None),
    ("b13", 0, "No thanks", '<div role="dialog"><p>Get 10% off your first order.</p><button>Sign up</button><button>No thanks</button></div>'),
    ("b14", 0, "Reject optional cookies", None),
    ("b15", 0, "We use cookies to remember your preferences.", None),
    ("b16", 0, "Accept all cookies", '<div role="dialog" aria-label="Cookie choices"><p>We use optional analytics cookies.</p><button>Accept all cookies</button><button>Reject all cookies</button></div>'),
    ("b17", 0, "Reject all cookies", '<div role="dialog" aria-label="Cookie choices"><p>We use optional analytics cookies.</p><button>Accept all cookies</button><button>Reject all cookies</button></div>'),
    ("b18", 0, "Get 20% off", '<section><h2>Autumn sale</h2><p>20% off selected items until 30 November.</p><button>Get 20% off</button><button>Continue shopping</button></section>'),
    ("b19", 0, "25% off", '<div class="product-price"><del>$80</del><ins>$60</ins><span>25% off</span></div>'),
    ("b20", 0, "Offer ends 30 November at 23:59", '<div class="promotion"><p>Autumn sale. Save 20% on selected items.</p><time datetime="2026-11-30T23:59:00">Offer ends 30 November at 23:59</time></div>'),
    ("b21", 0, "Email me offers", '<form><label><input type="checkbox" name="marketing">Email me offers</label><button>Checkout</button></form>'),
    ("b22", 0, "Remember me", '<form><label><input type="checkbox" checked>Remember me</label><button>Sign in</button></form>'),
    ("b23", 0, "View all available sizes", None),
    ("b24", 0, "Coupon valid from 1 to 30 November", None),
    ("d01", 1, "No thanks, I hate saving money", None),
    ("d02", 1, "No thanks, I prefer paying full price", None),
    ("d03", 1, "Hurry! Only 2 left in stock.", None),
    ("d04", 1, "Sale ends in 02:14:09 - order now!", None),
    ("d05", 1, "12 people are viewing this right now", None),
    ("d06", 1, "Someone in your city bought this 2 minutes ago", None),
    ("d07", 1, "You will miss out forever if you leave this page", None),
    ("d08", 1, "Claim your offer before the timer runs out", None),
    ("d09", 1, "No, I do not care about my family's safety", None),
    ("d10", 1, "Only 1 room left at this price! Book now", None),
    ("d11", 1, "Everyone is buying this right now. Hurry!", None),
    ("d12", 1, "Your discount expires in 00:05:00", '<div class="offer" data-timer-reset="on-load"><p>Your discount expires in 00:05:00</p><button>Buy now</button></div>'),
    ("d13", 1, "No, I do not want to protect my account", '<div role="dialog"><p>Enable premium security.</p><button>Enable protection</button><button>No, I do not want to protect my account</button></div>'),
    ("d14", 1, "Add shipping protection for $4.99", '<form><label><input type="checkbox" checked name="shipping-protection">Add shipping protection for $4.99</label></form>'),
    ("d15", 1, "Accept all cookies", '<div role="dialog" aria-label="Cookie choices"><p>We use optional marketing cookies.</p><button>Accept all cookies</button><a href="/settings">Manage settings</a></div>'),
    ("d16", 1, "Get your free trial", '<div><p>Free for 7 days, then $19.99 per month. Cancellation is unavailable online.</p><button>Get your free trial</button></div>'),
]

# A second, smaller set written after the rule was specified but before its
# scores were inspected. It varies the words and markup and is kept separate
# from the exploratory set above. Still synthetic, so it is not a field test.
CHECK_CASES = [
    ("cb01", 0, "No thanks", '<div><p>Join our mailing list.</p><button>Join</button><button>No thanks</button></div>'),
    ("cb02", 0, "30% off", '<div><del>$100</del><ins>$70</ins><span>30% off</span></div>'),
    ("cb03", 0, "Get 15% off", '<section><p>15% off until 12 December.</p><button>Get 15% off</button><button>Skip offer</button></section>'),
    ("cb04", 0, "Sale ends 12 December", '<div><time datetime="2026-12-12">Sale ends 12 December</time></div>'),
    ("cb05", 0, "Reject all cookies", '<div role="dialog"><button>Accept all cookies</button><button>Reject all cookies</button></div>'),
    ("cb06", 0, "Free delivery on orders over $50", None),
    ("cb07", 0, "Offer ends in 03:00:00 on 12 December", None),
    ("cb08", 0, "Subscribe for weekly updates", None),
    ("cd01", 1, "No thanks, I hate discounts", '<div><button>Accept deal</button><button>No thanks, I hate discounts</button></div>'),
    ("cd02", 1, "Only 3 left in stock, order now", None),
    ("cd03", 1, "Sale ends in 03:00:00 - hurry!", None),
    ("cd04", 1, "12 people are viewing this item now", None),
    ("cd05", 1, "Your discount expires in 00:05:00", '<div data-timer-reset="on-load"><time>Your discount expires in 00:05:00</time></div>'),
    ("cd06", 1, "Add warranty for $9", '<label><input type="checkbox" checked>Add warranty for $9</label>'),
    ("cd07", 1, "Get your free trial", '<div><p>Cancellation is unavailable online.</p><button>Get your free trial</button></div>'),
    ("cd08", 1, "Accept all cookies", '<div role="dialog"><button>Accept all cookies</button><a href="/settings">Manage settings</a></div>'),
]

DOM_CONSENT_CASES = [
    ("ec01", 0, "Strictly necessary cookies", '<div role="dialog"><label><input type="checkbox" checked disabled>Strictly necessary cookies</label><button>Reject optional cookies</button></div>'),
    ("ec02", 0, "Essential cookies (always active)", '<div role="dialog"><label><input type="checkbox" checked disabled>Essential cookies (always active)</label><button>Save preferences</button></div>'),
    ("ec03", 1, "Marketing cookies", '<div role="dialog"><label><input type="checkbox" checked>Marketing cookies</label><button>Save preferences</button></div>'),
    ("ec04", 1, "Share my data with partners", '<div role="dialog"><label><input type="checkbox" checked>Share my data with partners</label><button>Save preferences</button></div>'),
]


def metrics(y: list[int], flags: list[bool]) -> dict:
    tp = sum(a == 1 and b for a, b in zip(y, flags))
    fp = sum(a == 0 and b for a, b in zip(y, flags))
    fn = sum(a == 1 and not b for a, b in zip(y, flags))
    tn = sum(a == 0 and not b for a, b in zip(y, flags))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
            "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0}


def target_candidate(target: str, html: str | None):
    cands = extract_html(html)[0] if html else segment_text(target)
    matches = [c for c in cands if c.text == target]
    if len(matches) != 1:
        raise ValueError(f"Expected one candidate for {target!r}; got {len(matches)}")
    return matches[0]


def evaluate() -> dict:
    clf = TextClassifier.load(ARTIFACTS)
    auditor = Auditor(clf, promotion_context_gate=False, neutral_choice_gate=False)
    improved = Auditor(clf)
    cands = [target_candidate(text, html) for _, _, text, html in CASES]
    preds = clf.predict([c.text for c in cands])
    y = [label for _, label, _, _ in CASES]
    results = {}
    rows = []
    for tau in THRESHOLDS:
        ps = [dict(p, flagged=p["p_dark"] >= tau and not p["abstained"]) for p in preds]
        auditor.apply_gates(cands, ps, tau)
        flags = [p["flagged"] for p in ps]
        results[str(tau)] = metrics(y, flags)
        if tau == THRESHOLDS[0]:
            rows = [{"id": case[0], "label": case[1], "text": case[2], "p_dark": round(p["p_dark"], 4),
                     "category": p["category"], "gate": p.get("gate"), "flagged": p["flagged"]}
                    for case, p in zip(CASES, ps)]
            with_context = [dict(p) for p in ps]
            improved.apply_gates(cands, with_context, tau)
            context_decisions = [p["flagged"] for p in with_context]
            context_result = metrics(y, context_decisions)
            withheld_ids = [case[0] for case, p in zip(CASES, with_context)
                            if p.get("gate") in ("structured_promotion", "neutral_choice")]
    base = results[str(THRESHOLDS[0])]
    eligible = [t for t in THRESHOLDS[1:]
                if results[str(t)]["precision"] > base["precision"]
                and results[str(t)]["false_positive_rate"] < base["false_positive_rate"]
                and results[str(t)]["recall"] >= base["recall"] - MAX_RECALL_DROP]
    heldout = pd.read_csv(DATA_PROCESSED / "text_dataset.csv", keep_default_na=False)
    heldout = heldout[heldout.split == "test"].reset_index(drop=True)
    hcands = [Candidate(text=t, source_type="text") for t in heldout.text]
    hp = clf.predict(heldout.text.tolist())
    hy = [int(label != "benign") for label in heldout.label]
    safety = {}
    for tau in THRESHOLDS:
        ps = [dict(p, flagged=p["p_dark"] >= tau and not p["abstained"]) for p in hp]
        auditor.apply_gates(hcands, ps, tau)
        safety[str(tau)] = metrics(hy, [p["flagged"] for p in ps])
        if tau == THRESHOLDS[0]:
            after = [dict(p) for p in ps]
            improved.apply_gates(hcands, after, tau)
            heldout_improved = metrics(hy, [p["flagged"] for p in after])
    safe_thresholds = [t for t in eligible
                       if safety[str(t)]["precision"] > safety[str(THRESHOLDS[0])]["precision"]
                       and safety[str(t)]["false_positive_rate"] < safety[str(THRESHOLDS[0])]["false_positive_rate"]
                       and safety[str(t)]["recall"] >= safety[str(THRESHOLDS[0])]["recall"] - MAX_RECALL_DROP]
    check_cands = [target_candidate(text, html) for _, _, text, html in CHECK_CASES]
    check_preds = clf.predict([c.text for c in check_cands])
    check_y = [label for _, label, _, _ in CHECK_CASES]
    check_before = [dict(p, flagged=p["p_dark"] >= THRESHOLDS[0] and not p["abstained"])
                    for p in check_preds]
    auditor.apply_gates(check_cands, check_before, THRESHOLDS[0])
    check_after = [dict(p) for p in check_before]
    improved.apply_gates(check_cands, check_after, THRESHOLDS[0])
    check = {"before": metrics(check_y, [p["flagged"] for p in check_before]),
             "after": metrics(check_y, [p["flagged"] for p in check_after]),
             "withheld_ids": [case[0] for case, p in zip(CHECK_CASES, check_after)
                              if p.get("gate") in ("structured_promotion", "neutral_choice")]}
    dom_before, dom_after = [], []
    for _, _, target, html in DOM_CONSENT_CASES:
        candidates, _ = extract_html(html)
        for aud, flags in ((Auditor(clf, promotion_context_gate=False, neutral_choice_gate=False,
                                   essential_consent_gate=False), dom_before), (improved, dom_after)):
            result = aud.audit(candidates, "html")
            flags.append(any(f.detection_method == "dom_observation" and f.evidence == target
                             for f in result.findings))
    dy = [label for _, label, _, _ in DOM_CONSENT_CASES]
    dom_check = {"before": metrics(dy, dom_before), "after": metrics(dy, dom_after)}
    return {"description": __doc__.strip().splitlines()[0], "model": "served v1", "n_benign": y.count(0),
            "n_dark": y.count(1), "max_acceptable_recall_drop": MAX_RECALL_DROP,
            "threshold_sweep": results, "synthetic_eligible_thresholds": eligible,
            "heldout_threshold_sweep": safety, "heldout_improved": heldout_improved,
            "thresholds_passing_both": safe_thresholds,
            "baseline_rows": rows,
            "improved_decision": {"metrics": context_result, "withheld_ids": withheld_ids},
            "second_synthetic_check": check, "essential_consent_check": dom_check}


def main() -> None:
    result = evaluate()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "baseline_rows"}, indent=2))


if __name__ == "__main__":
    main()
