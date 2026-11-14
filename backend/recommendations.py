"""Explanations + ethical redesign suggestions.

Rewrites are conservative, deterministic transformations of the user's own evidence text
(remove pressure words, neutralise guilt framing). They never add facts such as dates,
stock numbers or prices that are not in the input. When a safe rewrite is not possible we
return None and give guidance only.
"""
from __future__ import annotations

import re
from typing import Optional

CATEGORY_LABELS = {
    "urgency": "Misleading urgency",
    "scarcity": "Misleading scarcity",
    "social_proof": "Social proof pressure",
    "misdirection": "Misdirection / confirmshaming",
    "preselected_option": "Preselected option",
    "benign": "No dark pattern",
}

EXPLANATIONS = {
    "urgency": "The wording creates time pressure (a deadline, countdown or 'ends soon' claim). When the deadline is not "
               "genuine or not verifiable, it pushes shoppers to decide before comparing options.",
    "scarcity": "The text signals limited stock or high demand. Unverified scarcity claims raise the perceived value of an "
                "item and pressure an immediate purchase.",
    "social_proof": "The text uses other shoppers' activity (views, purchases, 'people are looking') to pressure the "
                    "decision. Such messages are often unverifiable, generic or automatically generated.",
    "misdirection": "The wording steers the choice through framing rather than information - e.g. a decline option that "
                    "shames the user ('No thanks, I don't want to save money') or an upsell framed as the default path.",
    "preselected_option": "This option is already selected in the page markup. Shoppers who do not notice it opt in by "
                          "default, which can add charges or consents they did not actively choose.",
}

SHORT_REASONS = {
    "urgency": "Deadline or countdown pressure pushes a rushed decision",
    "scarcity": "Low-stock or high-demand claim pressures an immediate buy",
    "social_proof": "Other shoppers' activity used as pressure",
    "misdirection": "Guilt-tripping or steering wording instead of a neutral choice",
    "preselected_option": "Option is pre-ticked, so shoppers opt in by default",
}

RECOMMENDATIONS = {
    "urgency": "Only show a deadline if it is real, and state the exact end date/time and time zone. Remove countdowns that "
               "reset, and avoid pressure words such as 'Hurry' or 'Last chance'.",
    "scarcity": "Show stock levels only when they come from live inventory, as a neutral fact (e.g. 'In stock: 3'). Drop "
                "'Hurry' / 'Almost gone' framing and never display fabricated low-stock messages.",
    "social_proof": "Use verifiable, aggregated information (e.g. rating and number of reviews with a link to them). Avoid "
                    "real-time 'X people are viewing' or 'Y just bought' pop-ups unless they are accurate and auditable.",
    "misdirection": "Give accept and decline options equal visual weight and neutral wording. A decline option should simply "
                    "say what it does, e.g. 'No thanks' or 'Continue without the discount'.",
    "preselected_option": "Leave optional add-ons, subscriptions and marketing consents unchecked by default so the shopper "
                          "makes an explicit, informed choice. Show any cost next to the option.",
}

_PRESSURE = re.compile(
    r"\b(hurry( up)?|act (fast|now)|don'?t miss( out)?( on)?|last chance|before it'?s (too late|gone)|"
    r"quick(ly)?|right now|now or never|limited time only|today only|while (stocks|supplies) last|grab it)\b[!.,:]*",
    re.I,
)
_DECLINE_HEAD = re.compile(r"^\s*(no(,)?( thanks| thank you)?|nope|i'?ll pass|skip)\b[\s,.!]*", re.I)


def _tidy(s: str) -> str:
    s = re.sub(r"!{1,}", ".", s)
    s = re.sub(r"\s+([.,:;])", r"\1", s)
    s = re.sub(r"([.,:;]){2,}", r"\1", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" -|,.:;")
    if s.isupper() and len(s) > 4:
        s = s.capitalize()
    return s.strip()


def suggest_rewrite(category: str, evidence: str) -> Optional[str]:
    text = evidence.strip()
    if category == "misdirection":
        m = _DECLINE_HEAD.match(text)
        if m:
            return "No thanks"
        if re.match(r"^\s*i\s+(don'?t|do not|would rather|prefer|'?ll)\b", text, re.I):
            return "No thanks"
        return None
    if category in ("urgency", "scarcity"):
        out = _tidy(_PRESSURE.sub("", text))
        if not out or out.lower() == text.lower() or len(out) < 3:
            return None
        return out
    if category == "social_proof":
        return None  # can't rewrite without inventing a data source
    return None


def explanation_for(category: str, key_terms: list[str]) -> str:
    base = EXPLANATIONS.get(category, "")
    if key_terms:
        quoted = ", ".join(f"“{t}”" for t in key_terms[:3])
        base += f" Words in this element that most increased the model's score: {quoted}."
    return base


def preselected_explanation(label_text: str, cost_or_consent: bool) -> str:
    s = EXPLANATIONS["preselected_option"]
    if cost_or_consent:
        s += " The label mentions a price, subscription, add-on or consent, so the default may cost the shopper money or data."
    return s
