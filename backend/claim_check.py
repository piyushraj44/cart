"""Claim specificity check: is a pressure claim stated in checkable terms or in vague, unverifiable ones?

A genuine sale and a fake one can use the same words, so the text model cannot separate them. What differs is
usually *specificity*: honest offers tend to state terms a shopper can check (a calendar deadline, a reference
price, a code, a shipping threshold), while manufactured pressure tends to be vague ("hurry", "selling fast",
an undated countdown, "only a few left") or unverifiable from the page (stock counts, live viewer numbers).

This module is rule-based, uses no training data and never changes a model probability. It reports which
markers it found. `backend/auditor.py` uses `verdict == "checkable"` to withhold an urgency/scarcity score for
review (gate `checkable_claim`). Checkable terms can still be false: the gate says the claim is checkable,
never that it is honest.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_MON = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_DAY = r"([0-3]?\d)(?:st|nd|rd|th)?"
_YEAR = r"(?:,?\s+(20\d\d))?"
_DATE_DM = re.compile(rf"\b{_DAY}\s+(?:of\s+)?{_MON}\b{_YEAR}", re.I)
_DATE_MD = re.compile(rf"\b{_MON}\s+{_DAY}\b{_YEAR}", re.I)
_DATE_NUM = re.compile(r"\b([0-3]?\d)[/.]([01]?\d)[/.](20\d\d)\b|\b(20\d\d)-([01]\d)-([0-3]\d)\b")
_WEEKDAY_TIME = re.compile(
    r"\b(?:until|till|ends?|through|by|before)\s+(?:this\s+)?(mon|tues|wednes|thurs|fri|satur|sun)day\b"
    r"(?:\s+(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?", re.I)

_CUR = r"(?:[$£€¥₹]|rs\.?|inr|usd|eur|gbp)\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s?(?:[$£€₹]|rs\.?|inr|usd|eur|gbp)\b"
_AMOUNT = re.compile(_CUR, re.I)
_REF_PRICE = re.compile(rf"\b(?:was|now|regular price|original price|list price|mrp|rrp|compare at)\b[:\s]*(?:{_CUR})", re.I)
_SAVE_AMOUNT = re.compile(rf"\bsave\s+(?:{_CUR})", re.I)
_PERCENT_OFF = re.compile(r"(?<!up to )(?<!up to\s)\b\d{1,2}\s?%\s?(?:off|discount)\b", re.I)
_UP_TO = re.compile(r"\bup to\s+\d{1,2}\s?%", re.I)
_PRICE_LABEL = re.compile(rf"\b(?:sale|our|your|member|regular)\s+price\b[:\s]*(?:{_CUR})", re.I)
_OFFER_TERMS = re.compile(
    r"\b(?:(?:with|use)\s+(?:the\s+)?code\s+[A-Z0-9]{3,}|on orders? (?:over|above|of)\s+\S+|"
    r"free (?:shipping|delivery|returns?)|(?:delivery|dispatch(?:ed)?|ships?)\s+(?:with)?in\s+\d+(?:\s*[-–]\s*\d+)?\s+(?:working\s+|business\s+)?days?|"
    r"\d+[- ]day (?:returns?|money[- ]back)|returns? within \d+ days|buy (?:one|\d+),? get)", re.I)

_VAGUE = [
    ("hurry", r"\bhurry\b|\brush\b|\bquick(?:ly)?\b"),
    ("act now", r"\b(?:act|order|buy|book|shop|grab it|claim it)\s+(?:it\s+)?now\b|\bdon'?t wait\b"),
    ("fear of missing out", r"\bmiss(?:ing)? out\b|\bdon'?t miss\b|\blast chance\b|\bbefore it'?s (?:too late|gone)\b|\bnow or never\b"),
    ("vague deadline", r"\b(?:ends?|ending|expires?|over)\s+(?:very\s+)?soon\b|\btoday only\b|\btonight\b|\bwhile (?:stocks?|supplies) last\b|\bruns? out\b|\bin \d+\s*(?:hours?|hrs?|minutes?|mins?)\b"),
    ("limited", r"\blimited(?:[- ]time| stock| quantit\w*| offer| availability)?\b"),
    ("undated countdown", r"\b\d{1,2}\s*:\s*\d{2}\s*:\s*\d{2}\b"),
    ("selling fast", r"\bselling (?:fast|out)\b|\bgoing fast\b|\balmost (?:gone|sold out)\b|\bsell(?:s|ing)? out\b|\bhigh demand\b|\bin demand\b"),
    ("stock count", r"\bonly\s+(?:\d+|a few|few)\b|\b\d+\s+(?:left|remaining)\b|\b(?:a )?few left\b|\blow (?:stock|in stock)\b"),
    ("live activity", r"\b(?:viewing|watching|looking at)\b|\bin (?:their|other|\d+) (?:people'?s )?carts?\b|\bjust (?:bought|purchased|ordered)\b|\b(?:minutes?|hours?) ago\b|\beveryone\b"),
    ("up to discount", _UP_TO.pattern),
]
_VAGUE_RE = [(name, re.compile(rx, re.I)) for name, rx in _VAGUE]


@dataclass
class ClaimCheck:
    verdict: str                                   # checkable | vague | mixed | none
    checkable: list[str] = field(default_factory=list)   # "label: matched text"
    vague: list[str] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict:
        return {"verdict": self.verdict, "checkable": self.checkable, "vague": self.vague, "note": self.note}


def _month(s: str) -> int:
    return _MONTHS[s[:3].lower()]


def _dates(text: str) -> list[tuple[str, date | None]]:
    """Calendar dates as (matched text, date if the year is stated and the date is valid, else None)."""
    out: list[tuple[str, date | None]] = []

    def mk(y, m, d):
        try:
            return date(int(y), int(m), int(d)) if y else None
        except ValueError:
            return None
    for m in _DATE_DM.finditer(text):
        out.append((m.group(0), mk(m.group(3), _month(m.group(2)), m.group(1))))
    for m in _DATE_MD.finditer(text):
        out.append((m.group(0), mk(m.group(3), _month(m.group(1)), m.group(2))))
    for m in _DATE_NUM.finditer(text):
        if m.group(3):
            out.append((m.group(0), mk(m.group(3), m.group(2), m.group(1))))
        else:
            out.append((m.group(0), mk(m.group(4), m.group(5), m.group(6))))
    return out


def check_claim(text: str, today: date | None = None) -> ClaimCheck:
    today = today or date.today()
    checkable: list[str] = []
    vague: list[str] = []
    dates = _dates(text)
    for s, d in dates:
        if d is not None and d < today:
            vague.append(f"deadline already passed: {s.strip()}")
        else:
            checkable.append(f"calendar deadline: {s.strip()}")
    if (m := _WEEKDAY_TIME.search(text)):
        checkable.append(f"named-day deadline: {m.group(0).strip()}")
    amounts = _AMOUNT.findall(text)
    for rx, label in ((_REF_PRICE, "reference price"), (_SAVE_AMOUNT, "stated saving"),
                      (_PERCENT_OFF, "stated discount"), (_PRICE_LABEL, "price label"), (_OFFER_TERMS, "offer terms")):
        if (m := rx.search(text)):
            checkable.append(f"{label}: {m.group(0).strip()}")
    if len(amounts) >= 2 and not any(c.startswith("reference price") for c in checkable):
        checkable.append(f"price comparison: {amounts[0].strip()} / {amounts[1].strip()}")
    has_date = any(c.startswith(("calendar deadline", "named-day deadline")) for c in checkable)
    for name, rx in _VAGUE_RE:
        if (m := rx.search(text)):
            # an undated "limited time" or countdown becomes checkable when a calendar deadline is stated beside it
            if name in ("limited", "undated countdown") and has_date:
                continue
            vague.append(f"{name}: {m.group(0).strip()}")
    if checkable and not vague:
        verdict, note = "checkable", ("States checkable terms (" + ", ".join(c.split(":")[0] for c in checkable) +
                                      ") and no vague pressure. ClearCart cannot verify that the terms are true.")
    elif vague and not checkable:
        verdict, note = "vague", ("Pressure without checkable terms (" + ", ".join(v.split(":")[0] for v in vague) + ").")
    elif vague:
        verdict, note = "mixed", ("States some checkable terms but also vague or unverifiable pressure (" +
                                  ", ".join(v.split(":")[0] for v in vague) + ").")
    else:
        verdict, note = "none", "No deadline, price terms or vague pressure markers found."
    return ClaimCheck(verdict, checkable, vague, note)
