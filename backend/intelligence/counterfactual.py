"""Counterfactual ethical redesign of individual elements.

For every finding we propose the smallest edit that removes the manipulative element:

* urgency / scarcity  -> ``rewrite``: pressure phrases ("Hurry", "Buy now", "Before it's too late",
  exclamation runs, ...) and emphasis words ("only"/"just" before a quantity) are removed, ALL-CAPS
  sentences are set in sentence case, punctuation is tidied. The quantity words are kept exactly as
  given; a bare "<quantity> left" is clarified as "<quantity> left in stock" (no number is changed).
* misdirection        -> ``rewrite`` to the neutral "No thanks" when the text is a decline/opt-out
  option; up-sell framing has no safe automatic rewrite (``no_safe_rewrite``).
* social_proof        -> ``no_safe_rewrite``: it cannot be rewritten without a verifiable data
  source (remove it, or replace it with verifiable aggregate data).
* preselected_option  -> ``uncheck_default`` (directly observed DOM state; wording unchanged).
* accept/decline choice sets (MCAI) not already covered by a finding -> ``equalise_choice``.

Guarantees and limits
---------------------
Rewrites ONLY remove or neutralise wording. They never invent stock figures, prices, dates,
discounts, guarantees, refunds, warranties or policies: every candidate rewrite must pass
``facts_preserved`` (no numeric fact or offer term that is absent from the original), otherwise no
rewrite is proposed. Redesigned scores are MODEL-ESTIMATED by the same classifier that produced the
original scores, so a lower redesigned score is not independent validation of the redesign and says
nothing measured about consumer behaviour.
"""
from __future__ import annotations

import difflib
import re
from collections import Counter
from typing import Callable, Optional

from backend.intelligence.schemas import CounterfactualEdit, DiffSegment

ScoreFn = Callable[[list[str]], list[dict]]

_A = "['’]"  # straight or curly apostrophe

# ------------------------------------------------------------------ urgency / scarcity rewriting
# Pressure phrases that are removed. They carry no checkable fact (they only add pressure).
_PRESSURE_RE = re.compile(
    r"(?<!\w)(?:"
    r"hurry(?:\s+up)?"
    r"|act\s+(?:fast|now|quickly)"
    r"|(?:buy|order|shop)\s+(?:it\s+|them\s+)?now"
    rf"|do(?:n{_A}?t|\s+not)\s+miss(?:\s+out)?(?:\s+on)?(?:\s+(?:this|it|these|the)(?:\s+(?:deals?|offers?|sale))?)?"
    rf"|do(?:n{_A}?t|\s+not)\s+wait"
    r"|last\s+chance(?:\s+to\s+(?:buy|grab|get|order|shop)(?:\s+(?:it|this|them|now))?)?"
    rf"|before\s+(?:it|they)(?:{_A}s|{_A}re|\s+is|\s+are)\s+(?:too\s+late|gone|over|sold\s+out)"
    r"|while\s+(?:stocks?|supplies|supply)\s+lasts?"
    r"|limited\s+time\s+only"
    r"|today\s+only"
    r"|ends\s+soon"
    r"|grab\s+(?:it|yours|one)(?:\s+(?:now|today|while\s+you\s+can))?"
    r"|now\s+or\s+never"
    r"|right\s+now"
    r")(?!\w)",
    re.I,
)
_MARK = "\x00"
# connector words left dangling next to a removed phrase ("Only 2 left so <removed>")
_DANGLING_AFTER = re.compile(_MARK + r"\s*(?:and|or|but|so|to|on)\b", re.I)
_DANGLING_BEFORE = re.compile(r"\b(?:and|or|but|so|to|on|for)\s*" + _MARK, re.I)

_NUM_WORDS = ("one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
              "sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred")
_QTY = rf"(?:\d[\d,]*|a\s+few|few|a\s+couple(?:\s+of)?|(?:{_NUM_WORDS}))"
# "only"/"just" directly before a quantity (optionally one noun) followed by left/remaining/in stock
_EMPHASIS_RE = re.compile(rf"\b(?:only|just)\s+(?={_QTY}(?:\s+[a-z]+)?\s+(?:left|remaining|in\s+stock)\b)", re.I)
# bare "<quantity> left" at the end of a clause (no noun between quantity and "left")
_BARE_LEFT_RE = re.compile(rf"\b({_QTY}\s+left)\b(?!\s+in\s+stock)(?=\s*(?:[.;,?]|$))", re.I)


def _collapse_punct(m: re.Match) -> str:
    run = m.group(0)
    if "?" in run:
        return "?"
    if "." in run:
        return "."
    return run.strip()[0]


def _tidy(s: str) -> str:
    s = re.sub(r"!+", ".", s)
    s = re.sub(r"[.,;:?](?:\s*[.,;:?])+", _collapse_punct, s)
    s = re.sub(r"\s+([.,;:?])", r"\1", s)
    s = re.sub(r"(?:\s*[-–—|]\s*){2,}", " - ", s)               # dash runs
    s = re.sub(r"\s*[-–—|]\s*(?=[.,;:?]|$)", "", s)              # dash before punctuation / end
    s = re.sub(r"(?<=[.?])\s*[-–—|,;:]\s*", " ", s)              # separator after sentence end
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"^[\s.,;:?\-–—|]+", "", s)
    s = re.sub(r"[\s,;:\-–—|]+$", "", s)
    return s.strip()


def _sentence_case(s: str) -> tuple[str, bool]:
    parts = re.split(r"(?<=[.?!])\s+", s)
    changed = False
    out = []
    for p in parts:
        letters = [c for c in p if c.isalpha()]
        if len(letters) >= 3 and all(c.isupper() for c in letters):
            low = p.lower()
            if low and low[0].isalpha():
                low = low[0].upper() + low[1:]
            out.append(low)
            changed = True
        else:
            out.append(p)
    return " ".join(out), changed


def _content_words(s: str) -> list[str]:
    return re.sub(r"[^\w%$£€₹]+", " ", s.lower()).split()


def _quote(xs: list[str]) -> str:
    return ", ".join(f"'{x}'" for x in xs)


def _rewrite_pressure(text: str) -> tuple[Optional[str], list[str]]:
    removed: list[str] = []

    def _rm(m: re.Match) -> str:
        removed.append(re.sub(r"\s+", " ", m.group(0).strip()))
        return _MARK

    s = _PRESSURE_RE.sub(_rm, text)
    notes: list[str] = []
    if removed:
        s = _DANGLING_AFTER.sub(_MARK, s)
        s = _DANGLING_BEFORE.sub(_MARK, s)
        s = s.replace(_MARK, " ")
        notes.append(f"removed pressure wording ({_quote(removed)})")
    emph = [m.group(0).strip() for m in _EMPHASIS_RE.finditer(s)]
    if emph:
        s = _EMPHASIS_RE.sub("", s)
        notes.append(f"removed emphasis ({_quote(emph)}) before the stated quantity")
    if "!" in text:
        notes.append("removed exclamation marks")
    s = _tidy(s)
    s, cased = _sentence_case(s)
    if cased:
        notes.append("converted all-caps text to sentence case")
    # only clarify "N left" when the text does not already mention stock elsewhere ("In stock | 44 left")
    n_left = 0
    if not re.search(r"\bin[\s-]+stock\b|\bstock\b", s, re.I):
        s, n_left = _BARE_LEFT_RE.subn(r"\1 in stock", s)
    if n_left:
        notes.append("made the stock reference explicit ('left' -> 'left in stock') without changing the quantity")
    s = s.strip()
    if s and s[0].isalpha() and s[0].islower():
        s = s[0].upper() + s[1:]
    if s and s[-1] not in ".?" and text.rstrip()[-1:] in (".", "!", "?"):
        s += "."
    return (s or None), notes


# ------------------------------------------------------------------ misdirection
_DECLINE_RE = re.compile(
    r"^\s*(?:"
    r"no(?=\s*[,.!]|\s*$|\s+thanks?\b|\s+thank\s+you\b|\s+way\b|\s+i\b|\s*-)"
    r"|nope\b|nah\b"
    rf"|i\s*(?:{_A}ll|\s+will)\s+pass\b"
    rf"|i\s+(?:don{_A}?t|do\s+not|won{_A}?t|will\s+not|would\s+rather|{_A}d\s+rather|prefer|{_A}m\s+not|am\s+not|{_A}ll\s+(?:skip|pass|stay|keep|pay|take|risk|go))\b"
    rf"|i{_A}d\s+rather\b|i{_A}ll\s+(?:skip|pass|stay|keep|pay|take|risk|go)\b|i{_A}m\s+not\b"
    r"|not\s+(?:now|today|interested)\b|maybe\s+later\b|skip\b|continue\s+without\b"
    r")",
    re.I,
)
NEUTRAL_DECLINE = "No thanks"

# ------------------------------------------------------------------ facts
_UNIT_NORM = (
    (re.compile(r"^(?:days?|d)$", re.I), "d"),
    (re.compile(r"^(?:hours?|hrs?|h)$", re.I), "h"),
    (re.compile(r"^(?:minutes?|mins?|m)$", re.I), "min"),
    (re.compile(r"^(?:seconds?|secs?|s)$", re.I), "s"),
    (re.compile(r"^(?:weeks?|wks?)$", re.I), "wk"),
    (re.compile(r"^(?:months?)$", re.I), "mo"),
    (re.compile(r"^(?:years?|yrs?)$", re.I), "yr"),
    (re.compile(r"^(?:%|percent)$", re.I), "%"),
)
_CUR_NORM = {"$": "$", "usd": "$", "£": "£", "gbp": "£", "€": "€", "eur": "€",
             "₹": "₹", "rs": "₹", "rs.": "₹", "inr": "₹"}
_UNITS = r"(?:%|percent\b|days?\b|d\b|hours?\b|hrs?\b|h\b|minutes?\b|mins?\b|m\b|seconds?\b|secs?\b|s\b|weeks?\b|wks?\b|months?\b|years?\b|yrs?\b)"
_FACT_RE = re.compile(
    r"(?P<time>(?<![\d:])\d{1,3}(?::\d{2}){1,3}(?![\d:]))"
    r"|(?P<cur>[$£€₹]|(?<![A-Za-z])(?:Rs\.?|INR|USD|EUR|GBP)(?![A-Za-z]))\s?(?P<camt>\d[\d,]*(?:\.\d+)?)"
    r"|(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<post>(?:INR|USD|EUR|GBP)(?![A-Za-z])|" + _UNITS + r")?"
    rf"|(?P<word>\b(?:{_NUM_WORDS})\b)\s*(?P<wpost>{_UNITS})?",
    re.I,
)
_WORD_VALUE = {w: str(v) for v, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
_WORD_VALUE.update({"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
                    "ninety": "90", "hundred": "100"})
_OFFER_TERMS = (
    ("guarantee", re.compile(r"\bguarantee", re.I)),
    ("free", re.compile(r"\bfree\b", re.I)),
    ("refund", re.compile(r"\brefund", re.I)),
    ("warranty", re.compile(r"\bwarrant(?:y|ies)\b", re.I)),
    ("discount", re.compile(r"\bdiscount", re.I)),
    ("% off", re.compile(r"(?:%|\bpercent)\s*off\b", re.I)),
    ("money back", re.compile(r"\bmoney[-\s]?back\b", re.I)),
)


def _norm_number(s: str) -> str:
    s = s.replace(",", "")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    intpart, _, frac = s.partition(".")
    intpart = intpart.lstrip("0") or "0"
    return intpart + ("." + frac if frac else "")


def _norm_unit(u: Optional[str]) -> str:
    if not u:
        return ""
    u = u.strip()
    if u.lower() in _CUR_NORM:
        return "cur:" + _CUR_NORM[u.lower()]
    for rx, name in _UNIT_NORM:
        if rx.match(u):
            return name
    return u.lower()


def _fact_tokens(text: str) -> Counter:
    toks: Counter = Counter()
    for m in _FACT_RE.finditer(text or ""):
        if m.group("time"):
            toks["time:" + ":".join(str(int(p)) for p in m.group("time").split(":"))] += 1
        elif m.group("cur"):
            toks[f"cur:{_CUR_NORM[m.group('cur').lower()]}:{_norm_number(m.group('camt'))}"] += 1
        elif m.group("num"):
            post = _norm_unit(m.group("post"))
            if post.startswith("cur:"):
                toks[f"{post}:{_norm_number(m.group('num'))}"] += 1
            else:
                toks[f"num:{_norm_number(m.group('num'))}:{post}"] += 1
        elif m.group("word"):
            toks[f"num:{_WORD_VALUE[m.group('word').lower()]}:{_norm_unit(m.group('wpost'))}"] += 1
    return toks


def facts_preserved(original: str, redesigned: str) -> bool:
    """True iff ``redesigned`` introduces no numeric fact and no offer term absent from ``original``.

    Numbers, percentages, currency amounts, clock-style durations and number+time-unit quantities are
    normalised (thousands separators, trailing zeros, currency aliases such as Rs/INR/₹, number words)
    and compared as a multiset: each one in ``redesigned`` must occur at least as often in
    ``original``. Offer terms (guarantee, free, refund, warranty, discount, "% off", money back) may only
    appear in ``redesigned`` if they already appear in ``original``.
    """
    o, r = _fact_tokens(original), _fact_tokens(redesigned)
    for tok, n in r.items():
        if o.get(tok, 0) < n:
            return False
    for _, rx in _OFFER_TERMS:
        if rx.search(redesigned or "") and not rx.search(original or ""):
            return False
    return True


# ------------------------------------------------------------------ public: rewrite
def rewrite(category: str, text: str) -> tuple[Optional[str], str]:
    """Return (redesigned_text or None, explanation). Never adds facts (see module docstring)."""
    text = (text or "").strip()
    if not text:
        return None, "Empty text: nothing to rewrite."
    if category in ("urgency", "scarcity"):
        out, notes = _rewrite_pressure(text)
        if not out or not re.search(r"[A-Za-z0-9]", out):
            return None, ("Removing the pressure wording leaves no informative content; remove the element, or show a "
                          "verifiable fact (e.g. a real end date or live stock level) in neutral wording.")
        if _content_words(out) == _content_words(text):
            return None, ("No removable pressure phrase was found (only case or punctuation would change); no safe "
                          "automatic rewrite. Verify the claim and state it neutrally, or remove it.")
        if not facts_preserved(text, out):
            return None, "A candidate rewrite would have altered the factual content, so no rewrite is proposed."
        return out, ("Neutral rewrite: " + "; ".join(notes) + ". No figures, dates or offers were added."
                     if notes else "Neutral rewrite. No figures, dates or offers were added.")
    if category == "misdirection":
        if _DECLINE_RE.match(text.replace("’", "'")):
            if _content_words(text) == _content_words(NEUTRAL_DECLINE):
                return None, "The decline option is already neutral ('No thanks')."
            return NEUTRAL_DECLINE, ("Decline option reworded neutrally: it now simply states the choice, without "
                                     "guilt or loss framing.")
        return None, ("Up-sell or steering framing: no safe automatic rewrite (it would require knowing the offer's "
                      "real terms). Present the offer and the decline with neutral wording and equal visual weight.")
    if category == "social_proof":
        return None, ("Social proof cannot be rewritten without a verifiable data source; remove it or replace it with "
                      "verifiable aggregate data (e.g. rating and number of reviews, linked to the reviews).")
    if category == "preselected_option":
        return None, "Pre-selected options are redesigned by leaving them unchecked by default; the wording is unchanged."
    return None, f"No rewrite rule for category '{category}'."


# ------------------------------------------------------------------ public: word_diff
def word_diff(a: str, b: str) -> list[DiffSegment]:
    """Word-level diff on whitespace tokens (difflib). 'replace' is emitted as delete then insert."""
    ta, tb = (a or "").split(), (b or "").split()
    segs: list[tuple[str, list[str]]] = []

    def push(op: str, toks: list[str]) -> None:
        if not toks:
            return
        if segs and segs[-1][0] == op:
            segs[-1][1].extend(toks)
        else:
            segs.append((op, list(toks)))

    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
        if tag == "equal":
            push("equal", ta[i1:i2])
        elif tag == "delete":
            push("delete", ta[i1:i2])
        elif tag == "insert":
            push("insert", tb[j1:j2])
        else:  # replace
            push("delete", ta[i1:i2])
            push("insert", tb[j1:j2])
    return [DiffSegment(op=op, text=" ".join(toks)) for op, toks in segs]


# ------------------------------------------------------------------ public: plan_edits
def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _r4(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(float(x), 4)


def plan_edits(findings: list, score_fn: ScoreFn, stage_index: Optional[int] = None,
               choice_sets: list | None = None) -> list[CounterfactualEdit]:
    """One counterfactual edit per finding, plus ``equalise_choice`` edits for uncovered choice sets.

    All texts that need a classifier score (originals without a stored ``model_score`` and every
    redesigned text) are scored in a single ``score_fn`` call.
    """
    plans: list[dict] = []
    for f in findings or []:
        if f.detection_method == "dom_observation" and f.category == "preselected_option":
            red = f"[unchecked by default] {f.evidence}"
            plans.append(dict(target=f.id, category=f.category, action="uncheck_default", original=f.evidence,
                              redesigned=red, score_original=False, model_score=None,
                              explanation=("Directly observed pre-selected option (DOM state, not a model score). "
                                           "Redesign: leave it unchecked so the shopper opts in explicitly; wording unchanged.")))
            continue
        if f.detection_method == "ml_classifier":
            red, expl = rewrite(f.category, f.evidence)
            plans.append(dict(target=f.id, category=f.category, action="rewrite" if red else "no_safe_rewrite",
                              original=f.evidence, redesigned=red, score_original=True, model_score=f.model_score,
                              explanation=expl))
            continue
        plans.append(dict(target=f.id, category=f.category, action="no_safe_rewrite", original=f.evidence,
                          redesigned=None, score_original=False, model_score=None,
                          explanation=f"No redesign rule for a {f.detection_method} finding of category '{f.category}'."))

    covered_ids = {f.id for f in findings or []}
    covered_txt = {_norm_text(f.evidence) for f in findings or []}
    for cs in choice_sets or []:
        d = cs.decline
        if d is None or (d.finding_id and d.finding_id in covered_ids) or _norm_text(d.text) in covered_txt:
            continue
        cand, _ = rewrite("misdirection", d.text)
        red = NEUTRAL_DECLINE if cand == NEUTRAL_DECLINE else d.text
        wording = ("decline reworded to the neutral 'No thanks'" if red != d.text
                   else "decline wording kept (already neutral or no safe automatic rewrite)")
        mcai = f" Measured asymmetry for this pair: MCAI {cs.mcai:.1f}/100." if getattr(cs, "mcai", None) is not None else ""
        plans.append(dict(target=cs.id, category="choice_asymmetry", action="equalise_choice", original=d.text,
                          redesigned=red, score_original=True, model_score=None,
                          explanation=("Accept and decline must get equal visual weight, equal effort and neutral "
                                       f"wording; {wording}.{mcai}")))

    # ---- one batched scoring call
    need: list[str] = []
    seen: set[str] = set()

    def want(t: str) -> None:
        if t not in seen:
            seen.add(t)
            need.append(t)

    for p in plans:
        if p["score_original"] and p["model_score"] is None:
            want(p["original"])
        if p["score_original"] and p["redesigned"] is not None:
            want(p["redesigned"])
    scores: dict[str, float] = {}
    if need:
        preds = score_fn(need)
        if len(preds) != len(need):
            raise ValueError(f"score_fn returned {len(preds)} predictions for {len(need)} texts")
        scores = {t: float(p["p_dark"]) for t, p in zip(need, preds)}

    edits: list[CounterfactualEdit] = []
    for p in plans:
        red = p["redesigned"]
        if p["score_original"]:
            o_score = _r4(p["model_score"] if p["model_score"] is not None else scores[p["original"]])
            r_score = _r4(scores[red]) if red is not None else None
        else:
            o_score = r_score = None
        edits.append(CounterfactualEdit(
            target=p["target"], stage_index=stage_index, category=p["category"], action=p["action"],
            original=p["original"], redesigned=red,
            diff=word_diff(p["original"], red) if red is not None else [],
            original_score=o_score, redesigned_score=r_score,
            facts_preserved=facts_preserved(p["original"], red) if red is not None else True,
            explanation=p["explanation"]))
    return edits
