"""Sequential evidence and temporal deception analysis across ordered journey stages.

The timeline separates three kinds of evidence (``TimelineEvent.kind``):

* ``observed_fact``   - something read directly from two or more observations, e.g. a countdown
  that showed MORE time remaining at a later observation, or a pre-selected option that first
  appears at a later stage. Only these carry ``deceptive_signal=True`` and only these are returned
  as pressure / obstruction facts for FSPM and CDI.
* ``observation``     - neutral, directly read changes or repetitions (stock count changed, price
  amounts changed, repeated scarcity claims, a countdown that ran down consistently). They may
  have legitimate explanations (restocks, shipping/tax added at checkout) and never enter a score.
* ``model_inference`` - the classifier's flags per stage. They already enter FSPM through the
  classifier scores, so they are listed for context only (``deceptive_signal=False``).

Rules
-----
* A countdown reset is ONLY claimed from two observations (two stages / snapshots), never from a
  single page. Readings are compared between consecutive stages that show a countdown. When the
  same countdown text (same wording, digits masked) appears in both stages, those readings are
  compared; otherwise all readings are compared conservatively (every later reading must exceed
  every earlier one). An increase must exceed the coarser reading resolution (1 s for clock-style
  displays, the smallest unit for "2 days 3 hours"-style displays).
* Time-based inferences (a countdown that did not run down at the real-time rate) are only made
  when EVERY stage carries a parseable ISO-8601 ``captured_at``; stages are then ordered by
  timestamp. Otherwise submission order is kept and no time-based inference is made.
* Pre-selected options: a DOM-observed pre-selection whose text was not pre-selected at any
  earlier stage is an observed fact at the stage where it appears (strength 1.0 when the label
  mentions a cost or consent, else 0.5). Pre-selections already present at the FIRST stage are
  reported as a neutral observation and do not create a temporal fact (the engine already counts
  per-page DOM observations separately).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from backend.intelligence.config import (COUNTDOWN_TOLERANCE_SECONDS, OBSERVED_FACT_STRENGTH,
                                         OBSERVED_WEAK_FACT_STRENGTH)
from backend.intelligence.schemas import EvidenceRef, TemporalResult, TimelineEvent

Fact = tuple[float, EvidenceRef]
_A = "['’]"

# =====================================================================  parsers
_CUE_RE = re.compile(r"\b(?:ends?|ending|left|remaining|expires?|expiring|countdown|offer|sale|hurry|timer)\b", re.I)
_COLON_RE = re.compile(r"(?<![\w:.,/$£€₹])(\d{1,3}(?::\d{2}){1,3})(?![\w:]|[.,/]\d)")
_CLOCK_AFTER_RE = re.compile(
    r"^\s*(?:[ap]\.?\s?m\b\.?|o" + _A + r"?clock\b|h\b|hrs\b|(?:utc|gmt|ist|est|edt|cst|cdt|mst|mdt|pst|pdt|cet|cest|bst|aest|jst|z)\b)",
    re.I)
_CLOCK_BEFORE_RE = re.compile(
    r"(?:\b(?:at|by|until|till|til|from|on|between|opens?|closes?|tonight|today|tomorrow)"
    r"|\b(?:mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)(?:day)?\.?,?"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?(?:\s+\d{4},?)?"
    r"|\d{4}-\d{2}-\d{2}T?|\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4},?)\s*$",
    re.I)
_DAYS_PREFIX_RE = re.compile(r"(?<![\w.:])(\d{1,3})\s*(?:d|days?)\s*,?\s*$", re.I)

_UNIT = r"(?:days?|d|hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)"
_UNIT_SEQ_RE = re.compile(
    rf"(?<![\w.:$£€₹-])\d{{1,4}}\s*{_UNIT}\b(?:(?:\s*,\s*|\s+and\s+|\s*:\s*|\s+)\d{{1,4}}\s*{_UNIT}\b)*", re.I)
_UNIT_COMP_RE = re.compile(rf"(\d{{1,4}})\s*({_UNIT})\b", re.I)
_UNIT_SECONDS = (("d", 86400), ("h", 3600), ("m", 60), ("s", 1))
# single-component unit readings ("Ends in 3 hours", "2 hours left") need a countdown phrase around them
_SINGLE_BEFORE_RE = re.compile(r"\b(?:ends?|ending|expires?|expiring|closes?|finishes|over)\s+in\s+(?:just\s+|only\s+)?$", re.I)
_SINGLE_AFTER_RE = re.compile(r"^\s*(?:left|remaining|to\s+go)\b", re.I)
# unit readings in these contexts are not countdowns (delivery times, policies, elapsed time)
_NOT_COUNTDOWN_BEFORE_RE = re.compile(
    r"\b(?:deliver(?:y|ed|s)?|ships?|shipping|shipped|dispatch(?:ed|es)?|arriv(?:e|es|al)|returns?|refunds?|"
    r"warranty|guarantee|trial|every|per|posted|updated|joined)\b[^.!?]{0,25}$", re.I)
_NOT_COUNTDOWN_AFTER_RE = re.compile(r"^\s*(?:ago|old)\b", re.I)


def _unit_seconds(u: str) -> int:
    c = u.lower()[0]
    for k, v in _UNIT_SECONDS:
        if c == k:
            return v
    raise ValueError(u)


def _countdown_readings(text: str) -> list[tuple[int, int, int]]:
    """[(position, remaining_seconds, resolution_seconds)] for countdown displays in ``text``."""
    text = text or ""
    out: list[tuple[int, int, int]] = []
    consumed: list[tuple[int, int]] = []
    has_cue = bool(_CUE_RE.search(text))

    for m in _COLON_RE.finditer(text):
        parts = [int(p) for p in m.group(1).split(":")]
        before, after = text[:m.start()], text[m.end():]
        if _CLOCK_AFTER_RE.match(after) or _CLOCK_BEFORE_RE.search(before):
            continue
        if len(parts) == 2:              # MM:SS - only with a countdown cue
            if not has_cue or parts[1] >= 60:
                continue
            secs = parts[0] * 60 + parts[1]
        elif len(parts) == 3:            # HH:MM:SS
            if parts[1] >= 60 or parts[2] >= 60:
                continue
            secs = parts[0] * 3600 + parts[1] * 60 + parts[2]
        else:                            # DD:HH:MM:SS
            if parts[1] >= 24 or parts[2] >= 60 or parts[3] >= 60:
                continue
            secs = parts[0] * 86400 + parts[1] * 3600 + parts[2] * 60 + parts[3]
        start = m.start()
        dm = _DAYS_PREFIX_RE.search(before)
        if dm and len(parts) in (2, 3):
            secs += int(dm.group(1)) * 86400
            start = dm.start()
        consumed.append((start, m.end()))
        out.append((start, secs, 1))

    for m in _UNIT_SEQ_RE.finditer(text):
        if any(s < m.end() and m.start() < e for s, e in consumed):
            continue
        comps = [(int(n), _unit_seconds(u)) for n, u in _UNIT_COMP_RE.findall(m.group(0))]
        sizes = [s for _, s in comps]
        if any(a <= b for a, b in zip(sizes, sizes[1:])):      # units must strictly decrease (d > h > m > s)
            continue
        before, after = text[:m.start()], text[m.end():]
        if _NOT_COUNTDOWN_BEFORE_RE.search(before) or _NOT_COUNTDOWN_AFTER_RE.match(after):
            continue
        if len(comps) == 1 and not (_SINGLE_BEFORE_RE.search(before) or _SINGLE_AFTER_RE.match(after)):
            continue
        out.append((m.start(), sum(n * s for n, s in comps), min(sizes)))
    out.sort()
    return out


def parse_countdowns(text: str) -> list[int]:
    """Remaining seconds of countdown displays in ``text`` (clock times, prices and dates excluded).

    Supported: "HH:MM:SS", "DD:HH:MM:SS", "MM:SS" (only with a countdown cue word in the text),
    unit sequences ("20 HRS 28 MINS 30 SECS", "2 days 3 hours", "10h 5m"), a day prefix before a
    clock-style display ("Ends in 3d 04:10:00") and single-unit phrases tied to a countdown phrase
    ("Ends in 3 hours", "2 hours left").
    """
    return [v for _, v, _ in _countdown_readings(text)]


_STOCK_RES = (
    re.compile(r"\b(?:only|just)\s+(\d[\d,]*)\s+(?:(?:items?|units?|pieces?|pcs)\s+)?left\b", re.I),
    re.compile(r"(?<![\w.:,$£€₹])(\d[\d,]*)\s+(?:(?:items?|units?|pieces?|pcs)\s+)?left\s+in\s+stock\b", re.I),
    re.compile(r"(?<![\w.:,$£€₹])(\d[\d,]*)\s+(?:(?:items?|units?|pieces?|pcs)\s+)?remaining\b", re.I),
    re.compile(r"\b(?:only|just)\s+(\d[\d,]*)\s+(?:(?:items?|units?|pieces?|pcs)\s+)?in\s+stock\b", re.I),
)


def parse_stock_counts(text: str) -> list[int]:
    """Stock quantities stated as "only N left", "N left in stock", "N remaining", "only N in stock"."""
    found: dict[int, int] = {}
    for rx in _STOCK_RES:
        for m in rx.finditer(text or ""):
            found.setdefault(m.start(1), int(m.group(1).replace(",", "")))
    return [found[k] for k in sorted(found)]


_CUR_NORM = {"$": "$", "usd": "$", "£": "£", "gbp": "£", "€": "€", "eur": "€",
             "₹": "₹", "rs": "₹", "rs.": "₹", "inr": "₹"}
_AMOUNT = r"(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d{1,2}(?:,\d{2})+,\d{3}(?:\.\d{1,2})?|\d+,\d{2}(?![\d,])|\d+(?:\.\d{1,2})?)(?!\d)(?![,.]\d)"
_PRICE_BEFORE_RE = re.compile(
    r"(?:([$£€₹])|(?<![A-Za-z])(R[sS]\.?|INR|USD|EUR|GBP)(?![A-Za-z]))\s?" + _AMOUNT)
_PRICE_AFTER_RE = re.compile(r"(?<![\w.,])" + _AMOUNT + r"\s?(INR|USD|EUR|GBP)(?![A-Za-z])")


def _amount(s: str) -> float:
    if re.fullmatch(r"\d+,\d{2}", s):          # decimal comma (e.g. 12,99)
        return float(s.replace(",", "."))
    return float(s.replace(",", ""))


def parse_prices(text: str) -> list[tuple[str, float]]:
    """Currency amounts as (currency, amount); currency normalised to $, £, €, ₹ (Rs/INR -> ₹, USD -> $)."""
    text = text or ""
    hits: list[tuple[int, str, float]] = []
    spans: list[tuple[int, int]] = []
    for m in _PRICE_BEFORE_RE.finditer(text):
        cur = m.group(1) or m.group(2)
        hits.append((m.start(), _CUR_NORM[cur.lower()], _amount(m.group(3))))
        spans.append(m.span())
    for m in _PRICE_AFTER_RE.finditer(text):
        if any(s < m.end() and m.start() < e for s, e in spans):
            continue
        hits.append((m.start(), _CUR_NORM[m.group(2).lower()], _amount(m.group(1))))
    return [(c, a) for _, c, a in sorted(hits)]


# =====================================================================  journey analysis
@dataclass
class StageObservation:
    index: int
    name: str
    captured_at: Optional[str]
    findings: list = field(default_factory=list)
    texts: list[str] = field(default_factory=list)   # ALL analysed element texts of the stage
    # optional journey context (filled by the engine): declared (accept_steps, decline_steps, label) and the
    # stage's MCAI choice sets, used to report changing interaction difficulty and accept/reject paths
    declared_steps: Optional[tuple] = None
    choice_sets: list = field(default_factory=list)


def _parse_ts(s: Optional[str]) -> Optional[datetime]:
    if not s or not isinstance(s, str):
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))   # same parsing as the engine's ordering
    except ValueError:
        return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _skeleton(s: str) -> str:
    return re.sub(r"\d+", "#", _norm(s))


def _fmt_dur(sec: float) -> str:
    sec = int(round(sec))
    sign = "-" if sec < 0 else ""
    sec = abs(sec)
    d, rem = divmod(sec, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    return f"{sign}{d}d {h:02d}:{m:02d}:{s:02d}" if d else f"{sign}{h:02d}:{m:02d}:{s:02d}"


def _fmt_price(c: str, a: float) -> str:
    return f"{c}{int(a)}" if float(a).is_integer() else f"{c}{a:.2f}"


def _stage_texts(st: StageObservation) -> list[str]:
    out, seen = [], set()
    for t in list(st.texts or []) + [f.evidence for f in st.findings or []]:
        k = _norm(t)
        if k and k not in seen:
            seen.add(k)
            out.append(t)
    return out


def _finding_id(st: StageObservation, text: str) -> Optional[str]:
    k = _norm(text)
    for f in st.findings or []:
        if _norm(f.evidence) == k:
            return f.id
    for f in st.findings or []:
        fe = _norm(f.evidence)
        if len(fe) >= 4 and len(k) >= 4 and (fe in k or k in fe):
            return f.id
    return None


def _ref(st: StageObservation, text: str) -> EvidenceRef:
    return EvidenceRef(stage_index=st.index, finding_id=_finding_id(st, text), text=text)


def _q(s: str, n: int = 80) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    return "“" + (s if len(s) <= n else s[: n - 1] + "…") + "”"


_KIND_PRIORITY = {"observed_fact": 0, "observation": 1, "model_inference": 2}


def analyse_journey(stages: list[StageObservation]) -> tuple[TemporalResult, dict[int, list[Fact]], dict[int, list[Fact]]]:
    """Timeline of facts / observations / model inferences across stages.

    Returns (result, pressure_facts_by_stage, obstruction_facts_by_stage); the fact dicts are keyed by
    ``StageObservation.index`` and hold (strength, EvidenceRef) tuples for observed facts only.
    """
    stages = list(stages or [])
    if len(stages) < 2:
        return (TemporalResult(available=False, reason="Temporal analysis needs at least two stages/snapshots."), {}, {})

    pressure: dict[int, list[Fact]] = {s.index: [] for s in stages}
    obstruction: dict[int, list[Fact]] = {s.index: [] for s in stages}
    events: list[tuple[int, int, TimelineEvent]] = []   # (position, seq, event)

    # ---------------------------------------------------------------- ordering
    ts = [_parse_ts(s.captured_at) for s in stages]
    all_ts = all(t is not None for t in ts) and len({t.tzinfo is None for t in ts}) == 1
    reordered = False
    if all_ts:
        order = sorted(range(len(stages)), key=lambda i: ts[i])
        ordering = "timestamp"
        reordered = order != list(range(len(stages)))
    else:
        order = list(range(len(stages)))
        ordering = "submission_order"
    seq_stages = [stages[i] for i in order]
    seq_ts = [ts[i] for i in order] if all_ts else [None] * len(stages)

    def emit(pos: int, kind: str, category: str, desc: str, evidence: list[EvidenceRef], deceptive: bool = False,
             prev: Optional[str] = None, cur: Optional[str] = None) -> None:
        st = seq_stages[pos]
        events.append((pos, len(events), TimelineEvent(
            stage_index=st.index, stage_name=st.name, timestamp=st.captured_at, kind=kind, category=category,
            description=desc, evidence=evidence, deceptive_signal=deceptive, previous_state=prev, current_state=cur)))

    if reordered:
        emit(0, "observation", "ordering",
             "Stages are shown in capture-timestamp order, which differs from the submission order (submitted: "
             + ", ".join(s.name for s in stages) + "; chronological: " + ", ".join(s.name for s in seq_stages) + ").",
             [])
    elif not all_ts and any(t is not None for t in ts):
        emit(0, "observation", "ordering",
             "Not every stage has a parseable ISO-8601 capture time (or time zones are mixed), so submission order is "
             "used and no time-based inference is made.", [])
    order_note = " (chronological order differs from submission order)" if reordered else ""

    # ---------------------------------------------------------------- model inferences
    for pos, st in enumerate(seq_stages):
        for f in st.findings or []:
            if f.detection_method != "ml_classifier":
                continue
            if f.model_score is not None:
                pct = f.confidence_pct if getattr(f, "confidence_pct", None) is not None else round(f.model_score * 100)
                score_txt = f"{pct}% calibrated"
            else:
                score_txt = "score unavailable"
            label = getattr(f, "category_label", None) or f.category
            emit(pos, "model_inference", f.category, f"Classifier flagged {label} ({score_txt}).",
                 [EvidenceRef(stage_index=st.index, finding_id=f.id, text=f.evidence)])

    # ---------------------------------------------------------------- countdowns
    readings: list[list[tuple[int, int, str]]] = []      # per position: (value, resolution, text)
    for st in seq_stages:
        rs = []
        for t in _stage_texts(st):
            for _, v, res in _countdown_readings(t):
                rs.append((v, res, t))
        readings.append(rs)

    prev_pos: Optional[int] = None
    for pos in range(len(seq_stages)):
        if not readings[pos]:
            continue
        if prev_pos is None:
            prev_pos = pos
            continue
        _compare_countdowns(prev_pos, pos, readings, seq_stages, seq_ts, emit, pressure, order_note)
        prev_pos = pos

    # ---------------------------------------------------------------- pre-selected options
    seen_pre: set[str] = set()
    for pos, st in enumerate(seq_stages):
        current: set[str] = set()
        for f in st.findings or []:
            if not (f.detection_method == "dom_observation" and f.category == "preselected_option"):
                continue
            k = _norm(f.evidence)
            ref = EvidenceRef(stage_index=st.index, finding_id=f.id, text=f.evidence)
            if pos == 0:
                emit(pos, "observation", "preselected_option_present",
                     f"Pre-selected option observed at the first stage: {_q(f.evidence)}. Counted by the per-page DOM "
                     "observation; no temporal fact is added.", [ref])
            elif k not in seen_pre and k not in current:
                strong = "cost or consent" in (f.short_reason or "")
                strength = OBSERVED_FACT_STRENGTH if strong else OBSERVED_WEAK_FACT_STRENGTH
                emit(pos, "observed_fact", "new_preselected_option",
                     f"Pre-selected option {_q(f.evidence)} first appears at stage '{st.name}'; it was not pre-selected at "
                     f"any earlier stage{' (the label mentions a cost or consent)' if strong else ''}.", [ref], True,
                     prev="not pre-selected", cur="pre-selected")
                obstruction[st.index].append((strength, ref))
            current.add(k)
        seen_pre |= current

    # ---------------------------------------------------------------- repeated scarcity
    scar = [(pos, st, [f for f in st.findings or [] if f.detection_method == "ml_classifier" and f.category == "scarcity"])
            for pos, st in enumerate(seq_stages)]
    scar = [x for x in scar if x[2]]
    if len(scar) >= 2:
        refs = []
        for _, st, fs in scar:
            top = max(fs, key=lambda f: f.model_score if f.model_score is not None else -1.0)
            refs.append(EvidenceRef(stage_index=st.index, finding_id=top.id, text=top.evidence))
        emit(scar[1][0], "observation", "repeated_scarcity",
             f"Scarcity claims were flagged at {len(scar)} stages: " + ", ".join(f"'{st.name}'" for _, st, _ in scar)
             + ". Each claim is already scored at its own stage; this note adds no extra score.", refs)

    # ---------------------------------------------------------------- stock counts
    stock: list[list[tuple[int, str]]] = []
    for st in seq_stages:
        stock.append([(n, t) for t in _stage_texts(st) for n in parse_stock_counts(t)])
    prev_pos = None
    for pos in range(len(seq_stages)):
        if not stock[pos]:
            continue
        if prev_pos is not None:
            a = sorted({n for n, _ in stock[prev_pos]})
            b = sorted({n for n, _ in stock[pos]})
            if a != b:
                sa, sb = seq_stages[prev_pos], seq_stages[pos]
                emit(pos, "observation", "stock_count_change",
                     f"Stated stock count changed from {', '.join(map(str, a))} at '{sa.name}' to {', '.join(map(str, b))} "
                     f"at '{sb.name}'. Observed change only: restocks, other items or variants are legitimate "
                     "explanations, so no inference is made.",
                     [_ref(sa, t) for _, t in stock[prev_pos]] + [_ref(sb, t) for _, t in stock[pos]],
                     prev="stock " + ", ".join(map(str, a)), cur="stock " + ", ".join(map(str, b)))
        prev_pos = pos

    # ---------------------------------------------------------------- prices
    prices: list[list[tuple[tuple[str, float], str]]] = []
    for st in seq_stages:
        prices.append([(p, t) for t in _stage_texts(st) for p in parse_prices(t)])
    for pos in range(1, len(seq_stages)):
        if not prices[pos - 1] or not prices[pos]:
            continue
        a = {p for p, _ in prices[pos - 1]}
        b = {p for p, _ in prices[pos]}
        added = sorted(b - a, key=lambda p: (p[0], p[1]))
        removed = sorted(a - b, key=lambda p: (p[0], p[1]))
        if not added and not removed:
            continue
        sa, sb = seq_stages[pos - 1], seq_stages[pos]
        parts = []
        if added:
            parts.append("new amounts " + ", ".join(_fmt_price(*p) for p in added))
        if removed:
            parts.append("amounts no longer shown " + ", ".join(_fmt_price(*p) for p in removed))
        ev = [_ref(sa, t) for p, t in prices[pos - 1] if p in removed] + [_ref(sb, t) for p, t in prices[pos] if p in added]
        emit(pos, "observation", "price_change",
             f"Price amounts differ between '{sa.name}' and '{sb.name}': " + "; ".join(parts)
             + ". Observed change; may be legitimate, e.g. shipping/tax added at checkout.", ev,
             prev=", ".join(_fmt_price(*p) for p in sorted(a, key=lambda p: (p[0], p[1]))),
             cur=", ".join(_fmt_price(*p) for p in sorted(b, key=lambda p: (p[0], p[1]))))

    # ---------------------------------------------------------------- interaction difficulty (declared)
    # Step counts are DECLARED by the auditor (ClearCart cannot click through sites), so these are observations of
    # declared data, never observed facts. They already enter the CDI through MCAI's interaction dimension, so they
    # are not added to any fact list (no double counting).
    prev = None
    for pos, st in enumerate(seq_stages):
        if not st.declared_steps:
            continue
        acc, dec, label = (list(st.declared_steps) + [None])[:3]
        lbl = f" ({label})" if label else ""
        if prev is None:
            if dec > acc:
                emit(pos, "observation", "interaction_difficulty",
                     f"Declared choice path at '{st.name}'{lbl}: accepting takes {acc} step(s), declining takes {dec}. "
                     "Step counts are declared by the auditor, not observed by ClearCart.", [])
        else:
            pa, pacc, pdec = prev
            if dec != pdec or acc != pacc:
                trend = "harder" if (dec - acc) > (pdec - pacc) else "easier" if (dec - acc) < (pdec - pacc) else "equally hard"
                emit(pos, "observation", "interaction_difficulty",
                     f"Declining became {trend} relative to accepting between '{seq_stages[pa].name}' and '{st.name}'{lbl}: "
                     f"decline {pdec}→{dec} step(s), accept {pacc}→{acc}. Step counts are declared by the auditor.", [],
                     prev=f"accept {pacc} / decline {pdec} steps", cur=f"accept {acc} / decline {dec} steps")
        prev = (pos, acc, dec)

    # ---------------------------------------------------------------- accept/reject paths (MCAI per stage)
    def worst(st):
        scored = [cs for cs in (st.choice_sets or []) if getattr(cs, "mcai", None) is not None]
        return max(scored, key=lambda cs: cs.mcai) if scored else None

    def cs_prev_decline(cs):
        return f"decline: {_q(cs.decline.text, 60)}"

    def pair_refs(st, cs):
        out = [EvidenceRef(stage_index=st.index, finding_id=cs.decline.finding_id, text=cs.decline.text)]
        if cs.accept is not None:
            out.insert(0, EvidenceRef(stage_index=st.index, finding_id=cs.accept.finding_id, text=cs.accept.text))
        return out

    prev = None
    for pos, st in enumerate(seq_stages):
        cs = worst(st)
        if cs is None:
            continue
        acc_txt = _q(cs.accept.text, 60) if cs.accept else "no detected accept option"
        if prev is None:
            if pos > 0:
                emit(pos, "observation", "choice_asymmetry",
                     f"A competing accept/decline choice first appears at '{st.name}': {acc_txt} vs {_q(cs.decline.text, 60)} "
                     f"(MCAI {cs.mcai:.0f}/100, coverage {cs.coverage * 100:.0f}%).", pair_refs(st, cs))
        else:
            ppos, pcs = prev
            sa = seq_stages[ppos]
            if abs(cs.mcai - pcs.mcai) >= 10:
                emit(pos, "observation", "choice_asymmetry",
                     f"Choice asymmetry {'rose' if cs.mcai > pcs.mcai else 'fell'} from {pcs.mcai:.0f} at '{sa.name}' to "
                     f"{cs.mcai:.0f} at '{st.name}' (worst accept/decline pair per stage).",
                     pair_refs(sa, pcs) + pair_refs(st, cs), prev=f"MCAI {pcs.mcai:.0f}", cur=f"MCAI {cs.mcai:.0f}")
            if _norm(cs.decline.text) != _norm(pcs.decline.text):
                emit(pos, "observation", "decline_path_change",
                     f"The decline option changed between '{sa.name}' ({_q(pcs.decline.text, 60)}) and '{st.name}' "
                     f"({_q(cs.decline.text, 60)}).", pair_refs(sa, pcs) + pair_refs(st, cs),
                     prev=cs_prev_decline(pcs), cur=cs_prev_decline(cs))
        prev = (pos, cs)

    events.sort(key=lambda x: (x[0], _KIND_PRIORITY[x[2].kind], x[1]))
    result = TemporalResult(available=True, ordering=ordering, events=[e for _, _, e in events])
    return result, pressure, obstruction


def _compare_countdowns(pa: int, pb: int, readings, seq_stages, seq_ts, emit, pressure, order_note: str) -> None:
    A, B = readings[pa], readings[pb]
    sa, sb = seq_stages[pa], seq_stages[pb]
    elapsed = (seq_ts[pb] - seq_ts[pa]).total_seconds() if seq_ts[pa] is not None and seq_ts[pb] is not None else None

    # pair readings: same wording (digits masked) if possible, else conservative all-vs-all
    pairs = []   # (a_value, a_res, a_text, b_value, b_res, b_text, single)
    sk_a: dict[str, list] = {}
    sk_b: dict[str, list] = {}
    for r in A:
        sk_a.setdefault(_skeleton(r[2]), []).append(r)
    for r in B:
        sk_b.setdefault(_skeleton(r[2]), []).append(r)
    common = [k for k in sk_a if k in sk_b]
    if common:
        for k in common:
            ra, rb = max(sk_a[k]), max(sk_b[k])
            pairs.append((ra, rb, True))
    else:
        ra, rb = max(A), min(B)            # a reset needs EVERY later reading above EVERY earlier one
        pairs.append((ra, rb, len(A) == 1 and len(B) == 1))

    for ra, rb, comparable in pairs:
        a, res_a, ta = ra
        b, res_b, tb = rb
        m = max(1, res_a, res_b)
        tol = COUNTDOWN_TOLERANCE_SECONDS + max(res_a, res_b)
        ev = [_ref(sa, ta), _ref(sb, tb)]
        head = (f"Countdown read {_fmt_dur(a)} at '{sa.name}' ({_q(ta)}) and {_fmt_dur(b)} at the later stage "
                f"'{sb.name}' ({_q(tb)})")
        increased = b > a + m
        slow = elapsed is not None and b > a - elapsed + tol
        if increased or slow:
            parts = [head + "."]
            if increased:
                parts.append(f"The later observation showed MORE time remaining (+{_fmt_dur(b - a)}), so the countdown "
                             "did not keep running down between the two observations.")
            if elapsed is not None:
                expected = a - elapsed
                if expected > 0:
                    parts.append(f"The captures are {_fmt_dur(elapsed)} apart; a countdown running in real time would show "
                                 f"about {_fmt_dur(expected)} (tolerance {COUNTDOWN_TOLERANCE_SECONDS} s), but "
                                 f"{_fmt_dur(b)} was shown" + ("." if increased else
                                                               ": it did not run down at the real-time rate."))
                else:
                    parts.append(f"The captures are {_fmt_dur(elapsed)} apart, so a countdown running in real time would "
                                 f"have reached zero, but {_fmt_dur(b)} was shown.")
            emit(pb, "observed_fact", "countdown_reset", " ".join(parts) + order_note, ev, True,
                 prev=f"{_fmt_dur(a)} remaining", cur=f"{_fmt_dur(b)} remaining")
            pressure[sb.index].append((OBSERVED_FACT_STRENGTH, _ref(sb, tb)))
            continue
        if not comparable:
            continue
        if elapsed is not None:
            expected = a - elapsed
            if b >= expected - tol:
                emit(pb, "observation", "countdown_consistent",
                     head + f"; the captures are {_fmt_dur(elapsed)} apart, so the reading is consistent with a "
                     f"countdown running in real time (tolerance {COUNTDOWN_TOLERANCE_SECONDS} s)." + order_note, ev,
                 prev=f"{_fmt_dur(a)} remaining", cur=f"{_fmt_dur(b)} remaining")
            else:
                emit(pb, "observation", "countdown_decreased",
                     head + f"; the captures are {_fmt_dur(elapsed)} apart, so the reading fell faster than real time "
                     "(it may be a different timer). No inference is made." + order_note, ev,
                 prev=f"{_fmt_dur(a)} remaining", cur=f"{_fmt_dur(b)} remaining")
        elif b < a:
            emit(pb, "observation", "countdown_consistent",
                 head + ": the later reading is lower, consistent with a running countdown. No capture timestamps, so "
                 "the rate was not checked.", ev,
                 prev=f"{_fmt_dur(a)} remaining", cur=f"{_fmt_dur(b)} remaining")
        else:
            emit(pb, "observation", "countdown_unchanged",
                 head + ": the reading is unchanged. Without capture timestamps the elapsed time is unknown, so no "
                 "inference is made.", ev,
                 prev=f"{_fmt_dur(a)} remaining", cur=f"{_fmt_dur(b)} remaining")
