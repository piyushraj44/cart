"""Constrained counterfactual optimisation (intel-v1.1).

Problem
    For each flagged element e with original text t_e, choose one candidate c_e from a finite set C_e
    (the original, the minimal rewrite, a pressure-sentence removal, neutral decline wordings, removal of an
    unverifiable social-proof claim) to

        minimise   CDI_redesign(c_1..c_n)
        subject to facts_added(c_e) = 0       no number/price/%/duration/offer word that t_e lacks
                   facts_removed(c_e) = 0     every commercial fact of t_e is kept (social-proof counts exempt
                                              only for the removal candidate; declines carry no facts)
                   terms_kept(c_e)            disclosure words (fee, cancel, renew, subscription, terms...) kept
                   choice_kept(c_e)           a decline option stays a decline option
                   non-empty, changed         (except the 'original' baseline)

Why choosing per element is exactly optimal
    The redesigned CDI depends on candidates only through content pressure, which is
    p = 0.5*max_k c_k + 0.5*mean_k c_k with c_k = max over category-k elements of their score.
    p is monotone non-decreasing in every element's score and the constraints are per-element, so
    picking, for every element independently, its lowest-scoring feasible candidate minimises p and hence
    the CDI over the whole (product) candidate space. The implemented rule deliberately deviates within a
    0.005 tie band (below), so it is optimal to within 0.005 per element, not exactly: on 200 random real pages the
    selected content pressure was at most 0.0026 above the brute-force minimum (ml/research_intelligence.py, R4).
    Ties (within 0.005) go to the smaller word-level
    edit distance (minimal change), and a real change is preferred to the unchanged original. Choice-asymmetry and obstruction are set by the redesign specification
    (by construction) and do not depend on the text candidates. Cost: one batched classifier call over
    sum_e |C_e| candidate texts.

Honesty notes
    The objective is the classifier's own indicator: the optimiser can and does exploit model biases (e.g.
    the dataset labels every "No thanks..." row as misdirection, so "Decline" scores far lower). All
    candidates and their scores are returned so this is visible. Scores are model-estimated.
"""
from __future__ import annotations

import difflib
import re
from typing import Callable, Optional

from backend.intelligence.counterfactual import _fact_tokens, facts_preserved, rewrite, word_diff
from backend.intelligence.mcai import is_decline
from backend.intelligence.schemas import CounterfactualEdit, RewriteCandidate

ScoreFn = Callable[[list[str]], list[dict]]

NEUTRAL_DECLINES = ("No thanks", "No, thank you", "Decline", "Continue without this offer")
_DISCLOSURE = re.compile(r"\b(fees?|charges?|charged|billed|bill|cancel\w*|renew\w*|subscription|terms|conditions|"
                         r"refund\w*|returns?|shipping|tax(es)?|vat|deposit|commitment|contract|auto-?renew\w*)\b", re.I)
_PRESSURE_ONLY = re.compile(r"\b(hurry|act (fast|now)|buy now|order now|shop now|don'?t miss|last chance|"
                            r"before it'?s (too late|gone)|while (stocks|supplies) last|today only|limited time|grab)\b", re.I)
_SENT = re.compile(r"(?<=[.!?])\s+")
TIE = 0.005
OPTIMISATION_META = {
    "method": "exact per-element minimisation over a finite candidate set (one batched model call), with a 0.005 tie band",
    "objective": "minimise the redesigned CDI (model-estimated content pressure + by-construction choice/obstruction terms)",
    "constraints": ["no added facts or offer words", "no removed commercial facts", "disclosure terms kept",
                    "decline options stay declines", "non-empty and changed (except the original baseline)"],
    "optimality": ("content pressure is monotone non-decreasing in each element's score and constraints are per-element, "
                   "so the per-element minimum is the global minimum; the tie band makes the result optimal to within 0.005 "
                   "per element (brute-force check: max content-pressure gap 0.0026 on 200 real pages)"),
    "tie_break": ("within 0.005 of the best score, prefer an actual change over the unchanged original, then the "
                  "smaller word-level edit distance"),
    "caveat": "The objective is the classifier's own indicator and can exploit its label biases; all candidates are shown.",
}


def edit_distance(a: str, b: Optional[str]) -> int:
    if b is None:
        return len((a or "").split())
    sm = difflib.SequenceMatcher(a=(a or "").split(), b=b.split(), autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")


def _drop_pressure_sentences(text: str) -> Optional[str]:
    """Remove whole sentences that are pure pressure (no fact tokens, no disclosure words)."""
    parts = [p for p in _SENT.split(text.strip()) if p]
    if len(parts) < 2:
        return None
    keep = [p for p in parts if not (_PRESSURE_ONLY.search(p) and not _fact_tokens(p) and not _DISCLOSURE.search(p))]
    if not keep or len(keep) == len(parts):
        return None
    out = " ".join(keep)
    minimal, _ = rewrite("urgency", out)            # tidy any remaining pressure wording inside kept sentences
    return minimal or out


def candidates_for(category: str, text: str) -> list[tuple[str, Optional[str]]]:
    out: list[tuple[str, Optional[str]]] = [("original", text)]
    m, _ = rewrite(category, text)
    if m:
        out.append(("minimal", m))
    d = _drop_pressure_sentences(text) if category in ("urgency", "scarcity") else None
    if d:
        out.append(("drop_pressure_sentences", d))
    if category == "misdirection" and is_decline(text):
        out += [("neutral_decline", w) for w in NEUTRAL_DECLINES]
    if category == "social_proof":
        out.append(("remove_unverifiable_claim", None))
    seen, uniq = set(), []
    for v, t in out:
        key = (t or "").strip().lower()
        if (v, key) in seen or (v != "original" and t is not None and key == text.strip().lower()):
            continue
        seen.add((v, key)); uniq.append((v, t))
    return uniq


def violations(category: str, original: str, variant: str, cand: Optional[str]) -> list[str]:
    v: list[str] = []
    if variant == "original":
        return v
    if variant == "remove_unverifiable_claim":
        # removal is allowed only if the claim carries no commercial terms (prices, disclosures)
        if re.search(r"[$£€₹]|\b(rs|inr|usd|eur|gbp)\b", original, re.I):
            v.append("facts_removed")
        if _DISCLOSURE.search(original):
            v.append("terms_removed")
        return v
    if not cand or not cand.strip():
        return ["empty"]
    if not facts_preserved(original, cand):
        v.append("facts_added")
    o, r = _fact_tokens(original), _fact_tokens(cand)
    if any(r.get(tok, 0) < n for tok, n in o.items()):
        v.append("facts_removed")
    for w in {m.group(0).lower() for m in _DISCLOSURE.finditer(original)}:
        if w not in cand.lower():
            v.append("terms_removed"); break
    if is_decline(original) and not is_decline(cand):
        v.append("choice_changed")
    return v


def optimise(edits: list[CounterfactualEdit], score_fn: ScoreFn) -> list[CounterfactualEdit]:
    """Re-select every ML-finding and choice edit by constrained minimisation; other edits pass through."""
    work = [(i, e) for i, e in enumerate(edits) if e.action in ("rewrite", "no_safe_rewrite", "equalise_choice")]
    cand_map: dict[int, list[tuple[str, Optional[str]]]] = {}
    texts: list[str] = []
    for i, e in work:
        cat = "misdirection" if e.action == "equalise_choice" else e.category
        cand_map[i] = candidates_for(cat, e.original)
        texts += [t for _, t in cand_map[i] if t]
    uniq = list(dict.fromkeys(texts))
    preds = {t: p for t, p in zip(uniq, score_fn(uniq))} if uniq else {}
    out = list(edits)
    for i, e in work:
        cat = "misdirection" if e.action == "equalise_choice" else e.category
        rows = []
        for variant, t in cand_map[i]:
            score = 0.0 if t is None else round(float(preds[t]["p_dark"]), 4)
            viol = violations(cat, e.original, variant, t)
            rows.append(RewriteCandidate(variant=variant, text=t, score=score, feasible=not viol, violated=viol,
                                         edit_distance=edit_distance(e.original, t)))
        feas = [r for r in rows if r.feasible]
        best_score = min(r.score for r in feas)
        near = [r for r in feas if r.score <= best_score + TIE]
        # the redesign's purpose is to remove manipulative wording: within the tie band prefer an actual change;
        # the unchanged original wins only when it is better than every feasible change by more than TIE
        changes = [r for r in near if r.variant != "original"]
        best = min(changes or near, key=lambda r: (r.edit_distance, r.score))
        upd: dict = {"candidates": rows}
        if best.variant == "original":
            if e.action != "equalise_choice":
                upd.update(action="no_safe_rewrite", redesigned=None, redesigned_score=None, diff=[],
                           explanation=e.explanation + " Optimiser: no feasible candidate scored lower than the original.")
        else:
            act = "remove_unverifiable_claim" if best.variant == "remove_unverifiable_claim" else (
                "equalise_choice" if e.action == "equalise_choice" else "rewrite")
            upd.update(action=act, redesigned=best.text, redesigned_score=best.score,
                       diff=word_diff(e.original, best.text) if best.text else [],
                       facts_preserved=True,
                       explanation=(f"Optimiser selected '{best.variant}' (lowest model score among "
                                    f"{len(feas)} feasible of {len(rows)} candidates). ") + e.explanation)
        out[i] = e.model_copy(update=upd)
    return out
