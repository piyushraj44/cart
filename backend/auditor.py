"""Turns extracted candidates + model predictions into an evidence-backed audit."""
from __future__ import annotations

import re
import time
import uuid
from datetime import datetime, timezone

from backend.claim_check import check_claim
from backend.extractors import Candidate
from backend.recommendations import (CATEGORY_LABELS, RECOMMENDATIONS, SHORT_REASONS, explanation_for,
                                     preselected_explanation, suggest_rewrite)
from backend.schemas import (AnalysedElement, AuditResponse, AuditSummary, BBox, CategoryBreakdown,
                             ClaimCheck, Finding, SimilarExample, SourceLocation)

DISCLAIMER = ("ClearCart is an assistive screening tool. Scores are calibrated model estimates that an element matches a "
              "dark-pattern category learned from the ec-darkpattern dataset - they are indications of potentially deceptive "
              "design, not proof of fraud or a legal finding. A human reviewer should verify every finding.")
HIGH_RISK = 0.90
_COST_OR_CONSENT = re.compile(
    r"[$£€¥₹]|\b(insurance|protect(ion)?|warranty|subscri\w*|member\w*|newsletter|e-?mails?|marketing|offers?|promotion\w*|"
    r"partners?|third.part\w*|donat\w*|add(-| )?on|extra|plan|premium|agree|consent|share)\b", re.I)
_ESSENTIAL = re.compile(r"\b(remember me|keep me (signed|logged) in|same as (billing|shipping)|show password)\b", re.I)
_ESSENTIAL_CONSENT = re.compile(r"\b(strictly necessary|essential cookies?|required cookies?)\b", re.I)
_OPTIONAL_CONSENT = re.compile(r"\b(marketing|advertising|analytics|personaliz\w*|partners?|third.part\w*)\b", re.I)


def display_pct(score: float) -> int:
    """Floor and cap at 99: a statistical model is never 100% certain."""
    return min(99, int(score * 100))


def confidence_statement(score: float, category_label: str) -> str:
    return f"ClearCart is {display_pct(score)}% sure this is a fraud/scam (dark pattern) tactic: {category_label}."


def _loc(c: Candidate) -> SourceLocation:
    return SourceLocation(
        source_type=c.source_type, char_start=c.char_start, char_end=c.char_end, line=c.line,
        css_selector=c.css_selector, html_tag=c.html_tag, html_snippet=c.html_snippet,
        bbox=BBox(**c.bbox) if c.bbox else None, ocr_confidence=c.ocr_confidence,
    )


def _risk(score: float) -> str:
    return "high" if score >= HIGH_RISK else "medium"


QUOTE_NOTE = "quoted example — treated as a mention, not a claim"
OOD_NOTE = "uncertain — unlike the training data"
PROMOTION_NOTE = "uncertain — promotional terms are present in nearby HTML; claim not verified"
NEUTRAL_NOTE = "uncertain — a neutral choice label alone does not establish manipulation"
CHECKABLE_NOTE = "checkable promotion — states specific terms and no vague pressure; terms not verified"
# A deadline or price comparison can make an urgency/scarcity claim checkable. It says nothing about whether
# social proof or a confirmshaming choice is honest, so those categories are never withheld by this gate.
CLAIM_GATE_CATEGORIES = ("urgency", "scarcity")
_NEUTRAL_CHOICE = re.compile(r"\s*(?:no thanks|not now|continue shopping|skip offer)[.!]?\s*", re.I)
_OOD_CACHE: dict[int, dict | None] = {}


def ood_config(clf) -> dict | None:
    """tau_ood for this classifier (validation rule in ml/model.py), computed once per loaded bundle."""
    key = id(clf.b)
    if key not in _OOD_CACHE:
        from ml.model import ood_threshold_from_val
        try:
            _OOD_CACHE[key] = ood_threshold_from_val(clf)
        except Exception:          # dataset missing / unreadable: the gate is reported as unavailable, never guessed
            _OOD_CACHE[key] = None
    return _OOD_CACHE[key]


class Auditor:
    """Apply calibrated model decisions, observable DOM findings, and conservative review gates.

    The quote/OOD and HTML context/neutral-choice gates can be disabled separately
    for paired evaluation. Disabling all gates reproduces the original text rule.
    """

    def __init__(self, clf, quote_gate: bool = True, ood_gate: bool = True,
                 promotion_context_gate: bool = True, neutral_choice_gate: bool = True,
                 essential_consent_gate: bool = True, claim_check_gate: bool = True) -> None:
        self.clf = clf
        self.threshold = clf.b.threshold
        self.quote_gate = quote_gate
        self.promotion_context_gate = promotion_context_gate
        self.neutral_choice_gate = neutral_choice_gate
        self.essential_consent_gate = essential_consent_gate
        self.claim_check_gate = claim_check_gate
        self.ood = ood_config(clf) if ood_gate else None
        self.ood_gate_requested = ood_gate

    def apply_gates(self, cands: list[Candidate], preds: list[dict], tau: float) -> None:
        """Withhold flagged scores that a false-positive gate explains (in place). Sets p['gate'] and clears p['flagged'].

        Order: quote-aware, out-of-distribution, structured promotion, neutral choice.
        Only elements that crossed tau are gated,
        so nothing that was below the threshold can change, and every gated element is reported (uncertain_elements).
        """
        from ml.model import max_train_similarity, quote_masked
        from backend.scoring import predict as _predict
        if preds:
            sims = max_train_similarity(self.clf.b, [p["_emb"] for p in preds])
            for p, s in zip(preds, sims if sims is not None else [None] * len(preds)):
                p["max_train_similarity"] = None if s is None else float(s)
        if self.quote_gate:
            masked = {i: m for i, p in enumerate(preds) if p["flagged"] and (m := quote_masked(cands[i].text))}
            if masked:
                mp = _predict(self.clf, list(masked.values()))
                for (i, m), q in zip(masked.items(), mp):
                    if q["p_dark"] < tau:
                        preds[i].update(flagged=False, gate="quoted_mention", masked_text=m, masked_p_dark=q["p_dark"])
        if self.ood is not None:
            t = self.ood["threshold"]
            for p in preds:
                if p["flagged"] and p.get("max_train_similarity") is not None and p["max_train_similarity"] < t:
                    p.update(flagged=False, gate="out_of_distribution")
        if self.promotion_context_gate:
            for c, p in zip(cands, preds):
                if p["flagged"] and c.source_type == "html" and c.extra.get("structured_promotion"):
                    p.update(flagged=False, gate="structured_promotion")
        if self.neutral_choice_gate:
            for c, p in zip(cands, preds):
                if (p["flagged"] and c.source_type == "html" and c.html_tag in ("button", "a")
                        and _NEUTRAL_CHOICE.fullmatch(c.text)):
                    p.update(flagged=False, gate="neutral_choice")
        # claim specificity: reported on every element that crossed tau; withholds checkable urgency/scarcity claims
        for c, p in zip(cands, preds):
            if p["flagged"] or p.get("gate"):
                p["claim_check"] = check_claim(c.text).as_dict()
            if (self.claim_check_gate and p["flagged"] and p["category"] in CLAIM_GATE_CATEGORIES
                    and p["claim_check"]["verdict"] == "checkable"):
                p.update(flagged=False, gate="checkable_claim")

    def _uncertain(self, c: Candidate, p: dict) -> AnalysedElement:
        q = p["gate"] == "quoted_mention"
        sim = p.get("max_train_similarity")
        return AnalysedElement(
            text=c.text, p_dark=round(p["p_dark"], 4), flagged=False, source_location=_loc(c), gate=p["gate"],
            gate_note=(QUOTE_NOTE if q else PROMOTION_NOTE if p["gate"] == "structured_promotion"
                       else NEUTRAL_NOTE if p["gate"] == "neutral_choice"
                       else CHECKABLE_NOTE if p["gate"] == "checkable_claim" else OOD_NOTE),
            masked_text=p.get("masked_text"),
            masked_p_dark=round(p["masked_p_dark"], 4) if q else None,
            max_train_similarity=None if sim is None else round(sim, 4),
            ood_threshold=None if q or self.ood is None else round(self.ood["threshold"], 4),
            claim_check=ClaimCheck(**p["claim_check"]) if p.get("claim_check") else None)

    def audit(self, cands: list[Candidate], input_type: str, warnings: list[str] | None = None,
              image_size: dict | None = None, threshold: float | None = None) -> AuditResponse:
        """`threshold` overrides the text threshold (screenshots use a stricter, DEV-selected OCR threshold)."""
        t0 = time.perf_counter()
        tau = self.threshold if threshold is None else threshold
        warnings = list(warnings or [])
        if len(cands) > 2000:
            warnings.append(f"Input produced {len(cands)} elements; only the first 2000 were analysed.")
            cands = cands[:2000]
        texts = [c.text for c in cands]
        from backend.scoring import predict as _predict      # request-scoped cache when active
        preds = _predict(self.clf, texts) if texts else []
        for p in preds:
            p["flagged"] = p["p_dark"] >= tau and not p.get("abstained", False)
        abstained = [cands[i].text for i, p in enumerate(preds) if p.get("abstained")]
        if abstained:
            ex = "; ".join(f"“{t[:40]}”" for t in abstained[:3])
            warnings.append(f"{len(abstained)} element(s) contain no words (only numbers, prices or symbols, e.g. {ex}) and were "
                            "not classified: the text model has no linguistic evidence for them, so it abstains instead of guessing.")
        self.apply_gates(cands, preds, tau)
        if self.ood_gate_requested and self.ood is None:
            warnings.append("The out-of-distribution check is unavailable (no training neighbour bank or validation split), "
                            "so scores on text unlike the training data are not withheld.")

        findings: list[Finding] = []
        analysed: list[AnalysedElement] = []
        uncertain: list[AnalysedElement] = []
        uncertain_boxes: list[dict] = []
        flagged_blocks_covered: list[dict] = []

        # screenshot: line-level candidates first, merged blocks only when none of their lines was flagged
        order = sorted(range(len(cands)), key=lambda i: 0 if cands[i].extra.get("unit") != "block" else 1)
        n_elements = 0
        quiet_blocks: list[int] = []         # merged blocks below the threshold and not gated
        for i in order:
            c, p = cands[i], preds[i]
            if p.get("abstained"):
                continue
            is_block = c.extra.get("unit") == "block"
            if not is_block:
                n_elements += 1
            if is_block:
                if p.get("gate") and not any(_inside(b, c.bbox) for b in flagged_blocks_covered + uncertain_boxes):
                    uncertain.append(self._uncertain(c, p))
                if not p["flagged"] and not p.get("gate"):
                    quiet_blocks.append(i)
                if not p["flagged"] or any(_inside(b, c.bbox) for b in flagged_blocks_covered):
                    continue
            if p.get("gate") and not is_block:
                uncertain.append(self._uncertain(c, p))
                if c.bbox:
                    uncertain_boxes.append(c.bbox)
            elif p["flagged"]:
                findings.append(self._ml_finding(c, p, len(findings)))
                if c.bbox:
                    flagged_blocks_covered.append(c.bbox)
            elif not is_block:
                analysed.append(AnalysedElement(text=c.text, p_dark=round(p["p_dark"], 4), flagged=False, source_location=_loc(c)))

            # observable DOM property, reported separately from the model and never scored by it
            essential_consent = (self.essential_consent_gate and c.extra.get("disabled")
                                 and bool(_ESSENTIAL_CONSENT.search(c.text))
                                 and not _OPTIONAL_CONSENT.search(c.text))
            if (c.preselected and c.input_type in ("checkbox", "radio")
                    and not _ESSENTIAL.search(c.text) and not essential_consent):
                cost = bool(_COST_OR_CONSENT.search(c.text))
                if c.input_type == "checkbox" or cost:
                    findings.append(self._dom_finding(c, cost, len(findings)))

        # A merged OCR paragraph can score higher than any of its lines. When one sets the page's highest score,
        # list it so the number in the summary statement can be traced to its text.
        if quiet_blocks:
            top = max(quiet_blocks, key=lambda i: preds[i]["p_dark"])
            if preds[top]["p_dark"] > max((a.p_dark for a in analysed), default=-1.0):
                c = cands[top]
                analysed.append(AnalysedElement(text=c.text, p_dark=round(preds[top]["p_dark"], 4), flagged=False,
                                                source_location=_loc(c)))
        findings.sort(key=lambda f: (f.model_score is None, -(f.model_score or 0)))
        analysed.sort(key=lambda a: a.p_dark)
        uncertain.sort(key=lambda a: -a.p_dark)
        warnings += _gate_warnings(uncertain, self.ood)
        # gated scores do not drive the page-level gauge: a 99% score withheld as a mention must not read as "99% risk"
        scored = [p for p in preds if not p.get("abstained")]
        summary = self._summary(findings, n_elements, [p for p in scored if not p.get("gate")], tau,
                                any_scored=bool(scored), n_uncertain=len(uncertain))
        summary.tier_counts = {"observed": sum(f.evidence_tier == "observed" for f in findings),
                               "pressure_claim": sum(f.evidence_tier == "pressure_claim" for f in findings),
                               "review": len(uncertain)}
        return AuditResponse(
            audit_id=uuid.uuid4().hex[:12], input_type=input_type,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            processing_ms=int((time.perf_counter() - t0) * 1000), summary=summary, findings=findings,
            benign_elements=analysed[:50], disclaimer=DISCLAIMER, image_size=image_size, warnings=warnings,
            uncertain_elements=uncertain[:50],
        )

    def _ml_finding(self, c: Candidate, p: dict, n: int) -> Finding:
        cat = p["category"]
        terms = self.clf.key_terms(c.text, cat)
        score = round(p["p_dark"], 4)
        label = CATEGORY_LABELS[cat]
        expl = explanation_for(cat, terms)
        if c.source_type == "screenshot_ocr":
            expl += " (Text read by OCR from the screenshot and classified by the text model; visual layout was not analysed.)"
        if p["category_share"] < 0.6:
            expl += f" The model is less certain about the exact category ({round(p['category_share']*100)}% of the dark-pattern probability)."
        return Finding(
            id=f"f{n+1}", category=cat, category_label=label, evidence=c.text, explanation=expl,
            detection_method="ml_classifier", model_score=score, confidence_pct=display_pct(score),
            confidence_statement=confidence_statement(score, label),
            category_probabilities={k: round(v, 4) for k, v in p["probabilities"].items()},
            risk_level=_risk(score), key_terms=terms,
            similar_examples=[SimilarExample(**s) for s in self.clf.neighbors(p["_emb"])],
            source_location=_loc(c), ethical_recommendation=RECOMMENDATIONS[cat],
            suggested_rewrite=suggest_rewrite(cat, c.text), short_reason=SHORT_REASONS[cat],
            evidence_tier="pressure_claim",
            claim_check=ClaimCheck(**(p.get("claim_check") or check_claim(c.text).as_dict())),
        )

    def _dom_finding(self, c: Candidate, cost: bool, n: int) -> Finding:
        return Finding(
            id=f"f{n+1}", category="preselected_option", category_label=CATEGORY_LABELS["preselected_option"],
            evidence=c.text, explanation=preselected_explanation(c.text, cost), detection_method="dom_observation",
            model_score=None, confidence_pct=None,
            confidence_statement=f"Directly observed in the HTML: this {c.input_type} is pre-selected (checked attribute). No model score - this is a factual DOM observation.",
            risk_level="medium" if cost else "low", source_location=_loc(c),
            ethical_recommendation=RECOMMENDATIONS["preselected_option"], suggested_rewrite=None,
            short_reason=SHORT_REASONS["preselected_option"] + (" (mentions a cost or consent)" if cost else ""),
            evidence_tier="observed",
        )

    def _summary(self, findings: list[Finding], n_elements: int, preds: list[dict], tau: float,
                 any_scored: bool | None = None, n_uncertain: int = 0) -> AuditSummary:
        ml = [f for f in findings if f.detection_method == "ml_classifier"]
        cats: dict[str, CategoryBreakdown] = {}
        for f in findings:
            cb = cats.setdefault(f.category, CategoryBreakdown(category=f.category, category_label=f.category_label, count=0, max_score=None))
            cb.count += 1
            if f.model_score is not None:
                cb.max_score = max(cb.max_score or 0, f.model_score)
        any_scored = bool(preds) if any_scored is None else any_scored
        overall_score = round(max((p["p_dark"] for p in preds), default=0.0), 4) if any_scored else None
        if any(f.risk_level == "high" for f in findings) or len(ml) >= 3:
            risk = "high"
        elif any(f.risk_level == "medium" for f in findings):
            risk = "medium"
        elif findings:
            risk = "low"
        else:
            risk = "none"
        no_concern = not findings
        if no_concern:
            stmt = (f"No suspicious patterns detected. The highest dark-pattern score among {n_elements} element(s) was "
                    f"{display_pct(overall_score or 0)}%, below the {round(tau*100)}% flagging threshold.")
            text = "No suspicious patterns detected. " + (f"All {n_elements} analysed element(s) scored below the threshold." if n_elements else "No analysable text was found in the input.")
            if n_uncertain:
                stmt = (f"No suspicious patterns detected. The highest score among elements that count as evidence was "
                        f"{display_pct(overall_score or 0)}%, below the {round(tau*100)}% flagging threshold.")
                text = (f"No suspicious patterns detected. {n_uncertain} element(s) scored above the threshold but were withheld "
                        "by a false-positive check (quoted example, structured or checkable promotion, neutral choice, or text unlike the "
                        "training data); review them under "
                        "uncertain elements.")
        else:
            top = max(ml, key=lambda f: f.model_score) if ml else None
            stmt = top.confidence_statement if top else findings[0].confidence_statement
            names = ", ".join(f"{c.category_label} ({c.count})" for c in sorted(cats.values(), key=lambda c: -c.count))
            text = (f"{len(findings)} potentially deceptive element(s) found among {n_elements} analysed: {names}. "
                    f"Overall risk is {risk}. Review each finding's evidence before drawing conclusions.")
        return AuditSummary(
            elements_analysed=n_elements, suspicious_findings=len(findings), categories_detected=sorted(cats.values(), key=lambda c: -c.count),
            overall_risk=risk, overall_score=overall_score, overall_confidence_statement=stmt, no_concern=no_concern, summary_text=text,
            risk_score=_gauge(ml, findings, any_scored), threshold=tau,
        )


def _gauge(ml: list[Finding], findings: list[Finding], any_scored: bool) -> int | None:
    """Gauge = highest calibrated score among FLAGGED model findings, so it is 0 when nothing is flagged.

    The highest score of any element (overall_score) stays in the statement text, but a sub-threshold score
    must not read as "82/100 risk". Only DOM observations (no model score): None, shown as "—".
    """
    if ml:
        return display_pct(max(f.model_score for f in ml))
    if findings:
        return None
    return 0 if any_scored else None


def _gate_warnings(uncertain: list[AnalysedElement], ood: dict | None) -> list[str]:
    out = []
    q = [u for u in uncertain if u.gate == "quoted_mention"]
    o = [u for u in uncertain if u.gate == "out_of_distribution"]
    sp = [u for u in uncertain if u.gate == "structured_promotion"]
    nc = [u for u in uncertain if u.gate == "neutral_choice"]
    cc = [u for u in uncertain if u.gate == "checkable_claim"]
    if q:
        ex = "; ".join(f"“{u.text[:60]}”" for u in q[:2])
        out.append(f"{len(q)} element(s) scored above the threshold only because of quoted example phrases inside explanatory "
                   f"text (e.g. {ex}). With the quoted spans masked they score below the threshold, so each is a {QUOTE_NOTE}. "
                   "They are listed under uncertain elements, not findings.")
    if o:
        ex = "; ".join(f"“{u.text[:60]}”" for u in o[:2])
        out.append(f"{len(o)} element(s) scored above the threshold but are unlike anything in the training data (similarity to "
                   f"the closest training text below {ood['threshold']:.2f}, e.g. {ex}). The calibrated percentage is only "
                   f"reliable for text resembling the training data, so each is marked {OOD_NOTE} and listed under uncertain "
                   "elements, not findings. Review them manually.")
    if sp:
        out.append(f"{len(sp)} promotional element(s) had nearby HTML terms (such as a stated deadline, price comparison, "
                   "or neutral alternative). Their claims cannot be verified from markup alone, so they are listed under "
                   "uncertain elements for review instead of findings.")
    if nc:
        out.append(f"{len(nc)} neutral choice label(s) gave no textual evidence of manipulation. They are listed under "
                   "uncertain elements for review instead of findings.")
    if cc:
        ex = "; ".join(f"“{u.text[:60]}”" for u in cc[:2])
        out.append(f"{len(cc)} promotional element(s) state checkable terms (a calendar deadline, reference price, code or "
                   f"delivery terms) with no vague pressure, e.g. {ex}. Genuine sales are usually worded like this, so they "
                   "are listed under uncertain elements for review instead of findings. ClearCart has not verified the terms.")
    return out


def _inside(a: dict, b: dict | None) -> bool:
    """True if box a overlaps box b substantially (a flagged line inside block b)."""
    if not b:
        return False
    x0, y0 = max(a["x"], b["x"]), max(a["y"], b["y"])
    x1, y1 = min(a["x"] + a["width"], b["x"] + b["width"]), min(a["y"] + a["height"], b["y"] + b["height"])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter > 0.5 * a["width"] * a["height"]
