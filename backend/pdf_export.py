"""Server-side PDF export of an audit (ReportLab, no external services)."""
from __future__ import annotations

import io
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from backend.schemas import AuditResponse

RISK_COLORS = {"high": "#b91c1c", "medium": "#b45309", "low": "#475569", "none": "#15803d"}


def render_audit_pdf(a: AuditResponse) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title=f"ClearCart audit {a.audit_id}", author="ClearCart")
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12.5)
    small = ParagraphStyle("s", parent=body, fontSize=7.5, leading=10, textColor=colors.HexColor("#475569"))
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=18, alignment=0, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4)
    s = a.summary
    els = [Paragraph("ClearCart audit report", h1),
           Paragraph(f"Audit {escape(a.audit_id)} · input: {escape(a.input_type)} · {escape(a.created_at)}", small), Spacer(1, 6)]
    risk = f'<font color="{RISK_COLORS[s.overall_risk]}"><b>Overall risk: {s.overall_risk.upper()}</b></font>'
    els += [Paragraph(risk, body), Paragraph(f"<b>{escape(s.overall_confidence_statement)}</b>", body), Paragraph(escape(s.summary_text), body), Spacer(1, 4)]
    stats = [["Elements analysed", "Suspicious findings", "Categories", "Processing"],
             [str(s.elements_analysed), str(s.suspicious_findings), str(len(s.categories_detected)), f"{a.processing_ms} ms"]]
    t = Table(stats, colWidths=[42 * mm] * 4)
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8.5), ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#475569")),
                           ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"), ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9"))]))
    els.append(t)
    if s.categories_detected:
        els.append(Paragraph("Category breakdown", h2))
        rows = [["Category", "Count", "Max model score"]] + [[c.category_label, str(c.count), f"{round(c.max_score*100)}%" if c.max_score is not None else "DOM observation"] for c in s.categories_detected]
        ct = Table(rows, colWidths=[80 * mm, 25 * mm, 45 * mm])
        ct.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8.5), ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#94a3b8"))]))
        els.append(ct)
    els.append(Paragraph("Findings" if a.findings else "Result", h2))
    if not a.findings:
        els.append(Paragraph('<font color="#15803d"><b>No suspicious patterns detected.</b></font>', body))
    for f in a.findings:
        loc = f.source_location
        where = (f"line {loc.line}, chars {loc.char_start}–{loc.char_end}" if loc.source_type == "text" else
                 f"selector <font face='Courier'>{escape(loc.css_selector or '')}</font>" if loc.source_type == "html" else
                 f"OCR box x={loc.bbox.x}, y={loc.bbox.y}, {loc.bbox.width}×{loc.bbox.height}px" if loc.bbox else "n/a")
        block = [Paragraph(f"<b>{escape(f.id)} · {escape(f.category_label)}</b> · risk {f.risk_level} · {'ML classifier' if f.detection_method == 'ml_classifier' else 'DOM observation'}", body),
                 Paragraph(f"Evidence: <i>“{escape(f.evidence)}”</i>", body),
                 Paragraph(escape(f.confidence_statement), body),
                 Paragraph(f"Where: {where}", small),
                 Paragraph(f"Why: {escape(f.explanation)}", body),
                 Paragraph(f"Ethical redesign: {escape(f.ethical_recommendation)}", body)]
        if f.suggested_rewrite:
            block.append(Paragraph(f"Suggested rewrite: “{escape(f.evidence)}” → “{escape(f.suggested_rewrite)}”", body))
        block.append(Spacer(1, 6))
        els.append(KeepTogether(block))
    if a.intelligence is not None:
        i = a.intelligence
        els += intelligence_blocks(i.cdi, i.mcai, i.counterfactual, i.fspm, i.temporal, i.config_version, body, small, h2)
    if a.warnings:
        els.append(Paragraph("Notes", h2))
        els += [Paragraph(escape(w), small) for w in a.warnings]
    els += [Spacer(1, 8), Paragraph(escape(a.disclaimer), small)]
    doc.build(els)
    return buf.getvalue()


def _tbl(rows, widths):
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#94a3b8")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1"))]))
    return t


def _fmt(v, nd=1, suffix=""):
    return "unavailable" if v is None else f"{v:.{nd}f}{suffix}"


def intelligence_blocks(cdi, mcai, cf, fspm, temporal, version, body, small, h2) -> list:
    """Shared PDF section for the intel-v1.0 indices (single audits and journeys)."""
    els = [Paragraph(f"Deception intelligence ({escape(version)})", h2),
           Paragraph("Interpretable screening indices built on the calibrated classifier and direct observations. "
                     "Evaluated only on synthetic, explicitly labelled benchmarks; not probabilities of fraud.", small)]
    if cdi.cdi is None:
        els.append(Paragraph("CDI: unavailable (no component could be computed).", body))
    else:
        els.append(Paragraph(f"<b>CDI {cdi.cdi:.1f}/100</b> · band {escape(cdi.band or '')} · data coverage {cdi.coverage*100:.0f}% · "
                             f"weights {escape(cdi.weights_version)}", body))
        if cdi.sensitivity:
            els.append(Paragraph(f"Weight sensitivity: Dirichlet 5–95% range {cdi.sensitivity.dirichlet_p05:.1f}–{cdi.sensitivity.dirichlet_p95:.1f}", small))
    rows = [["Component", "Value", "Weight → effective", "Contribution", "Provenance"]]
    for c in cdi.components:
        rows.append([Paragraph(escape(c.label), small), _fmt(c.value, 3), f"{c.weight:.2f} → {_fmt(c.effective_weight, 2)}",
                     _fmt(c.contribution, 1), Paragraph(escape(c.provenance if c.available else 'unavailable: ' + c.explanation), small)])
    els.append(_tbl(rows, [52 * mm, 18 * mm, 30 * mm, 22 * mm, 52 * mm]))
    if fspm.available:
        els.append(Paragraph("Fibonacci-weighted sequential pressure (FSPM)", h2))
        els.append(Paragraph(" · ".join(f"{k}: {v:.1f}" for k, v in fspm.scores.items()) +
                             f" · positions: {escape(fspm.positions_mode)}", body))
        rows = [["Stage", "Pos.", "Pressure p_i", "w Fib", "w Lin", "w Eq", "Fib contribution"]]
        for st in fspm.stages:
            rows.append([Paragraph(escape(st.name), small), str(st.position), f"{st.pressure:.3f}", f"{st.weights['fibonacci']:.3f}",
                         f"{st.weights['linear']:.3f}", f"{st.weights['equal']:.3f}", f"{st.contributions['fibonacci']:.1f}"])
        els.append(_tbl(rows, [50 * mm, 12 * mm, 22 * mm, 18 * mm, 18 * mm, 18 * mm, 28 * mm]))
    else:
        els.append(Paragraph(f"FSPM: {escape(fspm.reason or 'unavailable')}", small))
    els.append(Paragraph("Choice asymmetry (MCAI)", h2))
    if not mcai.available:
        els.append(Paragraph(escape(mcai.reason or "unavailable"), small))
    for cs in mcai.choice_sets:
        dims = " · ".join(f"{d.name} {_fmt(None if d.score is None else d.score * 100, 0)} ({d.provenance})" for d in cs.dimensions)
        els.append(Paragraph(f"<b>{escape(cs.id)}</b> accept “{escape(cs.accept.text if cs.accept else 'not found')}” vs decline "
                             f"“{escape(cs.decline.text)}”: MCAI {_fmt(cs.mcai, 1)} (coverage {cs.coverage*100:.0f}%) — {dims}", small))
    els.append(Paragraph("Counterfactual ethical redesign", h2))
    els.append(Paragraph(f"CDI {_fmt(cf.original_cdi)} → {_fmt(cf.redesigned_cdi)}; estimated reduction "
                         f"{_fmt(cf.risk_reduction_pct, 1, '%')}. {escape(cf.label)}", body))
    for e in cf.edits[:30]:
        after = escape(e.redesigned) if e.redesigned else "(no safe automatic rewrite)"
        sc = f" · P(dark) {e.original_score:.2f} → {e.redesigned_score:.2f}" if e.original_score is not None and e.redesigned_score is not None else ""
        st = f"stage {e.stage_index + 1} · " if e.stage_index is not None else ""
        els.append(Paragraph(f"{st}{escape(e.action)}: “{escape(e.original)}” → “{after}”{sc}", small))
    if temporal.available:
        els.append(Paragraph("Temporal evidence timeline", h2))
        for ev in temporal.events:
            ts = escape(ev.timestamp) if ev.timestamp else "order only"
            mark = " [deceptive signal]" if ev.deceptive_signal else ""
            els.append(Paragraph(f"Stage {ev.stage_index + 1} ({escape(ev.stage_name)}, {ts}) · {escape(ev.kind)}{mark}: "
                                 f"{escape(ev.description)}", small))
    else:
        els.append(Paragraph(f"Temporal analysis: {escape(temporal.reason or 'unavailable')}", small))
    return els


def render_journey_pdf(j) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title=f"ClearCart journey {j.journey_id}", author="ClearCart")
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12.5)
    small = ParagraphStyle("s", parent=body, fontSize=7.5, leading=10, textColor=colors.HexColor("#475569"))
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=18, alignment=0, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4)
    els = [Paragraph("ClearCart journey audit", h1),
           Paragraph(f"Journey {escape(j.journey_id)} · {len(j.stages)} stage(s) · {escape(j.created_at)}", small), Spacer(1, 6)]
    rows = [["#", "Stage", "Input", "Captured", "Findings", "Pressure"]]
    for st in j.stages:
        rows.append([str(st.index + 1), Paragraph(escape(st.name) + (" (synthetic)" if st.synthetic else ""), small), st.input_type,
                     escape(st.captured_at or "—"), str(st.audit.summary.suspicious_findings), f"{st.pressure:.3f}"])
    els.append(_tbl(rows, [8 * mm, 60 * mm, 22 * mm, 40 * mm, 20 * mm, 20 * mm]))
    els += intelligence_blocks(j.cdi, j.mcai, j.counterfactual, j.fspm, j.temporal, j.config_version, body, small, h2)
    els.append(Paragraph("Stage findings", h2))
    for st in j.stages:
        for f in st.audit.findings:
            els.append(Paragraph(f"Stage {st.index + 1} · {escape(f.id)} · {escape(f.category_label)}: “{escape(f.evidence)}” — "
                                 f"{escape(f.confidence_statement)}", small))
    if j.warnings:
        els.append(Paragraph("Notes", h2))
        els += [Paragraph(escape(w), small) for w in j.warnings]
    els += [Spacer(1, 8), Paragraph(escape(j.disclaimer), small)]
    doc.build(els)
    return buf.getvalue()
