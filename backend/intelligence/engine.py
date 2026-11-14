"""Orchestrates the intelligence layer on top of the existing audit pipeline.

The existing classifier/auditor output is used as-is (its findings, ids and provenance are not
modified). This module only reads it and adds CDI, MCAI, counterfactual, FSPM and temporal analysis.
"""
from __future__ import annotations

from typing import Optional

from backend.intelligence.cdi import ComponentInput, compute_cdi, confidence as cdi_confidence
from backend.intelligence.config import CONFIG_VERSION
from backend.intelligence.counterfactual import plan_edits
from backend.intelligence.fspm import compute_fspm
from backend.intelligence.optimiser import OPTIMISATION_META, optimise
from backend.intelligence.mcai import _semantic_dimension, combine, compute_mcai
from backend.intelligence.pressure import blend, content_pressure, obstruction_facts
from backend.intelligence.schemas import (CounterfactualResult, EvidenceRef, FSPMResult, FusionGroup, IntelligenceReport,
                                          MCAIResult, TemporalResult)
from backend.intelligence.temporal import StageObservation, analyse_journey

SINGLE_FSPM_REASON = "Single-page audit: sequential pressure needs a journey of two or more stages (use a journey audit)."
SINGLE_TEMPORAL_REASON = "Single-page audit: temporal comparison needs two or more snapshots or journey stages."


def _asymmetry_component(mcai: MCAIResult, stage_index: Optional[int] = None) -> ComponentInput:
    best, best_set = None, None
    for cs in mcai.choice_sets:
        v, _cov = combine(cs.dimensions, only=("visual", "interaction"))
        if v is not None and (best is None or v > best):
            best, best_set = v, cs
    if best is None:
        why = mcai.reason or "Detected choice sets have no measurable visual or interaction properties (e.g. pasted text)."
        return ComponentInput(key="choice_asymmetry", available=False, explanation=why)
    provs = sorted({d.provenance for d in best_set.dimensions if d.available and d.name in ("visual", "interaction")})
    ev = [EvidenceRef(stage_index=stage_index, finding_id=best_set.decline.finding_id, text=best_set.decline.text)]
    if best_set.accept:
        ev.insert(0, EvidenceRef(stage_index=stage_index, finding_id=best_set.accept.finding_id, text=best_set.accept.text))
    return ComponentInput(key="choice_asymmetry", available=True, value=best / 100.0, provenance="+".join(provs),
                          explanation=f"Worst choice set ({best_set.id}): visual/interaction asymmetry {best:.0f}/100.", evidence=ev)


def _semantic_mean(mcai: MCAIResult) -> Optional[float]:
    vals = [d.score for cs in mcai.choice_sets for d in cs.dimensions if d.name == "semantic" and d.available]
    return round(sum(vals) / len(vals), 4) if vals else None


def _redesigned_content(findings, edits, tau: float, stage_index: Optional[int] = None) -> float:
    """Content pressure after applying rewrites: a rewrite re-scored below tau is no longer flagged."""
    return blend(_redesigned_best(findings, edits, tau))


def _redesigned_best(findings, edits, tau: float) -> dict:
    """Per-category maxima after applying the selected redesign edits."""
    by_target = {e.target: e for e in edits}
    best: dict[str, float] = {}
    for f in findings:
        if f.detection_method != "ml_classifier" or f.model_score is None:
            continue
        e = by_target.get(f.id)
        score = f.model_score
        if e is not None and e.action == "remove_unverifiable_claim":
            score = 0.0                                        # element removed in the redesign
        elif e is not None and e.action == "rewrite" and e.redesigned_score is not None:
            score = e.redesigned_score if e.redesigned_score >= tau else 0.0
        best[f.category] = max(best.get(f.category, 0.0), score)
    return best


def _fusion_groups(stages: list[dict]) -> list[list[int]]:
    """Union stages linked by fuse_with (always to an EARLIER stage); groups ordered by their first stage."""
    parent = list(range(len(stages)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for s in stages:
        j = s.get("fuse_with")
        if j is not None:
            a, b = find(s["index"]), find(j)
            parent[max(a, b)] = min(a, b)
    groups: dict[int, list[int]] = {}
    for i in range(len(stages)):
        groups.setdefault(find(i), []).append(i)
    return [groups[k] for k in sorted(groups)]


def _merged_maxima(stages, members):
    """Per-category maxima across the members of a fusion unit, keeping each finding's own stage in its ref."""
    from backend.intelligence.pressure import category_maxima
    best, src = {}, {}
    for m in members:
        b, sr = category_maxima(stages[m]["audit"].findings, stages[m]["index"])
        for k, v in b.items():
            if v > best.get(k, -1):
                best[k], src[k] = v, sr[k]
    return best, src


def _tokens(t: str) -> set:
    import re as _re
    return set(_re.findall(r"[a-z0-9]+", (t or "").lower()))


def _duplicate_pairs(stages, members, threshold: float = 0.6) -> list[dict]:
    """Findings that appear in more than one modality of the same page (token Jaccard >= threshold)."""
    out = []
    for x in range(len(members)):
        for y in range(x + 1, len(members)):
            sa, sb = stages[members[x]], stages[members[y]]
            for fa in sa["audit"].findings:
                ta = _tokens(fa.evidence)
                for fb in sb["audit"].findings:
                    tb = _tokens(fb.evidence)
                    if not ta or not tb:
                        continue
                    j = len(ta & tb) / len(ta | tb)
                    if j >= threshold:
                        out.append({"a": EvidenceRef(stage_index=sa["index"], finding_id=fa.id, text=fa.evidence).model_dump(),
                                    "b": EvidenceRef(stage_index=sb["index"], finding_id=fb.id, text=fb.evidence).model_dump(),
                                    "similarity": round(j, 3)})
    return out


def _temporal_metrics(stages, groups, order, unit_p, temporal) -> dict:
    """Simple, interpretable temporal measures over the units in timeline order."""
    import re as _re
    cats = ("urgency", "scarcity", "social_proof", "misdirection")
    n = len(groups)
    present = {c: 0 for c in cats}
    seen: dict[str, set] = {}
    for u in order:
        flagged = [f for m in groups[u] for f in stages[m]["audit"].findings if f.detection_method == "ml_classifier"]
        for c in cats:
            if any(f.category == c for f in flagged):
                present[c] += 1
        for f in flagged:
            key = _re.sub(r"\s+", " ", _re.sub(r"[^a-z ]", " ", _re.sub(r"\d+", "0", f.evidence.lower()))).strip()
            if key:
                seen.setdefault(key, set()).add(stages[groups[u][0]]["name"])
    deltas = [{"from": stages[groups[order[k - 1]][0]]["name"], "to": stages[groups[order[k]][0]]["name"],
               "delta": round(unit_p[order[k]] - unit_p[order[k - 1]], 4)} for k in range(1, len(order))]
    d = [x["delta"] for x in deltas]
    count = lambda cat: sum(1 for e in temporal.events if e.category == cat)  # noqa: E731
    declared = [stages[m]["declared"][0] for u in order for m in groups[u][:1] if stages[m].get("declared")]
    return {
        "units": n,
        "category_persistence": {c: round(present[c] / n, 4) for c in cats},      # share of stages where the tactic appears
        "repeated_signals": [{"signal": k, "stages": sorted(v)} for k, v in seen.items() if len(v) > 1],
        "pressure_deltas": deltas,
        "mean_abs_pressure_change": round(sum(abs(x) for x in d) / len(d), 4) if d else None,
        "max_pressure_increase": round(max(d), 4) if d else None,
        "countdown_resets": count("countdown_reset"),
        "new_preselections": count("new_preselected_option"),
        "price_change_events": count("price_change"),
        "stock_change_events": count("stock_count_change"),
        "declared_step_gap_change": (declared[-1][1] - declared[-1][0]) - (declared[0][1] - declared[0][0]) if len(declared) >= 2 else None,
    }


class IntelligenceEngine:
    def __init__(self, clf) -> None:
        self._clf = clf

    @property
    def score_fn(self):
        """Classifier scoring through the request-scoped cache (backend/scoring.py) when one is active."""
        from backend.scoring import current_scorer
        s = current_scorer()
        return s.predict if s is not None else self._clf.predict

    # ------------------------------------------------------------------ single page
    def single(self, audit, input_type: str, cands, html: Optional[str] = None, image=None) -> IntelligenceReport:
        findings = audit.findings
        tau = audit.summary.threshold or 0.5
        mcai = compute_mcai(input_type, self.score_fn, cands=cands, html=html, image=image, findings=findings)
        cval, csrc = content_pressure(findings)
        content = ComponentInput(key="content_pressure", available=True, value=cval, provenance="model",
                                 explanation=("Intensity-breadth pressure: 0.5 x strongest flagged category + 0.5 x mean over the "
                                              f"four categories ({len(csrc)} of 4 categories have flagged evidence)."),
                                 evidence=csrc)
        asym = _asymmetry_component(mcai)
        obst = self._obstruction(input_type == "html", obstruction_facts(findings))
        cdi = compute_cdi([content, asym, obst])
        cdi.confidence = cdi_confidence(cdi, {(None, f.id): f.model_score for f in findings if f.model_score is not None})

        edits = optimise(plan_edits(findings, self.score_fn, None, mcai.choice_sets), self.score_fn)
        red_content = _redesigned_content(findings, edits, tau)
        cf = self._counterfactual(cdi, edits, content_r=red_content, asym=asym, obst=obst, mcai=mcai)
        return IntelligenceReport(
            config_version=CONFIG_VERSION, cdi=cdi, mcai=mcai, counterfactual=cf,
            fspm=FSPMResult(available=False, reason=SINGLE_FSPM_REASON),
            temporal=TemporalResult(available=False, reason=SINGLE_TEMPORAL_REASON))

    # ------------------------------------------------------------------ journey
    def journey(self, stages: list[dict]) -> dict:
        """stages: [{index, name, stage_type, input_type, captured_at, audit, cands, html, image, declared, fuse_with}]

        Cross-modal fusion: stages that declare `fuse_with` (an earlier stage showing the same page in another
        modality) form one UNIT. FSPM and the timeline run over units, so the same manipulation captured as text,
        HTML and screenshot counts once; MCAI still runs per stage because each modality measures different things.
        """
        groups = _fusion_groups(stages)                       # list of member-index lists, ordered by representative
        rep_of = {m: g[0] for g in groups for m in g}

        # MCAI per stage first: the timeline compares choice sets and declared step counts across stages
        mcais = []
        for s in stages:
            mtype = "screenshot_ocr" if s["input_type"] == "screenshot" else s["input_type"]
            mcais.append(compute_mcai(mtype, self.score_fn, cands=s["cands"], html=s.get("html"), image=s.get("image"),
                                      findings=s["audit"].findings, declared_paths=s.get("declared")))
        obs = []
        for g in groups:
            r = stages[g[0]]
            obs.append(StageObservation(
                index=r["index"], name=r["name"], captured_at=r.get("captured_at"),
                findings=[f for m in g for f in stages[m]["audit"].findings],
                texts=[c.text for m in g for c in stages[m]["cands"] if c.extra.get("unit") != "block"],
                declared_steps=next((stages[m]["declared"][0] for m in g if stages[m].get("declared")), None),
                choice_sets=[cs for m in g for cs in mcais[m].choice_sets]))
        temporal, pfacts, ofacts = analyse_journey(obs)

        unit_p, unit_src, unit_best = [], [], []
        for g in groups:
            best, src = _merged_maxima(stages, g)
            facts = pfacts.get(g[0]) or []
            channels = dict(best)
            for strength, ref in facts:
                channels["urgency"] = max(channels.get("urgency", 0.0), float(strength))
            unit_p.append(blend(channels)); unit_best.append(best)
            unit_src.append([src[k] for k in sorted(src, key=lambda k: -best[k])] + [ref for _, ref in facts])
        names = [stages[g[0]]["name"] + (f" (+{len(g) - 1} merged)" if len(g) > 1 else "") for g in groups]
        types = [stages[g[0]].get("stage_type") for g in groups]
        fspm = compute_fspm(names, types, unit_p, unit_src)
        # FSPMStage.stage_index must refer to the representative STAGE index, not the unit position
        if fspm.available:
            for st, g in zip(fspm.stages, groups):
                st.stage_index = g[0]
            fspm.top_stage_index = None if fspm.top_stage_index is None else groups[fspm.top_stage_index][0]

        # timeline pressure annotations, metrics and per-event pressure change, in the timeline's own order
        order = self._temporal_order([stages[g[0]] for g in groups], temporal.ordering)   # unit positions
        p_by_rep = {groups[u][0]: unit_p[u] for u in range(len(groups))}
        temporal.pressure_progression = [round(unit_p[u], 4) for u in order]
        temporal.progression_stage_indices = [groups[u][0] for u in order]
        prev_p = {}
        for k, u in enumerate(order):
            prev_p[groups[u][0]] = None if k == 0 else unit_p[order[k - 1]]
        for ev in temporal.events:
            rep = rep_of.get(ev.stage_index, ev.stage_index)
            ev.pressure = round(p_by_rep.get(rep, 0.0), 4)
            pp = prev_p.get(rep)
            ev.pressure_delta = None if pp is None else round(p_by_rep[rep] - pp, 4)
        if temporal.available:
            temporal.metrics = _temporal_metrics(stages, groups, order, unit_p, temporal)

        # merged MCAI across stages (ids prefixed with the stage)
        merged = []
        for i, m in enumerate(mcais):
            for cs in m.choice_sets:
                cs = cs.model_copy(update={"id": f"s{i + 1}-{cs.id}"})
                merged.append((i, cs))
        mcai_all = MCAIResult(available=bool(merged), weights=mcais[0].weights if mcais else {},
                              reason=None if merged else "No accept/decline choice pair was detected in any stage.",
                              choice_sets=[cs for _, cs in merged],
                              mcai=max((cs.mcai for _, cs in merged if cs.mcai is not None), default=None))

        fib = fspm.scores.get("fibonacci") if fspm.available else None
        content = ComponentInput(key="content_pressure", available=fib is not None, value=None if fib is None else fib / 100.0,
                                 provenance="model+observed", explanation=(
                                     "Journey content pressure = FSPM (Fibonacci-weighted stage pressures, incl. observed countdown resets)/100. "
                                     "Replaces, not adds to, page-level content pressure."),
                                 evidence=[r for src in unit_src for r in src][:12])
        asym_inputs = [(_asymmetry_component(m, i), i) for i, m in enumerate(mcais)]
        avail_asym = [a for a, _ in asym_inputs if a.available]
        asym = max(avail_asym, key=lambda a: a.value) if avail_asym else ComponentInput(
            key="choice_asymmetry", available=False, explanation="No stage had measurable visual or interaction asymmetry.")
        dom_capable = any(s["input_type"] == "html" for s in stages)
        # DOM observations and temporal "newly introduced pre-selection" facts can refer to the same
        # element: de-duplicate by (stage, finding) and keep the stronger value
        dedup: dict = {}
        for strength, ref in [f for s in stages for f in obstruction_facts(s["audit"].findings, s["index"])] + \
                [f for i in ofacts for f in ofacts[i]]:
            key = (ref.stage_index, ref.finding_id or ref.text)
            if key not in dedup or strength > dedup[key][0]:
                dedup[key] = (strength, ref)
        facts = list(dedup.values())
        # ofacts has a (possibly empty) list for every stage: only real facts make obstruction observable
        obst = self._obstruction(dom_capable or any(ofacts.values()), facts)
        cdi = compute_cdi([content, asym, obst])
        cdi.confidence = cdi_confidence(cdi, {(s["index"], f.id): f.model_score for s in stages
                                              for f in s["audit"].findings if f.model_score is not None})

        # counterfactual across stages (edits per stage; redesigned pressure merged per unit like the original)
        edits, stage_edits = [], {}
        for s, m in zip(stages, mcais):
            e = optimise(plan_edits(s["audit"].findings, self.score_fn, s["index"], m.choice_sets), self.score_fn)
            edits += e; stage_edits[s["index"]] = e
        red_p = []
        for g in groups:
            best: dict[str, float] = {}
            for m in g:
                tau = stages[m]["audit"].summary.threshold or 0.5
                for k, v in _redesigned_best(stages[m]["audit"].findings, stage_edits[m], tau).items():
                    best[k] = max(best.get(k, 0.0), v)
            red_p.append(blend(best))                          # observed resets removed by construction
        red_fspm = compute_fspm(names, types, red_p, [[] for _ in groups]) if groups else None
        red_fib = red_fspm.scores.get("fibonacci") if red_fspm and red_fspm.available else None
        cf = self._counterfactual(cdi, edits, content_r=None if red_fib is None else red_fib / 100.0, asym=asym, obst=obst,
                                  mcai=mcai_all, fspm_o=fib, fspm_r=red_fib)

        fusion = [FusionGroup(stage_indices=g, representative=g[0], duplicate_pairs=_duplicate_pairs(stages, g),
                              fused_pressure=round(unit_p[u], 4)) for u, g in enumerate(groups) if len(g) > 1]
        unit_of = {m: u for u, g in enumerate(groups) for m in g}
        positions = [st.position for st in fspm.stages] if fspm.available else list(range(1, len(groups) + 1))
        return {"fspm": fspm, "temporal": temporal, "cdi": cdi, "mcai": mcai_all, "counterfactual": cf,
                "pressures": [unit_p[unit_of[i]] for i in range(len(stages))],
                "positions": [positions[unit_of[i]] for i in range(len(stages))],
                "fused_into": [None if rep_of[i] == i else rep_of[i] for i in range(len(stages))],
                "fusion": fusion}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _temporal_order(stages, ordering: str) -> list[int]:
        idx = list(range(len(stages)))
        if ordering == "timestamp":
            from datetime import datetime
            try:
                return sorted(idx, key=lambda i: datetime.fromisoformat(stages[i]["captured_at"].replace("Z", "+00:00")))
            except Exception:
                return idx
        return idx

    @staticmethod
    def _obstruction(observable: bool, facts: list) -> ComponentInput:
        if not observable:
            return ComponentInput(key="observed_obstruction", available=False,
                                  explanation="Pre-selected options and other DOM states are only observable in HTML input or journey comparisons.")
        if not facts:
            return ComponentInput(key="observed_obstruction", available=True, value=0.0, provenance="observed",
                                  explanation="DOM was inspected: no pre-selected options or obstruction facts were observed.")
        strength, _ = max(facts, key=lambda f: f[0])
        return ComponentInput(key="observed_obstruction", available=True, value=strength, provenance="observed",
                              explanation=f"{len(facts)} observed obstruction fact(s); the strongest sets the value (max, not sum).",
                              evidence=[ref for _, ref in facts][:8])

    def _counterfactual(self, cdi, edits, *, content_r, asym, obst, mcai, fspm_o=None, fspm_r=None) -> CounterfactualResult:
        red_inputs = [
            ComponentInput(key="content_pressure", available=content_r is not None, value=content_r, provenance="model",
                           explanation="Original findings re-scored after their rewrites (model-estimated); rewrites scored below threshold no longer count."),
            ComponentInput(key="choice_asymmetry", available=asym.available, value=0.0 if asym.available else None,
                           provenance="by_construction",
                           explanation="Redesign specification gives both choices equal visual weight and equal step counts."),
            ComponentInput(key="observed_obstruction", available=obst.available, value=0.0 if obst.available else None,
                           provenance="by_construction", explanation="Redesign specification leaves optional add-ons and consents unchecked."),
        ]
        red = compute_cdi(red_inputs, with_sensitivity=False)
        o, r = cdi.cdi, red.cdi
        reduction = round(100.0 * (o - r) / o, 1) if (o is not None and r is not None and o > 0) else None

        sem_o = _semantic_mean(mcai)
        sem_r = None
        mcai_r = None
        if mcai.available:
            # finding ids (f1..) and choice-set ids (c1..) repeat across journey stages, so key by (stage, target)
            new_decline = {(e.stage_index, e.target): e.redesigned for e in edits
                           if e.action in ("equalise_choice", "rewrite") and e.redesigned}
            sems, scores = [], []
            for cs in mcai.choice_sets:
                stage = int(cs.id[1:].split("-")[0]) - 1 if cs.id.startswith("s") else None
                base_id = cs.id.split("-")[-1]
                d_txt = (new_decline.get((stage, base_id)) or new_decline.get((stage, cs.decline.finding_id))
                         or cs.decline.text)
                sd = _semantic_dimension(cs.accept.text if cs.accept else None, d_txt, self.score_fn)
                sems.append(sd.score)
                dims = [d.model_copy(update={"score": 0.0, "provenance": "by_construction"}) if d.available and d.name != "semantic" else d
                        for d in cs.dimensions if d.name != "semantic"] + [sd]
                v, _ = combine(dims)
                if v is not None:
                    scores.append(v)
            sem_r = round(sum(sems) / len(sems), 4) if sems else None
            mcai_r = max(scores) if scores else None
        return CounterfactualResult(edits=edits, original_cdi=o, redesigned_cdi=r, risk_reduction_pct=reduction,
                                    original_components=cdi.components, redesigned_components=red.components,
                                    original_semantic_pressure=sem_o, redesigned_semantic_pressure=sem_r,
                                    original_mcai=mcai.mcai, redesigned_mcai=mcai_r, original_fspm=fspm_o, redesigned_fspm=fspm_r,
                                    optimisation=OPTIMISATION_META)
