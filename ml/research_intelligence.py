"""Research validation of the ClearCart intelligence layer (config intel-v1.1): studies R1-R6.

Run from the project root:  .venv/bin/python -m ml.research_intelligence

Writes ml/artifacts/intelligence_research.json and report/figures/research_*.png.

What these numbers are (and are not)
  * No labelled dataset of journeys, choice sets or composite deception levels exists. R1-R3 are property /
    sensitivity studies on SYNTHETIC inputs built from real held-out TEST-split strings (or authored HTML); the
    "reference cases" of R1(e) and the "designed levels" of R2 encode the AUTHORS' DESIGN INTENT, not ground truth.
  * R4 runs the real optimiser on every dark TEST-split row; its scores are MODEL-ESTIMATED (the same classifier
    scores the original and the redesign). The exactness check is pure maths on the optimiser's real candidates.
  * R5/R6 run the real API code path (FastAPI TestClient) on the demo inputs.
  * All text that is scored is a verbatim TEST-split row (data/processed/text_dataset.csv, split == 'test') unless it is
    explicitly listed as AUTHORED. Seed: numpy default_rng(2027); one spawned child stream per study.
  * The real calibrated classifier (ml/artifacts) is used throughout. Within one study run a request-scoped
    CachedScorer (backend/scoring.py) memoises predictions by exact text - this changes speed only, not values.
"""
from __future__ import annotations

import ml  # noqa: F401  (sets OMP_NUM_THREADS=1 before torch/xgboost load - required on macOS)

import contextlib
import functools
import io
import itertools
import json
import math
import platform
import statistics
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ml.benchmark_intelligence import _style, jsonable, r
from ml.common import ARTIFACTS, DATA_PROCESSED, FIGURES, ROOT

OUT_JSON = ARTIFACTS / "intelligence_research.json"
SEED = 2027
MASTER = np.random.default_rng(SEED)
_STREAMS = MASTER.spawn(8)
RNG = dict(zip(("R1", "R1e", "R2", "R3", "R4", "R5", "R6", "spare"), _STREAMS))

PHI = (1 + 5 ** 0.5) / 2
ENGINE_SCHEMES = ("fibonacci", "linear", "equal")
R1_SCHEMES = ("fibonacci", "linear", "equal", "geometric_phi", "geometric_2", "fib_capped3", "max_pressure")
SCHEME_COLORS = {"fibonacci": "#2a78d6", "linear": "#eb6834", "equal": "#1baf7a", "geometric_phi": "#7b4fd6",
                 "geometric_2": "#c23b7a", "fib_capped3": "#8a8a2a", "max_pressure": "#52514e"}
SCHEME_JUSTIFICATION = {
    "fibonacci": "Engine default (proposed): w = F(pos), F(1)=F(2)=1.",
    "linear": "Engine baseline: w = pos.",
    "equal": "Engine baseline: w = 1 (order-free weighted mean).",
    "geometric_phi": ("w = phi^(pos-1), phi = 1.618. F(n) ~ phi^n / sqrt(5), so this is the smooth closed-form analogue of "
                      "Fibonacci; it differs only at small positions (F(1)=F(2) gives landing and product equal weight). "
                      "Tests whether anything is specific to the integer Fibonacci sequence."),
    "geometric_2": "w = 2^(pos-1): a stronger recency emphasis than Fibonacci (ratio 2 instead of ~1.618); upper bracket.",
    "fib_capped3": ("'recency-capped': w = min(F(pos), 3) - Fibonacci up to checkout (pos 4), flat afterwards. Bounds the "
                    "leverage of any single late stage at 3x the first stage, so a missing or duplicated late capture should "
                    "move the score less while keeping a recency ordering for early stages."),
    "max_pressure": "Order-free, non-compensatory baseline: 100 * max_i p_i (the single worst stage).",
}
CATS = ("urgency", "scarcity", "social_proof", "misdirection")


# ============================================================================ shared context
def load_test() -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / "text_dataset.csv", keep_default_na=False)
    t = df[df["split"] == "test"].reset_index(drop=True)
    assert len(t) > 0
    return t


class Ctx:
    def __init__(self):
        from backend.auditor import Auditor
        from ml.model import TextClassifier
        t0 = time.perf_counter()
        self.clf = TextClassifier.load(ARTIFACTS)
        self.clf.predict(["warm-up"])
        self.load_s = time.perf_counter() - t0
        # false-positive gates off: the saved intelligence artefacts were produced before the gates existed
        # (ml/eval_false_positives.py), so this keeps them reproducible.
        self.auditor = Auditor(self.clf, quote_gate=False, ood_gate=False)
        self.tau = float(self.clf.b.threshold)
        self.test = load_test()
        self._stage: dict[tuple, dict] = {}

    def score_fn(self, texts):
        from backend.scoring import current_scorer
        s = current_scorer()
        return s.predict(texts) if s is not None else self.clf.predict(texts)

    def audit_text(self, text: str):
        from backend.extractors import segment_text
        cands = segment_text(text)
        return self.auditor.audit(cands, "text"), cands

    def stage(self, strings: tuple[str, ...]) -> dict:
        """Stage pressure exactly as the engine computes it for a text stage:
        segment_text -> Auditor.audit(..., 'text') -> pressure.content_pressure(findings)."""
        from backend.intelligence.pressure import category_maxima, content_pressure
        if strings not in self._stage:
            audit, _ = self.audit_text("\n".join(strings))
            p, _ = content_pressure(audit.findings)
            best, _ = category_maxima(audit.findings)
            self._stage[strings] = {"p": float(p), "best": best,
                                    "n_ml": sum(f.detection_method == "ml_classifier" for f in audit.findings)}
        return self._stage[strings]


def scheme_weight(s: str, pos: int) -> float:
    from backend.intelligence.fspm import fibonacci, raw_weight
    if s in ENGINE_SCHEMES:
        return raw_weight(s, pos)
    if s == "geometric_phi":
        return PHI ** (pos - 1)
    if s == "geometric_2":
        return 2.0 ** (pos - 1)
    if s == "fib_capped3":
        return float(min(fibonacci(pos)[-1], 3))
    raise ValueError(s)


def scheme_score(p: list[float], pos: list[int], s: str) -> float:
    from backend.intelligence.fspm import weighted_score
    if s == "max_pressure":
        return 100.0 * max(p)
    if s in ENGINE_SCHEMES:
        return weighted_score(list(p), list(pos), s)          # the engine's own function
    w = [scheme_weight(s, q) for q in pos]
    return 100.0 * sum(wi * pi for wi, pi in zip(w, p)) / sum(w)


def stats(x) -> dict:
    x = np.asarray(x, float)
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "mean": r(float(x.mean())), "median": r(float(np.median(x))), "p95": r(float(np.percentile(x, 95))),
            "max": r(float(x.max())), "min": r(float(x.min()))}


# ============================================================================ R1 FSPM robustness
R1_TYPES = ("product", "cart", "checkout", "payment")


def r1_random_journeys(ctx: Ctx, rng, n_j=400, n_stages=4, per_stage=3, p_dark=0.35):
    t = ctx.test
    dark = t[t["label"] != "benign"][["row_id", "text", "label"]].values.tolist()
    benign = t[t["label"] == "benign"][["row_id", "text", "label"]].values.tolist()
    js = []
    for _ in range(n_j):
        stages = []
        for _ in range(n_stages):
            items, used = [], set()
            while len(items) < per_stage:
                pool = dark if rng.random() < p_dark else benign
                it = pool[int(rng.integers(len(pool)))]
                if it[0] in used:
                    continue
                used.add(it[0]); items.append(tuple(it))
            stages.append(items)
        js.append(stages)
    return js, {"dark": dark, "benign": benign}


def _texts(stage_items) -> tuple[str, ...]:
    return tuple(it[1] for it in stage_items)


def r1_scores(ctx, journey, pos) -> dict:
    p = [ctx.stage(_texts(st))["p"] for st in journey]
    return {s: scheme_score(p, pos, s) for s in R1_SCHEMES}, p


def exp_R1(ctx: Ctx):
    from scipy.stats import spearmanr
    from backend.intelligence.fspm import resolve_positions
    rng = RNG["R1"]
    js, pools = r1_random_journeys(ctx, rng)
    by_label = defaultdict(list)
    for it in pools["dark"] + pools["benign"]:
        by_label[it[2]].append(tuple(it))
    pos, mode = resolve_positions(list(R1_TYPES))
    assert mode == "canonical"
    base, P = [], []
    for j in js:
        sc, p = r1_scores(ctx, j, pos)
        base.append(sc); P.append(p)
    P = np.array(P)
    base_arr = {s: np.array([b[s] for b in base]) for s in R1_SCHEMES}
    sd = {s: float(base_arr[s].std(ddof=1)) for s in R1_SCHEMES}
    out = {"synthetic": True, "kind": "property/sensitivity study on synthetic journeys built from real TEST-split strings (no ground truth)",
           "design": (f"{len(js)} synthetic 4-stage journeys with declared stage types {list(R1_TYPES)} -> canonical positions {pos}. Each "
                      "stage = 3 distinct TEST-split strings, each drawn from the dark pool (label != benign) with probability 0.35, else "
                      "from the benign pool. Stage pressure = pressure.content_pressure(Auditor(clf).audit(segment_text(stage),'text').findings) "
                      "(the engine's per-stage quantity)."),
           "schemes": {s: SCHEME_JUSTIFICATION[s] for s in R1_SCHEMES},
           "positions": pos,
           "normalised_weights_at_positions": {s: [r(scheme_weight(s, q) / sum(scheme_weight(s, x) for x in pos)) for q in pos]
                                               for s in R1_SCHEMES if s != "max_pressure"},
           "stage_pressure_summary": {"mean": r(float(P.mean())), "share_zero": r(float((P == 0).mean())),
                                      "share_ge_0_5": r(float((P >= 0.5).mean()))},
           "score_summary": {s: {"mean": r(float(base_arr[s].mean()), 2), "sd": r(sd[s], 2)} for s in R1_SCHEMES}}

    # ---------------- (a) single-observation noise: flip one string per stage to another of the same label
    R = 3
    dA = {s: [] for s in R1_SCHEMES}
    pert_scores = {s: [] for s in R1_SCHEMES}
    for ji, j in enumerate(js):
        for rep in range(R):
            new = []
            for st in j:
                k = int(rng.integers(len(st)))
                lab = st[k][2]
                present = {it[0] for it in st}
                cand = [it for it in by_label[lab] if it[0] not in present]
                rep_it = cand[int(rng.integers(len(cand)))]
                st2 = list(st); st2[k] = rep_it
                new.append(st2)
            sc, _ = r1_scores(ctx, new, pos)
            for s in R1_SCHEMES:
                dA[s].append(abs(sc[s] - base[ji][s]))
                if rep == 0:
                    pert_scores[s].append(sc[s])
    out["a_single_observation_noise"] = {
        "design": f"For every journey, {R} replicates; in each replicate EVERY stage has one random string replaced by another TEST-split "
                  "string with the same gold label (not already in the stage). |delta score| on the 0-100 scale. noise_to_signal = "
                  "mean |delta| / SD of the scheme's scores across the 400 journeys (schemes have different spreads). "
                  "rank_spearman = Spearman rho between original and perturbed (replicate 1) scores across journeys.",
        "per_scheme": {s: {**stats(dA[s]), "noise_to_signal": r(float(np.mean(dA[s])) / sd[s]),
                           "rank_spearman": r(float(spearmanr(base_arr[s], pert_scores[s])[0]))} for s in R1_SCHEMES}}

    # ---------------- (b) missing stage
    from backend.intelligence.fspm import resolve_positions as _rp
    dB = {mode_: {s: [[] for _ in pos] for s in R1_SCHEMES} for mode_ in ("canonical", "submission_order")}
    dropped_scores = {mode_: {s: np.zeros((len(js), len(pos))) for s in R1_SCHEMES} for mode_ in dB}
    for ji, j in enumerate(js):
        p = P[ji]
        for i in range(len(pos)):
            keep = [k for k in range(len(pos)) if k != i]
            pk = [float(p[k]) for k in keep]
            pos_c, m = _rp([R1_TYPES[k] for k in keep])
            assert m == "canonical"
            pos_s = list(range(1, len(keep) + 1))
            for mode_, pp in (("canonical", pos_c), ("submission_order", pos_s)):
                for s in R1_SCHEMES:
                    v = scheme_score(pk, pp, s)
                    dB[mode_][s][i].append(abs(v - base[ji][s]))
                    dropped_scores[mode_][s][ji, i] = v
    n_pairs = 3000
    pairs = set()
    while len(pairs) < n_pairs:
        a, b = (int(x) for x in rng.choice(len(js), 2, replace=False))
        pairs.add((min(a, b), max(a, b)))
    pairs = sorted(pairs)
    one_side_drop = [int(rng.integers(len(pos))) for _ in pairs]
    flips = {}
    for mode_ in dB:
        flips[mode_] = {}
        for s in R1_SCHEMES:
            both = one = considered = 0
            for (a, b), d1 in zip(pairs, one_side_drop):
                d0 = base_arr[s][a] - base_arr[s][b]
                if abs(d0) < 1e-9:
                    continue
                considered += 1
                for i in range(len(pos)):
                    d = dropped_scores[mode_][s][a, i] - dropped_scores[mode_][s][b, i]
                    both += np.sign(d) != np.sign(d0)
                d = dropped_scores[mode_][s][a, d1] - base_arr[s][b]
                one += np.sign(d) != np.sign(d0)
            flips[mode_][s] = {"pairs_considered": considered,
                               "rank_change_rate_same_stage_missing_in_both": r(both / (considered * len(pos))),
                               "rank_change_rate_random_stage_missing_in_one": r(one / considered)}
    out["b_missing_stage"] = {
        "design": ("Each stage of each journey is dropped in turn (3-stage journey). 'canonical': the remaining stages keep their "
                   "canonical positions (what the engine does when stage types are declared); 'submission_order': positions are "
                   "renumbered 1..3 (engine behaviour without stage types). Pair ranking: 3000 random journey pairs (original ties "
                   "excluded); a change = the sign of score(A) - score(B) differs after the drop (a new tie counts as a change)."),
        "mean_abs_delta_by_dropped_stage": {mode_: {s: {R1_TYPES[i]: r(float(np.mean(dB[mode_][s][i])), 3) for i in range(len(pos))}
                                                    for s in R1_SCHEMES} for mode_ in dB},
        "mean_abs_delta_overall": {mode_: {s: r(float(np.mean([x for i in range(len(pos)) for x in dB[mode_][s][i]])), 3)
                                           for s in R1_SCHEMES} for mode_ in dB},
        "pair_ranking": flips}

    # ---------------- (c) repeated signals
    dC_cross = {s: [] for s in R1_SCHEMES}
    dC_by_target = {s: [[] for _ in pos] for s in R1_SCHEMES}
    dC_within = {s: [] for s in R1_SCHEMES}
    n_c = 0
    for ji, j in enumerate(js):
        darks = [(si, it) for si, st in enumerate(j) for it in st if it[2] != "benign"]
        flagged = [(si, it) for si, it in darks if ctx.stage((it[1],))["n_ml"] > 0]
        if not flagged:
            continue
        si, it = flagged[int(rng.integers(len(flagged)))]
        tj = int(rng.choice([k for k in range(len(pos)) if k != si]))
        new = [list(st) for st in j]; new[tj] = new[tj] + [it]
        sc, _ = r1_scores(ctx, new, pos)
        same = [list(st) for st in j]; same[si] = same[si] + [it]
        sc_same, _ = r1_scores(ctx, same, pos)
        n_c += 1
        for s in R1_SCHEMES:
            dC_cross[s].append(sc[s] - base[ji][s])
            dC_by_target[s][tj].append(sc[s] - base[ji][s])
            dC_within[s].append(sc_same[s] - base[ji][s])
    out["c_repeated_signals"] = {
        "design": ("Journeys containing at least one dark TEST string that the classifier flags on its own: one such string is "
                   "duplicated (appended) into a random OTHER stage ('cross-stage') and, separately, appended a second time to its OWN "
                   "stage ('within-stage'). delta = new - original score."),
        "n_journeys": n_c,
        "within_stage_duplicate_delta": {s: {"max_abs": r(float(np.max(np.abs(dC_within[s]))), 9)} for s in R1_SCHEMES},
        "within_stage_note": ("De-duplication rule inside a stage: c_k = max over findings of category k, so a repeated string "
                              "(same category, same score) cannot raise the stage pressure; delta is exactly 0 for every scheme."),
        "cross_stage_delta": {s: stats(dC_cross[s]) for s in R1_SCHEMES},
        "cross_stage_mean_delta_by_target_stage": {s: {R1_TYPES[i]: (r(float(np.mean(v)), 3) if v else None)
                                                       for i, v in enumerate(dC_by_target[s])} for s in R1_SCHEMES},
        "cross_stage_note": ("FSPM has NO cross-stage de-duplication: the same claim shown at two stages is treated as two "
                             "exposures. The inflation is proportional to the weight of the stage it is copied into."),
    }

    # ---------------- (d) order sensitivity
    dD = {s: [] for s in R1_SCHEMES}
    for p in P:
        esc, desc = sorted(p), sorted(p, reverse=True)
        for s in R1_SCHEMES:
            dD[s].append(scheme_score(list(esc), pos, s) - scheme_score(list(desc), pos, s))
    out["d_order_sensitivity"] = {
        "design": "Each journey's 4 stage pressures sorted ascending (escalating) vs descending (de-escalating) on positions "
                  f"{pos}; difference = escalating - de-escalating.",
        "per_scheme": {s: {"mean_diff": r(float(np.mean(dD[s])), 3), "median_diff": r(float(np.median(dD[s])), 3),
                           "max_diff": r(float(np.max(dD[s])), 3),
                           "frac_escalating_higher": r(float(np.mean(np.array(dD[s]) > 1e-9)))} for s in R1_SCHEMES}}

    out["e_reference_cases"] = exp_R1e(ctx)
    out["conclusion"] = r1_conclusion(out)
    figs = {"dA": dA, "dB": dB, "dC": dC_by_target, "dD": dD}
    return out, figs


def exp_R1e(ctx: Ctx):
    """Manually constructed reference journeys with explicitly stated expected orderings (AUTHORS' DESIGN INTENT)."""
    from backend.intelligence.fspm import resolve_positions
    rng = RNG["R1e"]
    t = ctx.test

    def pick(label, flagged_category=None):
        rows = t[t["label"] == label][["row_id", "text"]].values.tolist()
        rng.shuffle(rows)
        for rid, txt in rows:
            audit, cands = ctx.audit_text(txt)
            ml_f = [f for f in audit.findings if f.detection_method == "ml_classifier"]
            if len(cands) != 1:
                continue
            if flagged_category is None and not ml_f:
                return int(rid), txt
            if flagged_category is not None and len(ml_f) == 1 and ml_f[0].category == flagged_category:
                return int(rid), txt
        raise RuntimeError(f"no suitable {label} string")
    b1, b2 = pick("benign"), pick("benign")
    while b2[0] == b1[0]:
        b2 = pick("benign")
    su, sm, ss = pick("urgency", "urgency"), pick("misdirection", "misdirection"), pick("scarcity", "scarcity")
    B = (b1[1], b2[1]); D1 = (b1[1], su[1]); D2 = (b1[1], su[1], sm[1]); D3 = (su[1], sm[1], ss[1])
    content = {"B": B, "D1": D1, "D2": D2, "D3": D3}
    types = ["landing", "cart", "checkout", "exit"]
    pos, mode = resolve_positions(types)
    assert mode == "canonical"
    J = {"benign": "B B B B", "L1": "D1 B B B", "A1": "B D1 B B", "C1": "B B D1 B", "E1": "B B B D1",
         "LA": "D1 D1 B B", "CE": "B B D1 D1", "L2": "D2 B B B", "C2": "B B D2 B", "E2": "B B B D2",
         "ALL1": "D1 D1 D1 D1", "ESC": "B D1 D2 D3", "DESC": "D3 D2 D1 B", "LAC": "D1 D1 D1 B"}
    pressures = {k: [ctx.stage(content[c])["p"] for c in v.split()] for k, v in J.items()}
    stage_p = {c: r(ctx.stage(v)["p"]) for c, v in content.items()}
    assert stage_p["B"] == 0.0, "benign filler must not be flagged"
    dark_js = [k for k in J if k != "benign"]
    E = []
    E += [("G1_benign_floor", k, "benign", f"any dark journey ({k}) > all-benign") for k in dark_js]
    E += [("G2_position", "A1", "L1", "same tactic at cart > at landing"),
          ("G2_position", "C1", "A1", "same tactic at checkout > at cart"),
          ("G2_position", "E1", "C1", "same tactic at exit > at checkout"),
          ("G2_position", "E1", "L1", "same tactic at exit > at landing"),
          ("G2_position", "C2", "L2", "two tactics at checkout > two tactics at landing"),
          ("G2_position", "E2", "C2", "two tactics at exit > two tactics at checkout"),
          ("G2_position", "CE", "LA", "tactic at checkout+exit > tactic at landing+cart"),
          ("G2_position", "CE", "L1", "tactic only at checkout+exit > tactic only at landing")]
    E += [("G3_breadth", "L2", "L1", "two tactics in the landing stage > one"),
          ("G3_breadth", "C2", "C1", "two tactics in the checkout stage > one"),
          ("G3_breadth", "E2", "E1", "two tactics in the exit stage > one")]
    E += [("G4_coverage", "LA", "L1", "landing+cart > landing only"),
          ("G4_coverage", "CE", "C1", "checkout+exit > checkout only"),
          ("G4_coverage", "CE", "E1", "checkout+exit > exit only"),
          ("G4_coverage", "ALL1", "CE", "every stage > checkout+exit"),
          ("G4_coverage", "ALL1", "LA", "every stage > landing+cart"),
          ("G4_coverage", "LAC", "LA", "landing+cart+checkout > landing+cart")]
    E += [("G5_escalation", "ESC", "DESC", "escalating > de-escalating (same multiset of stage pressures)")]
    E += [("G6_contestable_tradeoff", "LAC", "E1", "pressure at 3 of 4 stages > one tactic only at the exit"),
          ("G6_contestable_tradeoff", "LAC", "C1", "pressure at 3 of 4 stages > one tactic only at checkout")]
    scores = {s: {k: scheme_score(pressures[k], pos, s) for k in J} for s in R1_SCHEMES}
    res = {}
    for s in R1_SCHEMES:
        per_group = defaultdict(lambda: {"concordant": 0, "ties": 0, "discordant": 0, "n": 0})
        failed = []
        for g, hi, lo, desc in E:
            d = scores[s][hi] - scores[s][lo]
            pg = per_group[g]; pg["n"] += 1
            if d > 1e-9:
                pg["concordant"] += 1
            elif d < -1e-9:
                pg["discordant"] += 1; failed.append(f"{hi} > {lo} ({desc}): {scores[s][hi]:.2f} vs {scores[s][lo]:.2f}")
            else:
                pg["ties"] += 1; failed.append(f"TIE {hi} = {lo} ({desc}): {scores[s][hi]:.2f}")

        def agg(groups):
            n = sum(per_group[g]["n"] for g in groups)
            c = sum(per_group[g]["concordant"] for g in groups); tie = sum(per_group[g]["ties"] for g in groups)
            return {"n": n, "agreement_ties_half": r((c + 0.5 * tie) / n), "strict_agreement": r(c / n)}
        groups = sorted(per_group)
        res[s] = {"overall": agg(groups), "excluding_G6_contestable": agg([g for g in groups if not g.startswith("G6")]),
                  "per_group": {g: {**per_group[g], **agg([g])} for g in groups}, "not_satisfied": failed}
    return {"synthetic": True, "reference_is_design_intent_not_ground_truth": True,
            "design": ("14 reference journeys over the declared stages landing, cart, checkout, exit (canonical positions "
                       f"{pos}). Stage contents: B = two unflagged benign TEST strings; D1 = B[0] + one urgency string; D2 = D1 + one "
                       "misdirection string; D3 = urgency + misdirection + scarcity strings (all TEST-split rows that the classifier flags "
                       "in their gold category; chosen at random with a fixed seed). 33 pairwise expectations in 6 groups, written by the "
                       "authors BEFORE computing scores; G6 is a deliberately contestable trade-off (breadth/persistence vs recency). "
                       "Kendall-style agreement: concordant = 1, tie = 0.5, discordant = 0 (strict: ties count 0)."),
            "strings": {"benign_1_row_id": b1[0], "benign_2_row_id": b2[0], "urgency_row_id": su[0], "misdirection_row_id": sm[0],
                        "scarcity_row_id": ss[0], "texts": {"b1": b1[1], "b2": b2[1], "urgency": su[1], "misdirection": sm[1], "scarcity": ss[1]}},
            "stage_pressures": stage_p, "journeys": J, "positions": pos,
            "expectations": [{"group": g, "higher": hi, "lower": lo, "statement": d} for g, hi, lo, d in E],
            "scores": {s: {k: r(v, 2) for k, v in scores[s].items()} for s in R1_SCHEMES},
            "agreement": res}


def r1_conclusion(R1) -> dict:
    a = R1["a_single_observation_noise"]["per_scheme"]
    b = R1["b_missing_stage"]
    e = R1["e_reference_cases"]["agreement"]
    d = R1["d_order_sensitivity"]["per_scheme"]
    ag = {s: e[s]["overall"]["agreement_ties_half"] for s in R1_SCHEMES}
    ag_x = {s: e[s]["excluding_G6_contestable"]["agreement_ties_half"] for s in R1_SCHEMES}
    pos_g = {s: e[s]["per_group"]["G2_position"]["agreement_ties_half"] for s in R1_SCHEMES}
    flip = {s: b["pair_ranking"]["canonical"][s]["rank_change_rate_same_stage_missing_in_both"] for s in R1_SCHEMES}
    miss = {s: b["mean_abs_delta_overall"]["canonical"][s] for s in R1_SCHEMES}
    nts = {s: a[s]["noise_to_signal"] for s in R1_SCHEMES}
    rank = lambda m, rev=False: sorted(m, key=lambda k: m[k], reverse=rev)  # noqa: E731
    lines = [
        f"Reference agreement (ties = 0.5), all {e['fibonacci']['overall']['n']} expectations: " + ", ".join(f"{s} {ag[s]}" for s in rank(ag, True)) + ".",
        f"Excluding the contestable G6 trade-off: " + ", ".join(f"{s} {ag_x[s]}" for s in rank(ag_x, True)) + ".",
        f"Position expectations (G2): " + ", ".join(f"{s} {pos_g[s]}" for s in rank(pos_g, True)) + ".",
        f"Noise-to-signal (lower = more stable): " + ", ".join(f"{s} {nts[s]}" for s in rank(nts)) + ".",
        f"Missing stage, mean |delta| (canonical positions): " + ", ".join(f"{s} {miss[s]}" for s in rank(miss)) + ".",
        f"Pair-ranking change rate when the same stage is missing in both journeys: " + ", ".join(f"{s} {flip[s]}" for s in rank(flip)) + ".",
        f"Escalating minus de-escalating (mean points): " + ", ".join(f"{s} {d[s]['mean_diff']}" for s in R1_SCHEMES) + ".",
    ]
    fib, lin = "fibonacci", "linear"
    adv_over_equal_max = pos_g[fib] > pos_g["equal"] and pos_g[fib] > pos_g["max_pressure"] and d[fib]["mean_diff"] > 0
    vs_linear = ag_x[fib] - ag_x[lin]
    verdict = []
    if adv_over_equal_max:
        verdict.append("Fibonacci (like every recency-increasing scheme) has a measurable advantage over EQUAL and MAX on the "
                       "position and escalation expectations, which those order-free schemes cannot satisfy (they tie).")
    if abs(vs_linear) < 1e-9:
        verdict.append("Against LINEAR (and the geometric-phi analogue) there is no measurable advantage on the reference "
                       "expectations: they satisfy the same pairs.")
    elif vs_linear > 0:
        verdict.append(f"Against LINEAR, Fibonacci satisfies more non-contestable expectations (+{r(vs_linear)}).")
    else:
        verdict.append(f"Against LINEAR, Fibonacci satisfies FEWER non-contestable expectations ({r(vs_linear)}).")
    raw = {s: a[s]["mean"] for s in R1_SCHEMES}
    last = R1_TYPES[-1]
    miss_last = {s: b["mean_abs_delta_by_dropped_stage"]["canonical"][s][last] for s in R1_SCHEMES}
    c = R1["c_repeated_signals"]
    dup_last = {s: c["cross_stage_mean_delta_by_target_stage"][s][last] for s in R1_SCHEMES}
    dup_max = {s: c["cross_stage_delta"][s]["max"] for s in R1_SCHEMES}
    verdict.append(
        f"Robustness: mean |delta| under single-observation noise is practically identical for the weighted schemes "
        f"(fibonacci {raw[fib]}, linear {raw[lin]}, equal {raw['equal']} points; noise-to-signal fibonacci {nts[fib]} vs linear "
        f"{nts[lin]} vs equal {nts['equal']}, lower for fibonacci only because its scores are more spread out). The cost of recency "
        f"weighting is concentrated leverage of the LAST stage: dropping the {last} stage moves fibonacci by {miss_last[fib]} points "
        f"(linear {miss_last[lin]}, equal {miss_last['equal']}); duplicating a flagged string into the {last} stage adds "
        f"{dup_last[fib]} points on average (linear {dup_last[lin]}, equal {dup_last['equal']}; worst case fibonacci {dup_max[fib]} vs "
        f"linear {dup_max[lin]} vs equal {dup_max['equal']}). The pair-ranking change rate with a missing stage is "
        f"{flip[fib]} for fibonacci vs {flip[lin]} linear and {flip['equal']} equal.")
    g6 = {s: e[s]["per_group"]["G6_contestable_tradeoff"]["agreement_ties_half"] for s in R1_SCHEMES}
    verdict.append(f"On the contestable G6 trade-off (persistence over 3 stages vs one tactic at the exit) fibonacci scores "
                   f"{g6[fib]} vs linear {g6[lin]}: Fibonacci's exit weight (F(7)=13) lets one exit tactic outrank pressure at "
                   "three earlier stages. Overall: Fibonacci's measurable advantage is over order-free schemes (equal, max); it is "
                   "not measurably better than linear or a smooth geometric weighting, and it is more sensitive to the final stage.")
    return {"summary_lines": lines, "verdict": verdict}


# ============================================================================ R2 MCAI aggregation
FACTORS = ("weight", "size", "fill", "link", "grey", "opacity")
CUES = ("pretick", "details", "navaway")
ACCEPT_TEXT = "Yes, add protection to my order"      # AUTHORED
NEUTRAL_DECLINE = "No thanks"                         # AUTHORED
PRETICK_LABEL = "Add extended warranty for $9.99"     # AUTHORED


def build_choice_html(vis: set, cues: set, decline_text: str) -> str:
    """AUTHORED synthetic choice set: benchmark-B visual factors + inferred-interaction cues."""
    acc = {"font-size": "20px" if "size" in vis else "16px", "font-weight": "700" if "weight" in vis else "400",
           "opacity": "1", "border": "1px solid #111111"}
    dec = {"font-size": "13px" if "size" in vis else "16px", "font-weight": "400",
           "opacity": "0.55" if "opacity" in vis else "1", "border": "1px solid #111111"}
    if "fill" in vis:
        acc.update({"background-color": "#0f766e", "color": "#ffffff"}); dec["background-color"] = "transparent"
    else:
        acc.update({"background-color": "#ffffff", "color": "#111111"}); dec["background-color"] = "#ffffff"
    dec["color"] = "#b0b0b0" if "grey" in vis else "#111111"
    css = lambda d: "; ".join(f"{k}: {v}" for k, v in d.items())  # noqa: E731
    if "navaway" in cues:     # decline is a link to ANOTHER page (an <a role=button> unless the 'link' factor is on)
        role = "" if "link" in vis else ' role="button"'
        dec_el = f'<a href="/cart/continue-without-protection" class="dec"{role}>{decline_text}</a>'
    elif "link" in vis:
        dec_el = f'<a href="#" class="dec">{decline_text}</a>'
    else:
        dec_el = f'<button type="button" class="dec">{decline_text}</button>'
    if "details" in cues:
        dec_el = f"<details><summary>More options</summary>{dec_el}</details>"
    pre = f'<label><input type="checkbox" name="warranty" checked> {PRETICK_LABEL}</label>' if "pretick" in cues else ""
    return (f"<!doctype html><html><head><style>\nbody{{background-color:#ffffff}}\n.acc{{{css(acc)}}}\n.dec{{{css(dec)}}}\n</style></head>"
            f"<body><main><div class=\"offer\"><p>Add purchase protection to this order?</p>{pre}"
            f"<button type=\"button\" class=\"acc\">{ACCEPT_TEXT}</button> {dec_el}</div></main></body></html>")


AGGS = ("weighted_mean (selected)", "geometric", "max", "power2", "noisy_or")


def exp_R2(ctx: Ctx):
    from scipy.stats import kendalltau, spearmanr
    from backend.intelligence.aggregation import alternatives
    from backend.intelligence.config import MCAI_WEIGHTS
    from backend.intelligence.mcai import compute_mcai, is_decline
    rng = RNG["R2"]
    t = ctx.test
    mis = t[(t["label"] == "misdirection") & t["text"].map(is_decline)][["row_id", "text"]].values.tolist()
    picks = [mis[i] for i in rng.choice(len(mis), 3, replace=False)]
    wordings = [("neutral", None, NEUTRAL_DECLINE)] + [("misdirection", int(rid), txt) for rid, txt in picks]
    cases = []
    t0 = time.perf_counter()
    for vk in range(len(FACTORS) + 1):
        for vis in itertools.combinations(FACTORS, vk):
            for ck in range(len(CUES) + 1):
                for cues in itertools.combinations(CUES, ck):
                    for wname, wrow, dtxt in wordings:
                        html = build_choice_html(set(vis), set(cues), dtxt)
                        res = compute_mcai("html", ctx.score_fn, html=html)
                        c = {"vis": vis, "cues": cues, "wording": wname, "decline_row_id": wrow, "decline_text": dtxt,
                             "level": len(vis) + len(cues) + (1 if wname == "misdirection" else 0), "detected": False}
                        sets = [cs for cs in res.choice_sets if cs.accept is not None and cs.accept.text == ACCEPT_TEXT
                                and cs.decline.text == dtxt] if res.available else []
                        if sets:
                            cs = sets[0]
                            dims = {d.name: d for d in cs.dimensions}
                            vals = {k: d.score for k, d in dims.items() if d.available}
                            c.update({"detected": True, "dims": vals, "mcai": cs.mcai,
                                      "inferred_extras": [f.feature for f in dims["interaction"].features if f.feature != "steps"],
                                      "alts": alternatives(vals, MCAI_WEIGHTS)})
                            assert abs(c["alts"]["weighted_mean (selected)"] - cs.mcai) < 0.011, (cs.mcai, c["alts"])
                        cases.append(c)
    ok = [c for c in cases if c["detected"]]
    cue_feature = {"pretick": "preselected_optouts", "details": "collapsed_decline", "navaway": "decline_navigates_away"}
    cue_ok = sum(1 for c in ok if set(c["inferred_extras"]) == {cue_feature[q] for q in c["cues"]})
    lv = np.array([c["level"] for c in ok])
    out = {"synthetic": True, "labels_from_construction": True,
           "kind": "synthetic labelled benchmark; designed level = authors' construction, not ground truth",
           "design": ("All 64 subsets of the 6 benchmark-B visual factors (weight, size, fill, link, grey, opacity) x all 8 subsets of 3 "
                      "inferred-interaction cues {pretick: a pre-ticked 'Add extended warranty for $9.99' checkbox (AUTHORED) in the offer "
                      "container; details: decline inside a collapsed <details>; navaway: decline is a link to another page} x 4 decline "
                      "wordings (AUTHORED neutral 'No thanks' + 3 random TEST-split misdirection rows that the decline lexicon recognises) "
                      f"= {len(cases)} HTML choice sets. Designed level = #visual factors + #interaction cues + 1 if misdirection wording "
                      "(0..10). Dimension scores from compute_mcai; every aggregator applied to the same available dimensions with "
                      "MCAI weights via aggregation.alternatives()."),
           "n_cases": len(cases), "n_detected": len(ok), "n_interaction_cues_recovered_exactly": cue_ok,
           "misdirection_decline_row_ids": [w[1] for w in wordings[1:]], "seconds": r(time.perf_counter() - t0, 1),
           "dimension_means_by_cue_count": {k: r(float(np.mean([c["dims"]["interaction"] for c in ok if len(c["cues"]) == k])))
                                            for k in range(4)},
           "semantic_by_wording": {w[0] + ("" if w[1] is None else f"_{w[1]}"): r(float(np.mean([c["dims"]["semantic"] for c in ok
                                                                                                 if c["decline_text"] == w[2]])))
                                   for w in wordings},
           "per_aggregator": {}}
    idx = {(c["vis"], c["cues"], c["decline_text"]): c for c in ok}
    for agg in AGGS:
        sc = np.array([c["alts"][agg] for c in ok])
        rho, p = spearmanr(lv, sc)
        tau, _ = kendalltau(lv, sc)
        viol = flat = edges = 0
        for (vis, cues, dtxt), c in idx.items():
            adds = [(tuple(x for x in FACTORS if x in set(vis) | {f}), cues) for f in FACTORS if f not in vis] + \
                   [(vis, tuple(x for x in CUES if x in set(cues) | {q})) for q in CUES if q not in cues]
            for v2, c2k in adds:
                c2 = idx.get((v2, c2k, dtxt))
                if c2 is None:
                    continue
                edges += 1
                dlt = c2["alts"][agg] - c["alts"][agg]
                viol += dlt < -1e-9
                flat += abs(dlt) <= 1e-9
        w_dec = w_edges = 0
        for (vis, cues, dtxt), c in idx.items():
            if dtxt != NEUTRAL_DECLINE:
                continue
            for _, _, mtxt in wordings[1:]:
                c2 = idx.get((vis, cues, mtxt))
                if c2 is not None:
                    w_edges += 1; w_dec += c2["alts"][agg] < c["alts"][agg] - 1e-9
        dims_all = [c["dims"] for c in ok]
        any_pos = [max(d.values()) > 0 for d in dims_all]
        collapsed = [c["alts"][agg] == 0 and ap for c, ap in zip(ok, any_pos)]
        one_zero = [sum(v == 0 for v in d.values()) == 1 for d in dims_all]
        # dominance: does the max dimension alone explain the score?
        from backend.intelligence.aggregation import alternatives as _alt
        dom, multi, eq_max = 0, 0, 0
        for c in ok:
            d = c["dims"]
            nz = [k for k, v in d.items() if v > 0]
            top = max(d, key=d.get)
            eq_max += abs(c["alts"][agg] - 100 * d[top]) < 1.0
            if len(nz) == len(d) >= 2:                           # all dimensions > 0: no collapse confound
                multi += 1
                only = {k: (v if k == top else 0.0) for k, v in d.items()}
                dom += abs(c["alts"][agg] - _alt(only, MCAI_WEIGHTS)[agg]) < 1.0
        means = [float(np.mean(sc[lv == L])) for L in range(int(lv.max()) + 1)]
        rho_w = {}
        for _, _, dtxt in wordings:
            m = np.array([c["decline_text"] == dtxt for c in ok])
            rho_w[dtxt] = float(spearmanr(lv[m], sc[m])[0]) if np.ptp(sc[m]) > 0 else float("nan")   # nan: constant score
        out["per_aggregator"][agg] = {
            "spearman_rho_vs_level": r(float(rho)), "spearman_p": float(p), "kendall_tau_b": r(float(tau)),
            "spearman_rho_within_fixed_wording_mean": r(float(np.mean(list(rho_w.values())))),   # None = score constant within a wording
            "spearman_rho_within_fixed_wording": {k: r(v) for k, v in rho_w.items()},
            "per_level_mean": [r(m, 2) for m in means],
            "adjacent_level_mean_violations": int(sum(means[i + 1] < means[i] - 1e-9 for i in range(len(means) - 1))),
            "factor_edges": {"edges": edges, "violations": int(viol), "flat_share": r(flat / edges), "strictly_increasing_share": r(1 - (flat + viol) / edges)},
            "wording_edges_neutral_to_misdirection": {"edges": w_edges, "decreases": int(w_dec),
                                                      "note": "dimension-level effect (semantic sub-scores), not an aggregator property"},
            "collapse_to_zero_share": r(float(np.mean(collapsed))),
            "collapse_share_among_cases_with_exactly_one_zero_dimension": r(float(np.mean([cl for cl, oz in zip(collapsed, one_zero) if oz]))
                                                                            if any(one_zero) else None),
            "cases_with_exactly_one_zero_dimension": int(sum(one_zero)),
            "dominance_share": r(dom / multi) if multi else None,
            "score_equals_max_dimension_share": r(eq_max / len(ok)),
            "mean_score": r(float(sc.mean()), 2), "range": [r(float(sc.min()), 2), r(float(sc.max()), 2)],
        }
    out["metric_definitions"] = {
        "factor_edges": "(case, same case + one visual factor or one interaction cue), identical decline text; violation = score decreases; flat = unchanged.",
        "collapse_to_zero_share": "share of cases where the aggregate is 0 although at least one dimension is > 0.",
        "dominance_share": ("among cases where every dimension is > 0: share where zeroing every dimension except the largest changes "
                            "the aggregate by < 1 point (the max dimension alone explains the score)."),
        "score_equals_max_dimension_share": "share of all cases where the aggregate is within 1 point of 100 x the largest dimension.",
    }
    pa = out["per_aggregator"]
    best_rho = max(AGGS, key=lambda a: pa[a]["spearman_rho_vs_level"])
    best_rho_w = max(AGGS, key=lambda a: pa[a]["spearman_rho_within_fixed_wording_mean"] if pa[a]["spearman_rho_within_fixed_wording_mean"] is not None else -2)
    sel = "weighted_mean (selected)"
    sem = out["semantic_by_wording"]
    out["semantic_dimension_finding"] = (
        f"The semantic dimension scores the AUTHORED neutral decline 'No thanks' at {sem['neutral']} but the TEST-split "
        f"misdirection declines at {', '.join(str(v) for k, v in sem.items() if k != 'neutral')}: the classifier assigns high "
        "P(misdirection) to 'No thanks' (dataset label bias, see benchmark D) and the short neutral decline also maximises the "
        "word-count imbalance sub-score. So switching to misdirection wording LOWERS the semantic score on every wording edge; this "
        "is a property of the semantic dimension, not of any aggregator, and it caps every aggregator's rank agreement with the "
        "designed level (which counts misdirection wording as +1).")
    out["recommendation"] = {
        "best_spearman": best_rho,
        "best_spearman_within_fixed_wording": best_rho_w,
        "text": (f"Spearman rho vs designed level: " + ", ".join(f"{a} {pa[a]['spearman_rho_vs_level']}" for a in AGGS) + ". "
                 f"Mean rho within a fixed decline wording (removes the semantic confound): "
                 + ", ".join(f"{a} {pa[a]['spearman_rho_within_fixed_wording_mean']}" for a in AGGS) + ". "
                 f"Geometric collapses to 0 in {pa['geometric']['collapse_to_zero_share']} of cases (every case with a zero dimension, "
                 f"e.g. no interaction cue or no visual factor); max is flat on {pa['max']['factor_edges']['flat_share']} of factor edges "
                 f"(dominance {pa['max']['dominance_share']}). The selected weighted mean has 0 monotonicity violations, "
                 f"{pa[sel]['factor_edges']['flat_share']} flat edges, no collapse and dominance {pa[sel]['dominance_share']}. "
                 + ("Recommendation: keep the weighted arithmetic mean - it has the highest rank agreement and is the only one with "
                    "exact additive attribution." if best_rho == sel else
                    f"{best_rho} has the highest overall rank agreement ({pa[best_rho]['spearman_rho_vs_level']} vs "
                    f"{pa[sel]['spearman_rho_vs_level']} for the selected mean), but it achieves this by mapping every design with a zero "
                    f"dimension to 0 (collapse share {pa[best_rho]['collapse_to_zero_share']}, flat factor edges "
                    f"{pa[best_rho]['factor_edges']['flat_share']}): adding a visual factor to a page with no interaction cue does not "
                    "move the score at all. Recommendation: keep the weighted arithmetic mean (no collapse, strictly monotone on every "
                    "factor edge, exact additive attribution); the small rho gap is the price of not collapsing. Fix the semantic "
                    "dimension's 'No thanks' bias separately."))}
    return out, ok


# ============================================================================ demo client
def make_client(ctx: Ctx, ocr: bool):
    from fastapi.testclient import TestClient
    import backend.main as main
    from backend.intelligence.engine import IntelligenceEngine
    main.STATE.update({"clf": ctx.clf, "auditor": ctx.auditor, "intel": IntelligenceEngine(ctx.clf),
                       "metrics": json.loads((ARTIFACTS / "metrics.json").read_text()), "ocr": ocr})
    return TestClient(main.app, raise_server_exceptions=True), main


# ============================================================================ R3 CDI aggregation / double counting
def _cdi_from(values: dict, avail_keys) -> float | None:
    from backend.intelligence.cdi import ComponentInput, compute_cdi
    from backend.intelligence.config import CDI_WEIGHTS
    inputs = [ComponentInput(key=k, available=k in avail_keys and values.get(k) is not None, value=values.get(k)) for k in CDI_WEIGHTS]
    return compute_cdi(inputs, with_sensitivity=False).cdi


def _zero_filled(values: dict) -> float:
    from backend.intelligence.config import CDI_WEIGHTS
    return round(100 * sum(CDI_WEIGHTS[k] * v for k, v in values.items() if v is not None), 2)


def _asym_variants(mcai):
    from backend.intelligence.mcai import combine
    vi = [combine(cs.dimensions, only=("visual", "interaction"))[0] for cs in mcai.choice_sets]
    full = [combine(cs.dimensions)[0] for cs in mcai.choice_sets]
    vi = [v for v in vi if v is not None]; full = [v for v in full if v is not None]
    return (max(vi) / 100 if vi else None), (max(full) / 100 if full else None)


def _strength(f) -> float:
    from backend.intelligence.config import OBSERVED_FACT_STRENGTH, OBSERVED_WEAK_FACT_STRENGTH
    return OBSERVED_FACT_STRENGTH if "cost or consent" in (f.short_reason or "") else OBSERVED_WEAK_FACT_STRENGTH


def exp_R3(ctx: Ctx):
    from backend.intelligence.aggregation import alternatives
    from backend.intelligence.config import CDI_WEIGHTS
    from backend.intelligence.pressure import obstruction_facts
    from backend.schemas import AuditResponse, JourneyResponse
    from scipy.stats import spearmanr
    client, _ = make_client(ctx, ocr=False)
    samples = json.loads((ROOT / "data" / "samples" / "samples.json").read_text())
    journeys = json.loads((ROOT / "data" / "samples" / "journeys.json").read_text())
    demo = []
    for s in samples:
        if s["input_type"] == "text":
            resp = client.post("/api/audit/text", json={"text": s["content"]})
        else:
            resp = client.post("/api/audit/html", data={"html": s["content"]})
        a = AuditResponse.model_validate(resp.json())
        intel = a.intelligence
        comps = {c.key: c for c in intel.cdi.components}
        vals = {k: c.value for k, c in comps.items() if c.available}
        asym_vi, asym_full = _asym_variants(intel.mcai) if intel.mcai.available else (None, None)
        facts = obstruction_facts(a.findings)
        row = {"id": s["id"], "kind": "single page", "input_type": s["input_type"],
               "components": {k: r(v) for k, v in vals.items()}, "cdi_selected": intel.cdi.cdi,
               "alternatives": intel.cdi.alternatives,
               "missing_components": [k for k in CDI_WEIGHTS if k not in vals],
               "cdi_zero_filled": _zero_filled(vals) if len(vals) < 3 else None}
        # (ii) MCAI semantic inside choice asymmetry
        v2 = dict(vals)
        if asym_full is not None:
            v2["choice_asymmetry"] = asym_full
        row["ii_semantic_in_asymmetry"] = {"asym_visual_interaction": r(asym_vi), "asym_with_semantic": r(asym_full),
                                           "cdi_if_semantic_included": _cdi_from(v2, set(v2)) if asym_full is not None else None,
                                           "inflation_points": (r(_cdi_from(v2, set(v2)) - intel.cdi.cdi, 2) if asym_full is not None else None)}
        # (iii) obstruction max vs sum
        if facts:
            ssum = sum(x for x, _ in facts)
            v3 = dict(vals); v3["observed_obstruction"] = min(1.0, ssum)
            row["iii_obstruction"] = {"n_facts": len(facts), "strengths": [x for x, _ in facts], "max_selected": max(x for x, _ in facts),
                                      "sum_unclipped": ssum, "cdi_with_sum_clipped": _cdi_from(v3, set(v3)),
                                      "cdi_with_sum_unclipped_raw": r(100 * sum(CDI_WEIGHTS[k] * (ssum if k == "observed_obstruction" else v)
                                                                                for k, v in vals.items()) / sum(CDI_WEIGHTS[k] for k in vals), 2)}
        demo.append(row)
    for jd in journeys:
        resp = client.post("/api/audit/journey", data={"stages": json.dumps(jd["stages"])})
        j = JourneyResponse.model_validate(resp.json())
        comps = {c.key: c for c in j.cdi.components}
        vals = {k: c.value for k, c in comps.items() if c.available}
        p = [st.pressure for st in j.stages]
        fib = j.fspm.scores["fibonacci"] / 100
        row = {"id": jd["id"], "kind": "journey (SYNTHETIC demo)", "components": {k: r(v) for k, v in vals.items()},
               "cdi_selected": j.cdi.cdi, "alternatives": j.cdi.alternatives, "stage_pressures": p, "fspm": j.fspm.scores,
               "missing_components": [k for k in CDI_WEIGHTS if k not in vals],
               "cdi_zero_filled": _zero_filled(vals) if len(vals) < 3 else None}
        naive_a = dict(vals); naive_a["content_pressure"] = min(1.0, fib + float(np.mean(p)))
        naive_b = dict(vals); naive_b["content_pressure"] = min(1.0, float(np.sum(p)))
        row["i_fspm_instead_of_page_content"] = {
            "selected_content": r(fib), "naive_fspm_plus_mean_page_pressure": r(min(1.0, fib + float(np.mean(p)))),
            "naive_sum_of_stage_pressures": r(min(1.0, float(np.sum(p)))), "naive_sum_unclipped": r(float(np.sum(p))),
            "cdi_selected": j.cdi.cdi, "cdi_naive_fspm_plus_page": _cdi_from(naive_a, set(naive_a)),
            "cdi_naive_sum": _cdi_from(naive_b, set(naive_b))}
        asym_vi, asym_full = _asym_variants(j.mcai) if j.mcai.available else (None, None)
        v2 = dict(vals)
        if asym_full is not None:
            v2["choice_asymmetry"] = asym_full
        row["ii_semantic_in_asymmetry"] = {"asym_visual_interaction": r(asym_vi), "asym_with_semantic": r(asym_full),
                                           "cdi_if_semantic_included": _cdi_from(v2, set(v2)) if asym_full is not None else None,
                                           "inflation_points": r(_cdi_from(v2, set(v2)) - j.cdi.cdi, 2) if asym_full is not None else None}
        dom = [(x, (k, ref.finding_id)) for k, st in enumerate(j.stages) for x, ref in obstruction_facts(st.audit.findings, k)]
        temporal = []
        for ev in j.temporal.events:
            if ev.category == "new_preselected_option" and ev.deceptive_signal:
                for ref in ev.evidence:
                    f = next((f for f in j.stages[ref.stage_index].audit.findings if f.id == ref.finding_id), None)
                    if f is not None:
                        temporal.append((_strength(f), (ref.stage_index, ref.finding_id)))
        if dom or temporal:
            allf = dom + temporal
            uniq = {}
            for x, key in allf:
                uniq[key] = max(x, uniq.get(key, 0))
            row["iii_obstruction"] = {"dom_facts": len(dom), "temporal_facts": len(temporal), "unique_elements": len(uniq),
                                      "max_selected": max(uniq.values()), "sum_without_dedup": sum(x for x, _ in allf),
                                      "sum_with_dedup": sum(uniq.values()), "engine_value": vals.get("observed_obstruction")}
        demo.append(row)

    # ---------------- synthetic component vectors
    rng = RNG["R3"]
    n = 5000
    rows = []
    for _ in range(n):
        v = {}
        for k, pa in (("content_pressure", 1.0), ("choice_asymmetry", 0.5), ("observed_obstruction", 0.6)):
            if rng.random() < pa:
                v[k] = 0.0 if rng.random() < 0.3 else float(rng.uniform())
        rows.append(v)
    A = {a: np.array([alternatives(v, CDI_WEIGHTS)[a] for v in rows]) for a in AGGS}
    sel = A["weighted_mean (selected)"]
    zf = np.array([_zero_filled(v) for v in rows])
    missing = np.array([len(v) < 3 for v in rows])
    from backend.intelligence.cdi import band
    band_change = np.mean([band(a) != band(b) for a, b, m in zip(sel, zf, missing) if m])
    syn = {"design": (f"{n} SYNTHETIC component vectors: content_pressure always available, choice_asymmetry available w.p. 0.5, "
                      "observed_obstruction w.p. 0.6; each available value is exactly 0 w.p. 0.3, else U(0,1)."),
           "per_aggregator": {a: {"mean": r(float(A[a].mean()), 2), "spearman_vs_selected": r(float(spearmanr(A[a], sel)[0])),
                                  "mean_minus_selected": r(float((A[a] - sel).mean()), 2),
                                  "collapse_to_zero_while_selected_positive": r(float(np.mean((A[a] == 0) & (sel > 0)))),
                                  "equals_max_component_share": r(float(np.mean(np.abs(A[a] - A["max"]) < 1.0)))} for a in AGGS},
           "missing_components": {"share_with_missing": r(float(missing.mean())),
                                  "mean_renormalised": r(float(sel[missing].mean()), 2), "mean_zero_filled": r(float(zf[missing].mean()), 2),
                                  "mean_deflation_points_if_zero_filled": r(float((sel - zf)[missing].mean()), 2),
                                  "max_deflation_points": r(float((sel - zf)[missing].max()), 2),
                                  "band_changes_if_zero_filled": r(float(band_change))}}
    # missing component demo: a text page has only content pressure
    out = {"kind": "demo inputs through the live API path + synthetic component vectors",
           "synthetic_note": "Journeys in journeys.json are SYNTHETIC demos (authored order/markup + verbatim TEST strings); synthetic vectors are random.",
           "demo": demo, "synthetic_vectors": syn}
    out["conclusion"] = r3_conclusion(out)
    return out


def r3_conclusion(R3) -> list[str]:
    lines = []
    for d in R3["demo"]:
        if "i_fspm_instead_of_page_content" in d:
            x = d["i_fspm_instead_of_page_content"]
            lines.append(f"{d['id']}: CDI {x['cdi_selected']} with FSPM as journey content; naive FSPM + mean page pressure would give "
                         f"{x['cdi_naive_fspm_plus_page']}, naive sum of stage pressures {x['cdi_naive_sum']}.")
        ii = d.get("ii_semantic_in_asymmetry") or {}
        if ii.get("inflation_points") is not None:
            lines.append(f"{d['id']}: including MCAI semantic in choice asymmetry would move CDI by {ii['inflation_points']} points "
                         f"(asymmetry {ii['asym_visual_interaction']} -> {ii['asym_with_semantic']}).")
        if d.get("cdi_zero_filled") is not None:
            lines.append(f"{d['id']}: missing {d['missing_components']}; renormalised CDI {d['cdi_selected']} vs zero-filled {d['cdi_zero_filled']}.")
    m = R3["synthetic_vectors"]["missing_components"]
    lines.append(f"Synthetic vectors with a missing component: zero-filling would deflate the CDI by {m['mean_deflation_points_if_zero_filled']} "
                 f"points on average (max {m['max_deflation_points']}) and change the display band in {m['band_changes_if_zero_filled']} of them.")
    return lines


# ============================================================================ R4 counterfactual optimiser
def _selected(e):
    """The candidate the optimiser's documented rule selects (re-derived from e.candidates)."""
    from backend.intelligence.optimiser import TIE
    feas = [c for c in e.candidates if c.feasible]
    best = min(c.score for c in feas)
    near = [c for c in feas if c.score <= best + TIE]
    changes = [c for c in near if c.variant != "original"]
    return min(changes or near, key=lambda c: (c.edit_distance, c.score))


def _chosen(e):
    """The candidate the returned edit actually REPORTS (action / redesigned text)."""
    if e.action == "no_safe_rewrite":
        return next(c for c in e.candidates if c.variant == "original")
    if e.action == "remove_unverifiable_claim":
        return next(c for c in e.candidates if c.variant == "remove_unverifiable_claim")
    if e.action == "equalise_choice" and e.redesigned == e.original:
        return next(c for c in e.candidates if c.variant == "original")
    return next(c for c in e.candidates if c.variant != "original" and c.text == e.redesigned)


def _engine_edits(ctx: Ctx, text: str):
    from backend.intelligence.counterfactual import plan_edits
    from backend.intelligence.mcai import compute_mcai
    from backend.intelligence.optimiser import optimise
    audit, cands = ctx.audit_text(text)
    mc = compute_mcai("text", ctx.score_fn, cands=cands, findings=audit.findings)
    base = plan_edits(audit.findings, ctx.score_fn, None, mc.choice_sets)
    return audit, base, optimise(base, ctx.score_fn)


def exp_R4(ctx: Ctx):
    from backend.intelligence.engine import _redesigned_content
    from backend.intelligence.pressure import blend
    from backend.intelligence.optimiser import TIE
    rng = RNG["R4"]
    t = ctx.test
    dark = t[t["label"] != "benign"].reset_index(drop=True)
    per = defaultdict(lambda: {"n": 0, "feasible_change": 0, "reductions": [], "reductions_changed": [], "below_tau": 0,
                               "variants": Counter(), "violations": Counter(), "candidates": 0, "infeasible_candidates": 0,
                               "tie_band_nonmin": 0, "baseline_reductions": [], "decline_texts": Counter()})
    rows_with = rows_changed = 0
    inconsistent = []
    t0 = time.perf_counter()
    for row in dark.itertuples():
        audit, base, edits = _engine_edits(ctx, row.text)
        work = [(b, e) for b, e in zip(base, edits) if e.candidates]
        if not any(f.detection_method == "ml_classifier" for f in audit.findings):
            continue
        rows_with += 1
        changed_any = False
        for b, e in work:
            cat = e.category
            P = per[cat]; A = per["ALL"]
            ch = _selected(e)
            rep = _chosen(e)
            if (rep.variant, rep.text) != (ch.variant, ch.text) and not (rep.text == ch.text and rep.score == ch.score):
                inconsistent.append({"row_id": int(row.row_id), "action": e.action, "category": cat, "original": e.original,
                                     "selected_by_rule": ch.variant, "selected_text": ch.text, "reported_redesigned": e.redesigned,
                                     "candidates": [(c.variant, c.text, c.score, c.feasible) for c in e.candidates]})
            feas = [c for c in e.candidates if c.feasible]
            orig_score = next(c.score for c in e.candidates if c.variant == "original")
            changed = ch.variant != "original"
            red = (orig_score - ch.score) if changed else 0.0
            bred = (b.original_score - b.redesigned_score) if (b.action == "rewrite" and b.redesigned_score is not None) else 0.0
            for Q in (P, A):
                Q["n"] += 1; Q["candidates"] += len(e.candidates)
                Q["infeasible_candidates"] += sum(not c.feasible for c in e.candidates)
                for c in e.candidates:
                    Q["violations"].update(c.violated)
                Q["variants"][ch.variant] += 1
                if ch.variant == "neutral_decline":
                    Q["decline_texts"][ch.text] += 1
                Q["reductions"].append(red); Q["baseline_reductions"].append(bred)
                if changed:
                    Q["feasible_change"] += 1; Q["reductions_changed"].append(red)
                    Q["below_tau"] += ch.score < ctx.tau
                Q["tie_band_nonmin"] += ch.score > min(c.score for c in feas) + 1e-12
            changed_any |= changed
        rows_changed += changed_any
    per_out = {}
    for cat, P in sorted(per.items()):
        per_out[cat] = {"n_edits": P["n"], "feasible_change_share": r(P["feasible_change"] / P["n"]),
                        "mean_score_reduction_all": r(float(np.mean(P["reductions"]))),
                        "mean_score_reduction_changed": r(float(np.mean(P["reductions_changed"]))) if P["reductions_changed"] else None,
                        "baseline_minimal_rewrite_mean_reduction_all": r(float(np.mean(P["baseline_reductions"]))),
                        "changed_below_threshold_share": r(P["below_tau"] / P["feasible_change"]) if P["feasible_change"] else None,
                        "winning_variants": dict(P["variants"].most_common()), "candidates": P["candidates"],
                        "winning_neutral_decline_texts": dict(P["decline_texts"].most_common()),
                        "infeasible_candidates": P["infeasible_candidates"], "violation_counts": dict(P["violations"].most_common()),
                        "chosen_not_exact_min_due_to_tie_band": P["tie_band_nonmin"]}
    out = {"synthetic": False, "kind": "model-estimated optimiser study on every dark TEST-split row (scores by the same classifier)",
           "design": ("Each TEST row with label != benign is audited alone (segment_text -> Auditor.audit 'text'), MCAI choice sets "
                      "are extracted, plan_edits + optimise run exactly as engine.single. Categories are FINDING (predicted) categories; "
                      "'choice_asymmetry' = equalise_choice edits for uncovered decline options. score reduction = P(dark) original - "
                      "P(dark) of the selected candidate (0 when the original is kept; removal scores 0). Baseline = the pre-optimiser "
                      "rule-based minimal rewrite from plan_edits."),
           "threshold_tau": ctx.tau, "dark_rows": int(len(dark)), "dark_rows_with_ml_findings": rows_with,
           "rows_with_at_least_one_feasible_change": rows_changed,
           "rows_with_change_share": r(rows_changed / rows_with), "per_category": per_out,
           "report_vs_selection_inconsistencies": {"count": len(inconsistent), "examples": inconsistent[:5],
                                                   "note": ("edits whose returned action/redesigned text differs from the candidate the "
                                                            "optimiser's own rule selects (re-derived from e.candidates)")},
           "seconds": r(time.perf_counter() - t0, 1)}

    # ---------------- exactness check (brute force over the full product space)
    t1 = time.perf_counter()
    texts = dark["text"].tolist()
    lam = 0.5
    mism_greedy = mism_opt = mism_engine = 0
    gaps, sizes, examples = [], [], []
    n_pages = 0
    while n_pages < 200:
        k = int(rng.integers(3, 7))
        page = "\n".join(texts[i] for i in rng.choice(len(texts), k, replace=False))
        audit, _, edits = _engine_edits(ctx, page)
        ml_f = [f for f in audit.findings if f.detection_method == "ml_classifier"]
        if len(ml_f) < 2:
            continue                                             # need a multi-element page to make the check non-trivial
        n_pages += 1
        by_t = {e.target: e for e in edits}
        elems = []                                               # (category, g-values of feasible candidates, chosen g)
        for f in ml_f:
            e = by_t[f.id]

            def g(c):
                if c.variant == "remove_unverifiable_claim":
                    return 0.0
                if c.variant == "original":
                    return float(f.model_score)
                return c.score if c.score >= ctx.tau else 0.0
            feas = [c for c in e.candidates if c.feasible]
            elems.append((f.category, np.array([g(c) for c in feas]), g(_chosen(e))))
        n = len(elems)
        sizes.append(int(np.prod([len(v) for _, v, _ in elems])))
        grids = []
        for i, (_, v, _) in enumerate(elems):
            shape = [1] * n; shape[i] = len(v)
            grids.append(v.reshape(shape))
        full = tuple(len(v) for _, v, _ in elems)
        chan = []
        for cat in CATS:
            mem = [grids[i] for i, (c, _, _) in enumerate(elems) if c == cat]
            chan.append(np.broadcast_to(functools.reduce(np.maximum, mem), full) if mem else np.zeros(full))
        stack = np.stack(chan)
        Pgrid = lam * stack.max(axis=0) + (1 - lam) * stack.sum(axis=0) / len(CATS)
        p_bf = float(Pgrid.min())
        best_g = {}
        for c, v, _ in elems:
            best_g[c] = max(best_g.get(c, 0.0), float(v.min()))
        p_greedy = blend(best_g)
        opt_g = {}
        for c, _, chg in elems:
            opt_g[c] = max(opt_g.get(c, 0.0), chg)
        p_opt = blend(opt_g)
        p_engine = _redesigned_content(audit.findings, edits, ctx.tau)
        mism_greedy += abs(p_greedy - p_bf) > 1e-12
        mism_opt += abs(p_opt - p_bf) > 1e-12
        mism_engine += abs(p_engine - p_opt) > 1e-12
        gaps.append(p_opt - p_bf)
        if abs(p_opt - p_bf) > 1e-12 and len(examples) < 5:
            examples.append({"page": page, "p_bruteforce": r(p_bf, 6), "p_optimiser": r(p_opt, 6)})
    out["exactness_check"] = {
        "design": ("200 random multi-element pages: 3-6 distinct dark TEST strings joined by newlines (pages with < 2 ML findings "
                   "redrawn). For every ML finding the optimiser's FEASIBLE candidates are taken with the engine's redesign rule "
                   "(kept original -> its model score; removal -> 0; rewrite -> its score, or 0 if below tau). Brute force = minimum "
                   "redesigned content pressure over the FULL product space of per-element feasible candidates (vectorised)."),
        "pages": n_pages, "product_space_size": stats(sizes), "total_combinations_evaluated": int(sum(sizes)),
        "mismatches_per_element_argmin_vs_bruteforce": int(mism_greedy),
        "mismatches_optimiser_selection_vs_bruteforce": int(mism_opt),
        "max_gap_optimiser_minus_bruteforce": r(float(max(gaps)), 6),
        "mismatches_engine_redesigned_content_vs_optimiser_selection": int(mism_engine),
        "tie_band": TIE,
        "note": ("Per-element argmin is the exact global minimum (theory: pressure is monotone non-decreasing in each element's "
                 "score and constraints are per-element). The optimiser's actual selection may differ from the argmin only through "
                 f"its documented tie band (prefer a real change / smaller edit within {TIE} of the best score)."),
        "examples_of_gap": examples, "seconds": r(time.perf_counter() - t1, 1)}
    return out


# ============================================================================ R5 cross-modal fusion
def _render_text_png(lines: list[str]) -> bytes:
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    W, lh = 1100, 40
    img = Image.new("RGB", (W, 40 + lh * len(lines)), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        d.text((30, 20 + i * lh), ln, fill=(17, 17, 17), font=font)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return buf.getvalue()


def exp_R5(ctx: Ctx):
    from backend.extractors import extract_html, ocr_available, ocr_lines
    from backend.schemas import JourneyResponse
    from PIL import Image
    ocr = ocr_available()
    client, _ = make_client(ctx, ocr=ocr)
    html = (ROOT / "data" / "samples" / "checkout_deceptive.html").read_text()
    cands, _ = extract_html(html)
    vis_lines = [c.text for c in cands]
    text_capture = "\n".join(vis_lines)
    png = _render_text_png(vis_lines)
    shot_ok = False
    ocr_lines_read = []
    if ocr:
        ocr_lines_read = [c.text for c in ocr_lines(Image.open(io.BytesIO(png))) if c.extra.get("unit") != "block"]
        shot_ok = len(ocr_lines_read) > 0
    benign_product = json.loads((ROOT / "data" / "samples" / "journeys.json").read_text())[1]["stages"][0]["content"]

    def stages(with_fusion: bool, product: bool, typed: bool):
        out = []
        if product:
            out.append({"name": "Product page", "input_type": "text", "content": benign_product, "synthetic": True,
                        **({"stage_type": "product"} if typed else {})})
        h = len(out)
        out.append({"name": "Checkout (HTML)", "input_type": "html", "content": html, "synthetic": True, **({"stage_type": "checkout"} if typed else {})})
        out.append({"name": "Checkout (text)", "input_type": "text", "content": text_capture, "synthetic": True,
                    **({"stage_type": "checkout"} if typed else {}), **({"fuse_with": h} if with_fusion else {})})
        if shot_ok:
            out.append({"name": "Checkout (screenshot)", "input_type": "screenshot", "file_field": "shot", "synthetic": True,
                        **({"stage_type": "checkout"} if typed else {}), **({"fuse_with": h} if with_fusion else {})})
        return out

    def run(st):
        files = {"shot": ("checkout.png", png, "image/png")} if shot_ok else None
        resp = client.post("/api/audit/journey", data={"stages": json.dumps(st)}, files=files)
        assert resp.status_code == 200, resp.text[:500]
        j = JourneyResponse.model_validate(resp.json())
        cp = next(c for c in j.cdi.components if c.key == "content_pressure")
        return {"units": len(j.fspm.stages), "positions": [s.position for s in j.fspm.stages], "positions_mode": j.fspm.positions_mode,
                "unit_pressures": [s.pressure for s in j.fspm.stages], "fspm": j.fspm.scores, "cdi": j.cdi.cdi,
                "content_pressure": cp.value, "duplicate_pairs": sum(len(g.duplicate_pairs) for g in j.fusion),
                "stage_pressures": [s.pressure for s in j.stages]}
    configs = {}
    for name, product, typed in (("same_page_only", False, False), ("product_then_checkout_typed", True, True),
                                 ("product_then_checkout_untyped", True, False)):
        a, b = run(stages(True, product, typed)), run(stages(False, product, typed))
        configs[name] = {"with_fusion": a, "without_fusion": b,
                         "inflation_avoided": {"fspm": {s: r(b["fspm"][s] - a["fspm"][s], 2) for s in ENGINE_SCHEMES},
                                               "cdi": r(b["cdi"] - a["cdi"], 2)}}
    return {"synthetic": True, "kind": "demo checkout page captured in 2-3 modalities (AUTHORED demo markup + verbatim TEST strings)",
            "design": ("data/samples/checkout_deceptive.html captured as HTML, as text (the visible element texts the HTML extractor "
                       "finds, one per line) and, if Tesseract is available, as a PIL-rendered screenshot of those lines (Arial 22px, "
                       "SYNTHETIC rendering). Each journey is posted to /api/audit/journey with and without fuse_with linking the text "
                       "and screenshot captures to the HTML stage. 'typed' declares stage types (product/checkout x3, so the UNFUSED "
                       "journey falls back to submission order because the types repeat); 'untyped' uses submission order in both."),
            "ocr_available": ocr, "screenshot_included": shot_ok, "ocr_lines_read": len(ocr_lines_read), "visible_lines": len(vis_lines),
            "configs": configs}


# ============================================================================ R6 cache / performance
class _NoCache:
    def stats(self):
        return {}


@contextlib.contextmanager
def _no_request_scorer(clf):
    yield _NoCache()


def exp_R6(ctx: Ctx):
    import backend.main as main
    from backend.extractors import ocr_available
    ocr = ocr_available()
    client, _ = make_client(ctx, ocr=ocr)
    samples = {s["id"]: s for s in json.loads((ROOT / "data" / "samples" / "samples.json").read_text())}
    journey = json.loads((ROOT / "data" / "samples" / "journeys.json").read_text())[0]
    shot = (ROOT / "data" / "samples" / "screenshots" / "www.gasbike.net.png").read_bytes()
    counter = {"calls": 0, "texts": 0}
    orig_predict = ctx.clf.predict

    def counting(texts):
        counter["calls"] += 1; counter["texts"] += len(texts)
        return orig_predict(texts)
    ctx.clf.predict = counting
    reqs = {
        "text": lambda: client.post("/api/audit/text", json={"text": samples["text-deceptive"]["content"]}),
        "html": lambda: client.post("/api/audit/html", data={"html": samples["html-checkout"]["content"]}),
        "journey": lambda: client.post("/api/audit/journey", data={"stages": json.dumps(journey["stages"])}),
    }
    if ocr:
        reqs["screenshot"] = lambda: client.post("/api/audit/screenshot", files={"file": ("www.gasbike.net.png", shot, "image/png")})
    RUNS = 5
    res, bodies = {}, {}
    orig_rs = main.request_scorer
    try:
        for cond in ("with_cache", "without_cache"):
            main.request_scorer = orig_rs if cond == "with_cache" else _no_request_scorer
            for name, fn in reqs.items():
                resp = fn(); assert resp.status_code == 200, (name, cond, resp.text[:300])      # warm-up
                times, calls, texts = [], [], []
                for _ in range(RUNS):
                    counter.update(calls=0, texts=0)
                    t0 = time.perf_counter(); resp = fn(); times.append(time.perf_counter() - t0)
                    assert resp.status_code == 200
                    calls.append(counter["calls"]); texts.append(counter["texts"])
                body = resp.json()
                bodies[(name, cond)] = body
                cdi = body["cdi"]["cdi"] if name == "journey" else (body.get("intelligence") or {}).get("cdi", {}).get("cdi")
                res.setdefault(name, {})[cond] = {"median_ms": r(1000 * statistics.median(times), 1),
                                                  "runs_ms": [r(1000 * x, 1) for x in times],
                                                  "classifier_calls": calls[-1], "texts_scored": texts[-1], "cdi": cdi}
    finally:
        main.request_scorer = orig_rs
        ctx.clf.predict = orig_predict
    for name in res:
        a, b = res[name]["with_cache"], res[name]["without_cache"]
        res[name]["speedup_x"] = r(b["median_ms"] / a["median_ms"], 2)
        res[name]["identical_cdi"] = a["cdi"] == b["cdi"]

        def findings(body):
            if name == "journey":
                return [[(f["id"], f["category"], f["model_score"]) for f in s["audit"]["findings"]] for s in body["stages"]]
            return [(f["id"], f["category"], f["model_score"]) for f in body["findings"]]
        res[name]["identical_findings"] = findings(bodies[(name, "with_cache")]) == findings(bodies[(name, "without_cache")])
    pdf = {}
    for name, ep, body in (("audit_pdf_html", "/api/export/pdf", bodies[("html", "with_cache")]),
                           ("journey_pdf", "/api/export/journey-pdf", bodies[("journey", "with_cache")])):
        client.post(ep, json=body)
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter(); resp = client.post(ep, json=body); times.append(time.perf_counter() - t0)
            assert resp.status_code == 200 and resp.content[:4] == b"%PDF"
        pdf[name] = {"median_ms": r(1000 * statistics.median(times), 1), "runs_ms": [r(1000 * x, 1) for x in times], "bytes": len(resp.content)}
    return {"kind": "latency of the real API code path (FastAPI TestClient, in-process; includes JSON (de)serialisation)",
            "design": (f"Median over {RUNS} runs after one warm-up per endpoint and condition. 'without_cache' monkeypatches "
                       "backend.main.request_scorer to a no-op context manager, so every component calls clf.predict directly. "
                       "clf.predict is wrapped to count batched classifier calls and texts scored (texts include repeats)."),
            "machine": {"python": platform.python_version(), "platform": platform.platform()},
            "ocr_available": ocr, "endpoints": res, "pdf_export": pdf}


# ============================================================================ figures
def fig_R1(R1, figs):
    plt = _style()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    S = list(R1_SCHEMES)
    x = np.arange(len(S))
    cols = [SCHEME_COLORS[s] for s in S]
    ax = axes[0, 0]
    a = R1["a_single_observation_noise"]["per_scheme"]
    ax.bar(x, [a[s]["noise_to_signal"] for s in S], color=cols)
    ax.set_xticks(x, S, rotation=30, ha="right"); ax.set_ylabel("mean |delta| / SD of scores")
    ax.set_title("(a) Single-observation noise (lower = stabler)", loc="left")
    ax = axes[0, 1]
    b = R1["b_missing_stage"]
    ax.bar(x - 0.2, [b["mean_abs_delta_overall"]["canonical"][s] for s in S], 0.4, color=cols, label="mean |delta| (points)")
    ax2 = ax.twinx()
    ax2.plot(x, [100 * b["pair_ranking"]["canonical"][s]["rank_change_rate_same_stage_missing_in_both"] for s in S], "ko", ms=4,
             label="pair-ranking change %")
    ax2.set_ylabel("pair ranking changed (%)"); ax2.grid(False); ax2.set_ylim(0, None)
    ax.set_xticks(x, S, rotation=30, ha="right"); ax.set_ylabel("mean |delta score| when one stage is missing")
    ax.set_title("(b) Missing stage (canonical positions)", loc="left")
    ax = axes[1, 0]
    w = 0.11
    for k, s in enumerate(S):
        vals = [R1["c_repeated_signals"]["cross_stage_mean_delta_by_target_stage"][s][t] or 0 for t in R1_TYPES]
        ax.bar(np.arange(4) + (k - 3) * w, vals, w * 0.95, color=SCHEME_COLORS[s], label=s)
    ax.set_xticks(np.arange(4), list(R1_TYPES), fontsize=8); ax.set_xlabel("stage the string is copied into")
    ax.set_ylabel("mean score increase (points)")
    ax.set_title("(c) Duplicating a flagged string into another stage", loc="left")
    ax.legend(fontsize=6.5, ncol=2)
    ax = axes[1, 1]
    e = R1["e_reference_cases"]["agreement"]
    groups = sorted(e["fibonacci"]["per_group"])
    M = np.array([[e[s]["per_group"][g]["agreement_ties_half"] for g in groups] for s in S])
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(groups)), [g.split("_", 1)[0] + "\n" + g.split("_", 1)[1].replace("_", " ")[:14] for g in groups], fontsize=7)
    ax.set_yticks(range(len(S)), S, fontsize=8); ax.grid(False)
    for i in range(len(S)):
        for j in range(len(groups)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] > 0.6 else "black")
    ax.set_title("(e) Agreement with authors' reference expectations", loc="left")
    fig.text(0.01, 0.005, "SYNTHETIC: journeys built from real TEST-split strings; reference expectations are design intent, not ground truth.",
             fontsize=7, color="#52514e")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIGURES / "research_fspm_robustness.png", dpi=200)
    plt.close(fig)


def fig_R2(R2, ok):
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    cols = {"weighted_mean (selected)": "#2a78d6", "geometric": "#eb6834", "max": "#52514e", "power2": "#1baf7a", "noisy_or": "#c23b7a"}
    ax = axes[0]
    for a in AGGS:
        m = R2["per_aggregator"][a]["per_level_mean"]
        ax.plot(range(len(m)), m, "o-", color=cols[a], ms=3.5, lw=1.6,
                label=f"{a} (rho {R2['per_aggregator'][a]['spearman_rho_vs_level']:.3f})")
    ax.set_xlabel("designed asymmetry level (visual factors + interaction cues + misdirection wording)")
    ax.set_ylabel("mean MCAI (0-100)"); ax.legend(fontsize=7); ax.set_title("MCAI aggregators vs designed level", loc="left")
    ax = axes[1]
    mets = [("collapse_to_zero_share", "collapse to 0"), ("dominance_share", "max-dim dominance")]
    x = np.arange(len(AGGS))
    for k, (key, lab) in enumerate(mets):
        ax.bar(x + (k - 0.5) * 0.38, [R2["per_aggregator"][a][key] for a in AGGS], 0.36, label=lab,
               color=["#eb6834", "#7b4fd6"][k])
    ax.plot(x, [R2["per_aggregator"][a]["factor_edges"]["flat_share"] for a in AGGS], "k^", ms=5, label="flat factor edges")
    ax.set_xticks(x, [a.replace(" (selected)", "*") for a in AGGS], rotation=25, ha="right"); ax.set_ylabel("share of cases / edges")
    ax.legend(fontsize=7); ax.set_title("Failure modes", loc="left")
    fig.text(0.01, 0.005, "SYNTHETIC authored HTML choice sets; designed level from construction (not ground truth).", fontsize=7, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURES / "research_mcai_aggregators.png", dpi=200)
    plt.close(fig)


def fig_R3_R5(R3, R5):
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax = axes[0]
    labels, sel, alt, lab_alt = [], [], [], []
    for d in R3["demo"]:
        if "i_fspm_instead_of_page_content" in d:
            x = d["i_fspm_instead_of_page_content"]
            labels.append(d["id"] + "\nnaive journey content"); sel.append(x["cdi_selected"]); alt.append(x["cdi_naive_sum"])
        ii = d.get("ii_semantic_in_asymmetry") or {}
        if ii.get("cdi_if_semantic_included") is not None:
            labels.append(d["id"] + "\n+MCAI semantic"); sel.append(d["cdi_selected"]); alt.append(ii["cdi_if_semantic_included"])
        if d.get("cdi_zero_filled") is not None:
            labels.append(d["id"] + "\nzero-filled missing"); sel.append(d["cdi_selected"]); alt.append(d["cdi_zero_filled"])
        if (d.get("iii_obstruction") or {}).get("cdi_with_sum_unclipped_raw") is not None:
            labels.append(d["id"] + "\nobstruction sum (unclipped)"); sel.append(d["cdi_selected"])
            alt.append(d["iii_obstruction"]["cdi_with_sum_unclipped_raw"])
    keep = [i for i in range(len(labels)) if sel[i] or alt[i]]           # all-zero benign rows carry no information
    labels, sel, alt = [labels[i] for i in keep], [sel[i] for i in keep], [alt[i] for i in keep]
    y = np.arange(len(labels))
    ax.barh(y - 0.2, sel, 0.4, color="#2a78d6", label="selected rule")
    ax.barh(y + 0.2, alt, 0.4, color="#eb6834", label="naive / alternative")
    ax.set_yticks(y, labels, fontsize=6.5); ax.invert_yaxis(); ax.set_xlabel("CDI (0-100)"); ax.legend(fontsize=7)
    ax.grid(axis="x"); ax.set_title("R3  De-duplication / missing-component rules", loc="left")
    ax = axes[1]
    names = list(R5["configs"])
    x = np.arange(len(names))
    for k, (cond, col) in enumerate((("with_fusion", "#2a78d6"), ("without_fusion", "#eb6834"))):
        ax.bar(x + (k - 0.5) * 0.38, [R5["configs"][n][cond]["fspm"]["fibonacci"] for n in names], 0.36, color=col, label=f"FSPM fib, {cond}")
        ax.plot(x + (k - 0.5) * 0.38, [R5["configs"][n][cond]["cdi"] for n in names], "kD" if k else "ko", ms=4, label=f"CDI, {cond}")
    ax.set_xticks(x, [n.replace("_then_", "\nthen ").replace("_", " ") for n in names], fontsize=7); ax.set_ylabel("score (0-100)")
    ax.legend(fontsize=6.5, loc="lower right", ncol=2); ax.set_ylim(0, 100)
    ax.set_title("R5  Cross-modal fusion (same checkout page)", loc="left")
    fig.text(0.01, 0.005, "Demo inputs (authored markup + verbatim TEST strings); journeys are SYNTHETIC.", fontsize=7, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURES / "research_cdi_fusion.png", dpi=200)
    plt.close(fig)


def fig_R4(R4):
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
    per = {k: v for k, v in R4["per_category"].items() if k != "ALL"}
    cats = list(per)
    x = np.arange(len(cats))
    ax = axes[0]
    ax.bar(x - 0.2, [per[c]["feasible_change_share"] for c in cats], 0.4, color="#2a78d6", label="feasible change found")
    ax.bar(x + 0.2, [per[c]["mean_score_reduction_all"] for c in cats], 0.4, color="#1baf7a", label="mean P(dark) reduction")
    ax.set_xticks(x, cats, rotation=20, ha="right", fontsize=8); ax.legend(fontsize=7)
    ax.set_title("R4  Optimiser on dark TEST rows (model-estimated)", loc="left")
    ax = axes[1]
    allv = R4["per_category"]["ALL"]["winning_variants"]
    ks = list(allv)
    ax.barh(np.arange(len(ks)), [allv[k] for k in ks], color="#7b4fd6")
    ax.set_yticks(np.arange(len(ks)), ks, fontsize=8); ax.invert_yaxis(); ax.grid(axis="x")
    ax.set_xlabel("edits"); ax.set_title(f"Winning variants (exactness mismatches: "
                                         f"{R4['exactness_check']['mismatches_per_element_argmin_vs_bruteforce']})", loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "research_optimiser.png", dpi=200)
    plt.close(fig)


def fig_R6(R6):
    plt = _style()
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    names = list(R6["endpoints"])
    x = np.arange(len(names))
    for k, (cond, col) in enumerate((("with_cache", "#2a78d6"), ("without_cache", "#eb6834"))):
        v = [R6["endpoints"][n][cond]["median_ms"] for n in names]
        ax.bar(x + (k - 0.5) * 0.38, v, 0.36, color=col, label=cond.replace("_", " "))
        for xi, vi, n in zip(x, v, names):
            ax.text(xi + (k - 0.5) * 0.38, vi, f"{R6['endpoints'][n][cond]['texts_scored']} txt", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x, names); ax.set_ylabel("median latency (ms)"); ax.legend(fontsize=7)
    ax.set_title("R6  Request cache: latency and texts scored by the classifier", loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "research_cache_latency.png", dpi=200)
    plt.close(fig)


# ============================================================================ main
def main():
    from backend.intelligence.config import CONFIG_VERSION
    from backend.scoring import request_scorer
    t_start = time.perf_counter()
    FIGURES.mkdir(parents=True, exist_ok=True)
    ctx = Ctx()
    result = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "config_version": CONFIG_VERSION,
              "command": ".venv/bin/python -m ml.research_intelligence", "seed": SEED, "python": platform.python_version(),
              "classifier_threshold": ctx.tau, "test_split_rows": int(len(ctx.test)),
              "test_split_label_counts": ctx.test["label"].value_counts().to_dict(),
              "caveats": ["No labelled dataset of journeys, choice sets or composite deception levels exists; R1-R3/R5 use SYNTHETIC "
                          "inputs built from real TEST-split strings or authored markup.",
                          "R1(e) reference expectations and R2 designed levels are the authors' design intent, not ground truth.",
                          "R4 score reductions are model-estimated by the same classifier (circular): they show the cues the model "
                          "relies on were removed, not that shoppers would be less pressured.",
                          "R6 latencies are in-process TestClient timings on one machine (single-threaded OpenMP)."],
              "studies": {}}
    S = result["studies"]
    errors = {}
    with request_scorer(ctx.clf) as scorer:
        scorer.predict(ctx.test["text"].tolist())          # one batched pass; every later audit hits the cache
        for key, fn in (("R1_fspm_robustness", exp_R1), ("R2_mcai_aggregation", exp_R2)):
            print(key, "...", flush=True)
            t0 = time.perf_counter()
            out = fn(ctx)
            S[key], extra = out
            S[key]["seconds"] = r(time.perf_counter() - t0, 1)
            if key.startswith("R1"):
                fig_R1(S[key], extra)
            else:
                fig_R2(S[key], extra)
        for key, fn in (("R3_cdi_aggregation", exp_R3), ("R4_counterfactual_optimiser", exp_R4), ("R5_cross_modal_fusion", exp_R5)):
            print(key, "...", flush=True)
            t0 = time.perf_counter()
            try:
                S[key] = fn(ctx)
                S[key].setdefault("seconds", r(time.perf_counter() - t0, 1))
            except Exception as e:
                errors[key] = traceback.format_exc()
                S[key] = {"error": f"{type(e).__name__}: {e}"}
                print(errors[key])
        result["research_cache_stats"] = scorer.stats()
    print("R6_cache_performance ...", flush=True)
    try:
        S["R6_cache_performance"] = exp_R6(ctx)               # outside the study-wide cache on purpose
    except Exception as e:
        errors["R6_cache_performance"] = traceback.format_exc()
        S["R6_cache_performance"] = {"error": f"{type(e).__name__}: {e}"}
        print(errors["R6_cache_performance"])
    if "error" not in S["R4_counterfactual_optimiser"]:
        fig_R4(S["R4_counterfactual_optimiser"])
    if "error" not in S["R3_cdi_aggregation"] and "error" not in S["R5_cross_modal_fusion"]:
        fig_R3_R5(S["R3_cdi_aggregation"], S["R5_cross_modal_fusion"])
    if "error" not in S["R6_cache_performance"]:
        fig_R6(S["R6_cache_performance"])
    result["errors"] = errors
    result["figures"] = sorted(p.name for p in FIGURES.glob("research_*.png"))
    result["runtime_seconds"] = round(time.perf_counter() - t_start, 1)
    result["model_load_seconds"] = round(ctx.load_s, 1)
    OUT_JSON.write_text(json.dumps(jsonable(result), indent=2, ensure_ascii=False))
    print(f"wrote {OUT_JSON} in {result['runtime_seconds']} s")


if __name__ == "__main__":
    main()
