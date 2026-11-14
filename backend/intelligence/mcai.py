"""MCAI - Multi-dimensional Choice Asymmetry Index.

For each detected choice set (an "accept" option competing with a "decline" option):

    D_visual      = mean_k clip((a_k - d_k) / max(a_k, d_k), 0, 1)      over prominence features k
    D_interaction = max(0, s_decline - s_accept) / max(s_decline, s_accept)   (declared step counts)
    D_semantic    = sum_j u_j s_j / sum_j u_j                           over available sub-scores
                    s_guilt            = calibrated P(misdirection | decline text)   (classifier)
                    s_accept_pressure  = calibrated P(dark | accept text)            (classifier)
                    s_wording          = |words_a - words_d| / max(words_a, words_d)
    MCAI = 100 * sum_{d available} w_d D_d / sum_{d available} w_d

Asymmetry only counts in the consumer-unfriendly direction (accept more prominent / easier than
decline). Unavailable dimensions are dropped (never zero-filled) and reported via `coverage`.
Choice sets are found with a rule-based role lexicon - candidate extraction only; all scores come
from measurements or the trained classifier.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

import numpy as np

from backend.extractors import Candidate, css_selector, prepare_soup
from backend.intelligence.aggregation import alternatives as _alternatives
from backend.intelligence.aggregation import contributions as _contributions
from backend.intelligence.aggregation import weight_sensitivity
from backend.intelligence.config import MCAI_WEIGHTS, SEMANTIC_WEIGHTS
from backend.intelligence.schemas import (ChoiceOption, ChoiceSetResult, DimensionScore, FeatureComparison,
                                          MCAIResult, WeightSensitivity)

ScoreFn = Callable[[list[str]], list[dict]]

DECLINE_RE = re.compile(
    r"^\s*(no\b|nope\b|not now\b|decline\b|reject\b|refuse\b|cancel\b|skip\b|dismiss\b|maybe later\b|i'?ll pass\b|"
    r"continue without\b|i\s+(don'?t|do not|would rather|'?d rather|prefer|'?ll)\b|unsubscribe\b|opt[- ]?out\b)"
    r"|\b(no,? thanks?|no thank you|reject all|decline all|without (the )?(discount|offer|protection|deal))\b",
    re.I)
ACCEPT_RE = re.compile(
    r"\b(yes|accept|agree|allow|continue|subscribe|add|buy|get|claim|join|sign ?up|start|ok(ay)?|save|upgrade|confirm|"
    r"check ?out|proceed|place order|apply|redeem|unlock|protect|enable|keep)\b", re.I)


def is_decline(text: str) -> bool:
    return bool(DECLINE_RE.search(text or ""))


def is_accept(text: str) -> bool:
    return bool(text) and not is_decline(text) and bool(ACCEPT_RE.search(text))


def rel_asym(a: Optional[float], d: Optional[float]) -> Optional[float]:
    """Consumer-unfriendly relative difference in [0,1]; None if either side is unavailable."""
    if a is None or d is None:
        return None
    m = max(a, d)
    if m <= 0:
        return 0.0
    return float(min(1.0, max(0.0, (a - d) / m)))


# ===================================================================== CSS (declared) parsing
_NAMED = {"white": (255, 255, 255), "black": (0, 0, 0), "gray": (128, 128, 128), "grey": (128, 128, 128),
          "silver": (192, 192, 192), "lightgray": (211, 211, 211), "lightgrey": (211, 211, 211),
          "darkgray": (169, 169, 169), "darkgrey": (169, 169, 169), "red": (255, 0, 0), "green": (0, 128, 0),
          "blue": (0, 0, 255), "orange": (255, 165, 0), "yellow": (255, 255, 0), "navy": (0, 0, 128),
          "gainsboro": (220, 220, 220), "whitesmoke": (245, 245, 245)}


_COLOR_TOKEN = re.compile(r"rgba?\([^)]*\)|#[0-9a-fA-F]{3,8}\b|\b[a-zA-Z]+\b")


def first_color(value: str) -> Optional[tuple[float, float, float]]:
    """First parseable colour inside a CSS value such as `#fff`, `rgb(15, 118, 110)` or `url(x) #eee no-repeat`.

    (An earlier version split on whitespace, which broke `rgb()` values written with spaces.)
    """
    for tok in _COLOR_TOKEN.findall(value or ""):
        c = parse_color(tok)
        if c is not None:
            return c
        if tok.lower() in ("transparent", "none"):
            return None
    return None


def parse_color(v: str) -> Optional[tuple[float, float, float]]:
    """CSS colour -> RGB (alpha composited over white). None if transparent/unknown."""
    v = (v or "").strip().lower()
    if not v or v in ("transparent", "none", "inherit", "initial", "currentcolor"):
        return None
    if v in _NAMED:
        return tuple(float(c) for c in _NAMED[v])
    m = re.fullmatch(r"#([0-9a-f]{3}|[0-9a-f]{6})", v)
    if m:
        h = m.group(1)
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return tuple(float(int(h[i:i + 2], 16)) for i in (0, 2, 4))
    m = re.fullmatch(r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+)\s*)?\)", v)
    if m:
        r, g, b = (float(m.group(i)) for i in (1, 2, 3))
        a = float(m.group(4)) if m.group(4) is not None else 1.0
        if a <= 0:
            return None
        return tuple(a * c + (1 - a) * 255.0 for c in (r, g, b))
    return None


def rel_luminance(rgb: tuple[float, float, float]) -> float:
    def ch(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (ch(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(fg, bg) -> float:
    l1, l2 = sorted((rel_luminance(fg), rel_luminance(bg)), reverse=True)
    return (l1 + 0.05) / (l2 + 0.05)


def parse_px(v: str) -> Optional[float]:
    v = (v or "").strip().lower()
    kw = {"xx-small": 9, "x-small": 10, "small": 13, "medium": 16, "large": 18, "x-large": 24, "xx-large": 32}
    if v in kw:
        return float(kw[v])
    m = re.fullmatch(r"([\d.]+)\s*(px|pt|em|rem|%)?", v)
    if not m:
        return None
    n = float(m.group(1)); u = m.group(2) or "px"
    return {"px": n, "pt": n * 4 / 3, "em": n * 16, "rem": n * 16, "%": n * 16 / 100}[u]


def parse_weight(v: str) -> Optional[float]:
    v = (v or "").strip().lower()
    if v in ("bold", "bolder"):
        return 700.0
    if v in ("normal", "lighter"):
        return 400.0
    return float(v) if re.fullmatch(r"\d{3}", v) else None


def parse_declarations(text: str) -> dict[str, str]:
    out = {}
    for part in (text or "").split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip().lower()] = v.strip().replace("!important", "").strip()
    return out


def _strip_at_blocks(css: str) -> tuple[str, int]:
    """Remove @media/@supports/... blocks (conditional; we cannot evaluate them without a viewport)."""
    out, i, n = [], 0, 0
    while True:
        j = css.find("@", i)
        if j < 0:
            out.append(css[i:]); break
        k = css.find("{", j)
        semi = css.find(";", j)
        if k < 0 or (0 <= semi < k):          # @import ...; / @charset ...;
            out.append(css[i:j]); i = (semi + 1) if semi >= 0 else len(css); continue
        out.append(css[i:j])
        depth, p = 0, k
        while p < len(css):
            if css[p] == "{":
                depth += 1
            elif css[p] == "}":
                depth -= 1
                if depth == 0:
                    break
            p += 1
        i, n = p + 1, n + 1
    return "".join(out), n


class CssRules:
    """Minimal cascade for simple compound selectors (tag, .class, #id, tag.class, tag#id, .a.b).

    Selectors with combinators, pseudo-classes or attributes are skipped (reported as a limitation).
    """

    _SIMPLE = re.compile(r"^([a-z][a-z0-9]*)?((?:[.#][A-Za-z_-][\w-]*)*)$")

    def __init__(self, css_text: str) -> None:
        self.rules: list[tuple[tuple[int, int, int], int, dict]] = []
        self.skipped = 0
        css = re.sub(r"/\*.*?\*/", "", css_text or "", flags=re.S)
        css, self.skipped_at_rules = _strip_at_blocks(css)
        order = 0
        for sel_block, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
            decls = parse_declarations(body)
            for sel in sel_block.split(","):
                sel = sel.strip()
                m = self._SIMPLE.match(sel)
                if not sel or not m:
                    self.skipped += 1
                    continue
                tag = m.group(1)
                ids = re.findall(r"#([\w-]+)", m.group(2) or "")
                classes = re.findall(r"\.([\w-]+)", m.group(2) or "")
                spec = (len(ids), len(classes), 1 if tag else 0)
                self.rules.append((spec, order, {"tag": tag, "ids": ids, "classes": classes, "decls": decls}))
                order += 1

    def declared(self, el) -> dict[str, str]:
        matched = []
        el_classes = set(el.get("class", []) or [])
        for spec, order, r in self.rules:
            if r["tag"] and r["tag"] != el.name:
                continue
            if r["ids"] and el.get("id") not in r["ids"]:
                continue
            if any(c not in el_classes for c in r["classes"]):
                continue
            matched.append((spec, order, r["decls"]))
        out: dict[str, str] = {}
        for _, _, decls in sorted(matched, key=lambda x: (x[0], x[1])):
            out.update(decls)
        out.update(parse_declarations(el.get("style", "")))     # inline wins
        return out


def _option_text(el) -> str:
    if el.name == "input":
        return (el.get("value") or el.get("aria-label") or "").strip()
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip() or (el.get("aria-label") or "").strip()


def _is_control(el) -> bool:
    classes = " ".join(el.get("class", []) or []).lower()
    return (el.name == "button" or (el.name == "input" and (el.get("type") or "").lower() in ("submit", "button", "reset"))
            or (el.get("role") or "").lower() == "button" or bool(re.search(r"\b(btn|button)\b", classes)))


def html_visual_features(el, rules: CssRules) -> dict[str, tuple[Optional[float], str, Optional[str]]]:
    """feature -> (value, provenance, note). Values are 'prominence' quantities (higher = more prominent)."""
    own = rules.declared(el)
    chain = [el] + [p for p in el.parents if getattr(p, "name", None) not in (None, "[document]")]

    def inherited(prop):
        for node in chain:
            d = rules.declared(node) if node is not el else own
            if prop in d:
                return d[prop], ("measured" if node is el else "measured")
        return None, None

    feats: dict[str, tuple[Optional[float], str, Optional[str]]] = {}
    control = _is_control(el)
    feats["control_kind"] = (1.0 if control else 0.35, "measured", "button-like control" if control else "text link")

    fs_raw, prov = inherited("font-size")
    fs = parse_px(fs_raw) if fs_raw else None
    if fs is None:
        fs, prov = (13.33 if el.name in ("button", "input") else 16.0), "default"
    feats["font_size_px"] = (round(fs, 2), prov, None)

    fw_raw, prov = inherited("font-weight")
    fw = parse_weight(fw_raw) if fw_raw else None
    if fw is None:
        fw, prov = 400.0, "default"
    feats["font_weight"] = (fw, prov, None)

    bg_decl = own.get("background-color") or own.get("background")
    if bg_decl is not None:
        fill_rgb = first_color(bg_decl)
        feats["background_fill"] = (1.0 if fill_rgb is not None else 0.0, "measured", f"background: {bg_decl}")
    else:
        fill_rgb = (239.0, 239.0, 239.0) if el.name in ("button", "input") else None
        feats["background_fill"] = (1.0 if fill_rgb is not None else 0.0, "default", "browser default")

    # effective background for contrast: own fill, else nearest ancestor fill, else white page
    bg_rgb, bg_prov = (fill_rgb, "measured" if bg_decl is not None else "default") if fill_rgb is not None else (None, None)
    if bg_rgb is None:
        for node in chain[1:]:
            d = rules.declared(node)
            v = d.get("background-color") or d.get("background")
            if v and first_color(v) is not None:
                bg_rgb, bg_prov = first_color(v), "measured"
                break
    if bg_rgb is None:
        bg_rgb, bg_prov = (255.0, 255.0, 255.0), "default"
    col_raw, col_prov = inherited("color")
    fg = parse_color(col_raw) if col_raw else None
    if fg is None:
        fg, col_prov = ((0.0, 0.0, 238.0) if el.name == "a" else (0.0, 0.0, 0.0)), "default"
    both = "measured" if (bg_prov == "measured" and col_prov == "measured") else "default"
    feats["contrast_ratio"] = (round(contrast_ratio(fg, bg_rgb), 3), both, None)

    op = 1.0; op_prov = "default"
    for node in chain:
        d = rules.declared(node) if node is not el else own
        if "opacity" in d:
            try:
                op *= float(d["opacity"]); op_prov = "measured"
            except ValueError:
                pass
    feats["opacity"] = (round(op, 3), op_prov, None)
    return feats


def _visual_dimension(a_feats: dict, d_feats: dict, features: tuple[str, ...], extra_unavailable: list[str]) -> DimensionScore:
    comps = []
    vals = []
    provs = set()
    for k in features:
        av, ap, an = a_feats.get(k, (None, "unavailable", None))
        dv, dp, dn = d_feats.get(k, (None, "unavailable", None))
        asym = rel_asym(av, dv)
        comps.append(FeatureComparison(feature=k, accept_value=av, decline_value=dv, accept_provenance=ap,
                                       decline_provenance=dp, asymmetry=None if asym is None else round(asym, 4),
                                       note=an or dn))
        if asym is not None:
            vals.append(asym); provs.update({ap, dp})
    for k in extra_unavailable:
        comps.append(FeatureComparison(feature=k, accept_value=None, decline_value=None, accept_provenance="unavailable",
                                       decline_provenance="unavailable", asymmetry=None,
                                       note="needs rendered layout; not computed from declared HTML/CSS"))
    if not vals:
        return DimensionScore(name="visual", available=False, score=None, provenance="unavailable",
                              reason="No comparable visual properties for both options.", features=comps)
    prov = "measured" if "measured" in provs else "default"
    return DimensionScore(name="visual", available=True, score=round(float(np.mean(vals)), 4), provenance=prov, features=comps)


# ===================================================================== screenshot measurements
def _srgb_lum(arr: np.ndarray) -> np.ndarray:
    c = arr / 255.0
    lin = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]


def screenshot_visual_features(img_rgb: np.ndarray, bbox: dict, page_median: np.ndarray) -> dict:
    H, W = img_rgb.shape[:2]
    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    feats = {"text_height_px": (float(h), "measured", "OCR box height"),
             "text_box_area_px": (float(w * h), "measured", "OCR text-box area (not full button area)")}
    crop = img_rgb[y0:y1, x0:x1].astype(float)
    if crop.size:
        lum = _srgb_lum(crop).ravel()
        lo, hi = np.percentile(lum, 10), np.percentile(lum, 90)
        feats["contrast_ratio"] = (round(float((hi + 0.05) / (lo + 0.05)), 3), "measured", "p90/p10 luminance inside OCR box")
    pad = max(4, int(0.6 * h))
    rx0, ry0, rx1, ry1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
    region = img_rgb[ry0:ry1, rx0:rx1].astype(float)
    if region.size:
        mask = np.ones(region.shape[:2], bool)
        mask[(y0 - ry0):(y1 - ry0), (x0 - rx0):(x1 - rx0)] = False     # ring around the text
        ring = region[mask]
        if ring.size:
            dist = float(np.linalg.norm(np.median(ring, axis=0) - page_median))
            feats["background_fill"] = (round(min(1.0, dist / 120.0), 4), "measured", "colour distance of surrounding ring vs page median")
    return feats


# ===================================================================== semantic + interaction
def _semantic_dimension(accept_text: Optional[str], decline_text: str, score_fn: ScoreFn) -> DimensionScore:
    texts = [decline_text] + ([accept_text] if accept_text else [])
    preds = score_fn(texts)
    subs: dict[str, tuple[Optional[float], str]] = {}
    subs["guilt"] = (float(preds[0]["probabilities"].get("misdirection", 0.0)), "model")
    if accept_text:
        subs["accept_pressure"] = (float(preds[1]["p_dark"]), "model")
        na, nd = len(accept_text.split()), len(decline_text.split())
        subs["wording_imbalance"] = (abs(na - nd) / max(na, nd), "measured")
    else:
        subs["accept_pressure"] = (None, "unavailable")
        subs["wording_imbalance"] = (None, "unavailable")
    comps = [FeatureComparison(feature=k, accept_value=None, decline_value=None,
                               accept_provenance=p, decline_provenance=p,
                               asymmetry=None if v is None else round(v, 4),
                               note={"guilt": "calibrated P(misdirection) of the decline wording",
                                     "accept_pressure": "calibrated P(dark pattern) of the accept wording",
                                     "wording_imbalance": "relative difference in word count (unequal disclosure)"}[k])
             for k, (v, p) in subs.items()]
    avail = {k: v for k, (v, _) in subs.items() if v is not None}
    wsum = sum(SEMANTIC_WEIGHTS[k] for k in avail)
    score = sum(SEMANTIC_WEIGHTS[k] * v for k, v in avail.items()) / wsum
    return DimensionScore(name="semantic", available=True, score=round(score, 4), provenance="model", features=comps,
                          reason=None if accept_text else "No competing accept option found; semantic score uses the decline wording only.")


def interaction_dimension(accept_steps: Optional[int], decline_steps: Optional[int], label: str | None = None,
                          provenance: str = "declared", extra_features: list[FeatureComparison] | None = None,
                          reason: str | None = None) -> DimensionScore:
    """I = max(0, s_decline - s_accept) / max(s_decline, s_accept).

    provenance 'declared' = step counts supplied by the auditor; 'inferred' = estimated from page structure
    (collapsed decline, decline navigating away, pre-ticked opt-outs). Inferred values are never presented as
    observed click depth.
    """
    if accept_steps is None or decline_steps is None:
        return DimensionScore(name="interaction", available=False, score=None, provenance="unavailable",
                              reason=reason or "Click/step counts are not observable from a single page; declare them in a journey audit.")
    m = max(accept_steps, decline_steps)
    score = max(0, decline_steps - accept_steps) / m
    feats = [FeatureComparison(feature="steps", accept_value=float(accept_steps), decline_value=float(decline_steps),
                               accept_provenance=provenance, decline_provenance=provenance,
                               asymmetry=round(score, 4), note=label)] + list(extra_features or [])
    return DimensionScore(name="interaction", available=True, score=round(score, 4), provenance=provenance, features=feats,
                          reason=reason)


_COST_OR_CONSENT = re.compile(
    r"[$£€¥₹]|\b(insurance|protect(ion)?|warranty|subscri\w*|member\w*|newsletter|e-?mails?|marketing|offers?|promotion\w*|"
    r"partners?|third.part\w*|donat\w*|add(-| )?on|extra|plan|premium|agree|consent|share)\b", re.I)


def inferred_interaction(accept, decline, soup) -> DimensionScore:
    """Structural (inferred) extra steps needed to decline, from the DOM only. Accept is assumed one step."""
    from backend.extractors import _label_for
    extras: list[FeatureComparison] = []

    def add(name, n, note):
        extras.append(FeatureComparison(feature=name, accept_value=0.0, decline_value=float(n), accept_provenance="inferred",
                                        decline_provenance="inferred", asymmetry=None, note=note))
    det = decline.find_parent("details")
    if det is not None and not det.has_attr("open"):
        add("collapsed_decline", 1, "decline option is inside a collapsed <details> section (one extra click to reveal)")
    href = (decline.get("href") or "").strip() if decline.name == "a" else ""
    if href and not href.startswith(("#", "javascript:")) and accept is not None and _is_control(accept):
        add("decline_navigates_away", 1, f"decline is a link to another page ({href[:60]}) while accept is an in-page control")
    # pre-ticked cost/consent add-ons inside the shared container must be unticked to decline them
    container = None
    if accept is not None:
        anc = set(id(p) for p in accept.parents)
        container = next((p for p in decline.parents if id(p) in anc and getattr(p, "name", None) not in (None, "[document]")), None)
    container = container or decline.find_parent("form")
    n_pre = 0
    if container is not None:
        for inp in container.find_all("input"):
            if (inp.get("type") or "").lower() == "checkbox" and inp.has_attr("checked"):
                text, _ = _label_for(soup, inp)
                if _COST_OR_CONSENT.search(text or ""):
                    n_pre += 1
    if n_pre:
        add("preselected_optouts", n_pre, f"{n_pre} pre-ticked cost/consent option(s) must be unticked to decline them")
    decline_steps = 1 + int(sum(f.decline_value for f in extras))
    return interaction_dimension(1, decline_steps, "structural estimate (accept = 1 step)", provenance="inferred",
                                 extra_features=extras,
                                 reason="Inferred from page structure, not from observed click depth." if extras else
                                 "Inferred from page structure: no extra decline steps found (not observed click depth).")


def finalise_set(dims: list[DimensionScore]) -> dict:
    """Contributions, top dimension, alternatives and weight sensitivity for one choice set."""
    vals = {d.name: d.score for d in dims if d.available and d.score is not None}
    if not vals:
        return {}
    w = {k: MCAI_WEIGHTS[k] for k in vals}
    contrib = _contributions(vals, w)
    sens = weight_sensitivity(vals, w)
    return {"dimension_contributions": contrib, "top_dimension": max(contrib, key=contrib.get),
            "alternatives": _alternatives(vals, w), "sensitivity": WeightSensitivity(**sens)}


def combine(dims: list[DimensionScore], weights: dict[str, float] = MCAI_WEIGHTS, only: tuple[str, ...] | None = None) -> tuple[Optional[float], float]:
    use = [d for d in dims if d.available and (only is None or d.name in only)]
    total = sum(weights[d.name] for d in (dims if only is None else [d for d in dims if d.name in only]))
    if not use:
        return None, 0.0
    wsum = sum(weights[d.name] for d in use)
    val = 100.0 * sum(weights[d.name] * d.score for d in use) / wsum
    return round(val, 2), round(wsum / total, 4) if total else 0.0


# ===================================================================== choice-set extraction
def _finding_id(text: str, findings) -> Optional[str]:
    for f in findings or []:
        if f.evidence.strip() == text.strip():
            return f.id
    return None


def html_choice_sets(html: str) -> tuple[list[tuple], CssRules]:
    pairs, rules, _soup = html_choice_sets_with_soup(html)
    return pairs, rules


def html_choice_sets_with_soup(html: str):
    soup, css_text, _ = prepare_soup(html)
    rules = CssRules(css_text)
    body = soup.body or soup
    actions = [el for el in body.find_all(True)
               if (_is_control(el) or (el.name == "a" and el.get("href") is not None))
               and not el.find_parent(["nav", "header", "footer"]) and _option_text(el)]
    pairs = []
    for d in actions:
        if not is_decline(_option_text(d)):
            continue
        node, accept = d.parent, None
        for _ in range(5):
            if node is None or getattr(node, "name", None) in (None, "[document]"):
                break
            # one accept may legitimately compete with several declines ("No thanks" and "Close")
            pool = [a for a in actions if a is not d and node in a.parents and not is_decline(_option_text(a))]
            lex = [a for a in pool if is_accept(_option_text(a))]
            ctrl = [a for a in pool if _is_control(a)]
            accept = (lex or ctrl or [None])[0]
            if accept is not None:
                break
            node = node.parent
        pairs.append((accept, d))
    return pairs, rules, soup


def _lines_choice_sets(cands: list[Candidate]) -> list[tuple[Optional[Candidate], Candidate]]:
    lines = [c for c in cands if c.extra.get("unit") != "block"]
    out = []
    for d in lines:
        if not is_decline(d.text):
            continue
        acc = [a for a in lines if a is not d and is_accept(a.text)]
        if d.bbox:   # screenshot: nearest accept within 4 line-heights vertically
            h = max(d.bbox["height"], 1)
            acc = [a for a in acc if a.bbox and abs((a.bbox["y"] + a.bbox["height"] / 2) - (d.bbox["y"] + h / 2)) <= 4 * max(h, a.bbox["height"]) + 40]
            acc.sort(key=lambda a: abs(a.bbox["y"] - d.bbox["y"]) + abs(a.bbox["x"] - d.bbox["x"]) / 4)
        elif d.line is not None:   # text: nearest accept within 3 lines
            acc = [a for a in acc if a.line is not None and abs(a.line - d.line) <= 3]
            acc.sort(key=lambda a: abs(a.line - d.line))
        else:
            acc = []
        out.append((acc[0] if acc else None, d))
    return out


# ===================================================================== public API
def compute_mcai(input_type: str, score_fn: ScoreFn, *, cands: list[Candidate] | None = None, html: str | None = None,
                 image=None, findings=None, declared_paths: list[tuple[int, int, str]] | None = None) -> MCAIResult:
    """declared_paths: optional [(accept_steps, decline_steps, label)] applied to every choice set of the page."""
    sets: list[ChoiceSetResult] = []
    decl = declared_paths[0] if declared_paths else None
    if input_type == "html" and html is not None:
        pairs, rules, soup = html_choice_sets_with_soup(html)
        for k, (a, d) in enumerate(pairs, 1):
            a_txt, d_txt = (_option_text(a) if a is not None else None), _option_text(d)
            dims = []
            if a is not None:
                extra = ["rendered_area"]
                dims.append(_visual_dimension(html_visual_features(a, rules), html_visual_features(d, rules),
                                              ("control_kind", "font_size_px", "font_weight", "background_fill", "contrast_ratio", "opacity"), extra))
            else:
                dims.append(DimensionScore(name="visual", available=False, score=None, provenance="unavailable", reason="No competing accept option found."))
            # declared step counts (auditor) take precedence over structural inference
            dims.append(interaction_dimension(*decl[:2], decl[2]) if decl else inferred_interaction(a, d, soup))
            dims.append(_semantic_dimension(a_txt, d_txt, score_fn))
            m, cov = combine(dims)
            sets.append(ChoiceSetResult(
                id=f"c{k}", accept=None if a is None else ChoiceOption(role="accept", text=a_txt, css_selector=css_selector(a), finding_id=_finding_id(a_txt, findings)),
                decline=ChoiceOption(role="decline", text=d_txt, css_selector=css_selector(d), finding_id=_finding_id(d_txt, findings)),
                dimensions=dims, mcai=m, coverage=cov, **finalise_set(dims)))
        if rules.skipped:
            pass
    elif cands is not None:
        img_rgb = page_median = None
        if input_type == "screenshot_ocr" and image is not None:
            img_rgb = np.asarray(image.convert("RGB"))
            page_median = np.median(img_rgb.reshape(-1, 3).astype(float), axis=0)
        for k, (a, d) in enumerate(_lines_choice_sets(cands), 1):
            dims = []
            if img_rgb is not None and a is not None and a.bbox and d.bbox:
                dims.append(_visual_dimension(screenshot_visual_features(img_rgb, a.bbox, page_median),
                                              screenshot_visual_features(img_rgb, d.bbox, page_median),
                                              ("text_height_px", "text_box_area_px", "contrast_ratio", "background_fill"), []))
            else:
                why = ("Pasted text has no visual properties." if input_type == "text"
                       else "No competing accept option found." if a is None else "Image measurements unavailable.")
                dims.append(DimensionScore(name="visual", available=False, score=None, provenance="unavailable", reason=why))
            dims.append(interaction_dimension(*(decl[:2] if decl else (None, None)), decl[2] if decl else None))
            dims.append(_semantic_dimension(a.text if a else None, d.text, score_fn))
            m, cov = combine(dims)
            extra = finalise_set(dims)
            sets.append(ChoiceSetResult(
                id=f"c{k}",
                accept=None if a is None else ChoiceOption(role="accept", text=a.text, bbox=a.bbox, finding_id=_finding_id(a.text, findings)),
                decline=ChoiceOption(role="decline", text=d.text, bbox=d.bbox, finding_id=_finding_id(d.text, findings)),
                dimensions=dims, mcai=m, coverage=cov, **extra))
    if not sets:
        reason = "No accept/decline choice pair was detected in this input."
        if input_type == "screenshot_ocr" and cands and any(is_accept(c.text) for c in cands):
            # Known blind spot (synthetic benchmark B): small light-grey decline text is often not read by OCR,
            # i.e. exactly the most asymmetric designs. Say so instead of failing silently.
            reason = ("OCR found an accept-like option but no decline option. Low-contrast or very small decline text "
                      "is often unreadable by OCR, so a hidden decline cannot be ruled out; choice asymmetry was not measured.")
        return MCAIResult(available=False, weights=MCAI_WEIGHTS, reason=reason)
    worst = max((s.mcai for s in sets if s.mcai is not None), default=None)
    return MCAIResult(available=True, weights=MCAI_WEIGHTS, choice_sets=sets, mcai=worst)
