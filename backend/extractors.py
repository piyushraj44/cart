"""Input layer: turn pasted text / HTML / screenshots into candidate elements.

Every candidate keeps a pointer back to where it came from (char offsets, CSS selector,
or OCR bounding box) so that each finding can be verified against the user's input.
Rules here only EXTRACT candidates and OBSERVABLE properties (e.g. a checkbox that is
`checked` in the markup). They never assign a dark-pattern category - the classifier does.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

MAX_SEGMENT_CHARS = 400      # training texts: p95 is ~200 chars; longer blocks are split
MIN_SEGMENT_CHARS = 2


@dataclass
class Candidate:
    text: str
    source_type: str                       # text | html | screenshot_ocr
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    line: Optional[int] = None
    css_selector: Optional[str] = None
    html_tag: Optional[str] = None
    html_snippet: Optional[str] = None
    bbox: Optional[dict] = None
    ocr_confidence: Optional[float] = None
    # observable DOM properties
    input_type: Optional[str] = None       # checkbox | radio
    preselected: bool = False
    extra: dict = field(default_factory=dict)


_CLOCKISH = re.compile(r"\d\s*:\s*\d")


def _useful(text: str) -> bool:
    t = text.strip()
    if len(t) < MIN_SEGMENT_CHARS:
        return False
    # `\d\s*:\s*\d` matches exactly the strings `\d+\s*:\s*\d+` matches, but in linear time: the `\d+` form
    # backtracks quadratically on long digit runs without a colon (200k digits took ~114 s - a DoS vector).
    return any(c.isalpha() for c in t) or (":" in t and bool(_CLOCKISH.search(t)))


# ------------------------------------------------------------------ pasted text
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“(])")


def chunk_text(text: str, limit: int = MAX_SEGMENT_CHARS) -> list[str]:
    """Split long text at sentence boundaries; hard-wrap any sentence still over `limit` at word boundaries."""
    out: list[str] = []
    for sent in _SENT_SPLIT.split(text.strip()):
        sent = sent.strip()
        while len(sent) > limit:
            cut = sent.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            out.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if sent:
            out.append(sent)
    return out


def segment_text(text: str) -> list[Candidate]:
    """Split pasted text into line/sentence segments, tracking exact char offsets.

    Lines are the natural unit of copied web text (a button, a banner, a label). Lines longer
    than MAX_SEGMENT_CHARS are split at sentence boundaries.
    """
    out: list[Candidate] = []
    pos = 0
    for line_no, line in enumerate(text.split("\n"), start=1):
        line_start = pos
        pos += len(line) + 1
        if not line.strip():
            continue
        pieces = [(0, len(line))]
        if len(line.strip()) > MAX_SEGMENT_CHARS:
            pieces, s = [], 0
            for m in _SENT_SPLIT.finditer(line):
                pieces.append((s, m.start())); s = m.end()
            pieces.append((s, len(line)))
        for a, b in pieces:
            seg = line[a:b]
            lead = len(seg) - len(seg.lstrip())
            seg_clean = seg.strip()
            if not _useful(seg_clean):
                continue
            start = line_start + a + lead
            out.append(Candidate(text=seg_clean, source_type="text", char_start=start,
                                 char_end=start + len(seg_clean), line=line_no))
    return out


# ------------------------------------------------------------------ HTML
SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "head", "meta", "link", "iframe", "canvas"}
BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "button", "a", "label", "span", "div", "td", "th",
              "strong", "em", "b", "small", "summary", "figcaption", "caption", "option", "legend", "dt", "dd", "section", "article", "aside", "header", "footer", "nav", "main", "form", "fieldset", "ul", "ol", "table", "tr", "tbody", "thead"}


def css_selector(el: Tag, _cache: dict | None = None) -> str:
    """Readable CSS path. `_cache` memoises each parent's same-tag child index so building
    selectors for every element of a large page stays linear instead of quadratic."""
    parts = []
    node = el
    while isinstance(node, Tag) and node.name not in ("[document]", "html"):
        if node.get("id"):
            parts.append(f"{node.name}#{node['id']}")
            break
        sel = node.name
        classes = [c for c in node.get("class", []) if re.match(r"^[A-Za-z_-][\w-]*$", c)][:2]
        if classes:
            sel += "." + ".".join(classes)
        parent = node.parent
        if isinstance(parent, Tag):
            key = (id(parent), node.name)
            index = _cache.get(key) if _cache is not None else None
            if index is None:
                same = parent.find_all(node.name, recursive=False)
                index = {id(s): i + 1 for i, s in enumerate(same)}
                index["__n__"] = len(same)
                if _cache is not None:
                    _cache[key] = index
            if index["__n__"] > 1:
                sel += f":nth-of-type({index[id(node)]})"
        parts.append(sel)
        node = node.parent
    return " > ".join(reversed(parts))


def _snippet(el: Tag, limit: int = 300) -> str:
    s = str(el)
    return s if len(s) <= limit else s[:limit] + "…"


def _own_text(el: Tag) -> str:
    """Text of `el` excluding descendants that are themselves block-ish elements with text."""
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()


_PRESSURE = re.compile(r"\b(hurry|act now|only \d+ left|don't miss|do not miss|everyone|people are viewing|"
                       r"last chance|buy now|order now)\b", re.I)


def structured_promotion_context(el: Tag, text: str) -> bool:
    """Whether a neutral promotion has nearby DOM terms worth human review.

    This is evidence of *context*, not proof that a price or deadline is true.
    The auditor withholds a matching ML finding as uncertain, and only in HTML.
    """
    if _PRESSURE.search(text):
        return False
    if el.name == "time" and el.has_attr("datetime"):
        try:
            datetime.fromisoformat(el["datetime"].replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return False
        return bool(re.search(r"\b(offer|sale|discount)\b", text, re.I))
    if re.fullmatch(r"\d{1,2}% off", text, re.I):
        return bool(el.parent and el.parent.find("del") and el.parent.find("ins"))
    if el.name == "button" and re.fullmatch(r"get \d{1,2}% off", text, re.I):
        parent = el.parent
        if parent is None:
            return False
        peers = [b.get_text(" ", strip=True) for b in parent.find_all("button") if b is not el]
        return (bool(re.search(r"\b(until|through)\b", parent.get_text(" ", strip=True), re.I))
                and any(re.search(r"\b(continue|skip|no thanks)\b", p, re.I) for p in peers))
    return False


def _label_for(soup: BeautifulSoup, inp: Tag) -> tuple[str, Optional[Tag]]:
    if inp.get("id"):
        lab = soup.find("label", attrs={"for": inp["id"]})
        if lab:
            return _own_text(lab), lab
    parent_label = inp.find_parent("label")
    if parent_label:
        return _own_text(parent_label), parent_label
    for attr in ("aria-label", "title", "value", "name"):
        if inp.get(attr):
            return str(inp[attr]), None
    sib = inp.find_next_sibling(string=True)
    if sib and sib.strip():
        return sib.strip(), None
    return "", None


def prepare_soup(html: str) -> tuple[BeautifulSoup, str, int]:
    """Parse + clean HTML exactly as the auditor sees it.

    Returns (soup, css_text, hidden_count). `css_text` is the concatenated content of <style>
    blocks, captured before they are removed, so the choice-asymmetry module can read declared
    CSS. Shared by extract_html and backend.intelligence.mcai so CSS selectors always agree.
    """
    soup = BeautifulSoup(html, "lxml")
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()
    css_text = "\n".join(t.get_text() for t in soup.find_all("style"))
    for t in soup.find_all(SKIP_TAGS):
        t.decompose()
    hidden = 0
    for t in soup.find_all(style=re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.I)):
        t.decompose(); hidden += 1
    for t in soup.find_all(attrs={"hidden": True}):
        t.decompose(); hidden += 1
    return soup, css_text, hidden


def extract_html(html: str) -> tuple[list[Candidate], list[str]]:
    warnings: list[str] = []
    soup, _css, hidden = prepare_soup(html)
    if hidden:
        warnings.append(f"{hidden} hidden element(s) (display:none / hidden) were ignored because users cannot see them.")

    cands: list[Candidate] = []
    seen_text: dict[str, int] = {}
    sel_cache: dict = {}
    label_elements: set[int] = set()

    # 1) form controls: observable pre-selected state + their label text
    for inp in soup.find_all("input"):
        typ = (inp.get("type") or "text").lower()
        if typ not in ("checkbox", "radio"):
            continue
        text, lab = _label_for(soup, inp)
        if lab is not None:
            label_elements.add(id(lab))
        cands.append(Candidate(
            text=text or f"[unlabelled {typ}]", source_type="html", css_selector=css_selector(inp), html_tag=f"input[type={typ}]",
            html_snippet=_snippet(lab or inp), input_type=typ, preselected=inp.has_attr("checked"),
            extra={"name": inp.get("name"), "value": inp.get("value"), "disabled": inp.has_attr("disabled")},
        ))
        if text:
            seen_text[text] = seen_text.get(text, 0) + 1
    for sel in soup.find_all("select"):
        opt = sel.find("option", selected=True)
        if opt is not None:
            txt = _own_text(opt)
            if txt:
                cands.append(Candidate(text=txt, source_type="html", css_selector=css_selector(opt), html_tag="option[selected]",
                                       html_snippet=_snippet(sel), input_type="select", preselected=False))

    # 2) visible text-bearing elements: smallest element that directly owns text
    body = soup.body or soup
    for el in body.find_all(True):
        if el.name in ("input", "select", "option", "textarea", "br", "img", "hr"):
            continue
        if id(el) in label_elements:
            continue
        direct = "".join(s for s in el.children if isinstance(s, NavigableString)).strip()
        if not direct:
            continue
        # use the full text of inline-ish containers (button, a, label, headings, p, li) so we don't
        # fragment "Only <b>2</b> left" into pieces
        if el.name in ("b", "strong", "em", "i", "span", "small", "u") and el.parent is not None and el.parent.name in ("p", "button", "a", "label", "li", "h1", "h2", "h3", "h4", "h5", "h6", "div", "span", "td"):
            parent_direct = "".join(s for s in el.parent.children if isinstance(s, NavigableString)).strip()
            if parent_direct:
                continue  # parent will capture this text
        text = _own_text(el)
        if not _useful(text):
            continue
        pieces = [text]
        if len(text) > MAX_SEGMENT_CHARS:
            has_text_children = any(
                isinstance(ch, Tag) and "".join(s for s in ch.children if isinstance(s, NavigableString)).strip()
                for ch in el.find_all(True)
            )
            # containers: their children are visited separately, so keep only this element's own
            # text; leaves: keep everything. Either way split into sentence-sized chunks, never drop.
            own = re.sub(r"\s+", " ", direct) if has_text_children else text
            pieces = [p for p in chunk_text(own) if _useful(p)]
        selector = None
        for k, piece in enumerate(pieces):
            if piece in seen_text:
                seen_text[piece] += 1
                continue
            seen_text[piece] = 1
            selector = selector or css_selector(el, sel_cache)
            c = Candidate(text=piece, source_type="html", css_selector=selector, html_tag=el.name, html_snippet=_snippet(el))
            if structured_promotion_context(el, piece):
                c.extra["structured_promotion"] = True
            if len(pieces) > 1:
                c.extra["chunk"] = f"{k + 1}/{len(pieces)}"
            cands.append(c)

    # attribute text that users see
    for el in body.find_all(["img"]):
        alt = (el.get("alt") or "").strip()
        if _useful(alt) and alt not in seen_text:
            seen_text[alt] = 1
            cands.append(Candidate(text=alt, source_type="html", css_selector=css_selector(el), html_tag="img[alt]", html_snippet=_snippet(el)))

    for c in cands:
        n = seen_text.get(c.text, 1)
        if n > 1:
            c.extra["occurrences"] = n
    return cands, warnings


# ------------------------------------------------------------------ screenshot OCR
def ocr_available() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def ocr_lines(image, merge_blocks: bool = True, upscale: bool = True, psm: int = 3) -> list[Candidate]:
    """Tesseract OCR -> line candidates with bounding boxes in ORIGINAL image pixels.

    Configurations compared on ContextRico DEV images (ml/eval_screenshots.py):
      v1_naive: raw image, psm 3, lines only.
      v4_psm3 (selected, default): grayscale + 2x upscale for small UI text, psm 3, and lines in
        the same Tesseract paragraph are additionally merged so banners split over two lines are
        classified as one message. (With psm 11 Tesseract puts every line in its own paragraph,
        so merging never fires - that is why psm 3 is the default.)
    """
    import pytesseract
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(image)     # honour camera/phone rotation
    if image.mode in ("RGBA", "LA", "P"):
        rgba = image.convert("RGBA")           # composite on white so transparent areas don't turn black
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    img = image.convert("RGB")
    scale = 1.0
    work = img
    if upscale:
        work = ImageOps.grayscale(img)
        if max(work.size) < 2200:
            scale = 2.0
            work = work.resize((int(work.width * scale), int(work.height * scale)))
    d = pytesseract.image_to_data(work, output_type=pytesseract.Output.DICT, config=f"--psm {psm}")
    lines: dict[tuple, dict] = {}
    for i, word in enumerate(d["text"]):
        w = (word or "").strip()
        conf = float(d["conf"][i])
        if not w or conf < 0:
            continue
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        x, y, ww, hh = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
        L = lines.setdefault(key, {"words": [], "x0": x, "y0": y, "x1": x + ww, "y1": y + hh, "confs": []})
        L["words"].append(w); L["confs"].append(conf)
        L["x0"], L["y0"] = min(L["x0"], x), min(L["y0"], y)
        L["x1"], L["y1"] = max(L["x1"], x + ww), max(L["y1"], y + hh)

    def to_cand(text, x0, y0, x1, y1, confs, kind):
        return Candidate(text=text, source_type="screenshot_ocr",
                         bbox={"x": int(x0 / scale), "y": int(y0 / scale), "width": int((x1 - x0) / scale), "height": int((y1 - y0) / scale)},
                         ocr_confidence=round(sum(confs) / len(confs) / 100, 3), extra={"unit": kind})

    out: list[Candidate] = []
    ordered = sorted(lines.items(), key=lambda kv: kv[0])
    for key, L in ordered:
        text = " ".join(L["words"])
        if _useful(text):
            out.append(to_cand(text, L["x0"], L["y0"], L["x1"], L["y1"], L["confs"], "line"))
    if merge_blocks:
        blocks: dict[tuple, list] = {}
        for key, L in ordered:
            blocks.setdefault(key[:2], []).append(L)
        for key, Ls in blocks.items():
            if len(Ls) < 2:
                continue
            text = " ".join(" ".join(L["words"]) for L in Ls)
            if not _useful(text) or len(text) > MAX_SEGMENT_CHARS:
                continue
            out.append(to_cand(text, min(L["x0"] for L in Ls), min(L["y0"] for L in Ls), max(L["x1"] for L in Ls),
                               max(L["y1"] for L in Ls), [c for L in Ls for c in L["confs"]], "block"))
    return out


# ------------------------------------------------------------------ second OCR engine: EasyOCR
# EasyOCR (JaidedAI, Apache-2.0, https://github.com/JaidedAI/EasyOCR) with its pretrained English models:
# CRAFT text detector (craft_mlt_25k.pth, ~83 MB) + English recogniser (english_g2.pth, ~15 MB), downloaded on first
# use to ~/.EasyOCR/model/. Compared against Tesseract in ml/eval_ocr_engines.py. Optional: if `easyocr` is not
# installed, Tesseract remains the only engine and nothing here is imported.
OCR_ENGINES = ("tesseract", "easyocr", "union")
_EASYOCR_READERS: dict = {}
_EASYOCR_LOCK = threading.Lock()


def easyocr_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("easyocr") is not None


def default_ocr_engine() -> str:
    """Serving engine from CLEARCART_OCR_ENGINE (tesseract | easyocr | union); default tesseract."""
    import os
    eng = os.environ.get("CLEARCART_OCR_ENGINE", "tesseract").strip().lower() or "tesseract"
    if eng not in OCR_ENGINES:
        raise ValueError(f"CLEARCART_OCR_ENGINE must be one of {OCR_ENGINES}, got {eng!r}")
    return eng


def _easyocr_device() -> str:
    """CLEARCART_EASYOCR_DEVICE = cpu (default) | mps | cuda."""
    import os
    return os.environ.get("CLEARCART_EASYOCR_DEVICE", "cpu").strip().lower() or "cpu"


def _easyocr_reader(device: str | None = None):
    """Lazily build ONE EasyOCR reader per device and cache it at module level (loading takes seconds)."""
    device = device or _easyocr_device()
    with _EASYOCR_LOCK:
        if device not in _EASYOCR_READERS:
            import easyocr
            _EASYOCR_READERS[device] = easyocr.Reader(["en"], gpu=(False if device == "cpu" else device), verbose=False)
        return _EASYOCR_READERS[device]


def _prep_rgb(image):
    from PIL import Image, ImageOps
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA", "P"):
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    return image.convert("RGB")


def _merge_line_blocks(lines: list[dict]) -> list[list[dict]]:
    """Group line boxes into blocks by vertical proximity: a line joins the block above it when the vertical gap is
    <= 0.6 x the smaller line height, the boxes overlap horizontally and their heights differ by < 1.6x (so a big
    heading does not swallow the small print under it). Stand-in for Tesseract's paragraph grouping."""
    order = sorted(range(len(lines)), key=lambda i: (lines[i]["y0"], lines[i]["x0"]))
    parent = list(range(len(lines)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a_pos, i in enumerate(order):
        A = lines[i]
        ha = A["y1"] - A["y0"]
        for j in order[a_pos + 1:]:
            B = lines[j]
            hb = B["y1"] - B["y0"]
            gap = B["y0"] - A["y1"]
            if gap > 0.6 * min(ha, hb):
                if B["y0"] > A["y1"] + 2 * max(ha, hb):
                    break
                continue
            if B["y0"] < A["y0"] + 0.5 * ha:      # same row (side-by-side segments): not a vertical neighbour
                continue
            x_overlap = min(A["x1"], B["x1"]) - max(A["x0"], B["x0"])
            if x_overlap <= 0 or max(ha, hb) > 1.6 * max(min(ha, hb), 1):
                continue
            parent[find(j)] = find(i)
    groups: dict[int, list[dict]] = {}
    for i in order:
        groups.setdefault(find(i), []).append(lines[i])
    return [g for g in groups.values() if len(g) >= 2]


def ocr_lines_easyocr(image, merge_blocks: bool = True, device: str | None = None) -> list[Candidate]:
    """EasyOCR -> the same Candidate contract as `ocr_lines`.

    Line candidates: one per EasyOCR text segment (its detector already joins words on the same row whose gap is
    < 0.5 x text height), bbox = axis-aligned hull of the detected quadrilateral in ORIGINAL pixels, clipped to the
    image; ocr_confidence = EasyOCR's recognition confidence in [0, 1]; extra['unit'] = 'line'.
    Block candidates (merge_blocks): vertically adjacent lines grouped by `_merge_line_blocks`, text joined in
    reading order, confidence = mean of the line confidences; extra['unit'] = 'block'.
    No upscaling: EasyOCR resizes internally (canvas_size 2560, mag_ratio 1).
    """
    import numpy as np
    img = _prep_rgb(image)
    W, H = img.size
    reader = _easyocr_reader(device)
    with _EASYOCR_LOCK:                       # one inference at a time per process (MPS/torch modules are shared)
        res = reader.readtext(np.asarray(img), detail=1, paragraph=False)
    lines = []
    for pts, text, conf in res:
        t = re.sub(r"\s+", " ", str(text)).strip()
        if not t:
            continue
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        x0, y0 = max(0, int(min(xs))), max(0, int(min(ys)))
        x1, y1 = min(W, int(round(max(xs)))), min(H, int(round(max(ys))))
        if x1 <= x0 or y1 <= y0:
            continue
        lines.append({"text": t, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "conf": float(min(1.0, max(0.0, conf)))})

    def to_cand(text, x0, y0, x1, y1, confs, kind):
        return Candidate(text=text, source_type="screenshot_ocr",
                         bbox={"x": int(x0), "y": int(y0), "width": int(x1 - x0), "height": int(y1 - y0)},
                         ocr_confidence=round(sum(confs) / len(confs), 3), extra={"unit": kind})

    lines.sort(key=lambda L: (L["y0"], L["x0"]))
    out = [to_cand(L["text"], L["x0"], L["y0"], L["x1"], L["y1"], [L["conf"]], "line") for L in lines if _useful(L["text"])]
    if merge_blocks:
        for g in _merge_line_blocks(lines):
            g = sorted(g, key=lambda L: (L["y0"], L["x0"]))
            text = " ".join(L["text"] for L in g)
            if not _useful(text) or len(text) > MAX_SEGMENT_CHARS:
                continue
            out.append(to_cand(text, min(L["x0"] for L in g), min(L["y0"] for L in g), max(L["x1"] for L in g),
                               max(L["y1"] for L in g), [L["conf"] for L in g], "block"))
    return out


def _box_overlap(a: dict, b: dict) -> tuple[float, float]:
    """(IoU, intersection / area of the smaller box) for two {x, y, width, height} boxes."""
    ix = max(0, min(a["x"] + a["width"], b["x"] + b["width"]) - max(a["x"], b["x"]))
    iy = max(0, min(a["y"] + a["height"], b["y"] + b["height"]) - max(a["y"], b["y"]))
    inter = ix * iy
    aa, ab = max(a["width"] * a["height"], 1), max(b["width"] * b["height"], 1)
    return inter / (aa + ab - inter), inter / min(aa, ab)


UNION_IOU = 0.5
UNION_CONTAINMENT = 0.8


def union_candidates(primary: list[Candidate], secondary: list[Candidate]) -> list[Candidate]:
    """All `primary` candidates plus every `secondary` candidate whose box is not a duplicate of a primary candidate
    of the same unit (duplicate = IoU >= UNION_IOU or >= UNION_CONTAINMENT of the smaller box covered)."""
    def dup(p: Candidate, s: Candidate) -> bool:
        if not (p.bbox and s.bbox) or p.extra.get("unit") != s.extra.get("unit"):
            return False
        iou, contain = _box_overlap(p.bbox, s.bbox)
        return iou >= UNION_IOU or contain >= UNION_CONTAINMENT

    out = list(primary)
    for s in secondary:
        if not any(dup(p, s) for p in primary):
            out.append(Candidate(**{**s.__dict__, "extra": {**s.extra, "engine": "easyocr"}}))
    return out


def ocr_engine_lines(image, engine: str | None = None) -> list[Candidate]:
    """Dispatch screenshot OCR. engine: 'tesseract' (ocr_lines with its default v4_psm3 config), 'easyocr', or
    'union' (Tesseract candidates + non-duplicate EasyOCR candidates). None -> CLEARCART_OCR_ENGINE (default tesseract)."""
    engine = (engine or default_ocr_engine()).lower()
    if engine == "tesseract":
        return ocr_lines(image)
    if engine == "easyocr":
        return ocr_lines_easyocr(image)
    if engine == "union":
        return union_candidates(ocr_lines(image), ocr_lines_easyocr(image))
    raise ValueError(f"unknown OCR engine {engine!r}; expected one of {OCR_ENGINES}")
