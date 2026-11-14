"""OCR engine dispatcher: both engines return the screenshot Candidate contract."""
import pytest
from PIL import Image, ImageDraw, ImageFont

from backend.extractors import (Candidate, _box_overlap, default_ocr_engine, easyocr_available, ocr_available,
                                ocr_engine_lines, union_candidates)


def _font(size):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


@pytest.fixture(scope="module")
def big_text_image():
    img = Image.new("RGB", (900, 320), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 40), "Only 2 left in stock", fill="black", font=_font(48))
    d.text((40, 140), "Hurry, sale ends today", fill="black", font=_font(48))
    d.text((40, 240), "Add to cart", fill="black", font=_font(40))
    return img


def _check_contract(cands, img):
    assert cands and all(isinstance(c, Candidate) for c in cands)
    lines = [c for c in cands if c.extra.get("unit") == "line"]
    assert lines, "at least one line candidate"
    assert {c.extra.get("unit") for c in cands} <= {"line", "block"}
    W, H = img.size
    for c in cands:
        assert c.source_type == "screenshot_ocr" and c.text.strip()
        b = c.bbox
        assert set(b) == {"x", "y", "width", "height"}
        assert 0 <= b["x"] and 0 <= b["y"] and b["width"] > 0 and b["height"] > 0
        assert b["x"] + b["width"] <= W and b["y"] + b["height"] <= H
        assert 0.0 <= c.ocr_confidence <= 1.0
    joined = " ".join(c.text for c in lines).lower()
    assert "left" in joined and "stock" in joined


@pytest.mark.skipif(not ocr_available(), reason="tesseract missing")
def test_dispatch_tesseract(big_text_image):
    _check_contract(ocr_engine_lines(big_text_image, "tesseract"), big_text_image)


@pytest.mark.skipif(not easyocr_available(), reason="easyocr not installed")
def test_dispatch_easyocr(big_text_image):
    cands = ocr_engine_lines(big_text_image, "easyocr")
    _check_contract(cands, big_text_image)
    # the three rows are far apart vertically relative to their height except the first two (gap ~ 1 line height):
    # blocks, if any, must contain >= 2 lines' text
    for b in (c for c in cands if c.extra["unit"] == "block"):
        assert len(b.text.split()) >= 4


@pytest.mark.skipif(not (easyocr_available() and ocr_available()), reason="needs tesseract and easyocr")
def test_dispatch_union(big_text_image):
    t = ocr_engine_lines(big_text_image, "tesseract")
    u = ocr_engine_lines(big_text_image, "union")
    _check_contract(u, big_text_image)
    assert u[:len(t)] == t                      # union keeps every Tesseract candidate first, unchanged


def test_union_dedup_by_box():
    a = Candidate(text="Only 2 left", source_type="screenshot_ocr", bbox={"x": 10, "y": 10, "width": 100, "height": 20},
                  ocr_confidence=0.9, extra={"unit": "line"})
    near = Candidate(text="Only 2 lefl", source_type="screenshot_ocr", bbox={"x": 12, "y": 11, "width": 98, "height": 19},
                     ocr_confidence=0.8, extra={"unit": "line"})
    inside = Candidate(text="2 left", source_type="screenshot_ocr", bbox={"x": 50, "y": 12, "width": 40, "height": 16},
                       ocr_confidence=0.8, extra={"unit": "line"})
    other = Candidate(text="No thanks", source_type="screenshot_ocr", bbox={"x": 10, "y": 200, "width": 60, "height": 12},
                      ocr_confidence=0.7, extra={"unit": "line"})
    out = union_candidates([a], [near, inside, other])
    assert [c.text for c in out] == ["Only 2 left", "No thanks"]
    assert out[1].extra["engine"] == "easyocr" and other.extra == {"unit": "line"}   # input not mutated
    iou, contain = _box_overlap(a.bbox, a.bbox)
    assert iou == 1.0 and contain == 1.0


def test_engine_validation(monkeypatch):
    monkeypatch.delenv("CLEARCART_OCR_ENGINE", raising=False)
    assert default_ocr_engine() == "tesseract"
    monkeypatch.setenv("CLEARCART_OCR_ENGINE", "EasyOCR")
    assert default_ocr_engine() == "easyocr"
    monkeypatch.setenv("CLEARCART_OCR_ENGINE", "bogus")
    with pytest.raises(ValueError):
        default_ocr_engine()
    with pytest.raises(ValueError):
        ocr_engine_lines(Image.new("RGB", (10, 10), "white"), "bogus")
