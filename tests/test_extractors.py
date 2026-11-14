import random
import pytest
from PIL import Image, ImageDraw, ImageFont

from backend.extractors import extract_html, ocr_available, ocr_lines, segment_text, MAX_SEGMENT_CHARS


def test_segment_offsets_random():
    rnd = random.Random(0)
    pieces = ["Only 2 left!", "  indented line", "", "   ", "emoji 🔥🔥 sale", "“curly” quotes — dash", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 non-BMP 😀",
              "Hurry. Only today. Buy now! " * 30, "12:30:45", "\t tab line", "x", "---", "Ünïcödé 日本語 テキスト", "a\r", "é combining"]
    for _ in range(200):
        text = "\n".join(rnd.choice(pieces) for _ in range(rnd.randint(1, 12)))
        for c in segment_text(text):
            assert text[c.char_start:c.char_end] == c.text, (text, c)
            assert c.text == c.text.strip()


def test_long_lines_split():
    line = "This is a sentence about shopping. " * 40
    cands = segment_text(line)
    assert len(cands) > 1
    assert all(len(c.text) <= MAX_SEGMENT_CHARS for c in cands)
    for c in cands:
        assert line[c.char_start:c.char_end] == c.text


def test_blank_and_symbol_only():
    assert segment_text("\n\n   \n") == []
    assert segment_text("!!! ---") == []


HTML = """<html><head><title>t</title><style>.x{}</style><script>var s='Script text here';</script></head><body>
<h1 id="ttl">Big Heading Text</h1>
<button class="buy">Buy it now</button>
<a href="#" class="lnk">No thanks, I hate savings</a>
<form>
<input type="checkbox" id="c1" name="a" checked><label for="c1">Add insurance for $5</label>
<input type="checkbox" id="c2" name="b"><label for="c2">Unchecked option</label>
<label><input type="radio" name="r" checked> Wrapped radio premium plan</label>
</form>
<p style="display:none">Hidden inline text</p>
<div hidden>Hidden attr text</div>
<p style="visibility: hidden">Invisible text</p>
<p>Visible paragraph words</p>
</body></html>"""


def test_html_extraction():
    cands, warns = extract_html(HTML)
    by = {c.text: c for c in cands}
    texts = " | ".join(by)
    assert "Big Heading Text" in by and by["Big Heading Text"].css_selector == "h1#ttl" and by["Big Heading Text"].html_tag == "h1"
    assert "Buy it now" in by and by["Buy it now"].html_tag == "button" and "button.buy" in by["Buy it now"].css_selector
    assert "No thanks, I hate savings" in by and by["No thanks, I hate savings"].html_tag == "a"
    assert by["Add insurance for $5"].preselected is True and by["Add insurance for $5"].input_type == "checkbox"
    assert by["Unchecked option"].preselected is False
    assert by["Wrapped radio premium plan"].preselected is True and by["Wrapped radio premium plan"].input_type == "radio"
    for bad in ["Hidden inline text", "Hidden attr text", "Invisible text", "Script text", "var s", ".x{}"]:
        assert bad not in texts, bad
    assert "Visible paragraph words" in by
    assert warns and "hidden" in warns[0]


def test_html_label_for_not_duplicated():
    cands, _ = extract_html(HTML)
    assert sum(c.text == "Add insurance for $5" for c in cands) == 1


@pytest.mark.skipif(not ocr_available(), reason="tesseract missing")
def test_ocr_lines():
    img = Image.new("RGB", (800, 300), "white")
    d = ImageDraw.Draw(img)
    font = None
    for p in ["/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"]:
        try:
            font = ImageFont.truetype(p, 56); break
        except OSError:
            pass
    assert font is not None, "no TTF font available"
    d.text((40, 100), "Only 2 left in stock", fill="black", font=font)
    cands = ocr_lines(img)
    assert cands
    joined = [c.text.lower() for c in cands]
    assert any("only 2 left in stock" in t or ("only" in t and "left" in t and "stock" in t) for t in joined), joined
    for c in cands:
        b = c.bbox
        assert b["x"] >= 0 and b["y"] >= 0 and b["x"] + b["width"] <= img.width and b["y"] + b["height"] <= img.height, b
