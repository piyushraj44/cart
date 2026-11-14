"""Generate small feature-card PNGs (for slides/README) from REAL ClearCart audits.

Every number, crop, box, rewrite and example on the cards comes from running the actual
backend on the genuine samples in data/samples/. Run from project root:
    .venv/bin/python -m report.make_feature_cards
Output: report/feature_cards/*.png (2x resolution).
"""
from __future__ import annotations

import base64
import html
import io
import json
import subprocess
import tempfile
from pathlib import Path

import ml  # noqa: F401  (OMP fix before torch/xgboost)
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "feature_cards"
SHOTS = ROOT / "data" / "samples" / "screenshots"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
BG = (238, 242, 246)
HEX = {"urgency": "#dc2626", "scarcity": "#ea580c", "social_proof": "#7c3aed", "misdirection": "#d97706", "preselected_option": "#0369a1"}
E = html.escape
_ICON_PATHS = {
    "alert": "<path d='M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z'/><path d='M12 9v4'/><path d='M12 17h.01'/>",
    "box": "<path d='M21 8 12 3 3 8v8l9 5 9-5z'/><path d='m3 8 9 5 9-5'/><path d='M12 13v8'/>",
    "timer": "<circle cx='12' cy='14' r='8'/><path d='M12 10v4l2 2'/><path d='M10 2h4'/>",
    "sparkles": "<path d='M12 3l1.8 4.7L18.5 9.5l-4.7 1.8L12 16l-1.8-4.7L5.5 9.5l4.7-1.8z'/><path d='M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z'/>",
    "search": "<circle cx='11' cy='11' r='7'/><path d='m20 20-3.5-3.5'/><path d='M8 11h6'/>",
}


def icon(name: str) -> str:
    return (f"<svg width='19' height='19' viewBox='0 0 24 24' fill='none' stroke='#fff' stroke-width='2.2' "
            f"stroke-linecap='round' stroke-linejoin='round'>{_ICON_PATHS[name]}</svg>")

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{background:#eef2f6;font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","Inter",system-ui,sans-serif;color:#0f172a;padding:28px}
.card{width:620px;background:#fff;border:1px solid #e2e8f0;border-radius:18px;box-shadow:0 10px 30px rgba(15,23,42,.08);padding:22px 24px;overflow:hidden}
.head{display:flex;align-items:center;gap:10px}
.ico{width:34px;height:34px;border-radius:10px;display:flex;align-items:center;justify-content:center;color:#fff;font-weight:800;font-size:17px}
.title{font-size:19px;font-weight:800;letter-spacing:-.01em}
.sub{font-size:12.5px;color:#64748b;margin-top:1px}
.pill{margin-left:auto;font-size:13px;font-weight:800;padding:5px 11px;border-radius:999px}
.lbl{font-size:10.5px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#64748b}
.foot{margin-top:14px;font-size:11px;color:#94a3b8}
.chip{display:inline-block;font-family:ui-monospace,Menlo,monospace;font-size:12.5px;background:#f1f5f9;border:1px solid #e2e8f0;border-radius:7px;padding:3px 8px;margin:0 6px 6px 0;color:#0f172a}
.reason{font-size:14px;line-height:1.45;color:#334155}
"""


def page(body: str) -> str:
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"


def b64(img: Image.Image) -> str:
    buf = io.BytesIO(); img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def render(name: str, html_doc: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / f"{name}.html"; src.write_text(html_doc)
        raw = Path(td) / "raw.png"
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                        "--window-size=680,1000", f"--screenshot={raw}", src.as_uri()], check=True, capture_output=True)
        im = Image.open(raw).convert("RGB")
    # crop to the card + a 24px (48 device px) margin
    px = im.load(); W, H = im.size
    xs, ys = [], []
    for y in range(0, H, 2):
        for x in range(0, W, 2):
            if sum(abs(a - b) for a, b in zip(px[x, y], BG)) > 12:
                xs.append(x); ys.append(y)
    m = 40
    box = (max(0, min(xs) - m), max(0, min(ys) - m), min(W, max(xs) + m), min(H, max(ys) + m))
    out = OUT / f"{name}.png"
    im.crop(box).save(out)
    return out


def gauge_svg(v: int, color: str) -> str:
    import math
    r = 54; c = 2 * math.pi * r
    return (f"<svg width='150' height='150' viewBox='0 0 140 140' style='transform:rotate(-90deg)'>"
            f"<circle cx='70' cy='70' r='{r}' fill='none' stroke='#e2e8f0' stroke-width='13'/>"
            f"<circle cx='70' cy='70' r='{r}' fill='none' stroke='{color}' stroke-width='13' stroke-linecap='round' stroke-dasharray='{v/100*c:.1f} {c:.1f}'/></svg>")


# ------------------------------------------------------------------ cards
def card_risk(a: dict) -> str:
    s = a["summary"]; v = s["risk_score"]
    det = {c["category"]: c for c in s["categories_detected"]}
    names = [("urgency", "Misleading urgency"), ("scarcity", "Misleading scarcity"), ("social_proof", "Social proof pressure"), ("misdirection", "Misdirection / confirmshaming")]
    bars = ""
    for k, lab in names:
        c = det.get(k)
        val = min(99, int(c["max_score"] * 100)) if c else 0
        right = f"{val}%" + (f" · {c['count']}×" if c and c["count"] > 1 else "") if c else "Not detected"
        bars += (f"<div style='margin-bottom:11px'><div style='display:flex;justify-content:space-between;font-size:13px'>"
                 f"<b style='font-weight:600'>{lab}</b><span style='font-weight:700;font-variant-numeric:tabular-nums'>{right}</span></div>"
                 f"<div style='height:7px;background:#f1f5f9;border-radius:9px;margin-top:5px'><div style='height:7px;width:{val}%;background:{HEX[k]};border-radius:9px'></div></div></div>")
    return page(f"""<div class='card'>
      <div class='head'><div class='ico' style='background:#dc2626'>{icon('alert')}</div><div><div class='title'>High deception risk</div>
      <div class='sub'>One page-level score, broken down by category</div></div></div>
      <div style='display:flex;gap:22px;align-items:center;margin-top:16px'>
        <div style='position:relative;width:150px;height:150px;flex:none'>{gauge_svg(v, '#dc2626')}
          <div style='position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center'>
            <div><span style='font-size:40px;font-weight:800'>{v}</span><span style='font-size:17px;color:#64748b;font-weight:600'> /100</span></div>
            <div style='font-size:9px;font-weight:800;color:#b91c1c;text-align:center;line-height:1.2'>HIGH DECEPTION<br>RISK</div></div></div>
        <div style='flex:1;border-left:1px solid #e2e8f0;padding-left:20px'>{bars}</div>
      </div>
      <div style='margin-top:8px;padding-top:12px;border-top:1px solid #f1f5f9' class='reason'>Score = highest calibrated dark-pattern probability among the page's elements, with every category scored separately.</div>
      <div class='foot'>{s['elements_analysed']} elements analysed · {s['suspicious_findings']} flagged</div></div>""")


def card_shot(a: dict, img_path: Path, title: str, sub: str, cat: str, icon_name: str, pad=(150, 90, 90), src_note="") -> str:
    """pad = (horizontal, top, bottom) context around the flagged boxes, in image pixels."""
    fs = [f for f in a["findings"] if f["category"] == cat and f["source_location"].get("bbox")]
    bb = [f["source_location"]["bbox"] for f in fs]
    im = Image.open(img_path).convert("RGB")
    x0 = max(0, min(b["x"] for b in bb) - pad[0]); y0 = max(0, min(b["y"] for b in bb) - pad[1])
    x1 = min(im.width, max(b["x"] + b["width"] for b in bb) + pad[0]); y1 = min(im.height, max(b["y"] + b["height"] for b in bb) + pad[2])
    crop = im.crop((x0, y0, x1, y1))
    disp_w = 566; sc = disp_w / crop.width
    boxes = ""
    for i, (f, b) in enumerate(zip(fs, bb), 1):
        l, t = (b["x"] - x0) * sc - 4, (b["y"] - y0) * sc - 4
        boxes += (f"<div style='position:absolute;left:{l:.0f}px;top:{t:.0f}px;width:{b['width']*sc+8:.0f}px;height:{b['height']*sc+8:.0f}px;"
                  f"border:2.5px solid {HEX[cat]};background:{HEX[cat]}1f;border-radius:4px'>"
                  f"<span style='position:absolute;left:-13px;top:-13px;width:21px;height:21px;border-radius:50%;background:{HEX[cat]};color:#fff;"
                  f"font-size:11px;font-weight:800;display:flex;align-items:center;justify-content:center;box-shadow:0 0 0 2px #fff'>{i}</span></div>")
    top = max(fs, key=lambda f: f["model_score"])
    return page(f"""<div class='card'>
      <div class='head'><div class='ico' style='background:{HEX[cat]}'>{icon(icon_name)}</div><div><div class='title'>{title}</div><div class='sub'>{sub}</div></div>
      <div class='pill' style='background:{HEX[cat]}1a;color:{HEX[cat]}'>{top['confidence_pct']}% sure</div></div>
      <div style='position:relative;margin-top:16px;border:1px solid #e2e8f0;border-radius:10px;overflow:visible;width:{disp_w}px'>
        <img src='{b64(crop)}' style='display:block;width:{disp_w - 2}px;border-radius:9px'>{boxes}</div>
      <div style='margin-top:14px;display:flex;gap:10px;align-items:flex-start'>
        <span style='flex:none;font-size:12px;font-weight:800;color:{HEX[cat]};background:{HEX[cat]}14;border:1px solid {HEX[cat]}55;border-radius:8px;padding:4px 9px'>{E(top['category_label'])}</span>
        <div class='reason'>{E(top['short_reason'])}. Flagged {len(fs)} region{'s' if len(fs) > 1 else ''} at ≥ {round(a['summary']['threshold']*100)}% calibrated confidence.</div></div>
      <div class='foot'>Real screenshot{src_note} · text read by OCR and classified by ClearCart's model</div></div>""")


def card_redesign(a: dict) -> str:
    f = next(x for x in a["findings"] if x["evidence"].startswith("LAST CHANCE"))
    return page(f"""<div class='card' style='background:#f0fdf4;border-color:#bbf7d0'>
      <div class='head'><div class='ico' style='background:#15803d'>{icon('sparkles')}</div><div><div class='title' style='color:#14532d'>Ethical redesign</div>
      <div class='sub'>Every finding comes with a transparent alternative</div></div></div>
      <div class='reason' style='margin-top:14px;color:#14532d'>{E(f['ethical_recommendation'])}</div>
      <div style='display:flex;align-items:center;gap:12px;margin-top:16px'>
        <div style='flex:1'><div class='lbl'>Before</div><div style='margin-top:6px;background:#fff;border:1px solid #fecaca;border-radius:10px;padding:11px 13px;font-size:14.5px;color:#64748b;text-decoration:line-through;text-decoration-color:#dc2626'>{E(f['evidence'])}</div></div>
        <div style='font-size:22px;color:#15803d;padding-top:18px'>→</div>
        <div style='flex:1'><div class='lbl'>After</div><div style='margin-top:6px;background:#fff;border:1px solid #86efac;border-radius:10px;padding:11px 13px;font-size:14.5px;font-weight:600;color:#14532d'>{E(f['suggested_rewrite'])}</div></div></div>
      <div class='foot' style='color:#4d7c0f'>Rewrites only remove pressure language — they never invent dates, prices or stock numbers</div></div>""")


def card_explain(a: dict) -> str:
    f = next(x for x in a["findings"] if (x["source_location"].get("css_selector") or "").endswith("promo-banner"))
    c = HEX[f["category"]]; loc = f["source_location"]
    chips = "".join(f"<span class='chip'>{E(t)}</span>" for t in f["key_terms"])
    sims = "".join(f"<div style='display:flex;justify-content:space-between;gap:10px;font-size:13px;padding:7px 0;border-top:1px solid #f1f5f9'>"
                   f"<span>“{E(s['text'][:60])}”</span><span style='color:#64748b;white-space:nowrap'>{E(s['label'])} · {s['similarity']:.2f}</span></div>"
                   for s in f["similar_examples"])
    return page(f"""<div class='card'>
      <div class='head'><div class='ico' style='background:#0f766e'>{icon('search')}</div><div><div class='title'>Explainable evidence</div><div class='sub'>Why the model flagged it, and exactly where</div></div>
      <div class='pill' style='background:{c}1a;color:{c}'>{E(f['category_label'])}</div></div>
      <div style='margin-top:14px;padding:9px 12px;border-left:3px solid {c};background:{c}0f;border-radius:6px;font-size:14px;font-weight:600'>{E(f['evidence'])}</div>
      <div style='margin-top:12px;font-size:15px;font-weight:700'>{E(f['confidence_statement'])}</div>
      <div style='display:flex;justify-content:space-between;font-size:11.5px;color:#64748b;margin-top:8px'><span>Calibrated model confidence</span><b style='color:#0f172a'>{f['confidence_pct']}%</b></div>
      <div style='height:8px;background:#f1f5f9;border-radius:9px;margin-top:4px'><div style='height:8px;width:{f['confidence_pct']}%;background:{c};border-radius:9px'></div></div>
      <div style='margin-top:14px' class='lbl'>Words that drove the prediction</div><div style='margin-top:7px'>{chips}</div>
      <div style='margin-top:6px' class='lbl'>Source location</div>
      <div style='margin-top:7px;font-size:13.5px'>Selector <span class='chip'>{E(loc['css_selector'])}</span> Tag <span class='chip'>&lt;{E(loc['html_tag'])}&gt;</span></div>
      <div style='margin-top:4px' class='lbl'>Similar labelled training examples</div><div style='margin-top:4px'>{sims}</div>
      <div class='foot'>Key terms = TF-IDF × logistic-regression contributions · neighbours = MiniLM cosine similarity</div></div>""")


def main() -> None:
    samples = {s["id"]: s for s in json.loads((ROOT / "data/samples/samples.json").read_text())}
    with TestClient(app) as c:
        text = c.post("/api/audit/text", json={"text": samples["text-deceptive"]["content"]}).json()
        page_html = c.post("/api/audit/html", data={"html": samples["html-checkout"]["content"]}).json()
        shots = {n: c.post("/api/audit/screenshot", files={"file": (n, (SHOTS / n).read_bytes(), "image/png")}).json()
                 for n in ["www.aafnation.com.png", "www.gasbike.net.png"]}
    outs = [
        render("1_high_deception_risk", card_risk(text)),
        render("2_scarcity_signal", card_shot(shots["www.aafnation.com.png"], SHOTS / "www.aafnation.com.png", "Scarcity signal",
                                              "Low-stock pressure found on a real product page", "scarcity", "box", pad=(170, 70, 175), src_note=" (aafnation.com, ContextRico-DP)")),
        render("3_artificial_urgency", card_shot(shots["www.gasbike.net.png"], SHOTS / "www.gasbike.net.png", "Artificial urgency",
                                                 "Countdown timer pressure found on a real product page", "urgency", "timer", pad=(150, 45, 45), src_note=" (gasbike.net, ContextRico-DP)")),
        render("4_ethical_redesign", card_redesign(text)),
        render("5_explainable_evidence", card_explain(page_html)),
    ]
    for o in outs:
        print(o, Image.open(o).size)


if __name__ == "__main__":
    main()
