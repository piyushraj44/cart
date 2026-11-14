"""Build demo sample inputs from GENUINE dataset rows.

All visible strings come verbatim from the held-out TEST split of ec-darkpattern (never
trained on). Rows are chosen by dataset order (first N per category), not by whether the
model gets them right. The HTML wrapper markup is authored by us and is marked as such.

Run: python -m ml.make_samples
"""
from __future__ import annotations

import html
import json

import pandas as pd

from ml.common import DATA_PROCESSED, ROOT

OUT = ROOT / "data" / "samples"


def pick(df: pd.DataFrame, label: str, n: int, max_len: int = 160) -> list[dict]:
    rows = df[(df.label == label) & (df.text.str.len() <= max_len)].sort_values("row_id").head(n)
    return rows[["row_id", "text", "label"]].to_dict("records")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PROCESSED / "text_dataset.csv")
    test = df[df.split == "test"]

    dark = pick(test, "urgency", 2) + pick(test, "scarcity", 2) + pick(test, "social_proof", 2) + pick(test, "misdirection", 2)
    benign_mix = pick(test, "benign", 8)
    benign_only = test[(test.label == "benign") & (test.text.str.len() <= 160)].sort_values("row_id").iloc[8:22][["row_id", "text", "label"]].to_dict("records")

    # interleave so the deceptive sample looks like a page: benign, dark, benign, dark ...
    mixed = []
    for i in range(max(len(dark), len(benign_mix))):
        if i < len(benign_mix):
            mixed.append(benign_mix[i])
        if i < len(dark):
            mixed.append(dark[i])

    samples = []
    samples.append({
        "id": "text-deceptive", "title": "Product page text with dark patterns (test split)", "input_type": "text",
        "content": "\n".join(r["text"] for r in mixed),
        "provenance": "Every line is a verbatim row from ec-darkpattern dataset.tsv, held-out TEST split (row_ids "
                      + ", ".join(str(r["row_id"]) for r in mixed) + "). Lines chosen by dataset order, not by model output.",
        "expected": "Ground truth: " + "; ".join(f"“{r['text'][:40]}” = {r['label']}" for r in dark),
    })
    samples.append({
        "id": "text-benign", "title": "Benign storefront text (test split)", "input_type": "text",
        "content": "\n".join(r["text"] for r in benign_only),
        "provenance": "Verbatim 'Not Dark Pattern' rows from ec-darkpattern TEST split (row_ids "
                      + ", ".join(str(r["row_id"]) for r in benign_only) + ").",
        "expected": "Ground truth: all lines are labelled Not Dark Pattern.",
    })

    # HTML checkout: authored markup, genuine strings
    u, s, sp, m = (pick(test, "urgency", 1)[0], pick(test, "scarcity", 1)[0], pick(test, "social_proof", 1)[0], pick(test, "misdirection", 1)[0])
    b = pick(test, "benign", 6)
    e = lambda r: html.escape(r["text"])  # noqa: E731
    checkout = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Demo checkout</title>
<!-- Markup authored by the ClearCart team for demo purposes. Elements with data-source="test-split" contain verbatim
     strings from the ec-darkpattern TEST split. The two checkbox labels are authored (marked data-source="authored"). -->
<style>.banner{{background:#fee2e2}}</style></head>
<body>
<header><nav><a href="/">{e(b[0])}</a> <a href="/cart">{e(b[1])}</a></nav></header>
<main>
  <div class="banner" id="promo-banner" data-source="test-split">{e(u)}</div>
  <h1 class="product-title" data-source="test-split">{e(b[2])}</h1>
  <p class="stock" data-source="test-split">{e(s)}</p>
  <p class="activity" data-source="test-split">{e(sp)}</p>
  <p data-source="test-split">{e(b[3])}</p>
  <form id="checkout">
    <label data-source="authored"><input type="checkbox" name="protection" checked> Add shipping protection for $4.99</label>
    <label data-source="authored"><input type="checkbox" name="newsletter" checked> Send me marketing emails and partner offers</label>
    <label><input type="checkbox" name="remember"> Remember me</label>
    <button type="submit" class="btn-primary" data-source="test-split">{e(b[4])}</button>
    <a href="#" class="decline" data-source="test-split">{e(m)}</a>
  </form>
  <p class="hidden-note" style="display:none">This hidden note is ignored by the extractor.</p>
</main>
<footer><p data-source="test-split">{e(b[5])}</p></footer>
</body></html>
"""
    (OUT / "checkout_deceptive.html").write_text(checkout)
    samples.append({
        "id": "html-checkout", "title": "Checkout page HTML (authored markup, test-split strings)", "input_type": "html",
        "content": checkout,
        "provenance": f"HTML structure and the two pre-checked checkbox labels are authored by the ClearCart team for the demo. "
                      f"Text marked data-source=\"test-split\" is verbatim from ec-darkpattern TEST split (row_ids {u['row_id']}, {s['row_id']}, {sp['row_id']}, {m['row_id']}, "
                      + ", ".join(str(r['row_id']) for r in b) + ").",
        "expected": f"Ground truth: “{u['text'][:40]}” = urgency; “{s['text'][:40]}” = scarcity; “{sp['text'][:40]}” = social_proof; "
                    f"“{m['text'][:40]}” = misdirection; two pre-checked checkboxes are DOM observations.",
    })

    bb = test[(test.label == "benign") & (test.text.str.len() <= 160)].sort_values("row_id").iloc[22:30][["row_id", "text"]].to_dict("records")
    benign_html = "<!doctype html><html><body><!-- authored markup; text = verbatim ec-darkpattern TEST-split benign rows -->\n<main>\n" + \
        "\n".join(f'  <p data-source="test-split">{html.escape(r["text"])}</p>' for r in bb) + \
        '\n  <label><input type="checkbox" name="remember"> Remember me</label>\n  <button>Continue</button>\n</main></body></html>\n'
    (OUT / "benign_page.html").write_text(benign_html)
    samples.append({
        "id": "html-benign", "title": "Benign page HTML (authored markup, test-split strings)", "input_type": "html",
        "content": benign_html,
        "provenance": "Authored markup; paragraph text is verbatim 'Not Dark Pattern' rows from the TEST split (row_ids "
                      + ", ".join(str(r["row_id"]) for r in bb) + "). 'Remember me' and 'Continue' are authored.",
        "expected": "Ground truth: no dark patterns; the only checkbox is unchecked.",
    })

    (OUT / "samples.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False))
    (OUT / "text_deceptive.txt").write_text(samples[0]["content"] + "\n")
    (OUT / "text_benign.txt").write_text(samples[1]["content"] + "\n")
    print(f"wrote {len(samples)} samples to {OUT}")
    for smp in samples:
        print("---", smp["id"]); print(smp["content"][:700])


if __name__ == "__main__":
    main()
