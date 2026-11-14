"""Build demo JOURNEYS for the journey audit (data/samples/journeys.json).

The datasets contain single strings/screenshots, not journeys, so every journey here is SYNTHETIC
(authored stage order) and flagged `synthetic: true`. Real strings are taken verbatim from the
held-out TEST split of ec-darkpattern (asserted below). Authored text (marked AUTHORED in the
provenance) is limited to: page markup/CSS, a second countdown reading used to demonstrate reset
detection, a pre-selected add-on label, an accept-button label, and declared step counts.

Run: python -m ml.make_journey_samples
"""
from __future__ import annotations

import html
import json

import pandas as pd

from ml.common import DATA_PROCESSED, ROOT

OUT = ROOT / "data" / "samples" / "journeys.json"


def main() -> None:
    df = pd.read_csv(DATA_PROCESSED / "text_dataset.csv")
    test = df[df.split == "test"]

    def real(text: str) -> dict:
        rows = test[test.text == text]
        assert len(rows) == 1, f"not a unique TEST-split row: {text!r}"
        r = rows.iloc[0]
        return {"text": r.text, "row_id": int(r.row_id), "label": r.label}

    p1 = [real("Mystic Topaz: Rings, Earrings, Necklaces and Bracelets"), real("Only 5 item(s) left."),
          real("19 people viewed this product per day"), real("Shipping and Duties")]
    p2 = [real("My Account"), real("45 ADDED TO BAG IN THE PAST 24 HOURS"), real("OFFER ENDS 00:54:59"), real("Pay weekly")]
    decline = real("no thanks, I'll save money another day")
    benign_checkout = real("Take time to pay.")
    exit_decline = real("NO, I DON'T FEEL LUCKY")
    e = html.escape
    checkout_html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Checkout</title>
<!-- AUTHORED demo markup. data-source="test-split" = verbatim ec-darkpattern TEST row; data-source="authored" = written for this demo. -->
<style>
.timer{{color:#b91c1c;font-weight:700}}
.btn-primary{{background:#0f766e;color:#ffffff;font-size:20px;font-weight:700;padding:14px 28px;border-radius:8px}}
.decline{{color:#9ca3af;font-size:11px}}
</style></head><body><main>
<p class="timer" data-source="authored">OFFER ENDS 00:58:40</p>
<p data-source="test-split">{e(benign_checkout['text'])}</p>
<form id="checkout">
  <label data-source="authored"><input type="checkbox" name="protection" checked> Add shipping protection for $4.99</label>
  <button type="submit" class="btn-primary" data-source="authored">Claim discount and continue</button>
  <a href="#" class="decline" data-source="test-split">{e(decline['text'])}</a>
</form></main></body></html>"""

    t0 = "2026-11-12T10:00:00+00:00"
    j1 = {
        "id": "journey-subscription",
        "title": "Subscription checkout journey (synthetic)",
        "description": "Product → cart → checkout → exit with scarcity, social proof, a countdown observed to increase between snapshots, a pre-ticked add-on and a hard-to-cancel exit.",
        "synthetic": True,
        "provenance": (
            "SYNTHETIC journey: stage order and timestamps are authored. Verbatim TEST-split rows: "
            + ", ".join(f"#{r['row_id']} ({r['label']})" for r in p1 + p2 + [benign_checkout, decline, exit_decline])
            + ". AUTHORED: checkout markup/CSS, the second countdown reading 'OFFER ENDS 00:58:40' (to demonstrate reset "
              "detection against the cart reading 00:54:59 taken 9 minutes earlier), the pre-ticked 'Add shipping protection for $4.99', "
              "the accept labels 'Claim discount and continue' and 'Yes, keep my membership', and the declared 1-vs-5 step cancellation path."),
        "stages": [
            {"name": "Product page", "stage_type": "product", "input_type": "text", "synthetic": True,
             "captured_at": t0, "content": "\n".join(r["text"] for r in p1)},
            {"name": "Cart", "stage_type": "cart", "input_type": "text", "synthetic": True,
             "captured_at": "2026-11-12T10:04:00+00:00", "content": "\n".join(r["text"] for r in p2)},
            {"name": "Checkout", "stage_type": "checkout", "input_type": "html", "synthetic": True,
             "captured_at": "2026-11-12T10:13:00+00:00", "content": checkout_html},
            {"name": "Cancellation / exit", "stage_type": "exit", "input_type": "text", "synthetic": True,
             "captured_at": "2026-11-12T10:20:00+00:00",
             "content": "Yes, keep my membership\n" + exit_decline["text"],
             "choice_path": {"label": "Keep membership vs cancel membership (declared)", "accept_steps": 1, "decline_steps": 5}},
        ],
    }
    b = [real(t) for t in ["Women's Bags", "Coffee", "Sunflower", "My Account", "Shipping and Duties", "Pay weekly",
                           "Take time to pay.", "Product Recycling", "My Project Lists"]]
    j2 = {
        "id": "journey-benign",
        "title": "Benign three-step journey (synthetic order, real strings)",
        "description": "Product → cart → checkout using only 'Not Dark Pattern' TEST-split strings.",
        "synthetic": True,
        "provenance": "SYNTHETIC journey: stage order authored; every line is a verbatim TEST-split 'Not Dark Pattern' row: "
                      + ", ".join(f"#{r['row_id']}" for r in b) + ". No authored text.",
        "stages": [
            {"name": "Product page", "stage_type": "product", "input_type": "text", "synthetic": True, "content": "\n".join(r["text"] for r in b[0:3])},
            {"name": "Cart", "stage_type": "cart", "input_type": "text", "synthetic": True, "content": "\n".join(r["text"] for r in b[3:6])},
            {"name": "Checkout", "stage_type": "checkout", "input_type": "text", "synthetic": True, "content": "\n".join(r["text"] for r in b[6:9])},
        ],
    }
    OUT.write_text(json.dumps([j1, j2], indent=2, ensure_ascii=False))
    print("wrote", OUT, [len(j["stages"]) for j in (j1, j2)])


if __name__ == "__main__":
    main()
