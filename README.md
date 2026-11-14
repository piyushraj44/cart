# ClearCart

**2025–2026 false-positive study:** [research, synthetic evaluation, threshold decision, and implementation limits](docs/conservative_detection_2025_2026.md). The served text threshold remains 0.68; narrow HTML promotion and neutral-choice checks withhold uncertain findings for review.

**Evidence-first AI auditor for deceptive e-commerce interfaces.**

Paste webpage text, upload HTML, or upload a screenshot. ClearCart extracts every visible element, classifies each one with a trained ML model, and returns an audit. Each flagged finding includes the exact evidence, where it was found (character offsets, CSS selector, or OCR bounding box), why it may be manipulative, a calibrated confidence score ("ClearCart is 87% sure this is a fraud/scam (dark pattern) tactic"), and an ethical redesign suggestion. When nothing crosses the threshold it says **No suspicious patterns detected**.

> ClearCart is an assistive screening tool. Scores indicate *potentially* deceptive design learned from labelled examples. They are not proof of fraud and not legal findings.

---

## Project structure

```
clearcart/
├── frontend/   React 18 + Vite 5 + Tailwind 3 + lucide-react (landing, audit workspace, results, model page)
├── backend/    FastAPI app: extractors (text/HTML/OCR), auditor, explanations & redesign, PDF export
├── ml/         dataset inspection, preprocessing + leakage-safe split, training/eval, robustness, OCR eval
├── data/       raw/ (the two permitted datasets), processed/ (splits, reports), samples/ (demo inputs)
├── tests/      pytest suite (437 tests: core, intelligence layer, journey API, properties, false-positive gates, frontend render) + TEST_REPORT.md
├── report/     ClearCart_Report.pdf (8 pages), ClearCart_Math_Addendum.pdf (intel-v1.0), ClearCart_Update_Report.pdf (changes since; `python -m report.build_update`, which also rebuilds ClearCart_Full_Report.pdf), builders, figures/, feature_cards/
└── README.md
```

## Quick start (macOS)

Prerequisites: Python 3.12, Node 18+, Tesseract (only for screenshots).

```bash
brew install tesseract libomp node uv
```

```bash
git clone https://github.com/piyushraj44/clearCART.git && cd clearCART
```

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
```

The trained classifier (`ml/artifacts/clearcart_text_model.joblib`) and the processed splits are in the repo. On first start, the pretrained MiniLM encoder (`sentence-transformers/all-MiniLM-L6-v2`, about 90 MB) downloads once from Hugging Face; after that the app runs offline. Start the backend from the project root:

```bash
.venv/bin/uvicorn backend.main:app --port 8000
```

Start the frontend in a second terminal:

```bash
cd frontend && npm install && npm run dev
```

Open http://localhost:5173 and click **Start Audit**, then use **Load sample**, or click **Journey** for a multi-step audit. Saved real audits that work without the backend are at `/?mock=1`, `/?mock=benign` and `/?mock=journey`.

For a Python-capable container host, `Dockerfile` builds the frontend and serves it with the FastAPI model backend from one origin via `backend.hosted:app`. The container listens on `${PORT:-8000}`; `/health` is its health check. Build with `docker build -t clearcart .` and run with `docker run -p 8000:8000 clearcart`. The build downloads the frozen MiniLM encoder when it is absent from the build context. A static-only host cannot run live audits.

Linux and Windows: install Tesseract with your package manager. XGBoost needs an OpenMP runtime (`libgomp` / `vcomp`).

### Reproduce the ML pipeline from scratch

The raw datasets are not committed. Download them from their original sources (MD5-verified):

```bash
.venv/bin/python -m ml.download_data --images
```

```bash
.venv/bin/python -m ml.inspect_data
```
```bash
.venv/bin/python -m ml.preprocess
```
```bash
.venv/bin/python -m ml.train
```
```bash
.venv/bin/python -m ml.robustness
```
```bash
.venv/bin/python -m ml.eval_screenshots
```
```bash
.venv/bin/python -m ml.make_samples
```
```bash
.venv/bin/python -m pytest tests -q
```
```bash
.venv/bin/python -m report.build_report
```

Each step runs as a module from the project root, so the `ml` package can set `OMP_NUM_THREADS=1` before torch and XGBoost load. Without that, PyTorch's bundled libomp and Homebrew's libomp crash the process (exit 139) on macOS.

## Data (only the two permitted datasets)

| Dataset | Use | Licence |
|---|---|---|
| [yamanalab/ec-darkpattern](https://github.com/yamanalab/ec-darkpattern) `dataset/dataset.tsv`: 2,356 texts, `page_id, text, label, Pattern Category` | Training, validation, test | Apache-2.0 |
| [ContextRico-DP](https://zenodo.org/records/15036779) (Zenodo 15036779): 1,609 screenshots, `groundtrue.json` image-level labels | Screenshot/OCR evaluation of the text pipeline; training data only for the experimental, opt-in visual model (item 9) | CC-BY-4.0 |

The full inspection is in `data/processed/inspection_report.md`, generated by `ml/inspect_data.py`. Key facts:
- Categories: Not Dark Pattern 1178, Scarcity 418, Social Proof 312, Urgency 210, Misdirection 195, Obstruction 27, Sneaking 12, Forced Action 4.
- **366 rows are template duplicates** ("Only 2 left" vs "Only 7 left"). A random split leaks these templates across splits (see *Where we failed*).
- ContextRico has **no benign screenshots**, and its labels are image-level and mostly visual (Hard to Close 683, Default Choice 291, …). Only 125 label instances are text-observable (Limited Time, Countdown, Low Stock, High Demand, Activity). It is therefore used to evaluate the text pipeline and never trains the text model. Because the hackathon rules allow training on it, it also trains a separate, experimental visual pattern-type model (item 9 of *Where we failed*).

**Taxonomy.** We use 4 ML categories plus benign: *misleading urgency, misleading scarcity, social proof pressure, misdirection/confirmshaming*. Obstruction, Sneaking and Forced Action (43 rows in total) are too small to learn and are kept as an out-of-taxonomy probe set. A 5th category, *preselected option*, is a **DOM observation**: a `checked` checkbox in uploaded HTML. It is reported as rule-based, has no model score, and is never inferred from text or screenshots.

No synthetic, LLM-generated, scraped or external labelled data is used. MiniLM (`sentence-transformers/all-MiniLM-L6-v2`) is a **frozen, pretrained** feature extractor and is not fine-tuned. The optional second OCR engine, EasyOCR (Apache-2.0), uses its stock pretrained English models (CRAFT detector, MIT; `english_g2` recogniser). They are not fine-tuned, and the SHA-256 hashes are in `ml/artifacts/ocr_engine_eval.json`.

## Model

Inspired by the UC Berkeley Deceptive Pattern Detector (MiniLM embeddings → XGBoost), but selected by our own evaluation:

```
text ─┬─ TF-IDF (word 1–2g + char 2–5g) ─ Logistic Regression (C=1, balanced) ──── 0.7 ─┐
      └─ MiniLM-L6 384-d (frozen) + 7 surface feats ─ XGBoost (depth 5, lr .05) ─ 0.3 ─┴─> temperature scaling (T=0.448)
                                                    flag if calibrated P(dark) = 1 − P(benign) ≥ τ = 0.68
```

The split is grouped by page **and** text template (`StratifiedGroupKFold`, seed 42): train 1645 / val 328 / test 329, with zero text, template or page overlap. Model selection, calibration and the threshold all use **validation only**. The test set is used once.

### Held-out test results (329 rows)

| Metric | Value |
|---|---|
| Accuracy | 0.918 |
| Macro precision / recall / **F1** | 0.917 / 0.869 / **0.882** |
| Weighted F1 | 0.916 |
| ROC-AUC (one-vs-rest, macro) | 0.988 |
| Dark-vs-benign precision / recall / F1 at τ | 0.993 / 0.907 / 0.948 |
| Benign false-positive rate | 0.6% (1 of 167) |
| ECE before → after temperature scaling | 0.135 → 0.041 |

Per-class F1 values, the confusion matrix, FP/FN lists and all experiments are in `ml/artifacts/metrics.json` and the report. Repeated grouped CV over 5 seeds × 5 folds gives: ensemble 0.954 ± 0.011, TF-IDF+SVM 0.946 ± 0.013, TF-IDF+LR 0.939 ± 0.015, MiniLM+LR 0.926 ± 0.017, MiniLM+XGB (Berkeley-style) 0.923 ± 0.011. The ensemble had the best score in 18 of 25 folds.

### Where we failed and how we improved

1. **Leakage inflated the first score.** A naive random split scored macro-F1 0.947, but 63 test rows had a template twin in train. After grouping by page and template, the honest score is 0.895, and that is the number we report.
2. **The Berkeley-style model alone was not the best.** MiniLM+XGBoost scored 0.923 in CV, below TF-IDF baselines, on this short-text data. Soft-voting it with TF-IDF+LR raised the CV score to 0.954.
3. **Raw scores were over-confident** (ECE 0.135). One-parameter temperature scaling fit on validation reduced ECE to 0.041, so the displayed % is a calibrated probability. Displays are floored and capped at 99%.
4. **Too many false alarms at page scale.** Our first threshold rule chose τ = 0.30, which flagged 5 of 167 benign test strings. A real page has hundreds of benign elements, so the rule now breaks near-ties on validation toward the higher threshold (τ = 0.68). Test FPs dropped from 5 to 1, but recall fell from 0.963 to 0.907 (F1 0.966 → 0.948). The trade-off is deliberate, and it *lowered* the test score, so the rule was not tuned to the test set.
5. **Screenshots (OCR), evaluated on ContextRico only.** At the text threshold, naive OCR flagged 28.8% of test screenshots that have no text-observable label (status-bar clocks, timers). The OCR configuration and a stricter OCR threshold were selected on DEV images: psm 3, grayscale 2× upscale, paragraph merging, τ = 0.90. On TEST this halved negative flags to 14.4% (McNemar p = 0.0005) and raised F1 from 0.561 to 0.612, but category-correct recall fell from 0.634 to 0.561. The gain is modest, and we report it as such.
5b. **A second OCR engine (EasyOCR) did not win on DEV, so Tesseract stays the default.** `ml/eval_ocr_engines.py` compares Tesseract v4_psm3, EasyOCR 1.7.2 (Apache-2.0; pretrained CRAFT detector `craft_mlt_25k.pth` 83 MB + English recogniser `english_g2.pth` 15 MB, downloaded to `~/.EasyOCR/model/`) and their union on the same DEV/TEST images, the same current text model and the same selection rule. EasyOCR reads more of the text: at τ = 0.90 on DEV, category-correct recall is 0.929 vs 0.833 and F1 0.769 vs 0.745. But it flags 17.6% of negative DEV images (union: 18.4%), and no threshold up to 0.90 gets either below the 15% limit, so the rule keeps Tesseract@0.90. On TEST at τ = 0.90 the pattern is the same: EasyOCR category-correct recall is 0.683 vs 0.561 and F1 0.633 vs 0.612, but its negative flag rate is 0.208 vs 0.144. Neither difference is significant (paired McNemar p = 0.23 and 0.13). EasyOCR is also slower: 3.0 s/image on Apple MPS and 11.9 s on CPU (one OpenMP thread), against 0.57 s for Tesseract. On the synthetic choice screens it does read the faint grey decline text that Tesseract misses (S1 and S3 of S1–S3), and MCAI visual asymmetry becomes measurable on 7/8 screens instead of 5/8. Set `CLEARCART_OCR_ENGINE=easyocr|union` (and `CLEARCART_EASYOCR_DEVICE=mps`) to try it. Results: `ml/artifacts/ocr_engine_eval.json`, `report/figures/ocr_engines.png`.
6. **Real-world false positive: price lists flagged as "urgency".** A user audit of a fashion product page flagged a row of OCR-read prices (`£3.950.00 £39500 +2.980.00 …`) as 93% misleading urgency. A bare `₹ 2,990.00 ₹ 3,950.00` scored 0.947 and `$49.99` scored 0.85. The cause: only 10 of 2,302 training rows contain no words, so the model never learned that prices are benign, and it reads digits plus colons or periods as countdowns. Fix: an **abstention gate**. Elements with no alphabetic word are not classified, and the audit says so in a warning. On the held-out test split the gate changes nothing (macro-F1 0.882, dark-vs-benign F1 0.948, identical; `ml/artifacts/abstention_eval.json`). The cost is that a bare timer such as `00:12:47` with no words around it is no longer flagged on its own. **Not fixed:** prices or shipping/delivery text *with* words still produce false positives (`Sale price ₹ 2,990.00` 0.99, `Free shipping on orders over ₹2,990` 0.93, `Delivery in 3-5 working days` 0.85). Fixing this needs benign modern-retail training text, which the hackathon data rules do not allow.
7. **False positives on legitimate pages, including our own.** A screenshot of ClearCart's own *About us* page was audited as "Social proof pressure 97%". Tesseract reads the three pattern cards side by side as one line, `Countdowns and ‘hurry’ language with no real ‘Only 2 left!’ or 'selling fast' claims that cannot Crowd signals such as 'people are viewing this`, and that line scored 0.977. As pasted text the same page flagged `'Only 2 left!' or 'selling fast' claims that cannot be verified.` (scarcity 0.99), the confirmshaming example sentence (misdirection 0.99) and the badge `HTML only` (scarcity 0.73). There are three causes. (a) **Mention vs use:** the model scores a quoted example inside an explanation as if the page used it. (b) **Out-of-distribution text:** temperature scaling was fit on validation data, so a calibrated "99%" only holds for text that resembles the training data. (c) **Easy benign data:** most benign training rows are menu items, so any commerce-flavoured sentence leans dark. We added two post-hoc gates in `backend/auditor.py` (rules in `ml/model.py`). Neither changes the model, its probabilities or `metrics.json`. A gated element is never dropped silently: it is listed in the new optional `uncertain_elements` field, a warning explains it, and it no longer drives the risk gauge.
   - **Quote gate.** This applies when an element that crossed the threshold contains a quoted span (`"…"`, `“…”`, `‘…’`, `'…'`) **and** at least 4 words remain outside the quotes. In-word apostrophes such as don't are ignored. Mixed quote marks and a quotation cut off at the end of an OCR line also count; we added both after seeing the About-page OCR. The gate rescores the text with each quoted span replaced by `…`, a placeholder chosen on validation as the most score-neutral of 9 candidates. If the masked text falls below the threshold, the element is reported as a *quoted example — treated as a mention, not a claim*. Elements that are entirely a quotation, or have little context, are scored normally.
   - **Out-of-distribution (OOD) gate.** This uses the maximum cosine similarity (MiniLM) between the element and the 1,645 TRAIN texts. Its threshold τ_ood = 0.384 was set on the **validation split only**: it is the largest value with at most 2% of validation dark rows below it (3 of 161). An element that crossed the threshold but scores below τ_ood is reported as *uncertain — unlike the training data*. The same gate and threshold apply to text, HTML and OCR.
   - **Measured effect** (`ml/eval_false_positives.py` → `ml/artifacts/false_positive_eval.json`). *Held-out TEST, 329 rows:* the quote gate changes nothing. One test row qualifies (`… ENTER CODE \"CASES\" AT CHECKOUT`) and stays flagged after masking, as it should. The OOD gate withholds 3 true social-proof rows (e.g. `Chirag in Anoopshahr just got the JAYS … about 17 minutes ago`, similarity 0.34). Recall falls from 0.907 to 0.889 and macro-F1 from 0.882 to 0.874. Benign false positives stay at 1 of 167, because the only test false positive is in-distribution. So on the in-distribution test set the gates cost recall and buy nothing. *Probes (known-good text, not test data):* ClearCart's own About, Model and Audit pages, audited as HTML, as pasted text and as two About-page screenshots, go from 11 findings to 2 (1 HTML-mode and 1 text-mode). The remaining finding on each About-page text/HTML audit is `HTML only` (similarity 0.438, above τ_ood). The two About screenshots go to 0 findings. *ContextRico screenshots* (eval_screenshots protocol, τ = 0.90): compared with the previous served behaviour (no-words gate only), negative-image flag rate goes from 0.112 to 0.080 on DEV and from 0.104 to 0.096 on TEST, with no change in positive-image recall (paired McNemar p = 0.125 on DEV and 1.0 on TEST, so not significant). Two of the 4 DEV negatives that lost their flag show a real "Limited Time Offer" ad, so part of that reduction is unlabelled-urgency recall lost, not noise removed.
   - **Not fixed.** Short badge or label text close to scarcity templates (`HTML only` 0.73), commerce text with words (prices, shipping, delivery, see item 6), and quoted dark claims that do not survive masking will still produce errors. The OOD gate also suppresses some real dark patterns phrased unlike the training data. The real fix is **hard negatives**: benign explanatory, editorial and modern-retail text in training. The dataset rules do not allow adding it.

8. **Genuine sales flagged as urgency → evidence tiers and a claim-specificity check.** Wording alone cannot separate an honest sale from a fake one, but specificity often can. `backend/claim_check.py` (rules only, no training data, no change to any probability) lists *checkable* terms (calendar or named-day deadline, reference price, stated saving or discount, code, shipping/delivery/returns terms) and *vague or unverifiable* pressure (hurry, act now, fear of missing out, undated countdown, "limited", "up to X%", stock counts, live activity, a deadline that has already passed). An **urgency or scarcity** score above the threshold whose claim is checkable and has no vague marker is withheld as `checkable_claim` and listed under `uncertain_elements`; social proof and misdirection are never withheld this way. Every audit now reports three **evidence tiers**: *observed facts* (DOM observations, `evidence_tier: "observed"`), *pressure claims* (model findings, `"pressure_claim"`, each with its `claim_check` markers) and *held for review* (`uncertain_elements`), counted in `summary.tier_counts` and shown in the new Evidence tiers card.
   - **Measured effect** (`.venv/bin/python -m ml.eval_claim_check` → `ml/artifacts/claim_check_eval.json`; acceptance criteria fixed before scoring). *Held-out TEST (329):* recall 0.889 → 0.877 (2 rows, within the 5-point budget), benign FPs unchanged at 1 of 167, precision 0.993. *Fresh synthetic set* (24 invented strings, written before the rules were scored): FPR 0.75 → 0.08, precision 0.55 → 0.92, recall unchanged (0.917). *Exploratory synthetic text cases* (rules were written against these): FPR 0.69 → 0.38, precision 0.45 → 0.60, recall unchanged.
   - **Not fixed.** One withheld TEST row is a real countdown printed with spaced digits (`Ends in: 0 0 4 4 hours`), which the countdown pattern misses; it was left unchanged to avoid tuning on TEST. Price-only text that the model labels *misdirection* (`MRP ₹2,499. You save ₹500.`) is still flagged because the gate is limited to urgency/scarcity. Checkable terms can be false: the gate says a claim is checkable, never that it is honest. Synthetic sets are not field precision.

9. **The text pipeline cannot see visual patterns → an experimental, opt-in visual pattern-type model.** Most ContextRico labels are visual: a close control that is hard to find, a pre-selected option, an accept button styled far above the decline, a pop-up that interrupts. OCR followed by a text classifier cannot see layout, colour or prominence, so these were invisible to ClearCart. ContextRico had been evaluation-only; the hackathon rules allow training on it, so `ml/train_visual.py` fine-tunes ImageNet-pretrained CNNs (MobileNetV3-Large, EfficientNet-B0, ConvNeXt-Tiny) as a multi-label classifier over four required labels (Hard to Close, Default Choice, False Hierarchy, Nagging) and two optional ones (Privacy Zuckering, Pay to Avoid Ad). It does not touch the text model, its artifacts or its metrics.
   - **Data and split** (`ml/visual_data.py` → `data/processed/visual_split.csv`, `visual_split_stats.json`). 1,606 labelled images. `groundtrue.json` already reflects `correction.xlsx` for 94 of the 97 matching rows; the 3 rows where the correction sheet would add a label are reported and **not applied**, because the published ground truth is authoritative. Groups are web domain, mobile app name or Rico id, then 181 near-duplicate pairs (64-bit dHash within 4 bits) are merged, giving 1,414 groups. The split is 1,124 / 241 / 241 images with zero group overlap. The `ml/eval_screenshots` DEV/TEST images are these same ContextRico files (333 of them are in the visual split); the text model's OCR evaluation is unaffected because the text model is not trained on them.
   - **Protocol.** Acceptance criteria were written to `ml/artifacts_visual/criteria.json` before any model was trained: (a) beat both baselines on VAL macro-AP over the required labels, (b) for every required label, VAL AP > prevalence + 0.1, (c) TEST evaluated once, after selection, and reported whatever it shows. Baselines on VAL: label prior 0.234, frozen CLIP ViT-B/32 linear probe 0.762. Selection, per-label temperature scaling and thresholds use VAL only; a label without a threshold reaching VAL precision 0.7 is "not reliable" and never shown.
   - **Measured effect** (`ml/artifacts_visual/metrics.json`, report §4). Selected: mobilenet_v3_large, VAL macro-AP 0.760 (EfficientNet-B0 0.751; ConvNeXt-Tiny stopped before its first epoch and 3-split CV skipped, both at the user's request to finish early). Acceptance criteria: **not met** — (a) fails: the frozen CLIP probe scores 0.762 on VAL, 0.002 higher; (b) passes for all four required labels. The model is therefore **not ready to serve** and `CLEARCART_VISUAL` stays off. Held-out TEST (241 images, once): macro-AP 0.785 (CLIP probe 0.776, label prior 0.236; Mobile 0.791, Web 0.583). The TEST edge over CLIP changes nothing, because selection and criteria are decided on VAL. Grad-CAM pointing game on TEST: the peak falls in a ground-truth segment for 0.45 (20 TEST images have segments) of images, against 0.30; mean best IoU of the evidence box is 0.17 for the image centre. On ClearCart's own known-good page screenshots: `about_1280_viewport.png` fired *Default Choice* at 0.925 (a false alarm); `about_1440.png` fired nothing. The two optional labels passed the threshold rule on a single VAL positive each, which is degenerate; a minimum-support rule is needed next time.
   - **How it is served.** Off by default. `CLEARCART_VISUAL=1` adds `visual_signals` to screenshot audits (CPU, model files from `ml/artifacts_visual/` or `CLEARCART_VISUAL_DIR`). The signals are labelled `model (visual, experimental)` and never enter findings, the headline gauge, the CDI, the evidence tiers or the warnings.
   - **Not fixed.** ContextRico has **no benign screenshots**, so the model indicates a pattern *type*, never whether a page is deceptive; on an honest page it will still name a type. Only 58 / 12 / 13 split images are web pages, so web results rest on very few examples. Segments in `groundtrue.json` are image-level, not per label, so localisation can only be checked loosely. Labels are inherited from ContextRico without re-annotation.

## Deception intelligence layer (intel-v1.0, new)

This layer sits **on top of** the evaluated classifier. It does not change the classifier, its artifacts or its reported metrics. Every audit response gains an optional `intelligence` object, and a new journey endpoint audits multi-step shopping journeys. Status: implemented and unit-tested, and evaluated **only on synthetic, explicitly labelled benchmarks**, because no real-world ground truth exists for these indices. Full maths and results are in `report/ClearCart_Math_Addendum.pdf`.

| Module | What it computes | Formula (as implemented) |
|---|---|---|
| Stage / content pressure | Pressure of one page | p = 1 − Π_k (1 − c_k), where c_k is the max calibrated score of flagged findings in category k. Observed countdown resets enter as facts. |
| **FSPM** (`backend/intelligence/fspm.py`) | Sequential pressure across journey stages | FSPM = 100 · Σ F(pos_i) p_i / Σ F(pos_i), with F(1)=F(2)=1. Always reported next to **linear** and **equal** weighting. |
| **MCAI** (`mcai.py`) | Accept-vs-decline asymmetry | MCAI = 100 · Σ_avail w_d D_d / Σ_avail w_d over visual (0.4), interaction (0.3) and semantic (0.3). Unavailable dimensions are dropped and reported as `coverage`. |
| **CDI** (`cdi.py`) | Audit-level screening index | CDI = 100 · Σ_avail w_j x_j / Σ_avail w_j over content pressure (0.55), choice asymmetry (visual + interaction only, 0.25) and observed obstruction (0.20). Weights version `cdi-w1`. Sensitivity: ±50% one-at-a-time and 1,000 Dirichlet draws. |
| **Counterfactual** (`counterfactual.py`) | Ethical redesign and before/after | Rewrites only remove or neutralise wording; numbers, prices and offers must be preserved. Risk reduction = 100 · (CDI_orig − CDI_redesign) / CDI_orig, **model-estimated**. |
| **Temporal** (`temporal.py`) | Evidence timeline across snapshots | A countdown "reset" is claimed only when two observations show more time remaining later. Also reports new pre-selections, stock and price changes, changing declared interaction difficulty, and changing accept/decline paths (MCAI). Facts, observations and model inferences are kept separate. |

Rules the code enforces:
- Every value carries its provenance: `model`, `measured`, `default`, `declared`, `observed`, `by_construction` or `unavailable`.
- Visual measurements are never fabricated for pasted text.
- HTML visual features come from declared CSS. Rendered area needs a layout engine, so it is reported unavailable.
- Screenshot visual features come from OCR boxes and pixel contrast.
- Click and step counts are only used when declared in a journey.


### intel-v1.1 additions

- **Inferred interaction asymmetry (HTML).** A pre-ticked add-on, a collapsed decline, or a decline link that navigates away each add a declining step. These steps are labelled `inferred`; step counts you declare take precedence.
- **Separate indicators.** CDI severity, evidence **confidence** and data **coverage** are reported separately. Alternative aggregators are shown for transparency.
- **Constrained counterfactual optimiser** (`optimiser.py`). Each element gets its lowest-scoring feasible candidate, with no added or removed facts, kept disclosure terms and kept choices. This is optimal to within a 0.005 tie band, and every candidate is listed in the response.
- **Cross-modal fusion.** `fuse_with` merges captures of the same page, so a finding seen in several formats is counted once.
- **Replay.** `POST /api/audit/journey/replay` re-runs a journey from its replay record, verifying each input by SHA-256, and reports whether the results are identical. Every response carries a `versions` stamp.
- **Request-scoped inference cache.** No text is classified twice within a request, and nothing is kept between requests.
- **Hardening:**
  - Images over 40 megapixels are rejected before decoding.
  - Journey requests are capped at 40 MB.
  - Error logs no longer include user content.
  - A ReDoS bug is fixed (114 s down to milliseconds).
- **Validation studies:** `ml/research_intelligence.py` (R1–R6). The honest result: Fibonacci beats equal and max weighting but **not** linear weighting, so it stays the proposed scheme, with the baselines always shown beside it.

New API:

| Method | Path | Notes |
|---|---|---|
| POST | `/api/audit/journey` | multipart: `stages` = JSON list of `{name, stage_type?, input_type, content?, file_field?, captured_at?, choice_path?}` (1–8 stages), plus one file per screenshot stage |
| GET | `/api/samples/journeys` | Demo journeys. They are **synthetic**: authored order built from real test-split strings, with authored parts listed in `provenance` |
| POST | `/api/export/journey-pdf` | Journey PDF report |

Existing endpoints are unchanged apart from the new optional `intelligence` field.

## Opt-in v2 experiments (Mathur et al. 2019 data + fine-tuned transformer)

The default served model is unchanged. `CLEARCART_MODEL=v2 .venv/bin/uvicorn backend.main:app --port 8000` serves the
VAL-selected v2 model from `ml/artifacts_v2/` (`selected.json`); `ensemble_v2`, `transformer_v2` and `transformer_v1data`
can also be named directly. Audit `versions` now carry `model_name` and `model_version`.

```bash
.venv/bin/python -m ml.mathur      # inspect + match Mathur data, leakage-safe eligibility -> data/processed/mathur_*.{md,csv,json}
.venv/bin/python -m ml.train_v2    # ensemble_v2, transformer_v1data, transformer_v2, CV, Deceptive? model -> ml/artifacts_v2/metrics.json
```

* Mathur et al. (GPL-3.0) is the source of ec-darkpattern's dark rows (`page_id` = Mathur row, verified). Of its 1,818 rows,
  175 pass the leakage rule against the frozen VAL/TEST split, and every one of them repeats a text already in TRAIN,
  so the new data adds no new language. Details: `data/processed/mathur_inspection.md`.
* Selection is on VAL macro-F1. `ensemble_v2` ties the served model on VAL (0.9705) and gives the same TEST predictions
  (macro-F1 0.882). The fine-tuned distilroberta models score higher on TEST (0.902 and 0.911) but lower on VAL, have
  5-7x the benign false positives on TEST (5 and 7 vs 1 of 167), and match the ensemble in repeated grouped CV (0.950-0.953). So they were not selected.
* Transformer explanations: `key_terms` uses leave-one-word-out occlusion on the transformer, and `neighbors` uses the
  cosine similarity of its mean-pooled embeddings. Inference runs on CPU.
* The `Deceptive?` ("verified fake") text model is not exposed: its grouped-CV AUROC (0.85) is below a pattern-type
  prior (0.87), and for countdown timers it is below chance (0.44).

## Opt-in experimental visual pattern-type model (ContextRico-DP screenshots)

Off by default; see item 9 of *Where we failed* for what it can and cannot claim. Each stage runs as a module from the
project root, in this order. The acceptance criteria must be fixed before anything is trained, and TEST is read only by `finalize`.

```bash
.venv/bin/python -m ml.visual_data                               # labels, groups, split, image cache -> data/processed/visual_split*.{csv,json}
.venv/bin/python -m ml.train_visual criteria                      # fix acceptance criteria BEFORE training -> ml/artifacts_visual/criteria.json
.venv/bin/python -m ml.train_visual baselines                     # label prior + frozen CLIP linear probe (VAL only)
.venv/bin/python -m ml.train_visual train --arch mobilenet_v3_large   # also efficientnet_b0, convnext_tiny
.venv/bin/python -m ml.train_visual cv --arch convnext_tiny       # 3 grouped re-splits of TRAIN+VAL (variance only)
.venv/bin/python -m ml.train_visual finalize                      # select on VAL, calibrate, TEST once, figures, export -> metrics.json
```

Serve it on screenshot audits with `CLEARCART_VISUAL=1 .venv/bin/uvicorn backend.main:app --port 8000`
(model files from `ml/artifacts_visual/`, or set `CLEARCART_VISUAL_DIR`). Then `.venv/bin/python -m report.build_update`
adds the visual section to the update report; without `ml/artifacts_visual/metrics.json` the section is skipped.

## API

| Method | Path | Body |
|---|---|---|
| GET | `/health` | Status, model loaded, OCR available |
| GET | `/api/model-info` | Architecture, categories, threshold, test metrics, limitations |
| GET | `/api/samples` | Genuine demo inputs with provenance |
| POST | `/api/audit/text` | `{"text": "..."}` (≤ 200k chars) |
| POST | `/api/audit/html` | multipart `file` (.html, ≤ 2 MB) or form field `html` |
| POST | `/api/audit/screenshot` | multipart `file` (png/jpg/webp, ≤ 8 MB) |
| POST | `/api/export/pdf` | An `AuditResponse` JSON → PDF |

Each finding has `category`, `evidence`, `explanation`, `model_score` (calibrated P(dark)), `confidence_pct`, `confidence_statement`, `risk_level`, `source_location`, `key_terms` (words that pushed the TF-IDF model), `similar_examples` (nearest labelled *training* rows), `ethical_recommendation` and `suggested_rewrite`. Errors return `{"detail": ...}` with status 413, 415, 422 or 503. The model loads once at startup. Uploads are processed in memory and never stored. Nothing is sent to third parties.

## Samples

`data/samples/` contains `samples.json` (served to the UI), text and HTML files, and `screenshots/`. Every visible string is a **verbatim row from the held-out test split**, chosen by dataset order rather than by model output. Row ids and provenance are listed in each sample. The HTML markup and the two checkbox labels in the checkout demo are authored by us and marked as authored.

## Tests

```bash
.venv/bin/python -m pytest tests -q
```

The suite has 437 tests. 42 are the original core tests, covering data loading, split leakage, model reload, test-F1 reproduction, extractor offsets, HTML/OCR extraction, every endpoint and error code, the benign and deceptive samples, evidence-matches-input checks and PDF export. The rest cover the intelligence layer (FSPM, MCAI, CDI and sensitivity, counterfactual, temporal, regressions) and journey API integration (24 tests that recompute every index from the response). `tests/test_false_positive_gates.py` (24 tests) covers the two false-positive gates and checks that the reported TEST metrics are reproduced with the gates disabled. `tests/test_frontend_render.py` also runs the frontend render suite, which you can run on its own with `cd frontend && npm test`. See `tests/TEST_REPORT.md` for the live integration results.

## Limitations

- The model was trained on short English strings from 2019 shopping sites. Long paragraphs, other languages and other domains are out of distribution.
- The model can say text *resembles* labelled dark patterns. It cannot verify whether a countdown or stock claim is actually false.
- **Out-of-domain retail pages.** The benign training strings are 2019 navigation and menu text. Modern product pages with prices, shipping thresholds or delivery times can be flagged as urgency (see item 6 of the failure log). Treat findings on price and shipping text with caution.
- Category confusion between scarcity and urgency (e.g. "87% offers claimed. Hurry up!") accounts for most test errors. The categories genuinely overlap.
- Screenshot support is OCR followed by text classification. Layout, colour and button prominence are not modelled, except by the opt-in experimental visual model, which names a visual pattern type and never affects findings or scores (item 9).

## Future scalability

See section 8 of the report. In short: ONNX-quantised MiniLM in a browser extension (as Berkeley does), batching plus a Redis cache keyed by element-text hash, a stateless FastAPI service behind autoscaling, an active-learning loop fed by reviewer feedback, multilingual encoders, a CLIP-style visual branch trained on ContextRico regions, and multi-page journey crawling (product → cart → checkout) with diffing.
