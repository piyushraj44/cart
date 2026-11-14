# Conservative dark-pattern detection study (14 November 2026)

## Research and decision

Recent work supports treating a dark pattern as an interface behavior, not a keyword. [DPGuard (WWW 2025)](https://doi.org/10.1145/3696410.3714593) combines visual and textual UI information. [UMBRA (2026 preprint)](https://arxiv.org/abs/2603.21515) uses text, visual checks, interactions, and cookie state to detect consent patterns that static text misses. [Grossman et al. (ICWSM 2026)](https://ojs.aaai.org/index.php/ICWSM/article/view/42676) compare elements *within* cookie banners and find manipulation even in otherwise compliant banners. These sources motivate using nearby controls and markup; they do not validate ClearCart's particular rules.

[Rabanser and Papernot (2025 preprint)](https://arxiv.org/abs/2510.20242) show why calibration or a higher score cutoff alone may not improve selective classification: score ranking and distribution shift matter. ClearCart already uses temperature scaling and quote/OOD abstention. We therefore tested thresholds and a narrow context decision without retraining or changing calibrated probabilities.

**Implemented:** HTML extraction marks three forms of structured, relatively neutral promotion: a dated offer in a valid `<time datetime>`, a percent-off label next to `<del>` and `<ins>` prices, or a percent-off button next to stated offer duration and a neutral alternative. A scored element in those contexts is withheld from findings and listed under `uncertain_elements` as `structured_promotion`. An exact neutral refusal label such as `No thanks` is likewise withheld as `neutral_choice` **only when it is an HTML button or link**. Longer guilt-framed refusals still flag. These checks do not assert that a discount, expiry, or website is truthful; that cannot be established by static markup. DOM preselection findings still run independently.

A disabled, checked checkbox labelled “strictly necessary” or “essential cookies” is now excluded from the preselection observation: it is a standard required control, not an optional choice. Checked marketing and partner-sharing controls still produce the DOM finding. This is limited to observable `disabled` markup plus an essential-purpose label; it does not certify consent compliance.

**Retained:** the served v1 classifier, its probabilities, the 0.68 text/HTML threshold, the 0.90 OCR threshold, the quote and OOD gates, and all existing categories. OCR and pasted text lack reliable DOM context, so the new checks do not change their decisions. Standard balanced consent controls in the synthetic set were already below threshold. We did not add a blanket consent-banner exemption because the choice layout and behavior can be deceptive.

## Reproducible experiment

Run `.venv/bin/python -m ml.eval_conservative_detection`. The script stores exact case definitions and decision logic in [the evaluator](../ml/eval_conservative_detection.py) and writes [raw results](../ml/artifacts/conservative_detection_eval.json). It evaluates one named target per case. The first set has 24 invented benign and 16 invented deceptive elements, including promotions, discounts, deadlines, stock claims, consent choices, neutral refusals, confirmshaming, and a preselected paid add-on. The second set has 8 benign and 8 deceptive variants, written after the rule was specified but before its scores were inspected. Neither set contains real sites or independently verified commercial claims. Labels describe the stated fixture assumptions; they are not evidence that similarly worded live offers are genuine or fake.

Before scoring, we set a five percentage point maximum recall loss and required **both** higher precision and lower false-positive rate (FPR). The exploratory first set influenced the narrow rule design; the second set is an additional synthetic check, not an independent field sample. For the original grouped TEST text split (167 benign, 162 dark), we used its existing labels only as a safety veto on threshold changes and to check for regressions. It has no DOM features, so it cannot validate the HTML rule's real-world precision.

| Evaluation | Decision | Precision | Recall | F1 | FPR | TP / FP / FN / TN |
|---|---|---:|---:|---:|---:|---:|
| Synthetic exploratory (40) | Existing gates, 0.68 | 0.4444 | 0.7500 | 0.5581 | 0.6250 | 12 / 15 / 4 / 9 |
| Synthetic exploratory (40) | New context and neutral-choice gates | 0.5217 | 0.7500 | 0.6154 | 0.4583 | 12 / 11 / 4 / 13 |
| Synthetic second check (16) | Existing gates, 0.68 | 0.4545 | 0.6250 | 0.5263 | 0.7500 | 5 / 6 / 3 / 2 |
| Synthetic second check (16) | New gates | 0.7143 | 0.6250 | 0.6667 | 0.2500 | 5 / 2 / 3 / 6 |
| Original held-out text TEST (329) | Existing gates, 0.68 | 0.9931 | 0.8889 | 0.9381 | 0.0060 | 144 / 1 / 18 / 166 |
| Original held-out text TEST (329) | New gates | 0.9931 | 0.8889 | 0.9381 | 0.0060 | 144 / 1 / 18 / 166 |

The threshold-only experiment used the same model and existing gates. On the exploratory set, 0.85 improved precision to 0.5000 and FPR to 0.5000 at unchanged recall 0.7500. On the original TEST split, 0.85 left the sole benign false positive unchanged and lowered recall from 0.8889 to 0.7778 (a loss of 11.11 percentage points). At 0.75, TEST recall dropped to 0.8395 and FPR remained 0.0060. **No higher threshold passed both the synthetic improvement and TEST safety criteria**, so the deployment threshold remains 0.68. The 0.90 OCR threshold was selected in a separate earlier DEV experiment and was not changed here.

## Limits and next evidence

The observed synthetic improvement is small-sample and partly rule-development data; it cannot estimate precision or FPR on live websites. Price comparisons and `<time>` elements can be fabricated, and a real deadline may still be used deceptively. Withholding keeps these cases visible for review. The text model still misflags ordinary offers, shipping, reviews, and undated countdowns when it has no DOM context. Consent asymmetry and trial cancellation can also be missed when the target text is bland. A next model revision should use independently annotated, context-rich real pages and hard-negative training examples, with site-level splits and a separate untouched test set before considering retraining or broader suppression.

An additional four-case synthetic DOM check isolated the essential-cookie rule: before, 2 true positives and 2 false positives (precision 0.50, recall 1.00, F1 0.667, FPR 1.00); after, 2 true positives and 0 false positives (precision, recall and F1 1.00, FPR 0.00). This tiny constructed check only verifies the narrow rule's behavior.
