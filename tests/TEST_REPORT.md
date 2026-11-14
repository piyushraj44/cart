# ClearCart QA report

## Commands run
- `.venv/bin/python -m pytest tests -q -rx` (also `-s` for timings)
- Integration: killed old uvicorn (pid 20399), started `.venv/bin/uvicorn backend.main:app --port 8000` and `npm run dev` (frontend), curl through http://localhost:5173/api/..., `npm run build`, Claude_Browser UI run. Servers stopped afterwards.

## Pytest result
`40 passed, 2 xfailed in 11.78s` (files: tests/conftest.py, test_data.py, test_model.py, test_extractors.py, test_api.py).
The 2 xfails are strict-xfail tests encoding real BUGS below (they fail today; they will XPASS-fail once fixed, then remove the marker).
Two initial failures were test mistakes (pandas read "" template_key as NaN -> fixed with keep_default_na=False; the 2MB-pasted-HTML case, see BUG 1).

Verified: raw TSV 2356 rows/cols; groundtrue 1608; zero text/template_key/page_id/group_id overlap between all splits (recomputed); split_stats counts == recomputed; recomputed TEST macro-F1 with saved model + app decision rule matches metrics.json within 0.002; probabilities sum to 1, flagged == p_dark>=tau, deterministic, predict([])==[]; key_terms subset of text; neighbors are train texts; segment_text offsets hold on 200 random multi-line inputs (emoji, non-BMP, curly quotes, combining chars, \r); HTML extraction/checked/hidden/script/label-for; OCR on generated "Only 2 left in stock" image; all API endpoints and error codes (422/413/415) listed in the brief; every finding evidence is substring of input and offsets match; confidence_pct == min(99, floor(score*100)); DOM findings have model_score None; hidden note never surfaces; screenshot happy path (ContextRico beemission.com.png) bbox in bounds, image_size matches; PDF export starts %PDF.

## Integration (live)
- Through Vite proxy: /health 200; /api/samples OK; text audit (deceptive) 200, high, 7 findings, 48 ms; html file upload (checkout) 200, high, 6 findings; screenshot (knockaround.com.png, 1700x919) 200, 5 findings, image_size correct.
- `npm run build`: success (built in 1.20s; JS 220.9 kB, CSS 29.4 kB; output in frontend/dist).
- Browser (Claude_Browser available): Home -> Start Audit -> sample dropdown -> Analyze. Deceptive text: High risk, 7 findings (urgency 3, misdirection 2, scarcity 1, social proof 1), evidence/offsets/key terms/rewrites render. Benign text: "No concern", 14 elements, 0 findings. At 375px width: no horizontal scroll on Home, Audit, results (deceptive and benign), Model page (scrollWidth == 375, no element overflowing).
- Console showed ~192 `net::ERR_CONNECTION_REFUSED` errors in the tab buffer; not reproduced during my runs (all API calls succeeded, backend log has 0 errors). Likely stale from the older killed backend/previous tab state; unconfirmed.

## Timings (warm, TestClient, in-process)
- Deceptive text sample: ~20 ms (UI-reported processing 37 ms; curl via proxy 48 ms total)
- 300-line input: ~350 ms

## BUGS
1. Pasted HTML 1-2 MB rejected with HTTP 400 (documented limit 2 MB).
   Repro: `POST /api/audit/html` form `html=<p>Hurry</p>` + `<p>Only 2 left</p>`*70000 (~1.2MB).
   Expected: 200 (<=2MB) and 413 above 2MB. Actual: 400 `{"detail":"Field exceeded maximum size of 1024KB."}` (Starlette multipart non-file field cap). File upload path works (1.3MB -> 200). Suspect backend/main.py audit_html (Form field). Fix: parse with `request.form(max_part_size=...)`, or have the UI send pasted HTML as a file.  Test: tests/test_api.py::test_html_form_between_1_and_2MB (xfail).
2. A leaf HTML element with >800 chars of text is silently dropped.
   Repro: html=`<p>` + "Hurry, only 2 left in stock. "*60 + `</p>` -> 422 "No visible text elements were found" (in a mixed page it just vanishes, no warning). Expected: split into sentences (as segment_text does) or at least a warning. Suspect backend/extractors.py extract_html `if len(text) > MAX_SEGMENT_CHARS * 2: continue` (comment assumes children own the text, wrong for leaves). Test: test_html_long_paragraph_not_dropped (xfail).
3. UI score display inconsistent with the 99% cap. Summary "Highest model score" and category breakdown "max N%" use rounding (deceptive: "Highest model score 100%" and "max 100%" while the confidence statement says 99%; benign: "Highest model score 52%" while text says 51%). API overall_score=0.9998/1.0. Expected: floor-capped (display_pct) like everywhere else. Suspect frontend result summary component (round(overall_score*100) and CategoryBreakdown.max_score) - not located by me. Also API `summary.overall_score` can be 1.0 (rounded to 4 dp).
4. Model/limits (not code bugs, FYI): in text-deceptive the documented ground truth "Another babe in Alexandria..." (social_proof) is NOT flagged (p_dark 0.643 < threshold 0.68) in text and HTML; "87% offers claimed. Hurry up!" is flagged as urgency (74%) whereas the sample's ground truth says scarcity. The sample's expected strings therefore are not all reproduced; make sure UI/docs do not claim 8/8. In the checkout HTML, "Add shipping protection for $4.99" is flagged by the ML as misdirection (98%) in addition to the DOM preselected finding (two findings for one element).
5. Minor: data/samples/text_deceptive.txt contains a literal backslash-quote (`\"CASES\"`) that is shown verbatim in the UI evidence (comes from the dataset text).

## Not tested / caveats
- Real browser upload of screenshot/HTML file through the UI and Download PDF/JSON buttons (only API-level PDF tested). Mobile check was by viewport emulation + DOM overflow measurement, plus visual screenshot only on desktop width.
- Did not rerun training or touch OCR eval files. Backend and Vite servers are stopped; frontend/dist was (re)created by `npm run build`.

---

## Re-run after fixes (by the lead engineer, after this QA pass)

Bugs 1–3 above were fixed:
- Bug 1: `audit_html` now parses the multipart form itself with a 4 MB part limit and enforces the documented 2 MB limit (413).
- Bug 2: long elements are split into sentence chunks (`chunk_text`) that share one selector; nothing is dropped. The selector sibling index is now memoised, which removed an O(n²) slowdown on 70k-element pages.
- Bug 3: the frontend floors and caps displayed percentages at 99%, matching the backend.

The two strict-xfail tests were promoted to normal regression tests. Screenshot audits now use the DEV-selected OCR config (psm 3 + upscale + paragraph merge) and an OCR threshold of 0.90. The model was retrained deterministically (`final_test` metrics are bit-identical).

```
.venv/bin/python -m pytest tests -q
42 passed in 8.76s
```

---

## Re-run after the intel-v1.0 layer (FSPM, MCAI, CDI, counterfactual, temporal, journey API)

New test files:
- `tests/test_intelligence_core.py` (FSPM, pressure, CDI, MCAI)
- `tests/test_intelligence_cf_temporal.py` (counterfactual and temporal)
- `tests/test_intelligence_regressions.py` (bugs found by the benchmark pass)

The original 42 tests are unchanged and still pass, which confirms backward compatibility of the existing API.

```
.venv/bin/python -m pytest tests -q
328 passed in 10.18s
```

---

## Re-run after closing the report-vs-code gaps

Added:
- `tests/test_journey_api.py`: 24 API integration tests covering journeys with text, HTML and screenshot stages, validation errors, journey and single-audit PDF export, schema round-trip (the JSON export), backward compatibility, and independent recomputation of FSPM, CDI and risk reduction.
- `tests/test_frontend_render.py`: runs `npm test` in `frontend/`, which renders the five intelligence sections and the result views with react-dom/server from saved real API responses.

New timeline events: interaction difficulty and accept/reject path changes. One bug was found in the live UI check (obstruction claimed for a journey without HTML); it is fixed and covered by a regression test.

```
.venv/bin/python -m pytest tests -q
355 passed in 12.27s
```

---

## intel-v1.1 (validation, optimisation, fusion, hardening)

New test files:
- `tests/test_intelligence_v11.py`: API integration tests for versions, confidence, inferred interaction, the optimiser, fusion, replay (including tamper and screenshot cases), the request cache, and the security guards.
- `tests/test_intelligence_properties.py`: 22 seeded randomized property tests, each over at least 200 cases.

Frontend render suite: 20/20 (`cd frontend && npm test`).

```
.venv/bin/python -m pytest tests -q
396 passed in 13.18s
```
