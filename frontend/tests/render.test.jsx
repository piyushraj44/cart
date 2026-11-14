// Frontend rendering tests (no extra dependencies): esbuild (bundled with Vite) compiles this file and
// react-dom/server renders the real result views with SAVED REAL API responses from src/mock/.
// Run: npm test
import assert from 'node:assert/strict';
import { renderToString } from 'react-dom/server';
import Results from '../src/components/Results.jsx';
import JourneyResults from '../src/components/journey/JourneyResults.jsx';
import CDISection from '../src/components/intel/CDISection.jsx';
import MCAISection from '../src/components/intel/MCAISection.jsx';
import CounterfactualSection from '../src/components/intel/CounterfactualSection.jsx';
import TimelineSection from '../src/components/intel/TimelineSection.jsx';
import { EvidenceRefs, ProvenanceChip, ReproducibilityBlock } from '../src/components/intel/shared.jsx';
import EvidenceTiers from '../src/components/EvidenceTiers.jsx';
import FusionCard from '../src/components/journey/FusionCard.jsx';
import { ReplayBanner } from '../src/components/journey/ReplayPanel.jsx';
import { fuseProblem, newStage, toRequest, validateStage } from '../src/components/journey/journeyUtils.js';
import journey from '../src/mock/sampleJourney.json';
import audit from '../src/mock/sampleAudit.json';
import benign from '../src/mock/benignAudit.json';

const results = [];
function test(name, fn) {
  try { fn(); results.push([name, true]); } catch (e) { results.push([name, false, e]); }
}
// strip tags so assertions read the visible text
const text = (html) => html.replace(/<!-- -->/g, '').replace(/<[^>]+>/g, ' ').replace(/&amp;/g, '&').replace(/&#x27;/g, "'").replace(/\s+/g, ' ');
const attr = (s) => s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/'/g, '&#x27;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
// "CDI 73.5 → 41.2 · 43.9% lower (model-estimated)" (an sr-only "to" may follow the arrow)
const reNum = (v) => fmt1(v).replace('.', '\\.');
const cfLine = (cf) => new RegExp(`CDI ${reNum(cf.original_cdi)} → (to )?${reNum(cf.redesigned_cdi)} · ${reNum(cf.risk_reduction_pct)}% lower \\(model-estimated\\)`);
const noJunk = (t) => { for (const bad of ['NaN', 'undefined', '[object Object]']) assert.ok(!t.includes(bad), `rendered "${bad}"`); };
const model = { status: 'ready', data: { test_metrics: { calibration_ece: 0.0409 } } };
const fmt1 = (v) => v.toFixed(1);

test('journey results render every intelligence section from the real response', () => {
  const html = renderToString(<JourneyResults result={{ response: journey }} model={model} onNewJourney={() => {}} onEditJourney={() => {}} />);
  const t = text(html);
  noJunk(t);
  for (const h of ['Deception Intelligence Score (CDI)', 'Fibonacci Pressure Analysis (FSPM)', 'Temporal Evidence Timeline',
    'Choice Asymmetry (MCAI)', 'Counterfactual Ethical Redesign', 'Screening index, not a probability of fraud']) {
    assert.ok(t.includes(h), `missing section "${h}"`);
  }
  // numbers on screen are exactly the backend's numbers
  for (const s of ['fibonacci', 'linear', 'equal']) assert.ok(t.includes(fmt1(journey.fspm.scores[s])), `FSPM ${s} score not shown`);
  assert.ok(t.includes(fmt1(journey.cdi.cdi)), 'CDI value not shown');
  for (const st of journey.stages) assert.ok(t.includes(st.name), `stage "${st.name}" missing`);
  assert.ok(t.includes('Observed fact'), 'observed facts not distinguished');
  const cf = journey.counterfactual;
  assert.ok(html.includes(`title="${attr(cf.label)}"`), 'model-estimated caveat not shown verbatim (tooltip)');
  assert.ok(cfLine(cf).test(t), 'redesign summary line not the API values');
  // one green redesign card per edit, keyed by stage + finding, carrying that finding's recommendation
  for (const e of cf.edits) {
    assert.ok(html.includes(`id="redesign-s${e.stage_index}-${e.target}"`), `redesign card for stage ${e.stage_index} ${e.target} missing`);
    const f = journey.stages.find((st) => st.index === e.stage_index).audit.findings.find((x) => x.id === e.target);
    if (f?.ethical_recommendation) assert.ok(t.includes(f.ethical_recommendation.slice(0, 40)), `recommendation for ${e.target} missing`);
  }
  assert.ok(!t.includes('Findings in detail'), 'stage panels still show Findings in detail');
});

test('accordion: one top-level section open at a time (redesign on audits, FSPM on journeys; stage 1 preview open)', () => {
  const openTitles = (html) => [...html.matchAll(/data-collapsible="open"[\s\S]*?aria-expanded="true"[^>]*>([\s\S]*?)<\/button>/g)].map((m) => text(m[1]));
  const a = renderToString(<Results result={{ response: audit, inputType: 'text', inputText: '', isMock: true }} model={model} onNewAudit={() => {}} />);
  const ao = openTitles(a);
  assert.equal(ao.length, 1, `expected one open section, got ${ao.length}`);
  assert.ok(ao[0].includes('Counterfactual Ethical Redesign'), 'Ethical redesign not the open section');
  const j = renderToString(<JourneyResults result={{ response: journey }} model={model} onNewJourney={() => {}} />);
  const jo = openTitles(j);
  assert.equal(jo.length, 2, `expected FSPM + stage 1 open, got ${jo.length}`);
  assert.ok(jo.some((x) => x.includes('Fibonacci Pressure Analysis (FSPM)')), 'FSPM not open');
  assert.ok(jo.some((x) => x.includes(journey.stages.find((st) => st.index === Math.min(...journey.stages.map((s2) => s2.index))).name)), 'stage 1 not open');
});

test('timeline shows the observed countdown reset and the new declared/choice-path events', () => {
  const t = text(renderToString(<JourneyResults result={{ response: journey }} model={model} onNewJourney={() => {}} onEditJourney={() => {}} />));
  const reset = journey.temporal.events.find((e) => e.category === 'countdown_reset');
  assert.ok(reset, 'fixture has no countdown_reset event');
  assert.ok(t.includes(reset.description.slice(0, 30)), 'countdown reset description not rendered');
  for (const cat of ['interaction_difficulty', 'decline_path_change']) {
    const ev = journey.temporal.events.find((e) => e.category === cat);
    assert.ok(ev, `fixture has no ${cat} event`);
    assert.ok(t.includes(ev.description.slice(0, 30)), `${cat} event not rendered`);
  }
});

test('single audit renders intelligence, with journey-only sections shown as unavailable (not faked)', () => {
  const t = text(renderToString(<Results result={{ response: audit, inputType: 'text', inputText: '', isMock: true }} model={model} onNewAudit={() => {}} onRunJourney={() => {}} />));
  noJunk(t);
  assert.ok(t.includes('Deception Intelligence Score (CDI)'));
  assert.ok(t.includes(audit.intelligence.fspm.reason.slice(0, 30)), 'FSPM unavailable reason not shown');
  assert.ok(t.includes('Run a journey audit'));
});

test('audit saved before intel-v1.0 (no intelligence field) still renders', () => {
  const old = { ...audit }; delete old.intelligence;
  const t = text(renderToString(<Results result={{ response: old, inputType: 'text', inputText: '', isMock: true }} model={model} onNewAudit={() => {}} />));
  noJunk(t);
  assert.ok(t.includes('does not include intelligence indices'), 'missing-intelligence note not shown');
  assert.ok(!t.includes('Findings in detail'));
});

test('benign audit shows the no-concern state', () => {
  const t = text(renderToString(<Results result={{ response: benign, inputType: 'text', inputText: '', isMock: true }} model={model} onNewAudit={() => {}} />));
  assert.ok(t.includes('No suspicious patterns detected'));
});

test('unavailable CDI components say "Not measured" (reason in tooltip), never a 0 value', () => {
  const cdi = audit.intelligence.cdi;
  const html = renderToString(<CDISection cdi={cdi} />);
  const t = text(html);
  const missing = cdi.components.filter((x) => !x.available);
  assert.ok(missing.length > 0, 'fixture has no unavailable component');
  assert.ok(t.includes('Not measured for this input'));
  for (const c of missing) assert.ok(html.includes(`title="${attr(c.explanation)}"`), `tooltip reason for ${c.key} missing`);
});

test('MCAI on pasted text marks visual asymmetry unavailable', () => {
  const m = audit.intelligence.mcai;
  const t = text(renderToString(<MCAISection mcai={m} />));
  noJunk(t);
  if (m.available) assert.ok(/unavailable/i.test(t), 'visual dimension not marked unavailable');
  else assert.ok(t.includes(m.reason.slice(0, 30)));
});


/* ------------------------------------------------------------------ new contract fields (intel: confidence, fusion, replay …)
 * The objects below are HAND-BUILT TEST INPUTS shaped like backend/intelligence/schemas.py. They are not results and are
 * never shown in the app; they only check that every new field renders, and that older responses without them still do. */

const T_CONF = {
  evidence_by_provenance: { model: 3, observed: 2 }, model_evidence: 3, mean_model_confidence: 0.812, min_model_confidence: 0.64,
  level: 'medium', note: 'Confidence describes how sure the evidence is, not how severe the deception is.',
};
const withCdiExtras = (cdi) => ({ ...cdi, confidence: T_CONF, alternatives: { geometric_mean: 31.25, max_component: 77.5 } });

test('CDI shows severity, confidence and coverage as three separate indicators', () => {
  const t = text(renderToString(<CDISection cdi={withCdiExtras(journey.cdi)} />));
  noJunk(t);
  for (const h of ['Severity · CDI', 'Confidence', 'Coverage']) assert.ok(t.includes(h), `missing tile "${h}"`);
  assert.ok(t.includes('medium'), 'confidence level missing');
  assert.ok(t.includes('81%') && t.includes('64%'), 'mean/min model confidence missing (shown like confidence_pct: floor, max 99%)');
  assert.ok(t.includes(T_CONF.note), 'confidence note missing');
  assert.ok(t.includes('Study only') && t.includes('31.3') && t.includes('77.5'), 'alternative aggregators missing');
  assert.ok(t.includes('The weighted mean is what ClearCart reports'), 'study-only caveat missing');
});

test('CDI without confidence (older response) says "Not reported" instead of inventing a value', () => {
  const cdi = { ...journey.cdi }; delete cdi.confidence; delete cdi.alternatives;
  const t = text(renderToString(<CDISection cdi={cdi} />));
  noJunk(t);
  assert.ok(t.includes('Not reported'));
  assert.ok(!t.includes('Alternative aggregators'));
});

test('inferred provenance has its own chip and tooltip', () => {
  const html = renderToString(<ProvenanceChip value="inferred" />);
  assert.ok(text(html).includes('Inferred'));
  assert.ok(html.includes('Inferred from page structure, not observed clicks'), 'tooltip missing');
  const measured = renderToString(<ProvenanceChip value="measured" />);
  assert.notEqual(html.replace(/inferred|Inferred[^"<]*/g, ''), measured.replace(/measured|Measured[^"<]*/g, ''), 'inferred styled like measured');
});

test('MCAI choice set shows dimension contributions, top dimension, sensitivity and study-only alternatives', () => {
  const mcai = {
    available: true, weights: { visual: 0.4, interaction: 0.35, semantic: 0.25 }, mcai: 62.5,
    choice_sets: [{
      id: 's2-cs1', accept: { role: 'accept', text: 'Yes, keep my plan' }, decline: { role: 'decline', text: 'No thanks' },
      dimensions: [
        { name: 'interaction', available: true, score: 0.75, provenance: 'inferred', features: [] },
        { name: 'semantic', available: true, score: 0.5, provenance: 'model', features: [] },
        { name: 'visual', available: false, score: null, provenance: 'unavailable', reason: 'No styling in pasted text.' },
      ],
      mcai: 62.5, coverage: 0.6,
      dimension_contributions: { interaction: 43.75, semantic: 18.75 }, top_dimension: 'interaction',
      alternatives: { geometric_mean: 58.1 },
      sensitivity: { one_at_a_time: [{ component: 'interaction', low: 55.0, high: 68.2 }], dirichlet_p05: 51.2, dirichlet_p95: 70.4, samples: 2000 },
    }],
  };
  const html = renderToString(<MCAISection mcai={mcai} />);
  const t = text(html);
  noJunk(t);
  assert.ok(t.includes('Contribution by dimension'), 'contribution bar missing');
  assert.ok(t.includes('43.8 pts') && t.includes('18.8 pts'), 'contribution points missing');
  assert.ok(t.includes('Largest contributor: Interaction'), 'top dimension missing');
  assert.ok(t.includes('51.2 – 70.4') && t.includes('55.0 – 68.2'), 'sensitivity ranges missing');
  assert.ok(t.includes('Study only') && t.includes('58.1'), 'alternatives missing');
  assert.ok(html.includes('data-provenance="inferred"'), 'inferred chip missing');
});

test('counterfactual shows each edit as a compact grid card (chips + clamped before/after) whose hover/focus panel carries the finding recommendation (no diff, P(dark), optimiser or candidates)', () => {
  const findings = [
    { id: 'f1', category: 'urgency', category_label: 'Misleading urgency', ethical_recommendation: 'Only show a deadline if it is real.' },
    { id: 'f2', category: 'social_proof', category_label: 'Social proof pressure', ethical_recommendation: 'Use verifiable, aggregated information.' },
    { id: 'f3', category: 'preselected_option', category_label: 'Preselected option', ethical_recommendation: 'Leave optional add-ons unchecked.' },
    { id: 'f4', category: 'scarcity', category_label: 'Misleading scarcity', ethical_recommendation: 'Show live stock only.' },
  ];
  const cand = [{ variant: 'minimal', text: 'Today: 20% off.', score: 0.12, feasible: true, violated: [], edit_distance: 2 }];
  const cf = {
    ...journey.counterfactual,
    optimisation: { method: 'exhaustive search over rule-based variants', objective: 'minimise calibrated P(dark)' },
    edits: [
      { target: 'f1', stage_index: null, category: 'urgency', action: 'rewrite', original: 'Hurry! Only today: 20% off.', redesigned: 'Today: 20% off.',
        diff: [{ op: 'delete', text: 'Hurry! Only ' }, { op: 'equal', text: 'today: 20% off.' }], original_score: 0.91, redesigned_score: 0.12,
        facts_preserved: true, explanation: 'Optimiser selected the minimal variant.', candidates: cand },
      { target: 'f2', stage_index: null, category: 'social_proof', action: 'remove_unverifiable_claim', original: '1,203 people bought this today',
        redesigned: null, diff: [], original_score: 0.88, redesigned_score: 0.0, facts_preserved: true, explanation: 'y', candidates: [] },
      { target: 'f3', stage_index: null, category: 'preselected_option', action: 'uncheck_default', original: 'Add protection for $4.99',
        redesigned: '[unchecked by default] Add protection for $4.99', diff: [], facts_preserved: true },
      { target: 'f4', stage_index: null, category: 'scarcity', action: 'no_safe_rewrite', original: 'Only 5 left', redesigned: null, diff: [] },
      { target: 'cs1', stage_index: null, category: 'choice_asymmetry', action: 'equalise_choice', original: 'No, I hate saving', redesigned: 'No thanks', diff: [] },
    ],
  };
  const findingFor = (e) => findings.find((f) => f.id === e.target) || null;
  const html = renderToString(<CounterfactualSection cf={cf} findingFor={findingFor} />);
  const t = text(html);
  noJunk(t);
  for (const f of findings) {
    assert.ok(t.includes(`Ethical redesign: ${f.category_label}`), `Sparkles title for ${f.id} missing`);
    assert.ok(t.includes(f.ethical_recommendation), `recommendation for ${f.id} missing`);
    assert.ok(html.includes(`id="redesign-${f.id}"`), `card id for ${f.id} missing`);
  }
  assert.ok(t.includes('Give accept and decline equal visual weight and neutral wording.'), 'generic choice-set line missing');
  assert.equal((html.match(/lucide-sparkles/g) || []).length, 10, 'Sparkles icon missing from a card or its reason panel');
  assert.ok(/<ul class="[^"]*grid[^"]*md:grid-cols-2[^"]*xl:grid-cols-3/.test(html), 'cards are not laid out as a responsive grid');
  assert.equal((html.match(/data-redesign-card=""/g) || []).length, 5, 'one card per edit');
  for (const a of ['Rewrite', 'Remove claim', 'Uncheck default', 'No safe rewrite', 'Equal choice']) assert.ok(t.includes(a), `action chip "${a}" missing`);
  // at rest: clamped BEFORE (struck) and AFTER; the hover/focus panel repeats them un-clamped
  assert.equal((html.match(/line-through[^"]*line-clamp-3/g) || []).length, 5, 'every resting BEFORE is struck through and clamped');
  assert.ok(!t.includes('Why this change'), '"Why this change" hint should be gone');
  assert.equal((html.match(/line-through/g) || []).length, 10, 'BEFORE missing from resting card or reason panel');
  // the reason panel is wired for keyboard and screen readers
  assert.equal((html.match(/role="button" tabindex="0" aria-expanded="false" aria-describedby="[^"]+"/g) || []).length, 5, 'cards are not focusable toggles');
  for (const m of html.matchAll(/aria-describedby="([^"]+)"/g)) assert.ok(html.includes(`id="${m[1]}"`), 'aria-describedby target missing');
  assert.equal((html.match(/data-testid="redesign-reason"/g) || []).length, 5, 'reason panel missing');
  assert.ok(t.includes('Hurry! Only today: 20% off.') && t.includes('Today: 20% off.'), 'rewrite before/after missing');
  assert.ok(t.includes('(remove this claim)'), 'removal AFTER missing');
  assert.ok(t.includes('Add protection for $4.99 (unchecked by default)') && !t.includes('[unchecked by default]'), 'uncheck AFTER wrong');
  assert.ok(t.includes('No safe automatic rewrite — follow the recommendation'), 'no-rewrite AFTER missing');
  for (const gone of ['Candidates considered', 'How rewrites are chosen', 'P(dark pattern)', 'Optimiser selected', 'exhaustive search', '<del', '<ins']) {
    assert.ok(!t.includes(gone) && !html.includes(gone), `"${gone}" should no longer be shown`);
  }
  // one-line summary from the API values, caveat in the tooltip
  assert.ok(cfLine(cf).test(t), 'summary line not the API values');
  assert.ok(html.includes(`title="${attr(cf.label)}"`), 'label tooltip missing');
});

test('counterfactual summary leaves out values the API did not provide (never 0 or NaN)', () => {
  const cf = { edits: [{ target: 'f9', category: 'urgency', action: 'rewrite', original: 'a', redesigned: 'b' }], original_cdi: null, redesigned_cdi: null, risk_reduction_pct: null };
  const t = text(renderToString(<CounterfactualSection cf={cf} />));
  noJunk(t);
  assert.ok(!t.includes('CDI') && !t.includes('% lower'), 'unavailable summary values rendered');
});

test('timeline shows Previous → Current, signed Δ pressure in pp, and the metrics grid', () => {
  const temporal = {
    available: true, ordering: 'submission_order', pressure_progression: [0.2, 0.55], progression_stage_indices: [0, 1],
    metrics: { urgency_persistence: 0.667, repeated_categories: ['urgency', 'scarcity'], max_pressure_increase: 0.35, stages_with_pressure: 2 },
    events: [{
      stage_index: 1, stage_name: 'Checkout', timestamp: null, kind: 'observed_fact', category: 'countdown_reset',
      description: 'Countdown restarted.', evidence: [{ stage_index: 1, finding_id: 'f1', evidence_id: 's2:f1', text: '10:00' }],
      pressure: 0.55, deceptive_signal: true, previous_state: '04:12 remaining', current_state: '10:00 remaining', pressure_delta: 0.35,
    }, {
      stage_index: 0, stage_name: 'Product', timestamp: null, kind: 'observation', category: 'x', description: 'Calm.', evidence: [],
      pressure: 0.2, pressure_delta: -0.125,
    }],
  };
  const t = text(renderToString(<TimelineSection temporal={temporal} />));
  noJunk(t);
  assert.ok(t.includes('Previous 04:12 remaining') && t.includes('Current 10:00 remaining'), 'previous → current missing');
  assert.ok(t.includes('+35.0 pp') && t.includes('−12.5 pp'), 'signed Δ pressure missing');
  assert.ok(t.includes('Journey metrics') && t.includes('Urgency persistence') && t.includes('66.7%'), 'metrics grid missing');
  assert.ok(t.includes('urgency, scarcity') && t.includes('Stages with pressure 2'), 'list / count metrics wrong');
  assert.ok(t.includes('s2:f1'), 'evidence id not shown');
});

test('evidence chips show evidence_id, and fall back to finding_id on older responses', () => {
  assert.ok(text(renderToString(<EvidenceRefs refs={[{ finding_id: 'f3', evidence_id: 'f3', text: 'Only 2 left' }]} />)).includes('f3'));
  assert.ok(text(renderToString(<EvidenceRefs refs={[{ finding_id: 'f4', text: 'Only 2 left' }]} />)).includes('f4'));
});

const T_FUSION = [{
  stage_indices: [0, 1], representative: 0, fused_pressure: 0.42,
  duplicate_pairs: [{ a: { stage_index: 0, finding_id: 'f1', evidence_id: 's1:f1', text: 'Only 2 left!' },
    b: { stage_index: 1, finding_id: 'f1', evidence_id: 's2:f1', text: 'Only 2 left!' }, similarity: 0.97 }],
  note: 'Findings were merged across modalities.',
}];

test('fusion card lists merged stages, duplicate pairs with similarity, fused pressure and note', () => {
  const names = (i) => `${i + 1}. Stage`;
  const t = text(renderToString(<FusionCard fusion={T_FUSION} stageName={names} />));
  noJunk(t);
  assert.ok(t.includes('Merged same-page stages') && t.includes('1. Stage + 2. Stage'));
  assert.ok(t.includes('0.97') && t.includes('s1:f1') && t.includes('s2:f1'), 'duplicate pair missing');
  assert.ok(t.includes('42.0%') && t.includes('Findings were merged across modalities.'));
  assert.equal(renderToString(<FusionCard fusion={[]} stageName={names} />), '', 'empty fusion should render nothing');
});

test('journey results with fusion, fused_into, versions and replay render; screenshot replay explains missing files', () => {
  const stages = journey.stages.map((st, i) => (i === 1 ? { ...st, fused_into: 0 } : st));
  const response = {
    ...journey, stages, fusion: T_FUSION,
    versions: { model: 'clf-test-1', calibration: 'cal-test-1', config: 'intel-test' },
    replay: {
      inputs: journey.stages.map((st, i) => (i === 1 ? { name: st.name, input_type: 'screenshot', file_field: 'stage_1_file' }
        : { name: st.name, input_type: st.input_type, content: st.content })),
      input_sha256: journey.stages.map((_, i) => `hash${i}`), versions: { model: 'clf-test-1' }, expected: { cdi: 41.5 },
    },
  };
  const t = text(renderToString(<JourneyResults result={{ response }} model={model} onNewJourney={() => {}} />));
  noJunk(t);
  assert.ok(t.includes('merged into stage 1'), 'fused_into badge missing');
  assert.ok(t.includes('Merged same-page stages'), 'fusion card missing');
  assert.ok(t.includes('Reproducibility and replay') && t.includes('clf-test-1') && t.includes('intel-test'), 'versions block missing');
  assert.ok(t.includes('Verify by replay'), 'replay button missing');
  assert.ok(t.includes('Screenshot stages cannot be replayed without the original files'), 'missing-file explanation missing');
  // with the original File available the explanation disappears
  const t2 = text(renderToString(<JourneyResults result={{ response, files: { stage_1_file: { name: 'x.png' } } }} model={model} onNewJourney={() => {}} />));
  assert.ok(!t2.includes('cannot be replayed'));
});

test('replay banner: identical vs differences', () => {
  assert.ok(text(renderToString(<ReplayBanner outcome={{ identical: true, differences: [], journey: { journey_id: 'j1' } }} />)).includes('Replay reproduced identical results'));
  const t = text(renderToString(<ReplayBanner outcome={{ identical: false, differences: ['cdi: 41.5 != 41.6'], journey: {} }} />));
  assert.ok(t.includes('did not reproduce') && t.includes('cdi: 41.5 != 41.6'));
});

test('single audit shows a Reproducibility block only when versions are present', () => {
  const withV = { ...audit, versions: { model: 'clf-test-1', calibration: 'cal-test-1' } };
  const t = text(renderToString(<Results result={{ response: withV, inputType: 'text', inputText: '', isMock: true }} model={model} onNewAudit={() => {}} />));
  assert.ok(t.includes('Reproducibility') && t.includes('cal-test-1'));
  const old = { ...audit }; delete old.versions;
  assert.ok(!text(renderToString(<Results result={{ response: old, inputType: 'text', inputText: '', isMock: true }} model={model} onNewAudit={() => {}} />)).includes('Reproducibility'));
  assert.equal(renderToString(<ReproducibilityBlock versions={null} />), '');
});

test('journey builder: fuse_with is sent as an earlier 0-based index and must reference an earlier stage', () => {
  const a = newStage({ name: 'Product (text)', content: 'Only 2 left' });
  const b = newStage({ name: 'Product (HTML)', inputType: 'html', content: '<p>Only 2 left</p>' });
  b.fuseWith = a.uid;
  const { stages: payload } = toRequest([a, b]);
  assert.equal(payload[1].fuse_with, 0);
  assert.ok(!('fuse_with' in payload[0]));
  assert.equal(fuseProblem(b, [a, b]), null);
  assert.ok(/earlier stage/.test(fuseProblem(b, [b, a])), 'a later stage was accepted');
  assert.ok(validateStage(b, [b, a]).fuseWith, 'validation did not flag a non-earlier reference');
  assert.ok(!('fuse_with' in toRequest([b, a]).stages[0]), 'invalid reference was sent');
});


test('real responses: v1.1 fields render when present (and are optional when absent)', () => {
  const t = text(renderToString(<JourneyResults result={{ response: journey }} model={model} onNewJourney={() => {}} />));
  noJunk(t);
  const conf = journey.cdi.confidence;
  if (conf) assert.ok(t.includes(conf.note) && t.includes(conf.level), 'live confidence not shown');
  if (journey.versions) for (const v of Object.values(journey.versions)) assert.ok(t.includes(v), `version ${v} not shown`);
  if (journey.replay) assert.ok(t.includes('Verify by replay'));
  // redesign edits are shown as simple before/after cards: no candidates table or optimiser box
  assert.ok(!t.includes('Candidates considered') && !t.includes('How rewrites are chosen'), 'optimiser detail still shown');
  for (const e of journey.counterfactual.edits) assert.ok(t.includes(e.original), `edit original "${e.original}" not shown`);
  const change = journey.temporal.events.find((e) => e.previous_state && e.current_state);
  if (change) assert.ok(t.includes(change.previous_state) && t.includes(change.current_state), 'live previous → current missing');
  if (journey.temporal.metrics) assert.ok(t.includes('Journey metrics'));
  const aHtml = renderToString(<Results result={{ response: audit, inputType: 'text', inputText: '', isMock: true }} model={model} onNewAudit={() => {}} />);
  const a = text(aHtml);
  noJunk(a);
  assert.ok(!a.includes('Findings in detail'), 'Findings in detail still rendered');
  for (const e of audit.intelligence.counterfactual.edits) {
    assert.ok(aHtml.includes(attr(e.original)), `audit edit ${e.target} not shown`);
    assert.ok(aHtml.includes(`id="redesign-${e.target}"`), `redesign card id for ${e.target} missing`);
  }
  if (audit.versions) assert.ok(a.includes('Reproducibility') && a.includes(Object.values(audit.versions)[0]));
});

test('evidence tiers: observed facts, pressure claims with markers, and held-for-review promotions', () => {
  const claim = { verdict: 'vague', checkable: [], vague: ['hurry: Hurry', 'stock count: Only 2'], note: 'Pressure without checkable terms.' };
  const sale = { verdict: 'checkable', checkable: ['named-day deadline: ends Sunday at 23:59'], vague: [], note: '' };
  const response = {
    ...benign,
    findings: [
      { id: 'f1', category: 'scarcity', category_label: 'Misleading scarcity', evidence: 'Hurry! Only 2 left in stock.', detection_method: 'ml_classifier',
        model_score: 0.99, evidence_tier: 'pressure_claim', claim_check: claim, source_location: { source_type: 'text' } },
      { id: 'f2', category: 'preselected_option', category_label: 'Preselected option', evidence: 'Add protection for $4.99', detection_method: 'dom_observation',
        model_score: null, evidence_tier: 'observed', short_reason: 'Pre-ticked box', source_location: { source_type: 'html' } },
    ],
    uncertain_elements: [{ text: 'Winter sale ends Sunday at 23:59', p_dark: 0.994, flagged: false, gate: 'checkable_claim',
      gate_note: 'checkable promotion — terms not verified', claim_check: sale, source_location: { source_type: 'text' } }],
  };
  const t = text(renderToString(<EvidenceTiers response={response} />));
  noJunk(t);
  for (const s of ['Evidence tiers', 'Observed facts (1)', 'Pressure claims (1)', 'Held for review (1)', 'Add protection for $4.99',
    'vague pressure', 'hurry', 'stock count', 'Checkable promotion', 'named-day deadline', 'Winter sale ends Sunday at 23:59']) {
    assert.ok(t.includes(s), `missing "${s}"`);
  }
  // older responses without tier fields still split DOM observations from model claims
  const old = { ...response, findings: response.findings.map(({ evidence_tier, claim_check, ...f }) => f), uncertain_elements: [] };
  const o = text(renderToString(<EvidenceTiers response={old} />));
  assert.ok(o.includes('Observed facts (1)') && o.includes('Pressure claims (1)') && o.includes('Nothing withheld.'));
  assert.equal(renderToString(<EvidenceTiers response={{ ...benign, findings: [], uncertain_elements: [] }} />), '');
});

test('visual signals (experimental): listed with %, threshold and box; empty message for []; absent when undefined', () => {
  const caveat = 'Trained on screenshots that all contain dark patterns; indicates pattern type, not whether a page is deceptive.';
  const signals = [
    { label: 'Hard to Close', probability: 0.8731, threshold: 0.42, bbox: { x: 10, y: 20, width: 30, height: 40 }, provenance: 'model (visual, experimental)' },
    { label: 'Nagging', probability: 0.5, threshold: 0.35, bbox: null, provenance: 'model (visual, experimental)' },
  ];
  const t = text(renderToString(<Results result={{ response: { ...audit, visual_signals: signals } }} model={model} onNewAudit={() => {}} />));
  noJunk(t);
  for (const s of ['Visual signals (experimental)', caveat, 'Hard to Close', '87%', 'threshold 42%', 'x 10, y 20, 30×40 px',
    'Nagging', '50%', 'threshold 35%', 'no evidence region', '2 above threshold']) {
    assert.ok(t.includes(s), `missing "${s}"`);
  }
  const e = text(renderToString(<Results result={{ response: { ...audit, visual_signals: [] } }} model={model} onNewAudit={() => {}} />));
  assert.ok(e.includes('Visual signals (experimental)') && e.includes('Nothing above the threshold') && e.includes(caveat));
  const { visual_signals, ...noVisual } = audit;
  const n = text(renderToString(<Results result={{ response: noVisual }} model={model} onNewAudit={() => {}} />));
  assert.ok(!n.includes('Visual signals') && !n.includes(caveat), 'visual section shown without visual_signals');
});

const failed = results.filter((r) => !r[1]);
for (const [name, ok, e] of results) console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${ok ? '' : `\n     ${e.message}`}`);
console.log(`\n${results.length - failed.length} passed, ${failed.length} failed`);
process.exit(failed.length ? 1 : 0);
