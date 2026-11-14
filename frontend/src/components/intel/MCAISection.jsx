import { ChevronDown, Scale } from 'lucide-react';
import {
  AlternativesTable, AxisTicks, Formula, IntelCard, MethodNote, ProvenanceChip, UnavailableNote, WeightSensitivityChart, fmtIndex,
  fmtPoints, fmtShare, fmtValue, humanFeature, humanKeyLabel, isNum,
} from './shared.jsx';

const DIM_LABEL = { visual: 'Visual', interaction: 'Interaction', semantic: 'Semantic' };
const DIM_ORDER = ['visual', 'interaction', 'semantic'];
const DIM_HEX = { visual: '#0f766e', interaction: '#0369a1', semantic: '#7c3aed' };
const EXTRA_HEX = ['#b45309', '#475569', '#be123c'];
const dimLabel = (k) => DIM_LABEL[k] || humanKeyLabel(k);
const clamp100 = (v) => Math.max(0, Math.min(100, v));

/** Points of the 0..100 MCAI contributed by each available dimension (ChoiceSetResult.dimension_contributions). */
function DimensionContributionBar({ contributions, top, mcai }) {
  const keys = Object.keys(contributions || {}).filter((k) => isNum(contributions[k]));
  if (!keys.length) return null;
  const ordered = [...DIM_ORDER.filter((k) => keys.includes(k)), ...keys.filter((k) => !DIM_ORDER.includes(k))];
  let extra = 0;
  const hexOf = Object.fromEntries(ordered.map((k) => [k, DIM_HEX[k] || EXTRA_HEX[extra++ % EXTRA_HEX.length]]));
  const summary = ordered.map((k) => `${dimLabel(k)} ${fmtPoints(contributions[k])}`).join(', ');
  return (
    <div className="mt-3" data-testid="mcai-contributions">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h5 className="text-sm font-semibold text-slate-900">Contribution by dimension (points of 100)</h5>
        {top && (
          <p className="text-xs text-slate-600">
            Largest contributor: <b className="text-slate-900">{dimLabel(top)}</b>
          </p>
        )}
      </div>
      <div
        role="img"
        aria-label={`MCAI ${fmtIndex(mcai)} of 100, made up of: ${summary}.`}
        className="mt-2 flex h-4 w-full overflow-hidden rounded-md bg-slate-100 ring-1 ring-inset ring-slate-200"
      >
        {ordered.map((k) => (
          <div
            key={k}
            title={`${dimLabel(k)}: ${fmtPoints(contributions[k])}`}
            className="h-full border-r border-white/70 last:border-r-0"
            style={{ width: `${clamp100(contributions[k])}%`, background: hexOf[k] }}
          />
        ))}
      </div>
      <AxisTicks />
      <ul className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-700" aria-hidden="true">
        {ordered.map((k) => (
          <li key={k} className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm" style={{ background: hexOf[k] }} />
            {dimLabel(k)} <span className="tabular-nums text-slate-900">{fmtPoints(contributions[k])}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function bboxText(b) {
  if (!b || typeof b !== 'object') return null;
  const { x, y, width, height } = b;
  if ([x, y, width, height].every(isNum)) return `box at ${x},${y} · ${width}×${height} px`;
  return null;
}

function OptionBox({ role, option }) {
  const accept = role === 'accept';
  return (
    <div className={`min-w-0 rounded-lg p-3 ring-1 ring-inset ${accept ? 'bg-teal-50/50 ring-teal-200' : 'bg-slate-50 ring-slate-200'}`}>
      <p className={`text-[11px] font-bold uppercase tracking-wider ${accept ? 'text-teal-800' : 'text-slate-600'}`}>
        {accept ? 'Accept option' : 'Decline option'}
      </p>
      {option ? (
        <>
          <p className="mt-1 break-words text-sm font-semibold text-slate-900">“{option.text}”</p>
          <div className="mt-1 space-y-0.5 text-[11px] text-slate-500">
            {option.css_selector && <p className="break-all font-mono">{option.css_selector}</p>}
            {bboxText(option.bbox) && <p>OCR {bboxText(option.bbox)}</p>}
            {option.finding_id && <p>Linked finding <span className="font-mono">{option.finding_id}</span></p>}
          </div>
        </>
      ) : (
        <p className="mt-1 text-sm text-slate-500">No competing option was found.</p>
      )}
    </div>
  );
}

function DimensionBar({ dim, weight }) {
  const label = DIM_LABEL[dim.name] || dim.name;
  if (!dim.available || !isNum(dim.score)) {
    return (
      <li className="grid grid-cols-[6.5rem_minmax(0,1fr)] items-start gap-2 sm:grid-cols-[7.5rem_minmax(0,1fr)]">
        <span className="text-sm font-medium text-slate-400">{label}</span>
        <span className="text-xs text-slate-500">
          <ProvenanceChip value="unavailable" /> <span className="ml-1">{dim.reason || 'No data for this dimension.'}</span>
        </span>
      </li>
    );
  }
  const v = Math.max(0, Math.min(1, dim.score)) * 100;
  return (
    <li className="grid grid-cols-[6.5rem_minmax(0,1fr)] items-start gap-2 sm:grid-cols-[7.5rem_minmax(0,1fr)]">
      <span className="text-sm font-medium text-slate-800">
        {label}
        {isNum(weight) && <span className="block text-[11px] font-normal text-slate-500">weight {String(Number(weight.toFixed(3)))}</span>}
      </span>
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <div className="h-2 min-w-0 flex-1 overflow-hidden rounded-full bg-slate-100" aria-hidden="true">
            <div className="h-full rounded-full bg-teal-700" style={{ width: `${v}%` }} />
          </div>
          <span className="w-12 shrink-0 text-right text-xs font-semibold tabular-nums text-slate-800">{fmtShare(dim.score, 0)}</span>
          <ProvenanceChip value={dim.provenance} />
        </div>
        {dim.reason && <p className="mt-1 text-[11px] text-slate-500">{dim.reason}</p>}
      </div>
    </li>
  );
}

function FeatureTable({ dims }) {
  const rows = [];
  for (const name of DIM_ORDER) {
    const d = dims.find((x) => x.name === name);
    for (const f of d?.features || []) rows.push({ dim: name, f });
  }
  for (const d of dims) if (!DIM_ORDER.includes(d.name)) for (const f of d.features || []) rows.push({ dim: d.name, f });
  if (!rows.length) return null;
  return (
    <details className="group mt-3 rounded-lg border border-slate-200">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 rounded-lg px-3 py-2 text-sm font-medium text-slate-800 hover:bg-slate-50 [&::-webkit-details-marker]:hidden">
        Feature comparisons ({rows.length})
        <ChevronDown className="h-4 w-4 text-slate-500 transition-transform group-open:rotate-180" aria-hidden="true" />
      </summary>
      <div className="overflow-x-auto border-t border-slate-200">
        <table className="w-full min-w-[640px] text-left text-xs">
          <thead className="bg-slate-50 text-[11px] uppercase tracking-wide text-slate-500">
            <tr>
              <th scope="col" className="px-3 py-2 font-semibold">Dimension</th>
              <th scope="col" className="px-3 py-2 font-semibold">Feature</th>
              <th scope="col" className="px-3 py-2 font-semibold">Accept</th>
              <th scope="col" className="px-3 py-2 font-semibold">Decline</th>
              <th scope="col" className="px-3 py-2 text-right font-semibold">Asymmetry</th>
              <th scope="col" className="px-3 py-2 font-semibold">Note</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {rows.map(({ dim, f }, i) => (
              <tr key={`${dim}-${f.feature}-${i}`} className="align-top">
                <td className="px-3 py-2 text-slate-600">{DIM_LABEL[dim] || dim}</td>
                <td className="px-3 py-2 font-medium text-slate-800">{humanFeature(f.feature)}</td>
                <td className="px-3 py-2">
                  <span className="mr-1.5 tabular-nums text-slate-900">{fmtValue(f.accept_value)}</span>
                  <ProvenanceChip value={f.accept_provenance} />
                </td>
                <td className="px-3 py-2">
                  <span className="mr-1.5 tabular-nums text-slate-900">{fmtValue(f.decline_value)}</span>
                  <ProvenanceChip value={f.decline_provenance} />
                </td>
                <td className="px-3 py-2 text-right font-semibold tabular-nums text-slate-900">{isNum(f.asymmetry) ? fmtShare(f.asymmetry, 0) : '—'}</td>
                <td className="px-3 py-2 text-slate-600">{f.note || ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

function ChoiceSet({ cs, weights, label }) {
  const dims = cs.dimensions || [];
  const ordered = [...DIM_ORDER.map((n) => dims.find((d) => d.name === n)).filter(Boolean), ...dims.filter((d) => !DIM_ORDER.includes(d.name))];
  return (
    <li className="rounded-lg border border-slate-200 p-3 sm:p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-xs font-semibold text-slate-500">
          {label} <span className="font-mono font-normal">{cs.id}</span>
        </p>
        <p className="text-sm text-slate-700">
          MCAI <b className="text-lg tabular-nums text-slate-900">{isNum(cs.mcai) ? fmtIndex(cs.mcai) : '—'}</b>
          {isNum(cs.mcai) && <span className="text-slate-500"> /100</span>}
          <span className="ml-2 text-xs text-slate-500">coverage {fmtShare(cs.coverage, 0)}</span>
        </p>
      </div>
      <div className="mt-2 grid gap-2 sm:grid-cols-2">
        <OptionBox role="accept" option={cs.accept} />
        <OptionBox role="decline" option={cs.decline} />
      </div>
      <ul className="mt-3 space-y-2.5" aria-label="Asymmetry by dimension">
        {ordered.map((d) => <DimensionBar key={d.name} dim={d} weight={weights?.[d.name]} />)}
      </ul>
      <DimensionContributionBar contributions={cs.dimension_contributions} top={cs.top_dimension} mcai={cs.mcai} />
      {Object.keys(cs.dimension_contributions || {}).length === 0 && cs.top_dimension && (
        <p className="mt-3 text-xs text-slate-600">Largest contributor: <b className="text-slate-900">{dimLabel(cs.top_dimension)}</b></p>
      )}
      <WeightSensitivityChart s={cs.sensitivity} labelOf={dimLabel} reported={cs.mcai} indexName="MCAI" compact />
      <AlternativesTable alternatives={cs.alternatives} reported={cs.mcai} indexName="MCAI" compact />
      <FeatureTable dims={dims} />
      {cs.role_detection && <p className="mt-2 text-[11px] text-slate-500">Accept/decline roles: {cs.role_detection}.</p>}
    </li>
  );
}

/** Choice asymmetry section. `mcai` is an MCAIResult. */
export default function MCAISection({ mcai, headingId = 'mcai-heading' }) {
  if (!mcai) return null;
  const sets = mcai.choice_sets || [];
  return (
    <IntelCard
      id={headingId}
      icon={Scale}
      title="Choice Asymmetry (MCAI)"
      subtitle="Compares the accept and decline options of a choice: how they look, how many steps they take, and how they are worded."
      aside={
        mcai.available && isNum(mcai.mcai) ? (
          <div className="shrink-0 rounded-lg bg-slate-50 px-3 py-2 text-right ring-1 ring-inset ring-slate-200">
            <p className="text-[11px] font-bold uppercase tracking-wider text-slate-500">Worst case</p>
            <p className="tabular-nums text-slate-900"><span className="text-2xl font-bold">{fmtIndex(mcai.mcai)}</span><span className="text-sm text-slate-500"> /100</span></p>
          </div>
        ) : null
      }
    >
      {!mcai.available || sets.length === 0 ? (
        <UnavailableNote reason={mcai.reason || 'No accept/decline choice was found in this input.'} />
      ) : (
        <>
          {mcai.reason && <p className="mb-3 text-xs text-slate-600">{mcai.reason}</p>}
          <ul className="space-y-3">
            {sets.map((cs, i) => <ChoiceSet key={cs.id || i} cs={cs} weights={mcai.weights} label={`Choice ${i + 1}`} />)}
          </ul>
        </>
      )}
      <div className="mt-4 space-y-1 border-t border-slate-100 pt-3">
        <MethodNote>
          Provenance: <i>measured</i> values come from the submitted page, <i>declared</i> values from the person running the audit,
          and <i>inferred</i> values from page structure (not observed clicks).
        </MethodNote>
        <MethodNote>
          Unavailable dimensions are left out and the remaining weights renormalised; coverage is the share of weight that had data.
        </MethodNote>
        {mcai.formula && <MethodNote><Formula>{mcai.formula}</Formula></MethodNote>}
      </div>
    </IntelCard>
  );
}
