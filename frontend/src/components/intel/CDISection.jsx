import { Gauge, Layers, ShieldCheck, Target } from 'lucide-react';
import { pct } from '../../format.js';
import {
  AlternativesTable, AxisTicks, EvidenceRefs, Formula, IntelCard, MethodNote, ProvenanceChip, WeightSensitivityChart, fmtIndex,
  fmtPoints, fmtShare, isNum,
} from './shared.jsx';

// Fixed palette for component segments, assigned in API order.
const SEGMENT_HEX = ['#0f766e', '#0369a1', '#7c3aed', '#b45309', '#475569', '#be123c'];

const BAND_UI = {
  low: 'bg-slate-100 text-slate-800 ring-slate-300',
  moderate: 'bg-amber-50 text-amber-900 ring-amber-300',
  elevated: 'bg-orange-50 text-orange-900 ring-orange-300',
  high: 'bg-red-50 text-red-900 ring-red-300',
};

const fmtWeight = (v) => (isNum(v) ? String(Number(v.toFixed(3))) : '—');
const listJoin = (xs) => (xs.length <= 1 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`);
const clamp100 = (v) => Math.max(0, Math.min(100, v));

function ContributionBar({ components, cdi }) {
  const avail = components.map((c, i) => ({ ...c, hex: SEGMENT_HEX[i % SEGMENT_HEX.length] })).filter((c) => c.available && isNum(c.contribution));
  const summary = avail.map((c) => `${c.label} ${fmtPoints(c.contribution)}`).join(', ');
  return (
    <div>
      <div
        role="img"
        aria-label={`CDI ${fmtIndex(cdi)} of 100, made up of: ${summary || 'no available components'}.`}
        className="flex h-5 w-full overflow-hidden rounded-md bg-slate-100 ring-1 ring-inset ring-slate-200"
      >
        {avail.map((c) => (
          <div
            key={c.key}
            title={`${c.label}: ${fmtPoints(c.contribution)}`}
            className="h-full border-r border-white/70 last:border-r-0"
            style={{ width: `${clamp100(c.contribution)}%`, background: c.hex }}
          />
        ))}
      </div>
      <AxisTicks />
      <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-700" aria-hidden="true">
        {components.map((c, i) => (
          <li key={c.key} className={`flex items-center gap-1.5 ${c.available ? '' : 'text-slate-400'}`}>
            <span
              className="h-2.5 w-2.5 rounded-sm"
              style={c.available ? { background: SEGMENT_HEX[i % SEGMENT_HEX.length] } : { border: '1px dashed #94a3b8' }}
            />
            {c.label}
            {!c.available && ' (unavailable)'}
          </li>
        ))}
      </ul>
    </div>
  );
}

function ComponentRow({ c, hex, stageName }) {
  if (!c.available) {
    return (
      <li
        className="flex flex-wrap items-baseline justify-between gap-2 rounded-lg bg-slate-50 px-3 py-2"
        title={c.explanation || undefined}
      >
        <span className="text-sm text-slate-500">{c.label}</span>
        <span className="text-xs text-slate-400">Not measured for this input</span>
      </li>
    );
  }
  const maxPts = isNum(c.effective_weight) ? c.effective_weight * 100 : null;
  return (
    <li className="rounded-lg border border-slate-200 p-3">
      <div className="flex flex-wrap items-start justify-between gap-x-3 gap-y-1">
        <div className="flex min-w-0 items-center gap-2">
          <span className="h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: hex }} aria-hidden="true" />
          <p className="min-w-0 text-sm font-semibold text-slate-900">{c.label}</p>
          <ProvenanceChip value={c.provenance} />
        </div>
        <p className="shrink-0 text-sm font-bold tabular-nums text-slate-900">{fmtPoints(c.contribution)}</p>
      </div>
      <dl className="mt-1.5 flex flex-wrap gap-x-4 gap-y-0.5 text-xs text-slate-600">
        <div className="flex gap-1"><dt>Value</dt><dd className="font-semibold tabular-nums text-slate-800">{fmtShare(c.value)}</dd></div>
        <div className="flex gap-1">
          <dt>Weight</dt>
          <dd className="tabular-nums text-slate-800">
            {fmtWeight(c.weight)} <span aria-hidden="true">→</span><span className="sr-only">renormalised to</span> {fmtWeight(c.effective_weight)} effective
          </dd>
        </div>
      </dl>
      <div className="mt-2">
        <div
          className="relative h-2 w-full overflow-hidden rounded-full bg-slate-100"
          title={`${fmtPoints(c.contribution)} of a possible ${maxPts != null ? fmtPoints(maxPts) : '—'} (effective weight × 100)`}
          aria-hidden="true"
        >
          {maxPts != null && <div className="absolute inset-y-0 left-0 rounded-full bg-slate-200" style={{ width: `${clamp100(maxPts)}%` }} />}
          {isNum(c.contribution) && <div className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${clamp100(c.contribution)}%`, background: hex }} />}
        </div>
        {maxPts != null && (
          <p className="mt-1 text-[11px] text-slate-500">
            Contribution on the 0–100 scale; the lighter track is the most this component could add ({fmtPoints(maxPts)}).
          </p>
        )}
      </div>
      {c.explanation && <p className="mt-2 text-xs leading-relaxed text-slate-700">{c.explanation}</p>}
      <EvidenceRefs refs={c.evidence} stageName={stageName} />
    </li>
  );
}


const LEVEL_UI = {
  high: 'bg-teal-50 text-teal-900 ring-teal-300',
  medium: 'bg-sky-50 text-sky-900 ring-sky-300',
  low: 'bg-amber-50 text-amber-900 ring-amber-300',
  none: 'bg-slate-100 text-slate-700 ring-slate-300',
};

function TileHead({ icon: Icon, children }) {
  return (
    <p className="flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider text-slate-500">
      <Icon className="h-3.5 w-3.5 text-slate-400" aria-hidden="true" />
      {children}
    </p>
  );
}

function SeverityTile({ cdi, band }) {
  return (
    <div className="rounded-lg bg-slate-50 p-4 ring-1 ring-inset ring-slate-200" data-testid="cdi-severity">
      <TileHead icon={Gauge}>Severity · CDI</TileHead>
      {isNum(cdi.cdi) ? (
        <p className="mt-1 tabular-nums text-slate-900">
          <span className="text-4xl font-bold">{fmtIndex(cdi.cdi)}</span>
          <span className="text-lg font-medium text-slate-500"> /100</span>
        </p>
      ) : (
        <p className="mt-1 text-lg font-semibold text-slate-500">Not computed</p>
      )}
      <div className="mt-2 flex flex-wrap items-center gap-2">
        {band && (
          <span className={`rounded-full px-2.5 py-0.5 text-xs font-bold uppercase tracking-wide ring-1 ring-inset ${BAND_UI[band] || BAND_UI.low}`}>
            {band} band
          </span>
        )}
      </div>
      <p className="mt-2 text-xs font-medium text-slate-600">Screening index, not a probability of fraud</p>
      {!isNum(cdi.cdi) && <p className="mt-1 text-xs text-slate-600">No component had data for this input.</p>}
    </div>
  );
}

function ConfidenceTile({ confidence: c }) {
  if (!c) {
    return (
      <div className="rounded-lg p-4 ring-1 ring-inset ring-slate-200" data-testid="cdi-confidence">
        <TileHead icon={ShieldCheck}>Confidence</TileHead>
        <p className="mt-1 text-lg font-semibold text-slate-500">Not reported</p>
        <p className="mt-1 text-xs text-slate-600">This result was produced by a server version that did not report evidence confidence.</p>
      </div>
    );
  }
  const level = c.level ? String(c.level).toLowerCase() : null;
  const mix = Object.entries(c.evidence_by_provenance || {}).filter(([, n]) => isNum(n) && n > 0);
  return (
    <div className="rounded-lg p-4 ring-1 ring-inset ring-slate-200" data-testid="cdi-confidence">
      <TileHead icon={ShieldCheck}>Confidence</TileHead>
      <p className="mt-1.5">
        {level ? (
          <span className={`inline-flex rounded-full px-2.5 py-0.5 text-sm font-bold uppercase tracking-wide ring-1 ring-inset ${LEVEL_UI[level] || LEVEL_UI.none}`}>
            {level}
          </span>
        ) : (
          <span className="text-lg font-semibold text-slate-500">—</span>
        )}
      </p>
      <dl className="mt-2 grid grid-cols-2 gap-2 text-xs">
        <div>
          <dt className="text-slate-500">Mean model confidence</dt>
          <dd className="text-base font-bold tabular-nums text-slate-900">{pct(c.mean_model_confidence)}</dd>
        </div>
        <div>
          <dt className="text-slate-500">Lowest</dt>
          <dd className="text-base font-bold tabular-nums text-slate-900">{pct(c.min_model_confidence)}</dd>
        </div>
      </dl>
      <p className="mt-1 text-[11px] text-slate-500">
        Calibrated P(dark pattern) of the {isNum(c.model_evidence) ? <b className="tabular-nums">{c.model_evidence}</b> : ''} model-scored
        evidence item{c.model_evidence === 1 ? '' : 's'} used by the CDI.
      </p>
      {mix.length > 0 && (
        <div className="mt-2">
          <p className="text-[11px] font-semibold text-slate-600">Evidence by provenance</p>
          <ul className="mt-1 flex flex-wrap gap-1.5" aria-label="Evidence items by provenance">
            {mix.map(([k, n]) => (
              <li key={k} className="inline-flex items-center gap-1">
                <ProvenanceChip value={k} />
                <span className="text-xs font-semibold tabular-nums text-slate-800">{n}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {c.note && <p className="mt-2 text-[11px] text-slate-500">{c.note}</p>}
    </div>
  );
}

function CoverageTile({ cdi, unavailable }) {
  return (
    <div className="rounded-lg p-4 ring-1 ring-inset ring-slate-200" data-testid="cdi-coverage">
      <TileHead icon={Target}>Coverage</TileHead>
      <p className="mt-1 text-2xl font-bold tabular-nums text-slate-900">{fmtShare(cdi.coverage, 0)}</p>
      <p className="mt-0.5 text-xs text-slate-600">
        of the configured weight had data
        {unavailable > 0 ? ` — ${unavailable} component${unavailable === 1 ? '' : 's'} unavailable` : ' — all components available'}.
      </p>
      <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-100" aria-hidden="true">
        <div className="h-full rounded-full bg-slate-500" style={{ width: `${clamp100((cdi.coverage || 0) * 100)}%` }} />
      </div>
      {isNum(cdi.evidence_items) && (
        <p className="mt-2 text-xs text-slate-600">
          Based on <b className="tabular-nums">{cdi.evidence_items}</b> evidence item{cdi.evidence_items === 1 ? '' : 's'}.
        </p>
      )}
      <p className="mt-1 text-[11px] text-slate-500">Coverage describes how complete the data is, not how risky the page is.</p>
    </div>
  );
}

/**
 * CDI section. `cdi` is a CDIResult. `stageName` (optional) labels journey evidence.
 */
export default function CDISection({ cdi, stageName, headingId = 'cdi-heading' }) {
  if (!cdi) return null;
  const components = cdi.components || [];
  const unavailable = components.filter((c) => !c.available).length;
  const band = cdi.band ? String(cdi.band).toLowerCase() : null;
  return (
    <IntelCard
      id={headingId}
      icon={Layers}
      title="Deception Intelligence Score (CDI)"
      subtitle={components.length ? `Combines ${listJoin(components.map((c) => c.label))} into one screening index.` : undefined}
    >
      <div className="grid gap-3 md:grid-cols-3" data-testid="cdi-indicators">
        <SeverityTile cdi={cdi} band={band} />
        <ConfidenceTile confidence={cdi.confidence} />
        <CoverageTile cdi={cdi} unavailable={unavailable} />
      </div>

      {components.length > 0 && (
        <div className="mt-5">
          <h4 className="text-sm font-semibold text-slate-900">Contribution by component (points of 100)</h4>
          <div className="mt-2">
            <ContributionBar components={components} cdi={cdi.cdi} />
          </div>
          <ul className="mt-4 space-y-2.5">
            {components.map((c, i) => (
              <ComponentRow key={c.key} c={c} hex={SEGMENT_HEX[i % SEGMENT_HEX.length]} stageName={stageName} />
            ))}
          </ul>
        </div>
      )}

      <WeightSensitivityChart
        s={cdi.sensitivity}
        labelOf={(key) => components.find((c) => c.key === key || c.label === key)?.label || key}
        reported={cdi.cdi}
        indexName="CDI"
      />

      <AlternativesTable alternatives={cdi.alternatives} reported={cdi.cdi} indexName="CDI" />

      <div className="mt-5 space-y-1 border-t border-slate-100 pt-3">
        {cdi.note && <MethodNote>{cdi.note}</MethodNote>}
        <MethodNote>
          Weights {cdi.weights_version}
          {cdi.formula && <> · <Formula>{cdi.formula}</Formula></>}
        </MethodNote>
      </div>
    </IntelCard>
  );
}
