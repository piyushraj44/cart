// Shared building blocks for the intelligence sections. Every value rendered here is passed in
// from an API response; these helpers only format it.
import { createContext, useContext, useLayoutEffect, useState } from 'react';
import { AlertTriangle, ChevronDown, CircleSlash, Fingerprint, Info } from 'lucide-react';

/* ------------------------------------------------------------------ number formatting */

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

/** A 0..100 index value (CDI, MCAI, FSPM). One decimal, trailing ".0" kept for alignment. */
export function fmtIndex(v) {
  return isNum(v) ? v.toFixed(1) : '—';
}

/** A share in [0,1] shown as a percentage with one decimal (coverage, weights, pressure). */
export function fmtShare(v, digits = 1) {
  return isNum(v) ? `${(v * 100).toFixed(digits)}%` : '—';
}

/** Contribution points of a 0..100 index. */
export function fmtPoints(v) {
  return isNum(v) ? `${v.toFixed(1)} pts` : '—';
}

/** A raw measured feature value (px, contrast ratio, step count …). */
export function fmtValue(v) {
  if (!isNum(v)) return '—';
  if (Number.isInteger(v)) return v.toLocaleString();
  return String(Number(v.toPrecision(4)));
}

export { isNum };

/* ------------------------------------------------------------------ provenance */

const PROVENANCE = {
  model: { label: 'Model', cls: 'bg-violet-50 text-violet-800 ring-violet-200', title: 'Estimated by the calibrated text classifier' },
  measured: { label: 'Measured', cls: 'bg-teal-50 text-teal-800 ring-teal-200', title: 'Measured from the submitted input' },
  observed: { label: 'Observed', cls: 'bg-teal-50 text-teal-800 ring-teal-200', title: 'Directly observed in the submitted input' },
  // structural inference: deliberately a dashed outline so it never reads like a measurement
  inferred: {
    label: 'Inferred',
    cls: 'border-amber-400 bg-amber-50/60 text-amber-900',
    dashed: true,
    title: 'Inferred from page structure, not observed clicks',
  },
  declared: { label: 'Declared', cls: 'bg-sky-50 text-sky-800 ring-sky-200', title: 'Declared by the person who submitted the audit' },
  default: { label: 'Default', cls: 'bg-slate-100 text-slate-700 ring-slate-300', title: 'A configured default value, not measured' },
  by_construction: { label: 'By construction', cls: 'bg-slate-100 text-slate-700 ring-slate-300', title: 'Follows from how the index is defined' },
  unavailable: { label: 'Unavailable', cls: 'bg-white text-slate-500 ring-slate-300', title: 'Not available for this input' },
};

export function ProvenanceChip({ value }) {
  const p = PROVENANCE[value] || { label: value || 'unknown', cls: 'bg-slate-100 text-slate-700 ring-slate-300', title: 'Provenance' };
  const shape = p.dashed ? 'border border-dashed' : 'ring-1 ring-inset';
  return (
    <span
      title={p.title}
      data-provenance={value || 'unknown'}
      className={`inline-flex shrink-0 items-center rounded-full px-2 py-0.5 text-[11px] font-semibold ${shape} ${p.cls}`}
    >
      <span className="sr-only">Provenance: </span>
      {p.label}
    </span>
  );
}

/* ------------------------------------------------------------------ layout */

/**
 * Inside a <Collapsible> the collapsible header already carries the section title, so the intel
 * sections drop their own card chrome. `showTitles` keeps a small sub-heading (used when one
 * collapsible holds several sections).
 */
const BareContext = createContext(null);
export function BareIntelCards({ showTitles = false, children }) {
  return <BareContext.Provider value={{ showTitles }}>{children}</BareContext.Provider>;
}

export function IntelCard({ id, icon: Icon, title, subtitle, aside, children, className = '' }) {
  const bare = useContext(BareContext);
  if (bare) {
    return (
      <div className={className} data-intel-card="bare">
        {bare.showTitles && (
          <h3 id={id} className="text-sm font-semibold text-slate-900">{title}</h3>
        )}
        {subtitle && <p className={`text-sm text-slate-600 ${bare.showTitles ? 'mt-0.5' : ''}`}>{subtitle}</p>}
        <div className={subtitle || bare.showTitles ? 'mt-4' : ''}>{children}</div>
      </div>
    );
  }
  return (
    <section aria-labelledby={id} className={`card p-4 sm:p-5 ${className}`}>
      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          <h3 id={id} className="flex items-center gap-2 text-base font-bold text-slate-900">
            {Icon && <Icon className="h-4 w-4 shrink-0 text-teal-700" aria-hidden="true" />}
            {title}
          </h3>
          {subtitle && <p className="mt-0.5 text-sm text-slate-600">{subtitle}</p>}
        </div>
        {aside}
      </div>
      <div className="mt-4">{children}</div>
    </section>
  );
}

/** Calm "not available" state, carrying the API's own reason. */
export function UnavailableNote({ reason, children }) {
  return (
    <div className="flex items-start gap-3 rounded-lg border border-dashed border-slate-300 bg-slate-50 p-3 text-sm text-slate-600">
      <CircleSlash className="mt-0.5 h-4 w-4 shrink-0 text-slate-400" aria-hidden="true" />
      <div className="min-w-0 flex-1">
        <p>{reason || 'Not available for this input.'}</p>
        {children}
      </div>
    </div>
  );
}

export function InfoNote({ children, tone = 'slate' }) {
  const cls =
    tone === 'amber'
      ? 'border-amber-200 bg-amber-50 text-amber-950'
      : 'border-slate-200 bg-slate-50 text-slate-700';
  const Icon = tone === 'amber' ? AlertTriangle : Info;
  return (
    <div role="note" className={`flex items-start gap-2 rounded-lg border px-3 py-2 text-xs leading-relaxed ${cls}`}>
      <Icon className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
      <div className="min-w-0">{children}</div>
    </div>
  );
}

/** Small muted technical footnote (formula, weights version …). Text comes from the API. */
export function MethodNote({ children }) {
  return <p className="break-words text-[11px] leading-relaxed text-slate-500">{children}</p>;
}

export function Formula({ children }) {
  return <code className="break-words rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[11px] text-slate-600">{children}</code>;
}

/* ------------------------------------------------------------------ evidence references */

/**
 * EvidenceRef[] -> compact list of quoted snippets. `stageName(i)` resolves a stage index to a
 * label in journey results; single audits pass nothing.
 */
export function EvidenceRefs({ refs, stageName, max = 3 }) {
  if (!refs || refs.length === 0) return null;
  const shown = refs.slice(0, max);
  return (
    <ul className="mt-2 space-y-1">
      {shown.map((e, i) => (
        <li key={i} className="flex min-w-0 flex-wrap items-baseline gap-x-2 gap-y-0.5 text-xs">
          <span className="min-w-0 break-words text-slate-700">“{e.text}”</span>
          {(e.stage_index != null || e.evidence_id || e.finding_id) && (
            <span className="shrink-0 text-[11px] text-slate-500">
              {e.stage_index != null && (stageName ? stageName(e.stage_index) : `Stage ${e.stage_index + 1}`)}
              {e.stage_index != null && (e.evidence_id || e.finding_id) ? ' · ' : ''}
              {e.evidence_id ? (
                <span className="rounded bg-slate-100 px-1 font-mono text-slate-700" title="Evidence id">{e.evidence_id}</span>
              ) : (
                e.finding_id && <span className="font-mono">{e.finding_id}</span>
              )}
            </span>
          )}
        </li>
      ))}
      {refs.length > max && <li className="text-[11px] text-slate-500">+ {refs.length - max} more</li>}
    </ul>
  );
}

/* ------------------------------------------------------------------ charts */

export function useElementWidth(ref) {
  const [w, setW] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    setW(el.getBoundingClientRect().width);
    const ro = new ResizeObserver(([e]) => setW(e.contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return w;
}

/** Round an axis maximum up to a readable value (5, 10, 20, 25, 50, 100). */
export function niceMax(v) {
  if (!isNum(v) || v <= 0) return 10;
  for (const step of [5, 10, 20, 25, 50, 100]) if (v <= step) return step;
  return Math.ceil(v / 100) * 100;
}

/** Evenly spaced, round tick values from 0 to a niceMax() value. */
export function niceTicks(max) {
  const step = { 5: 1, 10: 2, 20: 5, 25: 5, 50: 10, 100: 20 }[max] || max / 5;
  const out = [];
  for (let t = 0; t <= max + 1e-9; t += step) out.push(Number(t.toFixed(6)));
  return out;
}

/** Horizontal 0..100 axis ticks drawn in HTML (used by bar-style charts). */
export function AxisTicks({ ticks = [0, 25, 50, 75, 100], unit = '' }) {
  return (
    <div className="relative mt-1 h-4 text-[10px] tabular-nums text-slate-500" aria-hidden="true">
      {ticks.map((t) => (
        <span
          key={t}
          className="absolute -translate-x-1/2"
          style={{ left: `${t}%`, transform: t === 0 ? 'none' : t === 100 ? 'translateX(-100%)' : undefined }}
        >
          {t}
          {unit}
        </span>
      ))}
    </div>
  );
}

export const SCHEME_UI = {
  fibonacci: { label: 'Fibonacci', hex: '#0f766e' },
  linear: { label: 'Linear', hex: '#64748b' },
  equal: { label: 'Equal', hex: '#cbd5e1' },
};

const FEATURE_LABEL = {
  control_kind: 'Control kind',
  font_size_px: 'Font size (px)',
  font_weight: 'Font weight',
  background_fill: 'Background fill',
  contrast_ratio: 'Contrast ratio',
  opacity: 'Opacity',
  text_height_px: 'Text height (px)',
  text_box_area_px: 'Text box area (px²)',
  steps: 'Steps to complete',
  guilt: 'Guilt wording',
  accept_pressure: 'Accept pressure',
  wording_imbalance: 'Wording imbalance',
};

export function humanKeyLabel(key) {
  return String(key).replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase());
}

export function humanFeature(key) {
  return FEATURE_LABEL[key] || String(key).replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase());
}

/* ------------------------------------------------------------------ signed deltas */

/** A change of a [0,1] share, in signed percentage points (e.g. +12.5 pp). */
export function fmtDeltaPp(v) {
  if (!isNum(v)) return '—';
  const pp = v * 100;
  const r = Number(pp.toFixed(1));
  if (r === 0) return '±0.0 pp';
  return `${r > 0 ? '+' : '−'}${Math.abs(r).toFixed(1)} pp`;
}

/* ------------------------------------------------------------------ weight sensitivity (CDI + MCAI) */

const clamp100 = (v) => Math.max(0, Math.min(100, v));

function RangeRow({ label, sub, low, high, reported, hex = '#0f766e' }) {
  const ok = isNum(low) && isNum(high);
  const lo = ok ? clamp100(Math.min(low, high)) : 0;
  const hi = ok ? clamp100(Math.max(low, high)) : 0;
  return (
    <li className="grid grid-cols-1 gap-1 sm:grid-cols-[11rem_minmax(0,1fr)_6.5rem] sm:items-center sm:gap-3">
      <p className="min-w-0 text-xs text-slate-700">
        <span className="font-medium text-slate-800">{label}</span>
        {sub && <span className="block text-[11px] text-slate-500">{sub}</span>}
      </p>
      <div className="relative h-4" title={ok ? `${fmtIndex(low)} – ${fmtIndex(high)}` : 'not available'}>
        <div className="absolute inset-x-0 top-1/2 h-px -translate-y-1/2 bg-slate-200" />
        {ok && (
          <div
            className="absolute top-1/2 h-2.5 -translate-y-1/2 rounded-full opacity-80"
            style={{ left: `${lo}%`, width: `max(${hi - lo}%, 3px)`, background: hex }}
          />
        )}
        {isNum(reported) && <div className="absolute inset-y-0 w-0.5 bg-slate-900" style={{ left: `calc(${clamp100(reported)}% - 1px)` }} />}
      </div>
      <p className="text-xs tabular-nums text-slate-700 sm:text-right">{ok ? `${fmtIndex(low)} – ${fmtIndex(high)}` : '—'}</p>
    </li>
  );
}

/**
 * One-at-a-time + Dirichlet weight sensitivity on a 0..100 axis. `s` is a CDISensitivity /
 * WeightSensitivity; `labelOf(component)` names a row; `reported` is the index the API reported.
 */
export function WeightSensitivityChart({ s, labelOf = humanKeyLabel, reported, indexName = 'CDI', title = 'Weight sensitivity', compact = false }) {
  if (!s) return null;
  const oat = (s.one_at_a_time || []).filter((r) => isNum(r?.low) && isNum(r?.high));
  // samples === 0 means the weights were not resampled (e.g. a single available weight): no band to show
  const hasDir = isNum(s.dirichlet_p05) && isNum(s.dirichlet_p95) && s.samples !== 0;
  if (!oat.length && !hasDir) return null;
  const summary = [
    hasDir ? `Dirichlet weight resampling 5th–95th percentile ${fmtIndex(s.dirichlet_p05)} to ${fmtIndex(s.dirichlet_p95)}` : null,
    ...oat.map((r) => `${labelOf(r.component)} weight varied: ${fmtIndex(r.low)} to ${fmtIndex(r.high)}`),
  ].filter(Boolean).join('; ');
  const Heading = compact ? 'h5' : 'h4';
  return (
    <div className={compact ? 'mt-4' : 'mt-5 border-t border-slate-100 pt-4'}>
      <Heading className="text-sm font-semibold text-slate-900">{title}</Heading>
      <p className="mt-0.5 text-xs text-slate-600">
        How much the index moves if the configured weights were different. The dark line marks the reported {indexName}
        {isNum(reported) ? ` (${fmtIndex(reported)})` : ''}.
      </p>
      <div role="img" aria-label={`Weight sensitivity of the ${indexName}: ${summary}.`} className="mt-3">
        <ul className="space-y-2.5" aria-hidden="true">
          {hasDir && (
            <RangeRow
              label="All weights resampled"
              sub={`Dirichlet, 5th–95th percentile${isNum(s.samples) ? ` of ${s.samples.toLocaleString()} samples` : ''}`}
              low={s.dirichlet_p05}
              high={s.dirichlet_p95}
              reported={reported}
              hex="#0f766e"
            />
          )}
          {oat.map((r, i) => (
            <RangeRow key={`${r.component}-${i}`} label={labelOf(r.component)} sub="this weight varied, others renormalised" low={r.low} high={r.high} reported={reported} hex="#64748b" />
          ))}
        </ul>
        <div className="grid grid-cols-1 sm:grid-cols-[11rem_minmax(0,1fr)_6.5rem] sm:gap-3">
          <span className="hidden sm:block" />
          <AxisTicks />
          <span className="hidden sm:block" />
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ alternative aggregators (study only) */

/**
 * The same inputs under alternative aggregators. Shown for transparency only: the reported index
 * is always the weighted mean. `alternatives` is {aggregator: value 0..100}.
 */
export function AlternativesTable({ alternatives, reported, indexName = 'CDI', compact = false }) {
  const rows = Object.entries(alternatives || {}).filter(([, v]) => isNum(v));
  if (!rows.length) return null;
  return (
    <details className={`group rounded-lg border border-slate-200 ${compact ? 'mt-3' : 'mt-5'}`} data-testid="alternatives">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 rounded-lg px-3 py-2 text-sm font-medium text-slate-800 hover:bg-slate-50 [&::-webkit-details-marker]:hidden">
        <span className="flex min-w-0 flex-wrap items-center gap-2">
          Alternative aggregators
          <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wide text-slate-600">Study only</span>
        </span>
        <ChevronDown className="h-4 w-4 text-slate-500 transition-transform group-open:rotate-180" aria-hidden="true" />
      </summary>
      <div className="border-t border-slate-200 px-3 pb-3 pt-2">
        <p className="text-xs text-slate-600">
          The weighted mean is what ClearCart reports{isNum(reported) ? ` (${indexName} ${fmtIndex(reported)})` : ''}; these
          values use the same inputs and are shown only to make the choice of aggregator transparent.
        </p>
        <table className="mt-2 w-full text-left text-xs">
          <thead className="text-[11px] uppercase tracking-wide text-slate-500">
            <tr>
              <th scope="col" className="py-1 pr-3 font-semibold">Aggregator</th>
              <th scope="col" className="py-1 text-right font-semibold">{indexName} (0–100)</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {rows.map(([k, v]) => {
              const selected = /\(selected\)/i.test(k);
              return (
                <tr key={k} className={selected ? 'bg-teal-50/60 font-semibold' : ''}>
                  <td className="py-1.5 pl-1 pr-3 text-slate-700">
                    {humanKeyLabel(k.replace(/\s*\(selected\)\s*/i, ''))}
                    {selected && <span className="ml-2 text-[10px] font-bold uppercase tracking-wide text-teal-800">Reported</span>}
                  </td>
                  <td className="py-1.5 pr-1 text-right tabular-nums text-slate-800">{fmtIndex(v)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </details>
  );
}

/* ------------------------------------------------------------------ generic values (metrics, optimisation) */

const COUNT_KEY = /(count|^n_|_n$|num|number|steps|stages|events|repeat|samples|total|index|units|resets|preselections)/i;

/**
 * Formats one value of an engine-defined dict. Non-integer numbers in [0,1] are shares (%); changes of a
 * share ("delta" keys) are signed percentage points; integers are counts unless the key names a rate/share.
 * Lists and nested objects are shown compactly.
 */
export function fmtGeneric(value, key = '') {
  if (value == null) return '—';
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return '—';
    if (/delta/i.test(key) && Math.abs(value) <= 1 && !Number.isInteger(value)) return fmtDeltaPp(value);
    if (/(change|increase|decrease)/i.test(key) && !COUNT_KEY.test(key) && Math.abs(value) <= 1 && !Number.isInteger(value)) {
      return `${(value * 100).toFixed(1)} pp`;
    }
    if (!Number.isInteger(value) && value >= 0 && value <= 1) return fmtShare(value);
    if (Number.isInteger(value) && (value === 0 || value === 1) && !COUNT_KEY.test(key) && /(rate|share|ratio|fraction|persist|pressure|prop|mean|max|min)/i.test(key)) return fmtShare(value);
    return fmtValue(value);
  }
  if (Array.isArray(value)) {
    if (value.length === 0) return 'none';
    const shown = value.slice(0, 6).map((v) => fmtGeneric(v, key));
    return `${shown.join(value.some((v) => v != null && typeof v === 'object') ? '; ' : ', ')}${value.length > 6 ? ` … +${value.length - 6}` : ''}`;
  }
  if (typeof value === 'object') {
    // {from, to, ...}: a transition between two stages
    if ('from' in value && 'to' in value) {
      const rest = Object.entries(value).filter(([k]) => k !== 'from' && k !== 'to');
      const tail = rest.map(([k, v]) => fmtGeneric(v, k)).join(' · ');
      return `${fmtGeneric(value.from)} → ${fmtGeneric(value.to)}${tail ? `: ${tail}` : ''}`;
    }
    return Object.entries(value).map(([k, v]) => `${humanKeyLabel(k)}: ${fmtGeneric(v, k)}`).join(' · ');
  }
  return String(value);
}

function MetricValue({ value, k }) {
  if (Array.isArray(value) && value.length > 1 && value.some((v) => v != null && typeof v === 'object')) {
    return (
      <ul className="space-y-0.5 text-xs font-medium">
        {value.map((v, i) => <li key={i} className="break-words">{fmtGeneric(v, k)}</li>)}
      </ul>
    );
  }
  if (value != null && typeof value === 'object' && !Array.isArray(value) && !('from' in value && 'to' in value)) {
    const entries = Object.entries(value);
    if (!entries.length) return 'none';
    return (
      <ul className="space-y-0.5 text-xs font-medium">
        {entries.map(([kk, v]) => (
          <li key={kk} className="flex justify-between gap-2">
            <span className="min-w-0 break-words font-normal text-slate-600">{humanKeyLabel(kk)}</span>
            <span className="shrink-0">{fmtGeneric(v, k)}</span>
          </li>
        ))}
      </ul>
    );
  }
  return fmtGeneric(value, k);
}

/** Small grid of an engine-defined metrics dict; keys and values come straight from the API. */
export function MetricsGrid({ metrics, label = 'Metrics' }) {
  const entries = Object.entries(metrics || {}).filter(([, v]) => v !== undefined);
  if (!entries.length) return null;
  return (
    <dl aria-label={label} className="grid grid-cols-1 gap-2 min-[420px]:grid-cols-2 lg:grid-cols-3">
      {entries.map(([k, v]) => (
        <div key={k} className="min-w-0 rounded-lg bg-slate-50 px-3 py-2 ring-1 ring-inset ring-slate-200">
          <dt className="truncate text-[11px] font-semibold uppercase tracking-wide text-slate-500" title={k}>{humanKeyLabel(k)}</dt>
          <dd className="mt-0.5 break-words text-sm font-semibold tabular-nums text-slate-900"><MetricValue value={v} k={k} /></dd>
        </div>
      ))}
    </dl>
  );
}

/** Renders any JSON value as readable nested text (used for optimisation details). */
export function GenericValue({ value, k = '' }) {
  if (value == null) return <span className="text-slate-400">—</span>;
  if (Array.isArray(value)) {
    if (value.every((v) => v == null || typeof v !== 'object')) {
      return (
        <ul className="list-disc space-y-0.5 pl-4">
          {value.map((v, i) => <li key={i} className="break-words">{typeof v === 'string' ? v : fmtGeneric(v, k)}</li>)}
        </ul>
      );
    }
    return (
      <ul className="space-y-1">
        {value.map((v, i) => <li key={i}><GenericValue value={v} k={k} /></li>)}
      </ul>
    );
  }
  if (typeof value === 'object') {
    return (
      <dl className="space-y-0.5">
        {Object.entries(value).map(([kk, v]) => (
          <div key={kk} className="flex flex-wrap gap-x-1.5">
            <dt className="font-medium text-slate-600">{humanKeyLabel(kk)}:</dt>
            <dd className="min-w-0 break-words text-slate-800"><GenericValue value={v} k={kk} /></dd>
          </div>
        ))}
      </dl>
    );
  }
  return <span className="break-words">{typeof value === 'string' ? value : fmtGeneric(value, k)}</span>;
}

/* ------------------------------------------------------------------ reproducibility */

/** Model / calibration / config versions as a compact definition list. */
export function VersionsList({ versions }) {
  const entries = Object.entries(versions || {});
  if (!entries.length) return null;
  return (
    <dl className="grid gap-x-6 gap-y-1.5 text-xs sm:grid-cols-2">
      {entries.map(([k, v]) => (
        <div key={k} className="flex min-w-0 flex-wrap items-baseline justify-between gap-x-3 border-b border-slate-100 pb-1">
          <dt className="text-slate-600">{humanKeyLabel(k)}</dt>
          <dd className="min-w-0 break-all font-mono text-slate-900">{typeof v === 'string' ? v : JSON.stringify(v)}</dd>
        </div>
      ))}
    </dl>
  );
}

/**
 * Collapsible list of the model / calibration / config versions behind a result
 * (AuditResponse.versions, JourneyResponse.versions). Renders nothing when absent.
 */
export function ReproducibilityBlock({ versions, children, defaultOpen = false, id = 'repro', bare = false, title = 'Reproducibility' }) {
  const entries = Object.entries(versions || {});
  if (!entries.length && !children) return null;
  return (
    <details className={`group ${bare ? '' : 'card'}`} open={defaultOpen || undefined} data-testid="versions-block">
      <summary
        id={`${id}-summary`}
        className="flex cursor-pointer list-none items-center justify-between gap-3 rounded-xl px-4 py-3 hover:bg-slate-50 [&::-webkit-details-marker]:hidden"
      >
        <span className="flex items-center gap-2 text-sm font-semibold text-slate-900">
          <Fingerprint className="h-4 w-4 text-teal-700" aria-hidden="true" />
          {title}
          {entries.length > 0 && <span className="text-xs font-normal text-slate-500">{entries.length} version{entries.length === 1 ? '' : 's'} recorded</span>}
        </span>
        <ChevronDown className="h-4 w-4 text-slate-500 transition-transform group-open:rotate-180" aria-hidden="true" />
      </summary>
      <div className="space-y-3 border-t border-slate-200 px-4 py-3">
        {entries.length > 0 ? (
          <VersionsList versions={versions} />
        ) : (
          <p className="text-xs text-slate-500">This result does not list model or configuration versions.</p>
        )}
        {children}
      </div>
    </details>
  );
}
