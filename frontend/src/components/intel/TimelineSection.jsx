import { useRef, useState } from 'react';
import { Activity, ArrowRight, Clock, Cpu, Eye, Flag, History, ListOrdered } from 'lucide-react';
import { formatDate, humanKey } from '../../format.js';
import { EvidenceRefs, IntelCard, MetricsGrid, fmtDeltaPp, fmtShare, isNum, useElementWidth } from './shared.jsx';
import { JourneyOnlyNote } from './FSPMSection.jsx';

const KIND_UI = {
  observed_fact: {
    label: 'Observed fact',
    icon: Eye,
    chip: 'bg-slate-900 text-white ring-slate-900',
    card: 'border border-slate-300 bg-white shadow-sm',
    dot: 'bg-slate-900 ring-slate-900',
    text: 'font-semibold text-slate-900',
  },
  observation: {
    label: 'Observation',
    icon: Activity,
    chip: 'bg-white text-slate-700 ring-slate-300',
    card: 'border border-slate-200 bg-white',
    dot: 'bg-white ring-slate-400',
    text: 'text-slate-800',
  },
  model_inference: {
    label: 'Model inference',
    icon: Cpu,
    chip: 'bg-violet-50 text-violet-800 ring-violet-200',
    card: 'border border-dashed border-violet-300 bg-violet-50/30',
    dot: 'bg-violet-100 ring-violet-400',
    text: 'text-slate-700',
  },
};

/* ------------------------------------------------------------------ progression chart */

/**
 * Labels for pressure_progression points. The API returns plain values in the temporal ordering;
 * stage names are attached only when that ordering can be reproduced exactly from the stages.
 */
function progressionLabels(values, temporal, stages) {
  const fallback = values.map((_, i) => `Step ${i + 1}`);
  if (!stages || stages.length !== values.length) return fallback;
  // the API states the stage behind each point explicitly
  const idx = temporal.progression_stage_indices;
  if (Array.isArray(idx) && idx.length === values.length) {
    const byIndex = Object.fromEntries(stages.map((s) => [s.index, s.name]));
    if (idx.every((i) => byIndex[i] != null)) return idx.map((i) => byIndex[i]);
  }
  if (temporal.ordering === 'submission_order') return [...stages].sort((a, b) => a.index - b.index).map((s) => s.name);
  if (temporal.ordering === 'timestamp') {
    const withTime = stages.map((s) => ({ s, t: s.captured_at ? Date.parse(s.captured_at) : NaN }));
    if (withTime.some((x) => Number.isNaN(x.t))) return fallback;
    const times = withTime.map((x) => x.t);
    if (new Set(times).size !== times.length) return fallback; // ties: order not reproducible
    return withTime.sort((a, b) => a.t - b.t).map((x) => x.s.name);
  }
  return fallback;
}

function ProgressionChart({ values, labels }) {
  const wrapRef = useRef(null);
  const width = useElementWidth(wrapRef);
  const [hover, setHover] = useState(null);
  const H = 190;
  const m = { top: 12, right: 16, bottom: 40, left: 40 };
  const plotW = Math.max(0, width - m.left - m.right);
  const plotH = H - m.top - m.bottom;
  const n = values.length;
  const step = n > 0 ? plotW / n : 0;
  const x = (i) => m.left + step * i + step / 2;
  const y = (v) => m.top + plotH - Math.max(0, Math.min(1, v)) * plotH;
  const maxChars = Math.floor(step / 6.2);
  const trunc = (s) => (s.length > maxChars ? `${s.slice(0, Math.max(1, maxChars - 1))}…` : s);
  // step-after path: horizontal through each stage's band, vertical at the boundary
  let d = '';
  values.forEach((v, i) => {
    const x0 = m.left + step * i;
    d += `${i === 0 ? 'M' : 'L'} ${x0} ${y(v)} L ${x0 + step} ${y(v)} `;
  });
  const summary = values.map((v, i) => `${labels[i]} ${fmtShare(v, 0)}`).join(', ');

  return (
    <div>
      <p className="min-h-[1rem] text-right text-xs tabular-nums text-slate-700" aria-hidden="true">
        {hover != null ? <><b>{labels[hover]}</b>: stage pressure {fmtShare(values[hover])}</> : <span className="text-slate-400">Hover or tap a step for its value</span>}
      </p>
      <div ref={wrapRef} className="mt-1 w-full">
        {width > 0 && (
          <svg width={width} height={H} role="img" aria-label={`Stage pressure progression through the journey: ${summary}.`}>
            {[0, 0.25, 0.5, 0.75, 1].map((t) => (
              <g key={t}>
                <line x1={m.left} x2={width - m.right} y1={y(t)} y2={y(t)} stroke="#e2e8f0" />
                <text x={m.left - 6} y={y(t)} textAnchor="end" dominantBaseline="middle" fontSize="10" fill="#64748b">{t * 100}%</text>
              </g>
            ))}
            {values.map((v, i) => (
              <rect
                key={i}
                x={m.left + step * i}
                y={m.top}
                width={step}
                height={plotH}
                fill={hover === i ? 'rgba(148,163,184,0.15)' : 'transparent'}
                onMouseEnter={() => setHover(i)}
                onMouseLeave={() => setHover(null)}
              >
                <title>{`${labels[i]}: ${fmtShare(v)}`}</title>
              </rect>
            ))}
            <path d={d} fill="none" stroke="#0f766e" strokeWidth="2" strokeLinejoin="round" pointerEvents="none" />
            {values.map((v, i) => (
              <g key={i} pointerEvents="none">
                <circle cx={x(i)} cy={y(v)} r={hover === i ? 5 : 4} fill="#fff" stroke="#0f766e" strokeWidth="2" />
                <text x={x(i)} y={m.top + plotH + 14} textAnchor="middle" fontSize="10" fontWeight="600" fill="#334155">{i + 1}</text>
                <text x={x(i)} y={m.top + plotH + 28} textAnchor="middle" fontSize="10" fill="#64748b">{trunc(labels[i])}</text>
              </g>
            ))}
            <line x1={m.left} x2={width - m.right} y1={m.top + plotH} y2={m.top + plotH} stroke="#94a3b8" />
          </svg>
        )}
      </div>
    </div>
  );
}

/** Pressure progression, rendered on its own card in journey results. */
export function PressureProgression({ temporal, stages, headingId = 'progression-heading' }) {
  const values = (temporal?.pressure_progression || []).filter(isNum);
  if (!temporal || values.length === 0) return null;
  const labels = progressionLabels(values, temporal, stages);
  return (
    <IntelCard
      id={headingId}
      icon={Activity}
      title="Pressure progression"
      subtitle={`Stage pressure at each step, in ${temporal.ordering === 'timestamp' ? 'capture-time' : 'submission'} order.`}
    >
      <ProgressionChart values={values} labels={labels} />
    </IntelCard>
  );
}

/* ------------------------------------------------------------------ timeline */

function EventItem({ ev, stageName, last }) {
  const ui = KIND_UI[ev.kind] || KIND_UI.observation;
  const Icon = ui.icon;
  return (
    <li className="relative pl-7">
      {!last && <span className="absolute -bottom-3 left-[7px] top-5 w-0.5 -translate-x-px bg-slate-300" aria-hidden="true" />}
      <span className={`absolute left-0 top-1.5 h-[15px] w-[15px] rounded-full ring-2 ${ui.dot}`} aria-hidden="true" />
      <div className={`rounded-lg p-3 ${ui.card}`}>
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1.5">
          <span className="inline-flex items-center gap-1 text-xs font-medium tabular-nums text-slate-600">
            {ev.timestamp ? (
              <><Clock className="h-3.5 w-3.5" aria-hidden="true" /><time dateTime={ev.timestamp}>{formatDate(ev.timestamp)}</time></>
            ) : (
              <><ListOrdered className="h-3.5 w-3.5" aria-hidden="true" />Order only</>
            )}
          </span>
          <span className="text-xs font-semibold text-slate-800">{ev.stage_name || stageName?.(ev.stage_index)}</span>
          <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-semibold ring-1 ring-inset ${ui.chip}`}>
            <Icon className="h-3 w-3" aria-hidden="true" /> {ui.label}
          </span>
          {ev.category && <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[11px] text-slate-700">{humanKey(ev.category)}</span>}
          {ev.deceptive_signal && (
            <span
              className="inline-flex items-center gap-1 rounded-full bg-red-50 px-2 py-0.5 text-[11px] font-semibold text-red-800 ring-1 ring-inset ring-red-300"
              title="This fact enters the FSPM / CDI calculation"
            >
              <Flag className="h-3 w-3" aria-hidden="true" /> Deceptive signal
            </span>
          )}
          {(isNum(ev.pressure) || isNum(ev.pressure_delta)) && (
            <span className="ml-auto flex items-center gap-2 text-[11px] tabular-nums text-slate-500">
              {isNum(ev.pressure) && <span>stage pressure {fmtShare(ev.pressure, 0)}</span>}
              {isNum(ev.pressure_delta) && (
                <span
                  className={`rounded px-1.5 py-0.5 font-semibold ${ev.pressure_delta > 0.0005 ? 'bg-red-50 text-red-800' : ev.pressure_delta < -0.0005 ? 'bg-emerald-50 text-emerald-800' : 'bg-slate-100 text-slate-700'}`}
                  title="Change in stage pressure from the previous stage, in percentage points"
                >
                  Δ pressure {fmtDeltaPp(ev.pressure_delta)}
                </span>
              )}
            </span>
          )}
        </div>
        <p className={`mt-1.5 text-sm leading-relaxed ${ui.text}`}>{ev.description}</p>
        {(ev.previous_state != null || ev.current_state != null) && (
          <div className="mt-2 flex flex-wrap items-center gap-2 text-xs" data-testid="state-change">
            <span className="min-w-0 rounded-md bg-slate-100 px-2 py-1 text-slate-700">
              <span className="mr-1 text-[10px] font-bold uppercase tracking-wide text-slate-500">Previous</span>
              <span className="break-words">{ev.previous_state ?? '—'}</span>
            </span>
            <ArrowRight className="h-3.5 w-3.5 shrink-0 text-slate-400" aria-label="changed to" />
            <span className="min-w-0 rounded-md bg-white px-2 py-1 text-slate-900 ring-1 ring-inset ring-slate-300">
              <span className="mr-1 text-[10px] font-bold uppercase tracking-wide text-slate-500">Current</span>
              <span className="break-words font-medium">{ev.current_state ?? '—'}</span>
            </span>
          </div>
        )}
        <EvidenceRefs refs={ev.evidence} stageName={stageName} />
      </div>
    </li>
  );
}

/** Temporal evidence timeline. `temporal` is a TemporalResult. */
export default function TimelineSection({ temporal, stageName, onRunJourney, headingId = 'timeline-heading' }) {
  if (!temporal) return null;
  const title = 'Temporal Evidence Timeline';
  if (!temporal.available) {
    return <JourneyOnlyNote id={headingId} icon={History} title={title} reason={temporal.reason} onRunJourney={onRunJourney} />;
  }
  const events = temporal.events || [];
  const kinds = Object.keys(KIND_UI);
  return (
    <IntelCard
      id={headingId}
      icon={History}
      title={title}
      subtitle={
        temporal.ordering === 'timestamp'
          ? 'Events in chronological order of the stages’ capture times.'
          : 'Events in the order the stages were submitted (no capture times were used).'
      }
    >
      {temporal.reason && <p className="mb-3 text-xs text-slate-600">{temporal.reason}</p>}
      <ul className="mb-4 flex flex-wrap gap-2 text-[11px]" aria-label="Event kinds">
        {kinds.map((k) => {
          const ui = KIND_UI[k];
          const Icon = ui.icon;
          return (
            <li key={k} className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 font-semibold ring-1 ring-inset ${ui.chip}`}>
              <Icon className="h-3 w-3" aria-hidden="true" /> {ui.label}
              <span className="font-normal opacity-80">
                {k === 'observed_fact' ? '— directly observed in the input' : k === 'model_inference' ? '— estimated by the classifier' : ''}
              </span>
            </li>
          );
        })}
      </ul>
      {temporal.metrics && Object.keys(temporal.metrics).length > 0 && (
        <div className="mb-4" data-testid="temporal-metrics">
          <h4 className="mb-2 text-sm font-semibold text-slate-900">Journey metrics</h4>
          <MetricsGrid metrics={temporal.metrics} label="Temporal metrics" />
        </div>
      )}
      {events.length === 0 ? (
        <p className="rounded-lg bg-slate-50 p-3 text-sm text-slate-600 ring-1 ring-inset ring-slate-200">No timeline events were produced for this journey.</p>
      ) : (
        <ol className="space-y-3">
          {events.map((ev, i) => <EventItem key={i} ev={ev} stageName={stageName} last={i === events.length - 1} />)}
        </ol>
      )}
    </IntelCard>
  );
}
