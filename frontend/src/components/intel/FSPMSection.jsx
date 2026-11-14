import { useMemo, useRef, useState } from 'react';
import { Route, TrendingUp } from 'lucide-react';
import {
  EvidenceRefs, Formula, InfoNote, IntelCard, MethodNote, SCHEME_UI, UnavailableNote, fmtIndex, fmtPoints, fmtShare, isNum,
  niceMax, niceTicks, useElementWidth,
} from './shared.jsx';

const SCHEMES = ['fibonacci', 'linear', 'equal'];

function truncate(s, max) {
  if (max <= 1) return '';
  return s.length > max ? `${s.slice(0, Math.max(1, max - 1))}…` : s;
}

/** Shown on single-page audits, where the API marks FSPM / temporal analysis unavailable. */
export function JourneyOnlyNote({ id, icon: Icon, title, reason, onRunJourney }) {
  return (
    <IntelCard id={id} icon={Icon} title={title}>
      <UnavailableNote reason={reason || 'Only available for multi-step journey audits.'}>
        {onRunJourney && (
          <button type="button" onClick={onRunJourney} className="btn-secondary mt-3 py-2">
            <Route className="h-4 w-4" aria-hidden="true" />
            Run a journey audit
          </button>
        )}
      </UnavailableNote>
    </IntelCard>
  );
}

/* ------------------------------------------------------------------ grouped bar chart */
function ContributionChart({ stages }) {
  const wrapRef = useRef(null);
  const width = useElementWidth(wrapRef);
  const [hover, setHover] = useState(null);
  const H = 230;
  const m = { top: 12, right: 8, bottom: 44, left: 40 };
  const plotW = Math.max(0, width - m.left - m.right);
  const plotH = H - m.top - m.bottom;
  const maxV = Math.max(0, ...stages.flatMap((s) => SCHEMES.map((k) => s.contributions?.[k]).filter(isNum)));
  const yMax = niceMax(maxV);
  const ticks = niceTicks(yMax);
  const n = stages.length || 1;
  const groupW = plotW / n;
  const barW = Math.max(3, Math.min(22, (groupW * 0.78) / 3));
  const y = (v) => m.top + plotH - (Math.max(0, v) / yMax) * plotH;
  const maxChars = Math.floor(groupW / 6.2);

  const summary = stages
    .map((s) => `${s.name}: ${SCHEMES.map((k) => `${SCHEME_UI[k].label} ${fmtPoints(s.contributions?.[k])}`).join(', ')}`)
    .join('; ');

  const hovered = hover && stages[hover.i];
  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-700" aria-hidden="true">
          {SCHEMES.map((k) => (
            <li key={k} className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-sm ring-1 ring-inset ring-black/10" style={{ background: SCHEME_UI[k].hex }} />
              {SCHEME_UI[k].label}
            </li>
          ))}
        </ul>
        <p className="min-h-[1rem] text-xs tabular-nums text-slate-700" aria-hidden="true">
          {hovered ? (
            <>
              <b>{hovered.name}</b> · {SCHEME_UI[hover.k].label}: {fmtPoints(hovered.contributions?.[hover.k])}
            </>
          ) : (
            <span className="text-slate-400">Hover or tap a bar for its value</span>
          )}
        </p>
      </div>
      <div ref={wrapRef} className="mt-2 w-full">
        {width > 0 && (
          <svg width={width} height={H} role="img" aria-label={`Per-stage contribution in points under each weighting scheme. ${summary}.`}>
            {ticks.map((t) => (
              <g key={t}>
                <line x1={m.left} x2={width - m.right} y1={y(t)} y2={y(t)} stroke="#e2e8f0" strokeWidth="1" />
                <text x={m.left - 6} y={y(t)} textAnchor="end" dominantBaseline="middle" fontSize="10" fill="#64748b">
                  {t}
                </text>
              </g>
            ))}
            <text x={10} y={m.top + plotH / 2} transform={`rotate(-90 10 ${m.top + plotH / 2})`} textAnchor="middle" fontSize="10" fill="#64748b">
              points
            </text>
            {stages.map((s, i) => {
              const gx = m.left + i * groupW + groupW / 2;
              const start = gx - (barW * 3) / 2;
              return (
                <g key={s.stage_index}>
                  {SCHEMES.map((k, j) => {
                    const v = s.contributions?.[k];
                    if (!isNum(v)) return null;
                    const top = y(v);
                    const active = hover && hover.i === i && hover.k === k;
                    return (
                      <rect
                        key={k}
                        x={start + j * barW + 0.5}
                        y={top}
                        width={barW - 1}
                        height={Math.max(0, m.top + plotH - top)}
                        fill={SCHEME_UI[k].hex}
                        stroke={active ? '#0f172a' : k === 'equal' ? '#94a3b8' : 'none'}
                        strokeWidth={active ? 1.5 : 0.75}
                        onMouseEnter={() => setHover({ i, k })}
                        onMouseLeave={() => setHover(null)}
                      >
                        <title>{`${s.name} · ${SCHEME_UI[k].label}: ${fmtPoints(v)}`}</title>
                      </rect>
                    );
                  })}
                  <text x={gx} y={m.top + plotH + 14} textAnchor="middle" fontSize="10" fontWeight="600" fill="#334155">
                    #{s.position}
                  </text>
                  <text x={gx} y={m.top + plotH + 28} textAnchor="middle" fontSize="10" fill="#64748b">
                    {truncate(s.name, maxChars)}
                  </text>
                </g>
              );
            })}
            <line x1={m.left} x2={width - m.right} y1={m.top + plotH} y2={m.top + plotH} stroke="#94a3b8" strokeWidth="1" />
          </svg>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ stage table */
function StageTable({ stages, topIndex, stageName }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-slate-200">
      <table className="w-full min-w-[720px] text-left text-xs">
        <caption className="sr-only">FSPM per-stage pressure, normalised weights and Fibonacci contribution</caption>
        <thead className="bg-slate-50 text-[11px] uppercase tracking-wide text-slate-500">
          <tr>
            <th scope="col" className="px-3 py-2 font-semibold">Pos.</th>
            <th scope="col" className="px-3 py-2 font-semibold">Stage</th>
            <th scope="col" className="px-3 py-2 text-right font-semibold">Pressure p<sub>i</sub></th>
            <th scope="col" className="px-3 py-2 text-right font-semibold">Fibonacci w</th>
            <th scope="col" className="px-3 py-2 text-right font-semibold">Linear w</th>
            <th scope="col" className="px-3 py-2 text-right font-semibold">Equal w</th>
            <th scope="col" className="px-3 py-2 text-right font-semibold">Fibonacci contribution</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {stages.map((s) => (
            <tr key={s.stage_index} className={`align-top ${s.stage_index === topIndex ? 'bg-teal-50/50' : ''}`}>
              <td className="px-3 py-2 font-semibold tabular-nums text-slate-700">{s.position}</td>
              <td className="max-w-[16rem] px-3 py-2">
                <p className="font-medium text-slate-900">
                  {s.name}
                  {s.stage_index === topIndex && <span className="ml-1.5 rounded bg-teal-700 px-1.5 py-px text-[10px] font-bold uppercase text-white">Top</span>}
                </p>
                <EvidenceRefs refs={s.pressure_sources} stageName={stageName} max={2} />
              </td>
              <td className="px-3 py-2 text-right font-semibold tabular-nums text-slate-900">{fmtShare(s.pressure)}</td>
              <td className="px-3 py-2 text-right tabular-nums text-slate-700">{fmtShare(s.weights?.fibonacci)}</td>
              <td className="px-3 py-2 text-right tabular-nums text-slate-700">{fmtShare(s.weights?.linear)}</td>
              <td className="px-3 py-2 text-right tabular-nums text-slate-700">{fmtShare(s.weights?.equal)}</td>
              <td className="px-3 py-2 text-right font-semibold tabular-nums text-teal-800">{fmtPoints(s.contributions?.fibonacci)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ------------------------------------------------------------------ main */
/**
 * FSPM section. `fspm` is an FSPMResult. On single audits (available=false) it renders a compact
 * note with the API's reason and, if `onRunJourney` is given, a button to the journey builder.
 */
export default function FSPMSection({ fspm, stageName, onRunJourney, headingId = 'fspm-heading' }) {
  const stages = useMemo(() => [...(fspm?.stages || [])].sort((a, b) => a.stage_index - b.stage_index), [fspm]);
  if (!fspm) return null;
  if (!fspm.available) {
    return <JourneyOnlyNote id={headingId} icon={TrendingUp} title="Fibonacci Pressure Analysis (FSPM)" reason={fspm.reason} onRunJourney={onRunJourney} />;
  }
  const scores = fspm.scores || {};
  const top = stages.find((s) => s.stage_index === fspm.top_stage_index);
  const fib = fspm.fibonacci_numbers || [];

  return (
    <IntelCard
      id={headingId}
      icon={TrendingUp}
      title="Fibonacci Pressure Analysis (FSPM)"
      subtitle="Sequential pressure across the journey, weighting each stage by its position."
    >
      {fspm.reason && <div className="mb-3"><InfoNote>{fspm.reason}</InfoNote></div>}

      <div className="grid gap-2 sm:grid-cols-3">
        {SCHEMES.map((k) => {
          const main = k === 'fibonacci';
          return (
            <div
              key={k}
              className={`rounded-lg p-3 ring-inset ${main ? 'bg-teal-50/60 ring-2 ring-teal-600' : 'bg-white ring-1 ring-slate-200'}`}
            >
              <p className={`text-[11px] font-bold uppercase tracking-wider ${main ? 'text-teal-800' : 'text-slate-500'}`}>
                {SCHEME_UI[k].label} weighting{main ? ' (FSPM)' : ' (baseline)'}
              </p>
              <p className="mt-1 tabular-nums text-slate-900">
                <span className={main ? 'text-3xl font-bold' : 'text-2xl font-semibold text-slate-700'}>{fmtIndex(scores[k])}</span>
                <span className="text-sm text-slate-500"> /100</span>
              </p>
            </div>
          );
        })}
      </div>
      <p className="mt-2 text-xs leading-relaxed text-slate-600">
        Fibonacci weighting is a proposed weighting that puts more emphasis on later stages, where decisions become binding. It is
        not an established law; equal and linear weighting are shown alongside so its effect is visible.
      </p>

      {top ? (
        <div className="mt-4 rounded-lg border-l-4 border-teal-700 bg-slate-50 p-3">
          <p className="text-[11px] font-bold uppercase tracking-wider text-slate-500">Highest-pressure stage (by Fibonacci contribution)</p>
          <p className="mt-1 text-sm text-slate-900">
            <b>{top.name}</b> at position {top.position} — stage pressure <b className="tabular-nums">{fmtShare(top.pressure)}</b>,
            contributing <b className="tabular-nums">{fmtPoints(top.contributions?.fibonacci)}</b> of the FSPM score.
          </p>
          <EvidenceRefs refs={top.pressure_sources} stageName={stageName} />
        </div>
      ) : (
        <p className="mt-4 rounded-lg bg-slate-50 p-3 text-sm text-slate-600 ring-1 ring-inset ring-slate-200">
          No stage carried measurable pressure, so there is no top-pressure stage.
        </p>
      )}

      <h4 className="mt-5 text-sm font-semibold text-slate-900">Contribution per stage under each weighting</h4>
      <p className="mt-0.5 text-xs text-slate-500">Bar height = 100 × normalised weight × stage pressure. Each scheme’s bars add up to its score.</p>
      <div className="mt-2">
        <ContributionChart stages={stages} />
      </div>

      <h4 className="mt-5 text-sm font-semibold text-slate-900">Stage breakdown</h4>
      <div className="mt-2">
        <StageTable stages={stages} topIndex={fspm.top_stage_index} stageName={stageName} />
      </div>

      <div className="mt-4 space-y-1 border-t border-slate-100 pt-3">
        <MethodNote>
          Positions: {fspm.positions_mode === 'canonical'
            ? 'canonical — each stage’s position comes from its declared stage type in the standard funnel order.'
            : 'submission order — stages are numbered in the order they were submitted.'}
        </MethodNote>
        {fib.length > 0 && (
          <MethodNote>
            Fibonacci numbers used, F(1)…F({fib.length}): <span className="tabular-nums">{fib.join(', ')}</span>
          </MethodNote>
        )}
        {fspm.formula && <MethodNote><Formula>{fspm.formula}</Formula></MethodNote>}
      </div>
    </IntelCard>
  );
}
