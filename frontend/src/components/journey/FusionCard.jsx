import { Combine } from 'lucide-react';
import { IntelCard, MethodNote, fmtShare, isNum } from '../intel/shared.jsx';

/** One side of a duplicate pair: an EvidenceRef ({stage_index, finding_id, evidence_id, text}) or a bare stage index. */
function PairSide({ side, stageName }) {
  if (side == null) return <span className="text-slate-400">—</span>;
  if (typeof side === 'number') return <span className="font-medium text-slate-800">{stageName(side)}</span>;
  if (typeof side !== 'object') return <span className="break-words text-slate-800">{String(side)}</span>;
  const id = side.evidence_id || side.finding_id;
  return (
    <span className="min-w-0">
      {side.text && <span className="block break-words text-slate-800">“{side.text}”</span>}
      <span className="text-[11px] text-slate-500">
        {side.stage_index != null && stageName(side.stage_index)}
        {side.stage_index != null && id ? ' · ' : ''}
        {id && <span className="rounded bg-slate-100 px-1 font-mono text-slate-700">{id}</span>}
      </span>
    </span>
  );
}

const fmtSimilarity = (v) => (isNum(v) ? v.toFixed(2) : '—');

/**
 * JourneyResponse.fusion: groups of stages that show the same page in different modalities
 * (declared with fuse_with). Renders nothing when there are no groups (older responses).
 */
export default function FusionCard({ fusion, stageName, headingId = 'j-fusion-heading' }) {
  const groups = Array.isArray(fusion) ? fusion.filter((g) => Array.isArray(g?.stage_indices)) : [];
  if (!groups.length) return null;
  return (
    <IntelCard
      id={headingId}
      icon={Combine}
      title="Merged same-page stages"
      subtitle="Stages declared as the same page in another format are merged, so one manipulation is not counted twice."
    >
      <ul className="space-y-3" data-testid="fusion-groups">
        {groups.map((g, gi) => {
          const pairs = Array.isArray(g.duplicate_pairs) ? g.duplicate_pairs : [];
          return (
            <li key={gi} className="rounded-lg border border-slate-200 p-3 sm:p-4">
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div className="min-w-0">
                  <p className="text-sm font-semibold text-slate-900">
                    {g.stage_indices.map((i) => stageName(i)).join(' + ')}
                  </p>
                  {isNum(g.representative) && (
                    <p className="mt-0.5 text-xs text-slate-600">
                      Representative stage (position used for FSPM): <b className="text-slate-800">{stageName(g.representative)}</b>
                    </p>
                  )}
                </div>
                <div className="shrink-0 rounded-lg bg-slate-50 px-3 py-1.5 text-right ring-1 ring-inset ring-slate-200">
                  <p className="text-[11px] font-bold uppercase tracking-wider text-slate-500">Fused pressure</p>
                  <p className="text-lg font-bold tabular-nums text-slate-900">{fmtShare(g.fused_pressure)}</p>
                </div>
              </div>
              {pairs.length > 0 ? (
                <div className="mt-3">
                  <p className="text-xs font-semibold text-slate-700">Duplicate findings ({pairs.length})</p>
                  <div className="mt-1 hidden grid-cols-[minmax(0,1fr)_minmax(0,1fr)_5rem] gap-3 bg-slate-50 px-3 py-2 text-[11px] font-semibold uppercase tracking-wide text-slate-500 sm:grid" aria-hidden="true">
                    <span>Evidence A</span><span>Evidence B</span><span className="text-right">Similarity</span>
                  </div>
                  <ul className="divide-y divide-slate-100 text-xs">
                    {pairs.map((p, pi) => (
                      <li key={pi} className="grid gap-1.5 px-3 py-2 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_5rem] sm:gap-3">
                        <PairSide side={p?.a} stageName={stageName} />
                        <PairSide side={p?.b} stageName={stageName} />
                        <span className="font-semibold tabular-nums text-slate-900 sm:text-right">
                          <span className="font-normal text-slate-500 sm:sr-only">Similarity </span>
                          {fmtSimilarity(p?.similarity)}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : (
                <p className="mt-2 text-xs text-slate-500">No duplicate finding pairs were reported for this group.</p>
              )}
              {g.note && <div className="mt-2"><MethodNote>{g.note}</MethodNote></div>}
            </li>
          );
        })}
      </ul>
    </IntelCard>
  );
}
