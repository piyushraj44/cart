import { useCallback, useMemo, useState } from 'react';
import { CheckCircle2, Combine, Download, EyeOff, FileJson, Layers3, Loader2, Pencil, Plus } from 'lucide-react';
import { exportJourneyPdf } from '../../api.js';
import { downloadBlob, formatDate, formatMs } from '../../format.js';
import Collapsible, { Chip } from '../Collapsible.jsx';
import { Notice, WarningsNotice } from '../ReportParts.jsx';
import { CdiCard, FspmCard, McaiCard, RedesignCard, TimelineCard } from '../intel/IntelligencePanel.jsx';
import { BareIntelCards, fmtIndex, fmtShare, isNum } from '../intel/shared.jsx';
import SyntheticBadge from './SyntheticBadge.jsx';
import StagePanel from './StagePanel.jsx';
import FusionCard from './FusionCard.jsx';
import ReplayPanel from './ReplayPanel.jsx';

/**
 * Journey results. `result` = { response: JourneyResponse, stageImages: {index: objectURL},
 * files?: {multipart field: File} (original screenshots, for replay), sample: {title, synthetic, provenance} | null,
 * isMock?: boolean }.
 */
export default function JourneyResults({ result, model, onNewJourney, onEditJourney }) {
  const r = result.response;
  const stages = useMemo(() => [...(r.stages || [])].sort((a, b) => a.index - b.index), [r.stages]);
  const warnings = r.warnings || [];
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState(null);
  const syntheticStages = stages.filter((s) => s.synthetic).length;
  const sample = result.sample;
  const showSynthetic = syntheticStages > 0 || !!sample?.synthetic;
  const fusionGroups = Array.isArray(r.fusion) ? r.fusion.filter((g) => Array.isArray(g?.stage_indices)) : [];

  // Top-level cards form one accordion group (FSPM open first); stage previews are a second group (stage 1 open).
  const [openId, setOpenId] = useState('fspm');
  const accordion = (key) => ({ open: openId === key, onOpenChange: (o) => setOpenId(o ? key : null) });
  const [openStage, setOpenStage] = useState(() => stages[0]?.index ?? null);
  const openRedesign = useCallback(() => setOpenId('redesign'), []);
  const findingFor = useCallback(
    (e) => stages.find((st) => st.index === e.stage_index)?.audit?.findings?.find((f) => f.id === e.target) || null,
    [stages],
  );

  const stageName = useCallback(
    (i) => {
      const st = stages.find((s) => s.index === i);
      return st ? `${i + 1}. ${st.name}` : `Stage ${i + 1}`;
    },
    [stages],
  );

  function downloadJson() {
    const blob = new Blob([JSON.stringify(r, null, 2)], { type: 'application/json' });
    downloadBlob(blob, `clearcart-journey-${r.journey_id}.json`);
  }

  async function downloadPdf() {
    setExporting(true);
    setExportError(null);
    try {
      const blob = await exportJourneyPdf(r);
      downloadBlob(blob, `clearcart-journey-${r.journey_id}.pdf`);
    } catch (err) {
      setExportError(err?.message || 'The PDF could not be created.');
    } finally {
      setExporting(false);
    }
  }

  return (
    <div className="space-y-4 sm:space-y-5">
      <p role="status" aria-live="polite" className="sr-only">
        Journey audit complete. {stages.length} stages audited.
      </p>

      <header className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div className="min-w-0">
          <h1 className="text-xl font-bold text-slate-900 sm:text-2xl">Journey Audit Report</h1>
          <p className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-sm text-slate-500">
            <span className="inline-flex items-center gap-1 font-medium text-slate-700">
              <Layers3 className="h-3.5 w-3.5 text-slate-500" aria-hidden="true" /> {stages.length} stage{stages.length === 1 ? '' : 's'}
            </span>
            <span aria-hidden="true">·</span>
            <span className="inline-flex items-center gap-1">
              <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" aria-hidden="true" /> Completed {formatDate(r.created_at)}
            </span>
            <span aria-hidden="true">·</span>
            <span>{formatMs(r.processing_ms)}</span>
          </p>
        </div>
        <div className="flex flex-wrap gap-2 lg:shrink-0">
          <button type="button" onClick={downloadJson} className="btn-secondary px-3 py-2">
            <FileJson className="h-4 w-4" aria-hidden="true" /> Download JSON
          </button>
          <button type="button" onClick={downloadPdf} disabled={exporting} className="btn-secondary px-3 py-2" aria-busy={exporting}>
            {exporting ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" /> : <Download className="h-4 w-4" aria-hidden="true" />}
            {exporting ? 'Preparing PDF…' : 'Download PDF'}
          </button>
          {onEditJourney && (
            <button type="button" onClick={onEditJourney} className="btn-secondary px-3 py-2">
              <Pencil className="h-4 w-4" aria-hidden="true" /> Edit journey
            </button>
          )}
          <button type="button" onClick={onNewJourney} className="btn-primary px-3 py-2">
            <Plus className="h-4 w-4" aria-hidden="true" /> New journey
          </button>
        </div>
      </header>

      {result.isMock && (
        <Notice tone="muted">
          <p className="text-xs">
            <b>Offline demo.</b> A saved real ClearCart journey audit of the synthetic demo journey, shown because the page was
            opened with ?mock=journey.
          </p>
        </Notice>
      )}

      {showSynthetic && (
        <div className="flex flex-col gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-950 sm:flex-row sm:items-center">
          <SyntheticBadge />
          <div className="min-w-0 text-xs leading-relaxed">
            {sample?.title && <span className="font-semibold">{sample.title}. </span>}
            {syntheticStages > 0 && (
              <span>
                {syntheticStages} of {stages.length} stage{stages.length === 1 ? '' : 's'} contain authored demonstration content, not a
                real merchant’s pages.{' '}
              </span>
            )}
            {sample?.provenance && <span title={sample.provenance} className="text-amber-900/80">Provenance: {sample.provenance}</span>}
          </div>
        </div>
      )}

      {exportError && (
        <Notice tone="error" role="alert">
          <p>PDF export failed: {exportError}</p>
        </Notice>
      )}
      <WarningsNotice warnings={warnings} />

      {/* 1. dashboard → 2. stage previews → 3. redesign, MCAI, FSPM, CDI → 4. details */}
      <JourneyDashboard response={r} stages={stages} />

      <section aria-labelledby="stages-heading" className="space-y-3">
        <h2 id="stages-heading" className="flex items-baseline gap-2 px-1 text-[15px] font-semibold text-slate-900">
          Stage previews <span className="text-sm font-normal text-slate-500">({stages.length})</span>
        </h2>
        {stages.map((st) => (
          <StagePanel
            key={st.index}
            stage={st}
            imageUrl={result.stageImages?.[st.index] || null}
            open={openStage === st.index}
            onOpenChange={(o) => setOpenStage(o ? st.index : null)}
            onOpenRedesign={openRedesign}
          />
        ))}
      </section>

      <RedesignCard cf={r.counterfactual} stageName={stageName} findingFor={findingFor} headingId="j-cf-heading" {...accordion('redesign')} />
      <McaiCard mcai={r.mcai} headingId="j-mcai-heading" {...accordion('mcai')} />
      <FspmCard fspm={r.fspm} stageName={stageName} headingId="j-fspm-heading" {...accordion('fspm')} />
      <CdiCard cdi={r.cdi} stageName={stageName} headingId="j-cdi-heading" {...accordion('cdi')} />

      <TimelineCard temporal={r.temporal} stages={stages} stageName={stageName} idPrefix="j-" {...accordion('timeline')} />

      {fusionGroups.length > 0 && (
        <Collapsible
          title="Merged same-page stages"
          icon={Combine}
          summary={<Chip>{fusionGroups.length} group{fusionGroups.length === 1 ? '' : 's'}</Chip>}
          {...accordion('fusion')}
        >
          <BareIntelCards>
            <FusionCard fusion={r.fusion} stageName={stageName} />
          </BareIntelCards>
        </Collapsible>
      )}

      <ReplayPanel response={r} files={result.files} stageName={stageName} {...accordion('repro')} />

      <footer className="space-y-1 pt-2">
        {r.config_version && <p className="text-xs text-slate-500">Configuration {r.config_version}</p>}
        {r.disclaimer && (
          <p className="flex items-start gap-2 text-xs leading-relaxed text-slate-500">
            <EyeOff className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
            {r.disclaimer}
          </p>
        )}
      </footer>
    </div>
  );
}

/* ------------------------------------------------------------------ dashboard */
function Stat({ label, value, sub, title }) {
  return (
    <div className="min-w-0 bg-white px-4 py-3 sm:px-5" title={title}>
      <dt className="truncate text-xs text-slate-500">{label}</dt>
      <dd className="mt-0.5 flex min-w-0 flex-wrap items-baseline gap-x-1.5">
        <span className="text-xl font-semibold tabular-nums text-slate-900">{value}</span>
        {sub && <span className="truncate text-xs text-slate-500">{sub}</span>}
      </dd>
    </div>
  );
}

/** One-card overview of a journey: FSPM headline, per-stage pressure, CDI, MCAI, stage and finding counts. */
function JourneyDashboard({ response: r, stages }) {
  const fspm = r.fspm;
  const scores = fspm?.scores || {};
  const fspmOk = fspm?.available && isNum(scores.fibonacci);
  const counted = stages.filter((st) => !Number.isInteger(st.fused_into));
  const merged = stages.length - counted.length;
  const findingCounts = counted.map((st) => st.audit?.summary?.suspicious_findings).filter(isNum);
  const totalFindings = findingCounts.length ? findingCounts.reduce((a, b) => a + b, 0) : null;
  const cdi = r.cdi;
  const band = cdi?.band ? String(cdi.band).toLowerCase() : null;
  const mcai = r.mcai;
  return (
    <section aria-labelledby="journey-dash-heading" className="card overflow-hidden">
      <h2 id="journey-dash-heading" className="sr-only">Journey overview</h2>
      <div className="grid gap-5 p-4 sm:grid-cols-[minmax(0,13rem)_minmax(0,1fr)] sm:gap-8 sm:p-5">
        <div className="min-w-0" title="Fibonacci weighting is a proposed weighting that emphasises later stages; equal and linear weighting are shown as baselines.">
          <p className="text-xs font-medium text-slate-500">Journey pressure · FSPM</p>
          {fspmOk ? (
            <>
              <p className="mt-1 tabular-nums text-slate-900">
                <span className="text-4xl font-bold">{fmtIndex(scores.fibonacci)}</span>
                <span className="text-sm text-slate-500"> /100</span>
              </p>
              <p className="mt-1 text-xs tabular-nums text-slate-500">
                linear {fmtIndex(scores.linear)} · equal {fmtIndex(scores.equal)}
              </p>
            </>
          ) : (
            <p className="mt-1 text-sm text-slate-500">{fspm?.reason || 'Not available for this journey.'}</p>
          )}
        </div>
        <StagePressureBars stages={stages} topIndex={fspm?.top_stage_index} />
      </div>
      <dl className="grid grid-cols-2 gap-px border-t border-slate-100 bg-slate-100 sm:grid-cols-4">
        <Stat
          label="Deception index (CDI)"
          value={isNum(cdi?.cdi) ? fmtIndex(cdi.cdi) : '—'}
          sub={isNum(cdi?.cdi) ? (band ? `${band} band` : undefined) : 'not computed'}
          title={cdi?.note || undefined}
        />
        <Stat
          label="Choice asymmetry (MCAI)"
          value={mcai?.available && isNum(mcai.mcai) ? fmtIndex(mcai.mcai) : '—'}
          sub={mcai?.available && isNum(mcai.mcai) ? 'worst case' : 'not available'}
          title={mcai?.reason || undefined}
        />
        <Stat label="Stages" value={stages.length} sub={merged ? `${merged} merged` : undefined} />
        <Stat
          label="Findings"
          value={totalFindings ?? '—'}
          sub={totalFindings != null ? `across ${counted.length} stage${counted.length === 1 ? '' : 's'}` : undefined}
          title={merged ? 'Stages merged into an earlier stage are not counted twice' : undefined}
        />
      </dl>
    </section>
  );
}

function StagePressureBars({ stages, topIndex }) {
  if (!stages.length) return null;
  return (
    <div className="min-w-0">
      <p className="text-xs font-medium text-slate-500">Pressure by stage</p>
      <ul className="mt-2 space-y-2" aria-label="Stage pressure">
        {stages.map((st) => {
          const ok = isNum(st.pressure);
          const top = st.index === topIndex;
          return (
            <li key={st.index} className="grid grid-cols-[minmax(0,1fr)_3.5rem] items-center gap-x-3 gap-y-1 sm:grid-cols-[minmax(0,13rem)_minmax(0,1fr)_3.5rem]">
              <span className="flex min-w-0 items-center gap-1.5 text-sm text-slate-800">
                <span className="min-w-0 truncate" title={st.name}>
                  <span className="tabular-nums text-slate-400">{st.index + 1}.</span> {st.name}
                </span>
                {top && <span className="shrink-0 rounded bg-teal-700 px-1 py-px text-[10px] font-bold uppercase text-white" title="Highest-pressure stage (by Fibonacci contribution)">Peak</span>}
              </span>
              <span className="col-span-2 row-start-2 h-1.5 overflow-hidden rounded-full bg-slate-100 sm:col-span-1 sm:col-start-2 sm:row-start-1" aria-hidden="true">
                {ok && <span className={`block h-full rounded-full ${top ? 'bg-teal-700' : 'bg-slate-400'}`} style={{ width: `${Math.max(0, Math.min(1, st.pressure)) * 100}%` }} />}
              </span>
              <span className="text-right text-xs font-semibold tabular-nums text-slate-700 sm:col-start-3 sm:row-start-1">{ok ? fmtShare(st.pressure, 0) : '—'}</span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
