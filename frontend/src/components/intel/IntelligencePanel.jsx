import { History, Layers, Scale, TrendingUp, Wand2 } from 'lucide-react';
import Collapsible, { Chip } from '../Collapsible.jsx';
import CDISection from './CDISection.jsx';
import MCAISection from './MCAISection.jsx';
import CounterfactualSection from './CounterfactualSection.jsx';
import FSPMSection from './FSPMSection.jsx';
import TimelineSection, { PressureProgression } from './TimelineSection.jsx';
import { BareIntelCards, fmtIndex, fmtShare, humanKeyLabel, isNum } from './shared.jsx';

/* ------------------------------------------------------------------ header summaries (always API values) */

export const BAND_TONE = { high: 'red', elevated: 'orange', moderate: 'amber', low: 'neutral' };
const DIM_LABEL = { visual: 'visual', interaction: 'interaction', semantic: 'semantic' };

function CfSummary({ cf }) {
  const edits = cf?.edits || [];
  if (!edits.length) return <Chip tone="muted">No changes needed</Chip>;
  return (
    <>
      {(isNum(cf.original_cdi) || isNum(cf.redesigned_cdi)) && (
        <Chip tone="value" title="CDI now → CDI if every edit were applied">
          CDI {fmtIndex(cf.original_cdi)} → {fmtIndex(cf.redesigned_cdi)}
        </Chip>
      )}
      {isNum(cf.risk_reduction_pct) && (
        <Chip tone="green" title={cf.label || 'Model-estimated reduction'}>
          {cf.risk_reduction_pct.toFixed(1)}% lower (est.)
        </Chip>
      )}
      <Chip>{edits.length} edit{edits.length === 1 ? '' : 's'}</Chip>
    </>
  );
}

function McaiSummary({ mcai }) {
  const sets = mcai?.choice_sets || [];
  if (!mcai?.available || !sets.length) return <Chip tone="muted" title={mcai?.reason || undefined}>Not available</Chip>;
  const worst = sets.filter((cs) => isNum(cs.mcai)).sort((a, b) => b.mcai - a.mcai)[0];
  const top = worst?.top_dimension;
  return (
    <>
      {isNum(mcai.mcai) && <Chip tone="value" title="Worst-case choice asymmetry">{fmtIndex(mcai.mcai)}/100</Chip>}
      {top && <Chip>largest: {DIM_LABEL[top] || humanKeyLabel(top).toLowerCase()}</Chip>}
      <Chip>{sets.length} choice{sets.length === 1 ? '' : 's'}</Chip>
    </>
  );
}

function FspmSummary({ fspm }) {
  if (!fspm?.available) return <Chip tone="muted" title={fspm?.reason || undefined}>Journey only</Chip>;
  const s = fspm.scores || {};
  const top = (fspm.stages || []).find((st) => st.stage_index === fspm.top_stage_index);
  return (
    <>
      {isNum(s.fibonacci) && <Chip tone="value" title="Fibonacci-weighted FSPM score">{fmtIndex(s.fibonacci)}/100</Chip>}
      {(isNum(s.linear) || isNum(s.equal)) && (
        <Chip title="Baseline weightings">linear {fmtIndex(s.linear)} · equal {fmtIndex(s.equal)}</Chip>
      )}
      {top && <Chip title="Highest-pressure stage by Fibonacci contribution">peak: {top.name}</Chip>}
    </>
  );
}

function CdiSummary({ cdi }) {
  if (!isNum(cdi?.cdi)) return <Chip tone="muted">Not computed</Chip>;
  const band = cdi.band ? String(cdi.band).toLowerCase() : null;
  const level = cdi.confidence?.level ? String(cdi.confidence.level).toLowerCase() : null;
  return (
    <>
      <Chip tone="value">{fmtIndex(cdi.cdi)}/100</Chip>
      {band && <Chip tone={BAND_TONE[band] || 'neutral'}>{band}</Chip>}
      {level && <Chip title={cdi.confidence?.note || undefined}>confidence {level}</Chip>}
      {isNum(cdi.coverage) && <Chip title="Share of the configured weight that had data">coverage {fmtShare(cdi.coverage, 0)}</Chip>}
    </>
  );
}

function TimelineSummary({ temporal }) {
  if (!temporal?.available) return <Chip tone="muted" title={temporal?.reason || undefined}>Journey only</Chip>;
  const events = temporal.events || [];
  const signals = events.filter((e) => e.deceptive_signal).length;
  return (
    <>
      <Chip>{events.length} event{events.length === 1 ? '' : 's'}</Chip>
      {signals > 0 && <Chip tone="red">{signals} deceptive signal{signals === 1 ? '' : 's'}</Chip>}
    </>
  );
}

/* ------------------------------------------------------------------ section cards */

/** Ethical redesign (counterfactual). */
export function RedesignCard({ cf, stageName, findingFor, defaultOpen = true, open, onOpenChange, headingId }) {
  if (!cf) return null;
  return (
    <Collapsible
      id="ethical-redesign"
      // visible overflow: a card's reason overlay may extend past the last row, and a clipping box would also
      // become a scroll container that focus/scrollIntoView can shift; the header keeps its rounded hover corners
      className="!overflow-visible [&>h2>button]:rounded-t-xl [&[data-collapsible=closed]>h2>button]:rounded-b-xl"
      title="Counterfactual Ethical Redesign" icon={Wand2} summary={<CfSummary cf={cf} />} defaultOpen={defaultOpen} open={open} onOpenChange={onOpenChange}>
      <BareIntelCards>
        <CounterfactualSection cf={cf} stageName={stageName} findingFor={findingFor} headingId={headingId} />
      </BareIntelCards>
    </Collapsible>
  );
}

export function McaiCard({ mcai, defaultOpen = false, open, onOpenChange, headingId }) {
  if (!mcai) return null;
  return (
    <Collapsible title="Choice Asymmetry (MCAI)" icon={Scale} summary={<McaiSummary mcai={mcai} />} defaultOpen={defaultOpen} open={open} onOpenChange={onOpenChange}>
      <BareIntelCards>
        <MCAISection mcai={mcai} headingId={headingId} />
      </BareIntelCards>
    </Collapsible>
  );
}

export function FspmCard({ fspm, stageName, onRunJourney, defaultOpen = false, open, onOpenChange, headingId }) {
  if (!fspm) return null;
  return (
    <Collapsible title="Fibonacci Pressure Analysis (FSPM)" icon={TrendingUp} summary={<FspmSummary fspm={fspm} />} defaultOpen={defaultOpen} open={open} onOpenChange={onOpenChange}>
      <BareIntelCards>
        <FSPMSection fspm={fspm} stageName={stageName} onRunJourney={onRunJourney} headingId={headingId} />
      </BareIntelCards>
    </Collapsible>
  );
}

export function CdiCard({ cdi, stageName, defaultOpen = false, open, onOpenChange, headingId }) {
  if (!cdi) return null;
  return (
    <Collapsible title="Deception Intelligence Score (CDI)" icon={Layers} summary={<CdiSummary cdi={cdi} />} defaultOpen={defaultOpen} open={open} onOpenChange={onOpenChange}>
      <BareIntelCards>
        <CDISection cdi={cdi} stageName={stageName} headingId={headingId} />
      </BareIntelCards>
    </Collapsible>
  );
}

/** Pressure progression + temporal evidence timeline; renders nothing unless the API marks it available. */
export function TimelineCard({ temporal, stages, stageName, defaultOpen = false, open, onOpenChange, idPrefix = '' }) {
  if (!temporal?.available) return null;
  return (
    <Collapsible title="Temporal Evidence Timeline" icon={History} summary={<TimelineSummary temporal={temporal} />} defaultOpen={defaultOpen} open={open} onOpenChange={onOpenChange}>
      <BareIntelCards showTitles>
        <PressureProgression temporal={temporal} stages={stages} headingId={`${idPrefix}progression-heading`} />
      </BareIntelCards>
      <BareIntelCards>
        <div className="mt-6 border-t border-slate-100 pt-5">
          <TimelineSection temporal={temporal} stageName={stageName} headingId={`${idPrefix}timeline-heading`} />
        </div>
      </BareIntelCards>
    </Collapsible>
  );
}

/**
 * Intelligence cards of a single-page audit, in reading order: ethical redesign, MCAI, FSPM, CDI.
 * `intelligence` is AuditResponse.intelligence and may be missing on results produced before intel-v1.0.
 * `accordion(key)` returns the controlled { open, onOpenChange } props of the page's one-open-at-a-time group.
 */
export default function IntelligencePanel({ intelligence, findings = [], onRunJourney, accordion }) {
  if (!intelligence) {
    return (
      <p className="card px-4 py-3 text-sm text-slate-600 sm:px-5">
        This result does not include intelligence indices (it was produced by a server version without them). Run the audit again
        to get the composite score, choice asymmetry and redesign analysis.
      </p>
    );
  }
  const acc = (key) => accordion?.(key) || {};
  const findingFor = (e) => findings.find((f) => f.id === e.target) || null;
  return (
    <>
      <RedesignCard cf={intelligence.counterfactual} findingFor={findingFor} {...acc('redesign')} />
      <McaiCard mcai={intelligence.mcai} {...acc('mcai')} />
      <FspmCard fspm={intelligence.fspm} onRunJourney={onRunJourney} {...acc('fspm')} />
      <CdiCard cdi={intelligence.cdi} {...acc('cdi')} />
    </>
  );
}
