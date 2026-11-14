import { useId, useState } from 'react';
import { ArrowDown, ArrowRight, ShieldAlert, Sparkles, Wand2 } from 'lucide-react';
import { categoryStyle, humanKey } from '../../format.js';
import { IntelCard, fmtIndex, isNum } from './shared.jsx';

const CATEGORY_LABEL = {
  urgency: 'Misleading urgency',
  scarcity: 'Misleading scarcity',
  social_proof: 'Social proof pressure',
  misdirection: 'Misdirection / confirmshaming',
  preselected_option: 'Preselected option',
  choice_asymmetry: 'Choice asymmetry',
};

const ACTION_LABEL = {
  rewrite: 'Rewrite',
  remove_unverifiable_claim: 'Remove claim',
  uncheck_default: 'Uncheck default',
  equalise_choice: 'Equal choice',
  no_safe_rewrite: 'No safe rewrite',
};

const CHOICE_RECOMMENDATION = 'Give accept and decline equal visual weight and neutral wording.';
const UNCHECK_PREFIX = /^\s*\[unchecked by default\]\s*/i;

/** DOM id of the redesign card for a finding (journeys prefix the stage: `redesign-s<stage>-<id>`). */
export function redesignId(target, stageIndex) {
  return `redesign-${stageIndex != null ? `s${stageIndex}-` : ''}${target}`;
}

/** What the AFTER box says for an edit. */
function afterContent(e) {
  if (e.action === 'remove_unverifiable_claim') return { text: '(remove this claim)', muted: true };
  if (e.action === 'uncheck_default') {
    const label = String(e.redesigned ?? e.original ?? '').replace(UNCHECK_PREFIX, '') || e.original || '';
    return { text: label, note: 'unchecked by default' };
  }
  if (e.action === 'no_safe_rewrite' || e.redesigned == null || !String(e.redesigned).trim()) {
    return { text: 'No safe automatic rewrite — follow the recommendation', muted: true };
  }
  if (e.action === 'equalise_choice' && e.redesigned === e.original) {
    return { text: e.redesigned, note: 'same weight as accept' };
  }
  return { text: e.redesigned };
}

/**
 * BEFORE → AFTER as two white boxes. `compact` (resting card): side by side from md up with the arrow centred
 * between them, each clamped to 3 lines and stretched to equal height. Otherwise (reason overlay): stacked, full text.
 */
function BeforeAfter({ before, after, compact }) {
  const lc = compact ? 'line-clamp-3' : '';
  const box = 'flex h-full min-w-0 flex-col rounded-lg bg-white p-2.5 ring-1 ring-inset ring-emerald-100';
  const cap = 'text-[10px] font-semibold uppercase tracking-wider text-slate-500';
  return (
    <div className={`grid h-full grid-cols-1 items-stretch gap-1.5 ${compact ? 'md:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)]' : ''}`}>
      <div className={box}>
        <p className={cap}>Before</p>
        <p className={`mt-1 whitespace-pre-wrap break-words text-sm text-slate-500 line-through decoration-red-400 ${lc}`}>{before}</p>
      </div>
      <span className="flex items-center justify-center" aria-hidden="true">
        <ArrowDown className={`h-4 w-4 text-emerald-600 ${compact ? 'md:hidden' : ''}`} />
        {compact && <ArrowRight className="hidden h-4 w-4 text-emerald-600 md:block" />}
      </span>
      <div className={box}>
        <p className={cap}>After</p>
        <p className={`mt-1 whitespace-pre-wrap break-words text-sm ${after.muted ? 'italic text-slate-600' : 'font-medium text-slate-900'} ${lc}`}>
          {after.text}
          {after.note && <> <span className="text-xs font-normal italic text-emerald-800">({after.note})</span></>}
        </p>
      </div>
    </div>
  );
}

/** Sparkles + category title, as in the original "Ethical redesign" box. */
function RedesignTitle({ label, className = '' }) {
  return (
    <span className={`flex min-w-0 items-center gap-1.5 text-sm font-semibold text-teal-800 ${className}`}>
      <Sparkles className="h-4 w-4 shrink-0 text-emerald-600" aria-hidden="true" />
      <span className="sr-only">Ethical redesign: </span>
      <span className="min-w-0 truncate">{label}</span>
    </span>
  );
}

/**
 * One edit as an equal-height grid card in the green "Ethical redesign" style: Sparkles + category, action chip,
 * then BEFORE → AFTER. Hover (pointer devices), keyboard focus, a tap/Enter/Space toggle, or a jump highlight
 * (`data-flash` on the <li>) reveals an overlay with the full recommendation and the un-clamped text.
 */
function RedesignEditCard({ e, finding, stageName, domId }) {
  const uid = useId();
  const [open, setOpen] = useState(false);
  const label = finding?.category_label || CATEGORY_LABEL[e.category] || humanKey(e.category || 'redesign');
  const action = ACTION_LABEL[e.action] || humanKey(e.action || 'edit');
  const recommendation = finding?.ethical_recommendation || (e.action === 'equalise_choice' ? CHOICE_RECOMMENDATION : null);
  const after = afterContent(e);
  const style = categoryStyle(e.category);
  const stage = e.stage_index != null && stageName ? stageName(e.stage_index) : null;
  const reasonId = `${uid}-reason`;

  function onKeyDown(ev) {
    if (ev.key === 'Enter' || ev.key === ' ') {
      ev.preventDefault();
      setOpen((o) => !o);
    } else if (ev.key === 'Escape') {
      setOpen(false);
    }
  }

  return (
    <li
      id={domId}
      tabIndex={domId ? -1 : undefined}
      data-redesign-card=""
      className="group/card relative h-full min-w-0 scroll-mt-28 rounded-xl focus:outline-none hover:z-30 focus-within:z-30 data-[flash]:z-30 data-[flash]:ring-2 data-[flash]:ring-teal-600 data-[flash]:ring-offset-2"
    >
      <div
        role="button"
        tabIndex={0}
        aria-expanded={open}
        aria-describedby={reasonId}
        data-open={open ? 'true' : 'false'}
        onClick={() => setOpen((o) => !o)}
        onKeyDown={onKeyDown}
        onBlur={(ev) => {
          if (!ev.currentTarget.contains(ev.relatedTarget)) setOpen(false);
        }}
        className="group/btn relative flex h-full cursor-pointer flex-col rounded-xl border border-emerald-200 bg-emerald-50/50 p-4 pl-5 text-left shadow-sm transition duration-200 ease-out focus:outline-none focus-visible:ring-2 focus-visible:ring-teal-600 [@media(hover:hover)]:group-hover/card:border-emerald-300 [@media(hover:hover)]:group-hover/card:shadow-md motion-safe:[@media(hover:hover)]:group-hover/card:-translate-y-0.5 focus-visible:border-emerald-300 focus-visible:shadow-md motion-safe:focus-visible:-translate-y-0.5 data-[open=true]:border-emerald-300 data-[open=true]:shadow-md"
      >
        {/* subtle category accent */}
        <span className={`absolute inset-y-4 left-0 w-1 rounded-r-full opacity-70 ${style.dot}`} aria-hidden="true" />
        <div className="flex items-center gap-2">
          <RedesignTitle label={label} className="flex-1" />
          {e.facts_preserved === false && (
            <span className="inline-flex shrink-0 items-center gap-1 rounded-full bg-amber-50 px-2 py-0.5 text-[11px] font-semibold text-amber-900 ring-1 ring-inset ring-amber-300" title="Facts may have changed — review">
              <ShieldAlert className="h-3 w-3" aria-hidden="true" /> Review facts
            </span>
          )}
          <span className="shrink-0 rounded-full bg-white px-2 py-0.5 text-[11px] font-semibold text-emerald-800 ring-1 ring-inset ring-emerald-200">
            {action}
          </span>
        </div>
        {stage && <p className="mt-0.5 truncate pl-[1.375rem] text-[11px] text-slate-500">{stage}</p>}
        <div className="mt-3 flex-1">
          <BeforeAfter before={e.original} after={after} compact />
        </div>

        {/* reason overlay */}
        <div
          className="invisible absolute inset-x-0 top-0 z-10 min-h-full max-h-[min(28rem,75vh)] overflow-y-auto rounded-xl border border-emerald-300 bg-emerald-50 p-4 opacity-0 shadow-xl transition-opacity duration-200 motion-reduce:transition-none [@media(hover:hover)]:group-hover/card:visible [@media(hover:hover)]:group-hover/card:opacity-100 group-focus-visible/btn:visible group-focus-visible/btn:opacity-100 group-data-[open=true]/btn:visible group-data-[open=true]/btn:opacity-100 group-data-[flash]/card:visible group-data-[flash]/card:opacity-100"
          data-testid="redesign-reason"
        >
          <p className="flex items-start gap-1.5 text-sm font-semibold text-teal-800">
            <Sparkles className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" aria-hidden="true" />
            <span className="min-w-0">Ethical redesign · {label}</span>
          </p>
          {stage && <p className="mt-0.5 pl-[1.375rem] text-[11px] text-slate-500">{stage}</p>}
          {recommendation ? (
            <p id={reasonId} className="mt-2 text-sm leading-relaxed text-emerald-950">{recommendation}</p>
          ) : (
            <p id={reasonId} className="mt-2 text-sm text-slate-600">{action}.</p>
          )}
          <div className="mt-3">
            <BeforeAfter before={e.original} after={after} />
          </div>
        </div>
      </div>
    </li>
  );
}

/**
 * Counterfactual ethical redesign. `cf` is a CounterfactualResult; `findingFor(edit)` returns the finding the edit
 * targets (for its category label and ethical_recommendation), or null for choice-set edits.
 */
export default function CounterfactualSection({ cf, stageName, findingFor, headingId = 'cf-heading' }) {
  if (!cf) return null;
  const edits = cf.edits || [];
  const hasCdi = isNum(cf.original_cdi) && isNum(cf.redesigned_cdi);
  const hasPct = isNum(cf.risk_reduction_pct);
  const usedIds = new Set();

  return (
    <IntelCard id={headingId} icon={Wand2} title="Counterfactual Ethical Redesign">
      {edits.length === 0 ? (
        <p className="rounded-lg bg-slate-50 p-3 text-sm text-slate-600 ring-1 ring-inset ring-slate-200">
          No redesign edits were proposed for this input.
        </p>
      ) : (
        <>
          <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
            {(hasCdi || hasPct) && (
              <p className="text-sm text-slate-700 tabular-nums" title={cf.label || undefined} data-testid="cf-summary">
                {hasCdi && (
                  <>
                    CDI <b className="font-semibold text-slate-900">{fmtIndex(cf.original_cdi)}</b>
                    {' '}<span aria-hidden="true">→</span><span className="sr-only">to</span>{' '}
                    <b className="font-semibold text-slate-900">{fmtIndex(cf.redesigned_cdi)}</b>
                  </>
                )}
                {hasCdi && hasPct && <span aria-hidden="true"> · </span>}
                {hasPct && (
                  <>
                    <b className="font-semibold text-emerald-800">{cf.risk_reduction_pct.toFixed(1)}% lower</b> (model-estimated)
                  </>
                )}
              </p>
            )}
            <p className="text-xs text-slate-500">Hover or tap a card for the reason</p>
          </div>
          <ul className="mt-3 grid grid-cols-1 gap-3 md:auto-rows-fr md:grid-cols-2 xl:grid-cols-3">
            {edits.map((e, i) => {
              const id = e.target != null ? redesignId(e.target, e.stage_index) : null;
              const domId = id && !usedIds.has(id) ? id : undefined;
              if (domId) usedIds.add(domId);
              return (
                <RedesignEditCard
                  key={`${e.stage_index ?? ''}-${e.target}-${i}`}
                  e={e}
                  finding={findingFor?.(e) || null}
                  stageName={stageName}
                  domId={domId}
                />
              );
            })}
          </ul>
        </>
      )}
    </IntelCard>
  );
}
