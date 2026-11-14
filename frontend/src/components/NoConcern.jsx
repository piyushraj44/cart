import { CheckCircle2, ChevronDown } from 'lucide-react';
import { pct } from '../format.js';

/** Plain list of analysed, unflagged elements with their dark-pattern score. */
export function BenignItems({ elements }) {
  if (!elements || elements.length === 0) return null;
  return (
    <ul className="divide-y divide-slate-100">
      {elements.map((el, i) => (
        <li key={i} className="flex items-start justify-between gap-4 py-2.5 first:pt-0 last:pb-0">
          <p className="min-w-0 break-words text-sm text-slate-700">{el.text}</p>
          <span className="shrink-0 rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium tabular-nums text-slate-700">
            {pct(el.p_dark)} dark
          </span>
        </li>
      ))}
    </ul>
  );
}

/** Collapsible list of elements that were analysed and not flagged. */
export function BenignList({ elements, title = 'Elements analysed and not flagged', defaultOpen = false }) {
  if (!elements || elements.length === 0) return null;
  return (
    <details className="group rounded-lg border border-slate-200 bg-white" open={defaultOpen}>
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-4 py-3 text-sm font-semibold text-slate-800 hover:bg-slate-50 [&::-webkit-details-marker]:hidden">
        <span>
          {title} <span className="font-normal text-slate-500">({elements.length})</span>
        </span>
        <ChevronDown className="h-4 w-4 shrink-0 text-slate-500 transition-transform group-open:rotate-180" aria-hidden="true" />
      </summary>
      <ul className="divide-y divide-slate-100 border-t border-slate-100">
        {elements.map((el, i) => (
          <li key={i} className="flex items-start justify-between gap-4 px-4 py-3">
            <p className="min-w-0 break-words text-sm text-slate-700">{el.text}</p>
            <span className="shrink-0 rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium tabular-nums text-slate-700">
              {pct(el.p_dark)} dark
            </span>
          </li>
        ))}
      </ul>
    </details>
  );
}

/** Shown when the audit found nothing above the flagging threshold. */
export default function NoConcern({ summary, benign }) {
  const n = summary.elements_analysed;
  return (
    <section
      aria-labelledby="no-concern-title"
      className="rounded-xl border border-emerald-200 bg-emerald-50 p-5 sm:p-6"
    >
      <div className="flex items-start gap-4">
        <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-emerald-100 text-emerald-700">
          <CheckCircle2 className="h-6 w-6" aria-hidden="true" />
        </span>
        <div className="min-w-0">
          <h2 id="no-concern-title" className="text-xl font-bold text-emerald-900">
            No suspicious patterns detected
          </h2>
          <p className="mt-1 text-sm text-emerald-900">
            ClearCart analysed {n} {n === 1 ? 'element' : 'elements'} and none crossed the flagging threshold.
          </p>
          <p className="mt-2 text-xs text-emerald-900/75">
            This is a screening result. It does not guarantee the page is free of deceptive design.
          </p>
        </div>
      </div>
      {benign?.length > 0 && (
        <div className="mt-5">
          <BenignList elements={benign} />
        </div>
      )}
    </section>
  );
}
