import { useId, useState } from 'react';
import { ChevronDown } from 'lucide-react';

const CHIP_TONE = {
  neutral: 'bg-slate-100 text-slate-700',
  value: 'bg-slate-100 font-semibold text-slate-900',
  teal: 'bg-teal-50 text-teal-800 ring-1 ring-inset ring-teal-200',
  red: 'bg-red-50 text-red-800 ring-1 ring-inset ring-red-200',
  orange: 'bg-orange-50 text-orange-900 ring-1 ring-inset ring-orange-200',
  amber: 'bg-amber-50 text-amber-900 ring-1 ring-inset ring-amber-200',
  green: 'bg-emerald-50 text-emerald-800 ring-1 ring-inset ring-emerald-200',
  muted: 'bg-white text-slate-500 ring-1 ring-inset ring-slate-200',
};

/** Small summary chip used in collapsible headers and the dashboards. */
export function Chip({ tone = 'neutral', title, children }) {
  return (
    <span
      title={title}
      className={`inline-flex max-w-full items-center gap-1 truncate rounded-full px-2 py-0.5 text-xs font-medium tabular-nums ${CHIP_TONE[tone] || CHIP_TONE.neutral}`}
    >
      {children}
    </span>
  );
}

/**
 * Accessible disclosure card (WAI-ARIA accordion pattern: a heading wrapping a button with
 * aria-expanded / aria-controls). The header always shows the title and `summary` chips so the
 * page stays scannable when collapsed.
 *
 * Content is rendered but `hidden` while collapsed, so it is present in server-side renders and
 * keeps its state; pass `lazy` to mount the content only after the first open (heavy previews).
 * Controlled when `open` + `onOpenChange` are given.
 */
export default function Collapsible({
  title,
  icon: Icon,
  summary,
  defaultOpen = false,
  open: openProp,
  onOpenChange,
  lazy = false,
  variant = 'card',
  headingLevel = 2,
  id,
  leading,
  className = '',
  bodyClassName = '',
  children,
}) {
  const uid = useId();
  const [openState, setOpenState] = useState(defaultOpen);
  const controlled = openProp !== undefined;
  const open = controlled ? openProp : openState;
  const [visited, setVisited] = useState(open);
  if (open && !visited) setVisited(true);

  const buttonId = `${uid}-button`;
  const panelId = `${uid}-panel`;
  const H = `h${Math.min(6, Math.max(2, headingLevel))}`;

  function toggle() {
    const next = !open;
    if (!controlled) setOpenState(next);
    onOpenChange?.(next);
  }

  const shell = variant === 'plain' ? 'rounded-lg border border-slate-200 bg-white' : 'card';

  return (
    <section id={id} className={`${shell} overflow-hidden ${className}`} data-collapsible={open ? 'open' : 'closed'}>
      <H className="m-0">
        <button
          id={buttonId}
          type="button"
          aria-expanded={open}
          aria-controls={panelId}
          onClick={toggle}
          className="group flex w-full items-center gap-3 px-4 py-3 text-left transition-colors hover:bg-slate-50 focus-visible:rounded-none focus-visible:ring-inset focus-visible:ring-offset-0 sm:px-5"
        >
          {leading}
          {Icon && <Icon className="h-4 w-4 shrink-0 text-teal-700" aria-hidden="true" />}
          <span className="flex min-w-0 flex-1 flex-col gap-1 sm:flex-row sm:flex-wrap sm:items-center sm:gap-x-3">
            <span className={`min-w-0 font-semibold text-slate-900 ${variant === 'plain' ? 'text-sm' : 'text-[15px]'}`}>{title}</span>
            {summary && <span className="flex min-w-0 flex-wrap items-center gap-1.5">{summary}</span>}
          </span>
          <ChevronDown
            className={`h-4 w-4 shrink-0 text-slate-400 transition-transform duration-200 group-hover:text-slate-600 ${open ? 'rotate-180' : ''}`}
            aria-hidden="true"
          />
        </button>
      </H>
      <div id={panelId} role="region" aria-labelledby={buttonId} hidden={!open}>
        {(!lazy || visited) && (
          <div className={`border-t border-slate-100 px-4 py-4 motion-safe:animate-collapsible-in sm:px-5 sm:py-5 ${bodyClassName}`}>
            {children}
          </div>
        )}
      </div>
    </section>
  );
}
