// Small pieces shared by the single-audit and journey result views.
import { useCallback, useEffect, useState } from 'react';
import { AlertTriangle, Info } from 'lucide-react';

/** Compact note banners (warnings, export errors, demo notice) shared by the result views. */
export function Notice({ tone = 'info', role, children }) {
  const cls = {
    info: 'border-sky-200 bg-sky-50 text-sky-950',
    error: 'border-red-200 bg-red-50 text-red-900',
    muted: 'border-slate-200 bg-slate-100 text-slate-600',
  }[tone];
  const Icon = tone === 'error' ? AlertTriangle : Info;
  return (
    <div role={role} className={`flex items-start gap-2.5 rounded-lg border px-3 py-2 text-sm ${cls}`}>
      <Icon className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
      <div className="min-w-0">{children}</div>
    </div>
  );
}

export function WarningsNotice({ warnings }) {
  if (!warnings?.length) return null;
  return (
    <Notice role="note">
      {warnings.length === 1 ? (
        <p>{warnings[0]}</p>
      ) : (
        <ul className="list-disc space-y-0.5 pl-4">
          {warnings.map((w, i) => <li key={i}>{w}</li>)}
        </ul>
      )}
    </Notice>
  );
}

const FLASH_MS = 2200;

/** Briefly marks an element (`data-flash`, styled by the element itself) without touching React-managed attributes. */
function flash(el) {
  el.setAttribute('data-flash', '');
  clearTimeout(el.__flashTimer);
  el.__flashTimer = setTimeout(() => el.removeAttribute('data-flash'), FLASH_MS);
}

/**
 * Selecting a numbered box in an annotated preview: briefly highlights the box (`activeId`) and, when the finding has
 * a redesign card (`redesign-<prefix><id>`), opens the Ethical redesign section via `openRedesign`, then scrolls to
 * that card and highlights it.
 */
export function useJumpToFinding(prefix = '', openRedesign) {
  const [activeId, setActiveId] = useState(null);

  // Clear the box highlight after a moment so it reads as a jump target, not a permanent state.
  useEffect(() => {
    if (!activeId) return undefined;
    const t = setTimeout(() => setActiveId(null), FLASH_MS);
    return () => clearTimeout(t);
  }, [activeId]);

  const jumpTo = useCallback(
    (id) => {
      setActiveId(id);
      if (!document.getElementById(`redesign-${prefix}${id}`)) return;
      openRedesign?.();
      // wait for the section to un-hide (React commits the click's update before this timer runs)
      setTimeout(() => {
        const el = document.getElementById(`redesign-${prefix}${id}`);
        if (!el) return;
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        el.focus({ preventScroll: true });
        flash(el);
      }, 0);
    },
    [prefix, openRedesign],
  );
  return { activeId, jumpTo };
}
