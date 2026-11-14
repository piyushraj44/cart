import { ScanEye } from 'lucide-react';
import Collapsible, { Chip } from './Collapsible.jsx';
import { pct } from '../format.js';

export const VISUAL_CAVEAT =
  'Trained on screenshots that all contain dark patterns; indicates pattern type, not whether a page is deceptive.';

const boxText = (b) => (b ? `x ${b.x}, y ${b.y}, ${b.width}×${b.height} px` : 'no evidence region');

/**
 * EXPERIMENTAL, opt-in visual pattern-type signals (backend CLEARCART_VISUAL=1). Rendered only when the response carries
 * a `visual_signals` array; they are separate from findings and never change the risk summary.
 */
export default function VisualSignals({ signals, ...collapsibleProps }) {
  if (!Array.isArray(signals)) return null;
  return (
    <Collapsible
      title="Visual signals (experimental)"
      icon={ScanEye}
      summary={
        <>
          <Chip tone="amber">experimental</Chip>
          <Chip tone="muted">{signals.length ? `${signals.length} above threshold` : 'none above threshold'}</Chip>
        </>
      }
      {...collapsibleProps}
    >
      <div className="space-y-3">
        <p className="text-xs leading-relaxed text-slate-500">{VISUAL_CAVEAT}</p>
        {signals.length === 0 ? (
          <p className="text-sm text-slate-600">Nothing above the threshold.</p>
        ) : (
          <ul className="divide-y divide-slate-100">
            {signals.map((s, i) => (
              <li key={`${s.label}-${i}`} className="flex flex-col gap-1 py-2.5 first:pt-0 last:pb-0 sm:flex-row sm:items-center sm:gap-3">
                <span className="min-w-0 flex-1 break-words text-sm font-medium text-slate-900">{s.label}</span>
                <span className="flex flex-wrap items-center gap-1">
                  <Chip tone="value" title="Calibrated probability">{pct(s.probability)}</Chip>
                  <Chip tone="muted" title="Per-label threshold">threshold {pct(s.threshold)}</Chip>
                  <Chip tone="neutral" title="Grad-CAM evidence box in original image pixels">{boxText(s.bbox)}</Chip>
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </Collapsible>
  );
}
