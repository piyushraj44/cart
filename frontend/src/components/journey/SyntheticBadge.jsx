import { FlaskConical } from 'lucide-react';

/** Visible marker for authored demo content (never real merchant data). */
export default function SyntheticBadge({ label = 'Synthetic demonstration journey' }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-100 px-2.5 py-1 text-xs font-semibold text-amber-900 ring-1 ring-inset ring-amber-300">
      <FlaskConical className="h-3.5 w-3.5" aria-hidden="true" />
      {label}
    </span>
  );
}
