import { AlertTriangle, AlertCircle, Info, CheckCircle2 } from 'lucide-react';

const LEVELS = {
  high: { label: 'High risk', cls: 'bg-red-50 text-red-900 ring-red-300', Icon: AlertTriangle },
  medium: { label: 'Medium risk', cls: 'bg-amber-50 text-amber-900 ring-amber-300', Icon: AlertCircle },
  low: { label: 'Low risk', cls: 'bg-yellow-50 text-yellow-900 ring-yellow-300', Icon: Info },
  none: { label: 'No concern', cls: 'bg-emerald-50 text-emerald-900 ring-emerald-300', Icon: CheckCircle2 },
};

/** Risk level badge. Colour is supplemented by an icon and text label. */
export default function RiskBadge({ level, size = 'md' }) {
  const cfg = LEVELS[level] || {
    label: `Risk: ${level || 'unknown'}`,
    cls: 'bg-slate-100 text-slate-800 ring-slate-300',
    Icon: Info,
  };
  const { Icon } = cfg;
  const sizing = size === 'lg' ? 'px-3 py-1.5 text-sm' : 'px-2.5 py-1 text-xs';
  const iconSize = size === 'lg' ? 'h-4 w-4' : 'h-3.5 w-3.5';
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full font-semibold ring-1 ring-inset ${sizing} ${cfg.cls}`}>
      <Icon className={iconSize} aria-hidden="true" />
      {cfg.label}
    </span>
  );
}
