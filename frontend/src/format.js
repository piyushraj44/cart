// Shared constants and small formatting helpers used across components.

export const LIMITS = {
  textChars: 200000,
  htmlBytes: 2 * 1024 * 1024,
  imageBytes: 8 * 1024 * 1024,
};

// Tailwind class names must appear literally in source so the JIT can find them.
export const CATEGORY_STYLES = {
  urgency: {
    dot: 'bg-red-600', chip: 'bg-red-50 text-red-900 ring-red-200', mark: 'bg-red-200',
    bar: 'bg-red-600', box: 'border-red-600 bg-red-500/15', tag: 'bg-red-700',
  },
  scarcity: {
    dot: 'bg-orange-600', chip: 'bg-orange-50 text-orange-900 ring-orange-200', mark: 'bg-orange-200',
    bar: 'bg-orange-600', box: 'border-orange-600 bg-orange-500/15', tag: 'bg-orange-700',
  },
  social_proof: {
    dot: 'bg-violet-600', chip: 'bg-violet-50 text-violet-900 ring-violet-200', mark: 'bg-violet-200',
    bar: 'bg-violet-600', box: 'border-violet-600 bg-violet-500/15', tag: 'bg-violet-700',
  },
  misdirection: {
    dot: 'bg-amber-600', chip: 'bg-amber-50 text-amber-900 ring-amber-200', mark: 'bg-amber-200',
    bar: 'bg-amber-600', box: 'border-amber-600 bg-amber-500/15', tag: 'bg-amber-700',
  },
  preselected_option: {
    dot: 'bg-sky-700', chip: 'bg-sky-50 text-sky-900 ring-sky-200', mark: 'bg-sky-200',
    bar: 'bg-sky-700', box: 'border-sky-700 bg-sky-500/15', tag: 'bg-sky-800',
  },
};

const FALLBACK_STYLE = {
  dot: 'bg-slate-500', chip: 'bg-slate-100 text-slate-800 ring-slate-200', mark: 'bg-slate-200',
  bar: 'bg-slate-500', box: 'border-slate-600 bg-slate-500/15', tag: 'bg-slate-700',
};

export function categoryStyle(key) {
  return CATEGORY_STYLES[key] || FALLBACK_STYLE;
}

export function pct(value) {
  if (value == null || Number.isNaN(Number(value))) return '—';
  // floor + cap at 99 to match the backend's confidence_pct: a model score is never "100% sure"
  return `${Math.min(99, Math.floor(Number(value) * 100))}%`;
}

export function clampPct(value) {
  const n = Number(value);
  if (Number.isNaN(n)) return 0;
  return Math.min(100, Math.max(0, n));
}

export function formatMs(ms) {
  if (ms == null) return '—';
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(2)} s`;
}

export function formatDate(iso) {
  const d = new Date(iso);
  if (!iso || Number.isNaN(d.getTime())) return iso || '';
  return d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}

export function humanKey(key) {
  return String(key)
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export function formatMetric(value) {
  if (value == null) return '—';
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (typeof value === 'number') {
    if (value >= 0 && value <= 1) return `${(value * 100).toFixed(1)}%`;
    if (Number.isInteger(value)) return value.toLocaleString();
    return String(Number(value.toFixed(3)));
  }
  if (typeof value === 'object') {
    const json = JSON.stringify(value);
    return json.length > 60 ? `${json.slice(0, 57)}…` : json;
  }
  return String(value);
}

export function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
