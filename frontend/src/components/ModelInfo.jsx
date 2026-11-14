import { AlertTriangle, RotateCcw } from 'lucide-react';
import { formatMetric, humanKey } from '../format.js';

function Section({ title, children }) {
  return (
    <section className="card p-5 sm:p-6" aria-labelledby={`sec-${title}`}>
      <h2 id={`sec-${title}`} className="text-base font-semibold text-slate-900">
        {title}
      </h2>
      <div className="mt-3 text-sm leading-relaxed text-slate-700">{children}</div>
    </section>
  );
}

function categoryParts(cat) {
  const entries = Object.entries(cat || {});
  if (entries.length === 0) return { label: 'Unnamed', rest: [] };
  const labelKey = ['label', 'name', 'key', 'category'].find((k) => cat[k] != null) ?? entries[0][0];
  const rest = entries.filter(([k]) => k !== labelKey);
  return { label: String(cat[labelKey]), rest };
}

export default function ModelInfo({ state, onRetry }) {
  if (state.status === 'loading') {
    return (
      <div className="space-y-4" aria-live="polite">
        <h1 className="text-2xl font-bold text-slate-900">Model</h1>
        <p className="text-sm text-slate-500">Loading model details…</p>
        <div className="h-32 animate-pulse rounded-xl bg-slate-200/70" />
        <div className="h-48 animate-pulse rounded-xl bg-slate-200/70" />
      </div>
    );
  }

  if (state.status === 'error' || !state.data) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-bold text-slate-900">Model</h1>
        <div role="alert" className="flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-900">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
          <div className="flex-1">
            <p className="font-semibold">Model details could not be loaded.</p>
            <p className="mt-1">{state.error || 'The ClearCart server did not respond.'}</p>
          </div>
        </div>
        <button type="button" onClick={onRetry} className="btn-secondary">
          <RotateCcw className="h-4 w-4" aria-hidden="true" />
          Try again
        </button>
      </div>
    );
  }

  const m = state.data;
  const metrics = Object.entries(m.test_metrics || {});
  const categories = Array.isArray(m.categories) ? m.categories : [];

  return (
    <div className="space-y-6">
      <header>
        <p className="text-xs font-semibold uppercase tracking-wider text-teal-700">Model card</p>
        <h1 className="mt-1 text-2xl font-bold text-slate-900 sm:text-3xl">
          {m.name}
          {m.version ? <span className="ml-2 text-base font-medium text-slate-500">v{m.version}</span> : null}
        </h1>
        <p className="mt-2 text-sm text-slate-600">
          {m.ocr_available ? 'Screenshot OCR is available.' : 'Screenshot OCR is not available on this server.'}
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-2">
        <Section title="Architecture">
          <p>{m.architecture || '—'}</p>
        </Section>
        <Section title="Embedder">
          <p>{m.embedder || '—'}</p>
        </Section>
        <Section title="Calibration">
          <dl className="grid grid-cols-2 gap-3">
            <div className="rounded-lg bg-slate-50 p-3 ring-1 ring-inset ring-slate-200">
              <dt className="text-xs text-slate-500">Decision threshold</dt>
              <dd className="mt-1 font-semibold tabular-nums">{formatMetric(m.threshold)}</dd>
            </div>
            <div className="rounded-lg bg-slate-50 p-3 ring-1 ring-inset ring-slate-200">
              <dt className="text-xs text-slate-500">Temperature</dt>
              <dd className="mt-1 font-semibold tabular-nums">{formatMetric(m.temperature)}</dd>
            </div>
          </dl>
        </Section>
        <Section title="Training data">
          <p>{m.training_data || '—'}</p>
        </Section>
      </div>

      <Section title="Categories">
        {categories.length === 0 ? (
          <p>No categories reported.</p>
        ) : (
          <ul className="grid gap-3 sm:grid-cols-2">
            {categories.map((cat, i) => {
              const { label, rest } = categoryParts(cat);
              return (
                <li key={`${label}-${i}`} className="rounded-lg border border-slate-200 p-3">
                  <p className="font-semibold text-slate-900">{label}</p>
                  {rest.map(([k, v]) => (
                    <p key={k} className="mt-1 text-xs text-slate-600">
                      <span className="font-medium">{humanKey(k)}:</span> {String(v)}
                    </p>
                  ))}
                </li>
              );
            })}
          </ul>
        )}
      </Section>

      <Section title="Test metrics">
        {metrics.length === 0 ? (
          <p>No test metrics reported.</p>
        ) : (
          <dl className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {metrics.map(([k, v]) => (
              <div key={k} className="rounded-lg bg-slate-50 p-3 ring-1 ring-inset ring-slate-200">
                <dt className="text-xs text-slate-500">{humanKey(k)}</dt>
                <dd className="mt-1 font-semibold tabular-nums text-slate-900">{formatMetric(v)}</dd>
              </div>
            ))}
          </dl>
        )}
      </Section>

      <Section title="Limitations">
        {Array.isArray(m.limitations) && m.limitations.length > 0 ? (
          <ul className="list-disc space-y-2 pl-5">
            {m.limitations.map((item, i) => (
              <li key={i}>{item}</li>
            ))}
          </ul>
        ) : (
          <p>No limitations listed.</p>
        )}
      </Section>
    </div>
  );
}
