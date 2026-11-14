import {
  ShieldCheck,
  ArrowRight,
  Clock,
  Flame,
  Users,
  Shuffle,
  CheckSquare,
  ScanLine,
  Cpu,
  Gauge,
  ClipboardCheck,
  FileText,
  FileCode,
  Image as ImageIcon,
  Info,
} from 'lucide-react';
import { formatMetric, humanKey } from '../format.js';

const PATTERNS = [
  {
    icon: Clock,
    title: 'Misleading urgency',
    body: "Countdowns and 'hurry' language with no real deadline behind them.",
  },
  {
    icon: Flame,
    title: 'Misleading scarcity',
    body: "'Only 2 left!' or 'selling fast' claims that cannot be verified.",
  },
  {
    icon: Users,
    title: 'Social proof pressure',
    body: "Crowd signals such as 'people are viewing this now' used to rush a decision.",
  },
  {
    icon: Shuffle,
    title: 'Misdirection / confirmshaming',
    body: "Decline buttons that guilt or shame you, such as 'No, I don't want to save money'.",
  },
  {
    icon: CheckSquare,
    title: 'Preselected options (HTML only, DOM-observed)',
    body: 'Add-ons, insurance or subscriptions already ticked in the markup, found by reading the DOM.',
    note: 'HTML only',
  },
];

const STEPS = [
  {
    icon: ScanLine,
    title: 'Extract elements',
    body: 'The page is split into the short text elements a shopper actually reads: banners, buttons and labels. Screenshots are read with OCR first.',
  },
  {
    icon: Cpu,
    title: 'Classify each element',
    body: 'A MiniLM + TF-IDF ensemble scores every element for each dark-pattern category.',
  },
  {
    icon: Gauge,
    title: 'Calibrate confidence',
    body: 'Scores are calibrated so a stated percentage means something. Weak signals are not flagged.',
  },
  {
    icon: ClipboardCheck,
    title: 'Evidence-linked audit',
    body: 'Each finding quotes the exact evidence, shows the words that drove it and proposes an ethical redesign.',
  },
];

const INPUTS = [
  {
    icon: FileText,
    title: 'Pasted text',
    body: 'Copy the visible text of a product or checkout page. Up to 200,000 characters.',
  },
  {
    icon: FileCode,
    title: 'HTML',
    body: 'Upload a .html or .htm file, or paste markup. Up to 2 MB. Preselected options are detected here.',
  },
  {
    icon: ImageIcon,
    title: 'Screenshot (OCR)',
    body: 'PNG, JPG or WEBP up to 8 MB. Text regions are read with OCR and classified by the text model.',
  },
];

function MetricsPanel({ model }) {
  if (model.status === 'loading') {
    return (
      <div className="space-y-3" aria-live="polite">
        <div className="h-4 w-1/2 animate-pulse rounded bg-slate-200" />
        <div className="h-10 animate-pulse rounded bg-slate-100" />
        <div className="h-10 animate-pulse rounded bg-slate-100" />
      </div>
    );
  }
  if (model.status === 'error' || !model.data) {
    return (
      <p className="text-sm text-slate-500">
        Model details are unavailable right now. The audit still works once the server is reachable.
      </p>
    );
  }
  const entries = Object.entries(model.data.test_metrics || {}).slice(0, 4);
  return (
    <>
      <p className="text-sm font-semibold text-slate-900">
        {model.data.name} {model.data.version ? `v${model.data.version}` : ''}
      </p>
      {entries.length === 0 ? (
        <p className="mt-2 text-sm text-slate-500">No held-out test metrics were reported.</p>
      ) : (
        <dl className="mt-4 grid grid-cols-2 gap-3">
          {entries.map(([key, value]) => (
            <div key={key} className="rounded-lg bg-slate-50 p-3 ring-1 ring-inset ring-slate-200">
              <dt className="text-xs font-medium text-slate-500">{humanKey(key)}</dt>
              <dd className="mt-1 text-lg font-bold tabular-nums text-slate-900">{formatMetric(value)}</dd>
            </div>
          ))}
        </dl>
      )}
    </>
  );
}

export default function Landing({ onStart, model }) {
  return (
    <div className="space-y-14 sm:space-y-16">
      <section className="grid gap-8 lg:grid-cols-12 lg:items-center">
        <div className="lg:col-span-7">
          <p className="inline-flex items-center gap-2 rounded-full bg-teal-50 px-3 py-1 text-xs font-semibold text-teal-800 ring-1 ring-inset ring-teal-200">
            <ShieldCheck className="h-4 w-4" aria-hidden="true" />
            Consumer protection screening
          </p>
          <h1 className="mt-4 text-4xl font-bold tracking-tight text-slate-900 sm:text-5xl">ClearCart</h1>
          <p className="mt-3 text-lg font-medium text-teal-800 sm:text-xl">
            Evidence-First AI Auditor for Deceptive E-Commerce Interfaces
          </p>
          <p className="mt-5 max-w-2xl leading-relaxed text-slate-600">
            Dark patterns are design choices that push shoppers into decisions they did not mean to make: fake
            deadlines, invented scarcity, crowd pressure, or decline buttons that shame. ClearCart reads the text or
            HTML you give it, quotes the exact words or elements behind each concern, and states how sure it is.
          </p>
          <div className="mt-8 flex flex-col gap-3 sm:flex-row sm:items-center">
            <button type="button" onClick={onStart} className="btn-primary px-6 py-3 text-base">
              Start Audit
              <ArrowRight className="h-4 w-4" aria-hidden="true" />
            </button>
            <p className="text-sm text-slate-500">Paste text, upload HTML, or drop a screenshot.</p>
          </div>
        </div>

        <aside className="card p-6 lg:col-span-5" aria-labelledby="model-glance-title">
          <h2 id="model-glance-title" className="text-sm font-semibold uppercase tracking-wide text-slate-500">
            Model at a glance
          </h2>
          <div className="mt-3">
            <MetricsPanel model={model} />
          </div>
        </aside>
      </section>

      <section aria-labelledby="patterns-title">
        <h2 id="patterns-title" className="text-2xl font-bold text-slate-900">
          What dark patterns look like
        </h2>
        <p className="mt-2 max-w-3xl text-slate-600">
          Dark patterns are interface tricks that steer choices against the shopper's interest. ClearCart screens for
          five common families.
        </p>
        <ul className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {PATTERNS.map(({ icon: Icon, title, body, note }) => (
            <li key={title} className="card p-5">
              <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-slate-100 text-slate-700">
                <Icon className="h-5 w-5" aria-hidden="true" />
              </span>
              <h3 className="mt-4 font-semibold text-slate-900">{title}</h3>
              <p className="mt-1.5 text-sm leading-relaxed text-slate-600">{body}</p>
              {note && (
                <span className="mt-3 inline-block rounded-full bg-sky-50 px-2 py-0.5 text-xs font-medium text-sky-900 ring-1 ring-inset ring-sky-200">
                  {note}
                </span>
              )}
            </li>
          ))}
        </ul>
      </section>

      <section aria-labelledby="how-title">
        <h2 id="how-title" className="text-2xl font-bold text-slate-900">
          How it works
        </h2>
        <ol className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {STEPS.map(({ icon: Icon, title, body }, i) => (
            <li key={title} className="card p-5">
              <div className="flex items-center gap-3">
                <span className="flex h-8 w-8 items-center justify-center rounded-full bg-teal-700 text-sm font-bold text-white">
                  {i + 1}
                </span>
                <Icon className="h-5 w-5 text-teal-700" aria-hidden="true" />
              </div>
              <h3 className="mt-4 font-semibold text-slate-900">{title}</h3>
              <p className="mt-1.5 text-sm leading-relaxed text-slate-600">{body}</p>
            </li>
          ))}
        </ol>
      </section>

      <section aria-labelledby="inputs-title">
        <h2 id="inputs-title" className="text-2xl font-bold text-slate-900">
          Supported inputs
        </h2>
        <ul className="mt-6 grid gap-4 md:grid-cols-3">
          {INPUTS.map(({ icon: Icon, title, body }) => (
            <li key={title} className="card flex gap-4 p-5">
              <Icon className="mt-0.5 h-6 w-6 shrink-0 text-teal-700" aria-hidden="true" />
              <div>
                <h3 className="font-semibold text-slate-900">{title}</h3>
                <p className="mt-1 text-sm leading-relaxed text-slate-600">{body}</p>
              </div>
            </li>
          ))}
        </ul>
      </section>

      <section className="flex flex-col items-start gap-4 rounded-xl border border-slate-200 bg-white p-6 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex gap-3">
          <Info className="mt-0.5 h-5 w-5 shrink-0 text-slate-500" aria-hidden="true" />
          <p className="text-sm text-slate-600">
            <span className="font-semibold text-slate-800">Assistive screening tool.</span> Findings are indications
            of potentially deceptive design, not legal findings.
          </p>
        </div>
        <button type="button" onClick={onStart} className="btn-primary shrink-0">
          Start Audit
        </button>
      </section>
    </div>
  );
}
