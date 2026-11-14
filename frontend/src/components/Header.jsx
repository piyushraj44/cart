import { ScanSearch } from 'lucide-react';

const NAV = [
  { id: 'workspace', label: 'Audit' },
  { id: 'landing', label: 'About us' },
  { id: 'model', label: 'Model' },
  { id: 'journey', label: 'Journey' },
];

const HEALTH = {
  checking: { label: 'Checking API…', dot: 'bg-slate-400' },
  online: { label: 'API online', dot: 'bg-emerald-500' },
  offline: { label: 'API offline', dot: 'bg-red-500' },
};

export default function Header({ view, onNavigate, health }) {
  const current = view === 'results' ? 'workspace' : view === 'journeyResults' ? 'journey' : view;
  const h = HEALTH[health] || HEALTH.checking;
  return (
    <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/95 backdrop-blur">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-3 focus:z-50 focus:rounded-md focus:bg-white focus:px-3 focus:py-2 focus:shadow"
      >
        Skip to content
      </a>
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-3 px-4 py-3 sm:px-6 lg:px-8">
        <button
          type="button"
          onClick={() => onNavigate('workspace')}
          className="flex items-center gap-2.5 rounded-lg"
          aria-label="ClearCart, go to Audit"
        >
          <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-teal-700 text-white">
            <ScanSearch className="h-5 w-5" aria-hidden="true" />
          </span>
          <span className="text-lg font-bold tracking-tight text-slate-900 max-[419px]:sr-only">ClearCart</span>
        </button>

        <div className="flex items-center gap-3">
          <span
            className="hidden items-center gap-1.5 text-xs text-slate-500 md:inline-flex"
            role="status"
          >
            <span className={`h-2 w-2 rounded-full ${h.dot}`} aria-hidden="true" />
            {h.label}
          </span>
          <nav aria-label="Main">
            <ul className="flex items-center gap-0.5 sm:gap-1">
              {NAV.map((item) => (
                <li key={item.id}>
                  <button
                    type="button"
                    onClick={() => onNavigate(item.id)}
                    aria-current={current === item.id ? 'page' : undefined}
                    className={`rounded-lg px-2 py-2 text-sm font-medium transition-colors sm:px-3 ${
                      current === item.id
                        ? 'bg-teal-50 text-teal-800'
                        : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
                    }`}
                  >
                    {item.label}
                  </button>
                </li>
              ))}
            </ul>
          </nav>
        </div>
      </div>
    </header>
  );
}
