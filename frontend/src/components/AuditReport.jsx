import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { CheckCircle2, Info, ScanSearch } from 'lucide-react';
import { pct, formatDate } from '../format.js';

// Literal hex values for SVG strokes (Tailwind classes cannot colour SVG lines dynamically).
const CAT_HEX = {
  urgency: '#dc2626',
  scarcity: '#ea580c',
  social_proof: '#7c3aed',
  misdirection: '#d97706',
  preselected_option: '#0369a1',
};
const CAT_LABEL = {
  urgency: 'Misleading urgency',
  scarcity: 'Misleading scarcity',
  social_proof: 'Social proof pressure',
  misdirection: 'Misdirection / confirmshaming',
  preselected_option: 'Preselected option',
};
const RISK_UI = {
  high: { hex: '#dc2626', label: 'High deception risk', text: 'text-red-700' },
  medium: { hex: '#d97706', label: 'Medium deception risk', text: 'text-amber-700' },
  low: { hex: '#64748b', label: 'Low deception risk', text: 'text-slate-600' },
  none: { hex: '#16a34a', label: 'No concern', text: 'text-green-700' },
};
const hex = (cat) => CAT_HEX[cat] || '#475569';
const CALLOUT_H = 68; // fixed callout height (px) used for collision-free stacking
const RAIL_W = 232;
const GUTTER = 30; // left margin so number badges don't cover content
const MIN_RAIL_CONTAINER = 560; // below this container width, callouts become a numbered legend

/* ------------------------------------------------------------------ callout stage */
// targets: [{ finding, n, rect: {x, y, w, h} }] in stage pixel coordinates.
function stackCallouts(targets) {
  const sorted = [...targets].sort((a, b) => a.rect.y + a.rect.h / 2 - (b.rect.y + b.rect.h / 2));
  let next = 0;
  return sorted.map((t) => {
    const want = Math.max(0, t.rect.y + t.rect.h / 2 - CALLOUT_H / 2);
    const y = Math.max(want, next);
    next = y + CALLOUT_H + 8;
    return { ...t, cy: y };
  });
}

function Stage({ contentHeight, stageWidth, showRail, targets: rawTargets, onSelect, activeId, children }) {
  // Number findings in on-screen reading order (top-to-bottom, then left-to-right) and offset
  // the badge when several findings point at the same element.
  const targets = useMemo(() => {
    const ordered = [...rawTargets].sort((a, b) => a.rect.y - b.rect.y || a.rect.x - b.rect.x);
    const seen = {};
    return ordered.map((t, i) => {
      const key = `${Math.round(t.rect.x)}:${Math.round(t.rect.y)}:${Math.round(t.rect.w)}`;
      seen[key] = (seen[key] || 0) + 1;
      return { ...t, n: i + 1, dup: seen[key] - 1 };
    });
  }, [rawTargets]);
  const placed = useMemo(() => stackCallouts(targets), [targets]);
  const railBottom = placed.length ? placed[placed.length - 1].cy + CALLOUT_H : 0;
  const height = Math.max(contentHeight, railBottom);
  return (
    <>
      <div className="relative flex" style={{ minHeight: height }}>
        <div className="relative min-w-0 flex-1" style={{ marginLeft: GUTTER }}>
          {children}
          {targets.map((t) => (
            <button
              key={t.finding.id}
              type="button"
              onClick={() => onSelect(t.finding.id)}
              aria-label={`Finding ${t.n}: ${t.finding.category_label}. ${t.finding.short_reason}. Show ethical redesign`}
              data-active={activeId === t.finding.id ? 'true' : undefined}
              className={`absolute rounded-[3px] border-2 transition hover:brightness-95 focus:outline-none focus-visible:ring-2 focus-visible:ring-teal-600 ${
                activeId === t.finding.id ? 'z-10 ring-4 ring-teal-500/50 ring-offset-1' : ''
              }`}
              style={{
                left: t.rect.x - 3, top: t.rect.y - 3, width: t.rect.w + 6, height: t.rect.h + 6,
                borderColor: hex(t.finding.category), background: `${hex(t.finding.category)}14`,
              }}
            >
              <span
                className="absolute -top-2.5 flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-bold text-white shadow ring-2 ring-white"
                style={{ background: hex(t.finding.category), left: -27 + t.dup * 22 }}
              >
                {t.n}
              </span>
            </button>
          ))}
        </div>
        {/* callout rail (only when the container is wide enough) */}
        {showRail && (
        <div className="relative shrink-0" style={{ width: RAIL_W }} aria-hidden="true">
          {placed.map((t) => (
            <button
              key={t.finding.id}
              type="button"
              tabIndex={-1}
              onClick={() => onSelect(t.finding.id)}
              className="absolute left-4 right-0 overflow-hidden rounded-lg border bg-white px-2.5 py-1.5 text-left shadow-sm transition hover:shadow-md"
              style={{ top: t.cy, height: CALLOUT_H, borderColor: `${hex(t.finding.category)}66`, background: `${hex(t.finding.category)}0d` }}
            >
              <span className="flex items-center justify-between gap-2">
                <span className="truncate text-[11px] font-bold" style={{ color: hex(t.finding.category) }}>
                  {t.n}. {t.finding.category_label}
                </span>
                <span className="shrink-0 text-[11px] font-bold tabular-nums text-slate-900">
                  {t.finding.confidence_pct != null ? `${t.finding.confidence_pct}%` : 'DOM'}
                </span>
              </span>
              <span className="mt-0.5 line-clamp-2 block text-[11px] leading-snug text-slate-700">{t.finding.short_reason}</span>
            </button>
          ))}
        </div>
        )}
        {/* leader lines */}
        {showRail && (
        <svg className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true">
          {placed.map((t) => {
            const x1 = GUTTER + t.rect.x + t.rect.w + 3;
            const y1 = t.rect.y + t.rect.h / 2;
            const x2 = GUTTER + stageWidth + 16;
            const y2 = t.cy + CALLOUT_H / 2;
            return (
              <g key={t.finding.id} stroke={hex(t.finding.category)} strokeWidth="1.5" fill="none">
                <path d={`M ${x1} ${y1} L ${GUTTER + stageWidth + 6} ${y1} L ${x2} ${y2}`} />
                <circle cx={x1} cy={y1} r="2.5" fill={hex(t.finding.category)} />
              </g>
            );
          })}
        </svg>
        )}
      </div>
      {/* compact legend for narrow containers, where the rail is hidden */}
      {!showRail && targets.length > 0 && (
        <ol className="mt-3 space-y-1.5">
          {[...targets].sort((a, b) => a.n - b.n).map((t) => (
            <li key={t.finding.id}>
              <button type="button" onClick={() => onSelect(t.finding.id)} className="flex w-full items-start gap-2 text-left text-xs">
                <span className="mt-px flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-[9px] font-bold text-white" style={{ background: hex(t.finding.category) }}>{t.n}</span>
                <span><b style={{ color: hex(t.finding.category) }}>{t.finding.category_label}</b>{t.finding.confidence_pct != null ? ` · ${t.finding.confidence_pct}%` : ''} — {t.finding.short_reason}</span>
              </button>
            </li>
          ))}
        </ol>
      )}
    </>
  );
}

function useWidth(ref) {
  const [w, setW] = useState(0);
  useLayoutEffect(() => {
    if (!ref.current) return undefined;
    const ro = new ResizeObserver(([e]) => setW(e.contentRect.width));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, [ref]);
  return w;
}

/* ------------------------------------------------------------------ text preview */
function TextPreview({ text, findings, numbers, onSelect, activeId }) {
  const wrapRef = useRef(null);
  const stageRef = useRef(null);
  const markRefs = useRef({});
  const width = useWidth(wrapRef);
  const [targets, setTargets] = useState([]);
  const [contentH, setContentH] = useState(0);

  const segments = useMemo(() => {
    const chars = Array.from(text); // code-point indexing, matches backend offsets
    const spans = findings
      .filter((f) => f.source_location?.char_start != null)
      .sort((a, b) => a.source_location.char_start - b.source_location.char_start);
    const out = [];
    let pos = 0;
    for (const f of spans) {
      const { char_start: s, char_end: e } = f.source_location;
      if (s < pos) continue; // overlapping span: keep the earlier one
      if (s > pos) out.push({ text: chars.slice(pos, s).join('') });
      out.push({ text: chars.slice(s, e).join(''), finding: f });
      pos = e;
    }
    if (pos < chars.length) out.push({ text: chars.slice(pos).join('') });
    return out;
  }, [text, findings]);

  useLayoutEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    const base = stage.getBoundingClientRect();
    const t = [];
    for (const seg of segments) {
      if (!seg.finding) continue;
      const el = markRefs.current[seg.finding.id];
      if (!el) continue;
      const r = el.getBoundingClientRect();
      t.push({ finding: seg.finding, n: numbers[seg.finding.id], rect: { x: r.left - base.left, y: r.top - base.top, w: r.width, h: r.height } });
    }
    setTargets(t);
    setContentH(stage.scrollHeight);
  }, [segments, width, numbers]);

  const showRail = width >= MIN_RAIL_CONTAINER;
  const stageWidth = Math.max(0, width - GUTTER - (showRail ? RAIL_W : 0));
  return (
    <div ref={wrapRef} className="relative">
      <Stage contentHeight={contentH} stageWidth={stageWidth} showRail={showRail} targets={targets} onSelect={onSelect} activeId={activeId}>
        <div ref={stageRef} className="whitespace-pre-wrap break-words px-1 py-1 text-[13px] leading-[2.1] text-slate-800">
          {segments.map((seg, i) =>
            seg.finding ? (
              <span key={i} ref={(el) => { markRefs.current[seg.finding.id] = el; }} className="font-semibold text-slate-900">
                {seg.text}
              </span>
            ) : (
              <span key={i}>{seg.text}</span>
            ),
          )}
        </div>
      </Stage>
    </div>
  );
}

/* ------------------------------------------------------------------ HTML preview */
const VIRTUAL_W = 800; // virtual viewport for the HTML render; scaled to fit the stage
function safeSrcDoc(html) {
  // No scripts (sandbox has no allow-scripts) and no network: the CSP blocks every external
  // request so previewing a page never contacts third-party servers.
  const csp = `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; font-src data:">`;
  const style = '<style>html,body{margin:0;padding:16px;font-family:Inter,system-ui,sans-serif;color:#0f172a;background:#fff}</style>';
  if (/<head[^>]*>/i.test(html)) return html.replace(/<head[^>]*>/i, (m) => `${m}${csp}${style}`);
  return `<!doctype html><html><head>${csp}${style}</head><body>${html}</body></html>`;
}

function HtmlPreview({ html, findings, numbers, onSelect, activeId }) {
  const wrapRef = useRef(null);
  const frameRef = useRef(null);
  const width = useWidth(wrapRef);
  const [docH, setDocH] = useState(400);
  const [targets, setTargets] = useState([]);
  const [missing, setMissing] = useState(0);
  const showRail = width >= MIN_RAIL_CONTAINER;
  const stageWidth = Math.max(0, width - GUTTER - (showRail ? RAIL_W : 0));
  const scale = stageWidth > 0 ? Math.min(1, stageWidth / VIRTUAL_W) : 1;
  const srcDoc = useMemo(() => safeSrcDoc(html), [html]);

  const measure = useCallback(() => {
    const doc = frameRef.current?.contentDocument;
    if (!doc?.body) return;
    const h = Math.max(doc.documentElement.scrollHeight, doc.body.scrollHeight);
    setDocH(h);
    const t = [];
    let miss = 0;
    for (const f of findings) {
      const sel = f.source_location?.css_selector;
      let el = null;
      try { el = sel ? doc.querySelector(sel) : null; } catch { el = null; }
      if (el && el.tagName === 'INPUT') {
        el = el.closest('label') || (el.id && doc.querySelector(`label[for="${CSS.escape(el.id)}"]`)) || el;
      }
      if (!el) { miss += 1; continue; }
      const r = el.getBoundingClientRect();
      t.push({ finding: f, n: numbers[f.id], rect: { x: r.left * scale, y: r.top * scale, w: r.width * scale, h: r.height * scale } });
    }
    setTargets(t);
    setMissing(miss);
  }, [findings, numbers, scale]);

  useEffect(() => { measure(); }, [measure, width]);

  return (
    <div ref={wrapRef} className="relative">
      <Stage contentHeight={docH * scale} stageWidth={stageWidth} showRail={showRail} targets={targets} onSelect={onSelect} activeId={activeId}>
        <div className="overflow-hidden rounded-md" style={{ height: docH * scale }}>
          <iframe
            ref={frameRef}
            title="Sandboxed preview of the audited HTML (scripts and network disabled)"
            sandbox="allow-same-origin"
            srcDoc={srcDoc}
            onLoad={measure}
            className="pointer-events-none origin-top-left border-0 bg-white"
            style={{ width: VIRTUAL_W, height: docH, transform: `scale(${scale})` }}
          />
        </div>
      </Stage>
      {missing > 0 && (
        <p className="mt-2 text-xs text-amber-800">{missing} finding(s) could not be located in the rendered preview; see the finding cards below.</p>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ screenshot preview */
function ShotPreview({ imageUrl, imageSize, findings, numbers, onSelect, activeId }) {
  const wrapRef = useRef(null);
  const width = useWidth(wrapRef);
  const showRail = width >= MIN_RAIL_CONTAINER;
  const stageWidth = Math.max(0, width - GUTTER - (showRail ? RAIL_W : 0));
  const W = imageSize?.width || 1;
  const H = imageSize?.height || 1;
  const scale = stageWidth / W;
  const targets = findings
    .filter((f) => f.source_location?.bbox)
    .map((f) => {
      const b = f.source_location.bbox;
      return { finding: f, n: numbers[f.id], rect: { x: b.x * scale, y: b.y * scale, w: b.width * scale, h: b.height * scale } };
    });
  return (
    <div ref={wrapRef} className="relative">
      <Stage contentHeight={H * scale} stageWidth={stageWidth} showRail={showRail} targets={targets} onSelect={onSelect} activeId={activeId}>
        <img src={imageUrl} alt="Uploaded screenshot with flagged text regions boxed" className="block h-auto w-full rounded-md" />
      </Stage>
      <p className="mt-2 text-xs text-slate-500">Boxes mark text read by OCR and classified by the text model. ClearCart does not analyse visual layout.</p>
    </div>
  );
}


/* ------------------------------------------------------------------ annotated preview card */
/** Finding numbers in the reading order of the preview (top-to-bottom). */
export function useFindingNumbers(response) {
  const findings = response.findings;
  const inputType = response.input_type;
  return useMemo(() => {
    const list = findings || [];
    const key = (f) => {
      const l = f.source_location || {};
      if (l.bbox) return l.bbox.y * 1e4 + l.bbox.x;
      if (l.char_start != null) return l.char_start;
      return 0;
    };
    const order = inputType === 'html' ? list : [...list].sort((a, b) => key(a) - key(b));
    return Object.fromEntries(order.map((f, i) => [f.id, i + 1]));
  }, [findings, inputType]);
}

/**
 * The audited input with every finding boxed and numbered, and its reason in a callout beside it
 * (numbered legend on narrow screens). One self-contained card; always visible.
 */
export function AnnotatedPreview({ result, onSelect, activeId = null, headingLevel = 2 }) {
  const { inputType, inputText, imageUrl, htmlSource, response, inputName } = result;
  const findings = response.findings || [];
  const numbers = useFindingNumbers(response);
  const headingId = useId();
  const H = `h${headingLevel}`;
  const kind = response.input_type === 'screenshot_ocr' ? 'shot' : inputType === 'html' || response.input_type === 'html' ? 'html' : 'text';
  let body;
  if (kind === 'text' && typeof inputText === 'string') body = <TextPreview text={inputText} findings={findings} numbers={numbers} onSelect={onSelect} activeId={activeId} />;
  else if (kind === 'html' && typeof htmlSource === 'string') body = <HtmlPreview html={htmlSource} findings={findings} numbers={numbers} onSelect={onSelect} activeId={activeId} />;
  else if (kind === 'shot' && imageUrl) body = <ShotPreview imageUrl={imageUrl} imageSize={response.image_size} findings={findings} numbers={numbers} onSelect={onSelect} activeId={activeId} />;
  else body = <p className="text-sm text-slate-600">The original input is not available for this result, so the annotated preview cannot be drawn. The Ethical redesign section still lists every flagged element.</p>;
  const kindLabel = kind === 'shot' ? 'Screenshot (OCR)' : kind === 'html' ? 'HTML' : 'Text';

  return (
    <section aria-labelledby={headingId} className="card overflow-hidden">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-slate-100 px-4 py-3 sm:px-5">
        <ScanSearch className="h-4 w-4 shrink-0 text-teal-700" aria-hidden="true" />
        <H id={headingId} className="text-[15px] font-semibold text-slate-900">Annotated preview</H>
        <span className="min-w-0 flex-1 truncate text-xs text-slate-500" title={inputName || undefined}>
          {kindLabel}{inputName ? ` · ${inputName}` : ''}
        </span>
        {findings.length > 0 ? (
          <span className="text-xs text-slate-500">{findings.length} marked · select a box to see its redesign</span>
        ) : response.summary.no_concern ? (
          <span className="inline-flex items-center gap-1 text-xs font-medium text-emerald-700">
            <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" /> Nothing flagged
          </span>
        ) : null}
      </div>
      <div className="max-h-[760px] overflow-auto bg-slate-50/40 p-4 sm:p-5">{body}</div>
    </section>
  );
}

/* ------------------------------------------------------------------ dashboard */
function RiskGauge({ score, risk }) {
  const ui = RISK_UI[risk] || RISK_UI.low;
  const has = typeof score === 'number' && Number.isFinite(score);
  const r = 50;
  const c = 2 * Math.PI * r;
  const v = has ? Math.max(0, Math.min(99, score)) : 0;
  return (
    <div
      className="relative h-28 w-28 shrink-0"
      role="img"
      aria-label={has ? `Deception risk score ${v} out of 100, ${ui.label}` : `Deception risk score not reported, ${ui.label}`}
    >
      <svg viewBox="0 0 120 120" className="h-full w-full -rotate-90">
        <circle cx="60" cy="60" r={r} fill="none" stroke="#f1f5f9" strokeWidth="10" />
        {has && (
          <circle cx="60" cy="60" r={r} fill="none" stroke={ui.hex} strokeWidth="10" strokeLinecap="round" strokeDasharray={`${(v / 100) * c} ${c}`} />
        )}
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center text-center tabular-nums">
        <span className="text-3xl font-bold leading-none text-slate-900">{has ? v : '—'}</span>
        <span className="mt-1 text-[11px] font-medium text-slate-500">/100</span>
      </div>
    </div>
  );
}

function CategoryBars({ summary, inputType }) {
  const detected = [...(summary.categories_detected || [])].sort((a, b) => (b.max_score ?? 0) - (a.max_score ?? 0));
  const keys = ['urgency', 'scarcity', 'social_proof', 'misdirection'];
  if (inputType === 'html') keys.push('preselected_option');
  const missing = keys.filter((k) => !detected.some((c) => c.category === k));
  if (!detected.length && !missing.length) return null;
  return (
    <div className="min-w-0">
      {detected.length > 0 && (
        <ul className="space-y-2" aria-label="Detected categories">
          {detected.map((c) => {
            const isDom = c.category === 'preselected_option' || c.max_score == null;
            const width = isDom ? 100 : Math.min(99, Math.floor(c.max_score * 100));
            const right = isDom ? `${c.count} observed` : `${pct(c.max_score)}${c.count > 1 ? ` · ${c.count}×` : ''}`;
            return (
              <li key={c.category} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1 sm:grid-cols-[14rem_minmax(0,1fr)_5.5rem]">
                <span className="min-w-0 truncate text-sm text-slate-800" title={c.category_label || CAT_LABEL[c.category]}>
                  {c.category_label || CAT_LABEL[c.category] || c.category}
                </span>
                <span className="col-span-2 row-start-2 h-1.5 overflow-hidden rounded-full bg-slate-100 sm:col-span-1 sm:col-start-2 sm:row-start-1" aria-hidden="true">
                  <span className="block h-full rounded-full" style={{ width: `${width}%`, background: hex(c.category) }} />
                </span>
                <span className="text-right text-xs font-semibold tabular-nums text-slate-700 sm:col-start-3 sm:row-start-1">{right}</span>
              </li>
            );
          })}
        </ul>
      )}
      {missing.length > 0 && (
        <p className={`text-xs text-slate-400 ${detected.length ? 'mt-2.5' : ''}`}>
          Not detected: {missing.map((k) => CAT_LABEL[k]).join(', ')}
        </p>
      )}
    </div>
  );
}

function Stat({ label, value, sub, title }) {
  return (
    <div className="min-w-0 bg-white px-4 py-3 sm:px-5" title={title}>
      <dt className="truncate text-xs text-slate-500">{label}</dt>
      <dd className="mt-0.5 flex min-w-0 flex-wrap items-baseline gap-x-1.5">
        <span className="text-xl font-semibold tabular-nums text-slate-900">{value}</span>
        {sub && <span className="truncate text-xs text-slate-500">{sub}</span>}
      </dd>
    </div>
  );
}

const isFiniteNum = (v) => typeof v === 'number' && Number.isFinite(v);

/**
 * One-card overview of a single-page audit: risk gauge, detected categories, the confidence line and
 * a few headline numbers. Every value comes from the AuditResponse.
 */
export function AuditDashboard({ response: r }) {
  const s = r.summary;
  const findings = r.findings || [];
  const risk = RISK_UI[s.overall_risk] || RISK_UI.low;
  // risk_score is based on flagged findings only (0 when nothing is flagged). Very old responses lack it.
  const score = s.risk_score !== undefined ? s.risk_score
    : s.no_concern ? 0 : (isFiniteNum(s.overall_score) ? Math.min(99, Math.floor(s.overall_score * 100)) : null);
  const thresholdPct = isFiniteNum(s.threshold) ? Math.round(s.threshold * 100) : null;
  const rewrites = findings.filter((f) => f.suggested_rewrite).length;
  const cdi = r.intelligence?.cdi;
  const cats = (s.categories_detected || []).length;
  const unit = r.input_type === 'screenshot_ocr' ? 'OCR text lines' : r.input_type === 'html' ? 'visible DOM elements' : 'text segments';
  const finePrint = `Score = highest calibrated dark-pattern probability among flagged findings, so it is 0 when nothing is flagged${thresholdPct != null ? ` (elements at or above ${thresholdPct}% are flagged)` : ''}. An assistive indication, not a legal finding.`;
  const headingId = useId();
  const heading = s.no_concern ? 'No suspicious patterns detected' : 'Audit summary';
  // the API's no-concern statement repeats the heading; show only the rest of it
  let statement = s.overall_confidence_statement || '';
  if (s.no_concern && statement.startsWith(`${heading}.`)) statement = statement.slice(heading.length + 1).trim();

  return (
    <section aria-labelledby={headingId} className="card overflow-hidden">
      <div className="grid gap-5 p-4 sm:grid-cols-[auto_minmax(0,1fr)] sm:items-center sm:gap-6 sm:p-5">
        <div className="flex items-center gap-4 sm:flex-col sm:gap-2">
          <RiskGauge score={score} risk={s.overall_risk} />
          <span
            className="inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold"
            style={{ color: risk.hex, background: `${risk.hex}14` }}
          >
            {risk.label}
          </span>
        </div>
        <div className="min-w-0 space-y-4">
          <div>
            <h2 id={headingId} className={`text-base font-semibold ${s.no_concern ? 'text-emerald-800' : 'text-slate-900'}`}>
              {heading}
            </h2>
            {statement && (
              <p className="mt-0.5 text-sm text-slate-600" title={finePrint}>
                {statement}
                <span className="sr-only"> {finePrint}</span>
                <Info className="ml-1 inline h-3.5 w-3.5 -translate-y-px text-slate-400" aria-hidden="true" />
              </p>
            )}
          </div>
          {!s.no_concern && <CategoryBars summary={s} inputType={r.input_type} />}
          {s.no_concern && (
            <p className="text-xs text-slate-500">A screening result: it does not guarantee the page is free of deceptive design.</p>
          )}
        </div>
      </div>
      <dl className="grid grid-cols-2 gap-px border-t border-slate-100 bg-slate-100 sm:grid-cols-4">
        <Stat
          label="Findings"
          value={s.suspicious_findings}
          sub={s.suspicious_findings ? `${cats} categor${cats === 1 ? 'y' : 'ies'}` : 'none flagged'}
        />
        <Stat label="Elements analysed" value={s.elements_analysed} sub={unit} title={isFiniteNum(r.processing_ms) ? `Processed in ${r.processing_ms} ms` : undefined} />
        {isFiniteNum(cdi?.cdi) ? (
          <Stat label="Deception index (CDI)" value={cdi.cdi.toFixed(1)} sub={cdi.band ? `${String(cdi.band).toLowerCase()} band` : undefined} title={cdi.note || undefined} />
        ) : (
          <Stat label="Deception index (CDI)" value="—" sub={r.intelligence ? 'not computed' : 'not reported'} />
        )}
        <Stat label="Fixes suggested" value={findings.length} sub={findings.length ? `${rewrites} with a rewrite` : 'no changes needed'} />
      </dl>
    </section>
  );
}

/* ------------------------------------------------------------------ header */
export function ReportTitle({ result, actions, title = 'Website Audit Report' }) {
  const r = result.response;
  return (
    <header className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
      <div className="min-w-0">
        <h1 className="text-xl font-bold text-slate-900 sm:text-2xl">{title}</h1>
        <p className="mt-1 flex min-w-0 flex-wrap items-center gap-x-2 gap-y-0.5 text-sm text-slate-500">
          <span className="min-w-0 max-w-full truncate font-medium text-slate-700" title={result.inputName || r.input_type}>
            {result.inputName || r.input_type}
          </span>
          <span aria-hidden="true" className="hidden sm:inline">·</span>
          <span className="inline-flex items-center gap-1">
            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" aria-hidden="true" /> Completed {formatDate(r.created_at)}
          </span>
        </p>
      </div>
      {actions && <div className="lg:shrink-0">{actions}</div>}
    </header>
  );
}

/** Annotated preview of one audit (kept as the default export for callers that only need the preview). */
export default function AuditReport({ result, onSelect }) {
  return <AnnotatedPreview result={result} onSelect={onSelect} />;
}
