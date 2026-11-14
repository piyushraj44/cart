import { useEffect, useRef, useState } from 'react';
import {
  Type,
  FileCode,
  Image as ImageIcon,
  UploadCloud,
  Loader2,
  X,
  AlertTriangle,
  Eraser,
  ScanSearch,
  BookOpen,
  Route,
} from 'lucide-react';
import { auditText, auditHtmlFile, auditHtmlText, auditScreenshot, getSamples } from '../api.js';
import { LIMITS, formatBytes } from '../format.js';

const TABS = [
  { id: 'text', label: 'Paste text', icon: Type },
  { id: 'html', label: 'HTML', icon: FileCode },
  { id: 'screenshot', label: 'Screenshot', icon: ImageIcon },
];

const STEPS = {
  text: ['Extracting elements…', 'Running classifier…', 'Calibrating confidence…'],
  html: ['Extracting elements…', 'Running classifier…', 'Calibrating confidence…'],
  screenshot: ['Reading text regions with OCR…', 'Running classifier…', 'Calibrating confidence…'],
};

const IMAGE_TYPES = ['image/png', 'image/jpeg', 'image/webp'];

function byteLength(str) {
  return new TextEncoder().encode(str).length;
}

function checkHtmlFile(file) {
  if (!/\.(html?|htm)$/i.test(file.name)) return 'Only .html or .htm files are supported.';
  if (file.size > LIMITS.htmlBytes) {
    return `That file is ${formatBytes(file.size)}. The HTML limit is ${formatBytes(LIMITS.htmlBytes)}.`;
  }
  return null;
}

function checkImageFile(file) {
  const typeOk = IMAGE_TYPES.includes(file.type) || /\.(png|jpe?g|webp)$/i.test(file.name);
  if (!typeOk) return 'Only PNG, JPG, JPEG or WEBP screenshots are supported.';
  if (file.size > LIMITS.imageBytes) {
    return `That image is ${formatBytes(file.size)}. The screenshot limit is ${formatBytes(LIMITS.imageBytes)}.`;
  }
  return null;
}

/** Drag-and-drop + click-to-browse file zone. */
function FileDrop({ id, accept, hint, icon: Icon, disabled, onFile, label }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  function handleDrop(e) {
    e.preventDefault();
    setDragOver(false);
    if (disabled) return;
    const file = e.dataTransfer?.files?.[0];
    if (file) onFile(file);
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      className={`flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed p-6 text-center transition-colors ${
        dragOver ? 'border-teal-600 bg-teal-50' : 'border-slate-300 bg-white'
      }`}
    >
      <Icon className="h-8 w-8 text-slate-400" aria-hidden="true" />
      <p className="text-sm text-slate-700">
        <span className="font-semibold">Drag and drop</span> a file here, or
      </p>
      <button
        type="button"
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
        className="btn-secondary py-2"
      >
        <UploadCloud className="h-4 w-4" aria-hidden="true" />
        Choose file
      </button>
      <p className="text-xs text-slate-500">{hint}</p>
      <input
        ref={inputRef}
        id={id}
        type="file"
        accept={accept}
        disabled={disabled}
        aria-label={label}
        className="sr-only"
        tabIndex={-1}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFile(file);
          e.target.value = '';
        }}
      />
    </div>
  );
}

export default function Workspace({ onResult, onJourney }) {
  const [tab, setTab] = useState('text');
  const [text, setText] = useState('');
  const [htmlText, setHtmlText] = useState('');
  const [htmlFile, setHtmlFile] = useState(null);
  const [shot, setShot] = useState(null);
  const [shotPreview, setShotPreview] = useState(null);

  const [samples, setSamples] = useState([]);
  const [samplesStatus, setSamplesStatus] = useState('loading');
  const [sample, setSample] = useState(null);

  const [fieldError, setFieldError] = useState(null);
  const [apiError, setApiError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [step, setStep] = useState(0);

  const steps = STEPS[tab];
  const busy = loading;

  // Load the sample list once.
  useEffect(() => {
    let cancelled = false;
    getSamples()
      .then((list) => {
        if (!cancelled) {
          setSamples(list);
          setSamplesStatus('ready');
        }
      })
      .catch(() => {
        if (!cancelled) setSamplesStatus('error');
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Preview URL for the selected screenshot; revoked when replaced or unmounted.
  useEffect(() => {
    if (!shot) {
      setShotPreview(null);
      return undefined;
    }
    const url = URL.createObjectURL(shot);
    setShotPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [shot]);

  // Approximate step text while the request is in flight (the API reports no progress).
  useEffect(() => {
    if (!loading) return undefined;
    const timer = setInterval(() => {
      setStep((s) => Math.min(s + 1, steps.length - 1));
    }, 1500);
    return () => clearInterval(timer);
  }, [loading, steps.length]);

  function resetMessages() {
    setFieldError(null);
    setApiError(null);
  }

  function switchTab(id) {
    if (busy) return;
    setTab(id);
    resetMessages();
  }

  function clearAll() {
    if (busy) return;
    setText('');
    setHtmlText('');
    setHtmlFile(null);
    setShot(null);
    setSample(null);
    resetMessages();
  }

  function chooseHtmlFile(file) {
    const problem = checkHtmlFile(file);
    if (problem) {
      setFieldError(problem);
      return;
    }
    resetMessages();
    setHtmlFile(file);
  }

  function chooseImage(file) {
    const problem = checkImageFile(file);
    if (problem) {
      setFieldError(problem);
      return;
    }
    resetMessages();
    setShot(file);
  }

  function loadSample(id) {
    const s = samples.find((x) => x.id === id);
    if (!s) {
      setSample(null);
      return;
    }
    resetMessages();
    setSample(s);
    if (s.input_type === 'text') {
      setText(s.content);
      setTab('text');
    } else {
      setHtmlFile(null);
      setHtmlText(s.content);
      setTab('html');
    }
  }

  function validate() {
    if (tab === 'text') {
      if (!text.trim()) return 'Paste the page text you want to audit.';
      if (text.length > LIMITS.textChars) {
        return `The text is ${text.length.toLocaleString()} characters. The limit is ${LIMITS.textChars.toLocaleString()}.`;
      }
      return null;
    }
    if (tab === 'html') {
      if (htmlFile) return checkHtmlFile(htmlFile);
      if (!htmlText.trim()) return 'Upload an .html file or paste HTML markup to audit.';
      const bytes = byteLength(htmlText);
      if (bytes > LIMITS.htmlBytes) {
        return `The pasted HTML is ${formatBytes(bytes)}. The limit is ${formatBytes(LIMITS.htmlBytes)}.`;
      }
      return null;
    }
    if (!shot) return 'Choose a screenshot (PNG, JPG or WEBP) to audit.';
    return checkImageFile(shot);
  }

  async function analyze() {
    const problem = validate();
    if (problem) {
      setFieldError(problem);
      return;
    }
    resetMessages();
    setStep(0);
    setLoading(true);
    try {
      // A loaded sample keeps its title only while its content is unchanged.
      const sampleName = (content) => (sample && sample.content === content ? sample.title : null);
      if (tab === 'text') {
        const response = await auditText(text);
        onResult({ response, inputType: 'text', inputText: text, inputName: sampleName(text) || 'Pasted text' });
      } else if (tab === 'html') {
        // Keep the markup client-side so the report can render it in a sandboxed, script-free preview.
        const htmlSource = htmlFile ? await htmlFile.text() : htmlText;
        const response = htmlFile ? await auditHtmlFile(htmlFile) : await auditHtmlText(htmlText);
        onResult({
          response, inputType: 'html', htmlSource,
          inputName: htmlFile ? htmlFile.name : sampleName(htmlText) || 'Pasted HTML',
        });
      } else {
        const response = await auditScreenshot(shot);
        onResult({ response, inputType: 'screenshot', imageUrl: URL.createObjectURL(shot), inputName: shot.name });
      }
    } catch (err) {
      setApiError(err?.message || 'Something went wrong while running the audit. Please try again.');
      setLoading(false);
    }
  }

  const canAnalyze =
    !busy &&
    (tab === 'text' ? text.trim().length > 0 : tab === 'html' ? !!htmlFile || htmlText.trim().length > 0 : !!shot);

  const messageId = 'workspace-field-error';

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <header className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-teal-700">Audit workspace</p>
          <h1 className="mt-1 text-2xl font-bold text-slate-900 sm:text-3xl">Check a page for dark patterns</h1>
          <p className="mt-2 text-sm text-slate-600">Choose an input type, then run the audit.</p>
        </div>
        {onJourney && (
          <button type="button" onClick={onJourney} className="btn-secondary shrink-0 self-start sm:self-auto">
            <Route className="h-4 w-4" aria-hidden="true" />
            Journey (multi-step)
          </button>
        )}
      </header>

      <div className="card p-4 sm:p-5">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div role="tablist" aria-label="Input type" className="grid grid-cols-3 rounded-lg bg-slate-100 p-1 sm:inline-flex">
            {TABS.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                id={`tab-${id}`}
                type="button"
                role="tab"
                aria-selected={tab === id}
                aria-controls={`panel-${id}`}
                tabIndex={tab === id ? 0 : -1}
                disabled={busy}
                onClick={() => switchTab(id)}
                onKeyDown={(e) => {
                  if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
                  const idx = TABS.findIndex((t) => t.id === tab);
                  const next = TABS[(idx + (e.key === 'ArrowRight' ? 1 : TABS.length - 1)) % TABS.length];
                  switchTab(next.id);
                  document.getElementById(`tab-${next.id}`)?.focus();
                }}
                className={`inline-flex items-center justify-center gap-1.5 whitespace-nowrap rounded-md px-3 py-2 text-sm font-medium transition-colors disabled:opacity-60 ${
                  tab === id ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <Icon className="h-4 w-4" aria-hidden="true" />
                {label}
              </button>
            ))}
          </div>

          <div className="sm:w-64">
            <label htmlFor="sample-select" className="label">
              Load sample
            </label>
            <select
              id="sample-select"
              className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 focus:border-teal-600 disabled:bg-slate-100"
              disabled={busy || samplesStatus !== 'ready' || samples.length === 0}
              value={sample?.id || ''}
              onChange={(e) => loadSample(e.target.value)}
            >
              <option value="">
                {samplesStatus === 'loading'
                  ? 'Loading samples…'
                  : samplesStatus === 'error'
                    ? 'Samples unavailable'
                    : samples.length === 0
                      ? 'No samples available'
                      : 'Choose a sample…'}
              </option>
              {samples.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.title} ({s.input_type === 'html' ? 'HTML' : 'text'})
                </option>
              ))}
            </select>
          </div>
        </div>

        {sample && (
          <div className="mt-4 flex items-start gap-3 rounded-lg bg-slate-50 p-3 text-xs text-slate-600 ring-1 ring-inset ring-slate-200">
            <BookOpen className="mt-0.5 h-4 w-4 shrink-0 text-slate-500" aria-hidden="true" />
            <div className="min-w-0 flex-1">
              <p className="font-semibold text-slate-800">{sample.title}</p>
              {sample.provenance && <p className="mt-0.5">Provenance: {sample.provenance}</p>}
              {sample.expected && <p className="mt-0.5">Expected: {sample.expected}</p>}
            </div>
            <button
              type="button"
              onClick={() => setSample(null)}
              className="rounded p-1 text-slate-500 hover:bg-slate-200 hover:text-slate-800"
              aria-label="Dismiss sample details"
            >
              <X className="h-4 w-4" aria-hidden="true" />
            </button>
          </div>
        )}

        <div id={`panel-${tab}`} role="tabpanel" aria-labelledby={`tab-${tab}`} className="mt-5">
          {tab === 'text' && (
            <div>
              <label htmlFor="text-input" className="label">
                Page text
              </label>
              <textarea
                id="text-input"
                value={text}
                disabled={busy}
                onChange={(e) => {
                  setText(e.target.value);
                  setSample(null);
                  resetMessages();
                }}
                rows={12}
                placeholder="Paste the visible text from a product, checkout or pop-up here."
                aria-describedby={`text-count ${fieldError ? messageId : ''}`.trim()}
                className="block w-full resize-y rounded-lg border border-slate-300 bg-white px-3 py-2.5 font-mono text-sm leading-relaxed text-slate-900 placeholder:text-slate-400 focus:border-teal-600 disabled:bg-slate-100"
              />
              <p
                id="text-count"
                className={`mt-1.5 text-right text-xs tabular-nums ${
                  text.length > LIMITS.textChars ? 'font-semibold text-red-700' : 'text-slate-500'
                }`}
              >
                {text.length.toLocaleString()} / {LIMITS.textChars.toLocaleString()} characters
              </p>
            </div>
          )}

          {tab === 'html' && (
            <div className="space-y-5">
              <div>
                <p className="label">HTML file</p>
                {htmlFile ? (
                  <div className="flex items-center justify-between gap-3 rounded-xl border border-slate-200 bg-white p-4">
                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium text-slate-900">{htmlFile.name}</p>
                      <p className="text-xs text-slate-500">{formatBytes(htmlFile.size)}</p>
                    </div>
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => setHtmlFile(null)}
                      className="btn-secondary py-1.5"
                    >
                      <X className="h-4 w-4" aria-hidden="true" />
                      Remove file
                    </button>
                  </div>
                ) : (
                  <FileDrop
                    id="html-file"
                    label="Upload an HTML file"
                    accept=".html,.htm,text/html"
                    icon={FileCode}
                    disabled={busy}
                    hint="Accepts .html or .htm, up to 2 MB"
                    onFile={chooseHtmlFile}
                  />
                )}
                {htmlFile && htmlText.trim() && (
                  <p className="mt-2 text-xs text-slate-500">The selected file is audited. Pasted HTML is ignored while a file is selected.</p>
                )}
              </div>

              <div>
                <label htmlFor="html-input" className="label">
                  Or paste HTML markup
                </label>
                <textarea
                  id="html-input"
                  value={htmlText}
                  disabled={busy}
                  onChange={(e) => {
                    setHtmlText(e.target.value);
                    setSample(null);
                    resetMessages();
                  }}
                  rows={8}
                  spellCheck={false}
                  placeholder="<div class='banner'>Hurry! Sale ends in 10 minutes</div>"
                  className="block w-full resize-y rounded-lg border border-slate-300 bg-white px-3 py-2.5 font-mono text-xs leading-relaxed text-slate-900 placeholder:text-slate-400 focus:border-teal-600 disabled:bg-slate-100"
                />
              </div>
            </div>
          )}

          {tab === 'screenshot' && (
            <div className="space-y-4">
              {shot && shotPreview ? (
                <div className="rounded-xl border border-slate-200 bg-white p-4">
                  <img
                    src={shotPreview}
                    alt={`Preview of ${shot.name}`}
                    className="mx-auto max-h-80 w-auto max-w-full rounded-lg border border-slate-200 object-contain"
                  />
                  <div className="mt-3 flex items-center justify-between gap-3">
                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium text-slate-900">{shot.name}</p>
                      <p className="text-xs text-slate-500">{formatBytes(shot.size)}</p>
                    </div>
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => setShot(null)}
                      className="btn-secondary py-1.5"
                    >
                      <X className="h-4 w-4" aria-hidden="true" />
                      Remove image
                    </button>
                  </div>
                </div>
              ) : (
                <FileDrop
                  id="shot-file"
                  label="Upload a screenshot"
                  accept=".png,.jpg,.jpeg,.webp,image/png,image/jpeg,image/webp"
                  icon={ImageIcon}
                  disabled={busy}
                  hint="PNG, JPG or WEBP, up to 8 MB"
                  onFile={chooseImage}
                />
              )}
              <p className="flex items-start gap-2 text-xs text-slate-500">
                <ScanSearch className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
                Text is read with OCR, then classified. Results are shown as boxes over the detected text regions.
              </p>
            </div>
          )}
        </div>

        {fieldError && (
          <p id={messageId} role="alert" className="mt-4 flex items-start gap-2 text-sm text-red-800">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
            {fieldError}
          </p>
        )}

        {apiError && (
          <div role="alert" className="mt-4 flex items-start gap-3 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-900">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
            <div>
              <p className="font-semibold">The audit did not complete.</p>
              <p className="mt-0.5">{apiError}</p>
            </div>
          </div>
        )}

        {loading && (
          <div role="status" aria-live="polite" className="mt-5 rounded-xl border border-teal-200 bg-teal-50/60 p-4">
            <div className="flex items-center gap-3 text-sm font-semibold text-teal-900">
              <Loader2 className="h-5 w-5 animate-spin" aria-hidden="true" />
              {steps[step]}
            </div>
            <div className="mt-4 space-y-2" aria-hidden="true">
              <div className="h-3 w-11/12 animate-pulse rounded bg-teal-200/70" />
              <div className="h-3 w-4/5 animate-pulse rounded bg-teal-200/70" />
              <div className="h-3 w-2/3 animate-pulse rounded bg-teal-200/70" />
            </div>
          </div>
        )}

        <div className="mt-6 flex flex-col-reverse gap-3 sm:flex-row sm:justify-end">
          <button type="button" onClick={clearAll} disabled={busy} className="btn-secondary">
            <Eraser className="h-4 w-4" aria-hidden="true" />
            Clear
          </button>
          <button
            type="button"
            onClick={analyze}
            disabled={!canAnalyze}
            className="btn-primary"
            aria-busy={loading}
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" /> : null}
            {loading ? 'Analyzing…' : 'Analyze'}
          </button>
        </div>
      </div>
    </div>
  );
}
