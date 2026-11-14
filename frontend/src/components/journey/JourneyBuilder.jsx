import { useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle, ArrowDown, ArrowUp, BookOpen, Copy, Eraser, FileCode, Image as ImageIcon, Loader2, Plus, Route, Trash2, Type,
  UploadCloud, X,
} from 'lucide-react';
import { auditJourney, getJourneySamples } from '../../api.js';
import { LIMITS, formatBytes } from '../../format.js';
import SyntheticBadge from './SyntheticBadge.jsx';
import {
  CHOICE_LABEL_MAX, MAX_STAGES, NAME_MAX, STAGE_TYPES, STEPS_MAX, STEPS_MIN, byteLength, checkHtmlFile, checkImageFile,
  fromSampleStages, fuseProblem, newStage, toRequest, validateStage,
} from './journeyUtils.js';

const INPUT_TYPES = [
  { id: 'text', label: 'Text', icon: Type },
  { id: 'html', label: 'HTML', icon: FileCode },
  { id: 'screenshot', label: 'Screenshot', icon: ImageIcon },
];

const inputCls =
  'block w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 placeholder:text-slate-400 focus:border-teal-600 disabled:bg-slate-100';

function FieldError({ id, msg }) {
  if (!msg) return null;
  return (
    <p id={id} className="mt-1 flex items-start gap-1.5 text-xs text-red-800">
      <AlertTriangle className="mt-px h-3.5 w-3.5 shrink-0" aria-hidden="true" />
      {msg}
    </p>
  );
}

function ShotPicker({ stage, disabled, onFile, onClear, error, errorId }) {
  const ref = useRef(null);
  const [url, setUrl] = useState(null);
  useEffect(() => {
    if (!stage.file) {
      setUrl(null);
      return undefined;
    }
    const u = URL.createObjectURL(stage.file);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [stage.file]);

  return (
    <div>
      {stage.file && url ? (
        <div className="flex items-center gap-3 rounded-lg border border-slate-200 bg-white p-3">
          <img src={url} alt={`Preview of ${stage.file.name}`} className="h-16 w-16 shrink-0 rounded border border-slate-200 object-cover" />
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium text-slate-900">{stage.file.name}</p>
            <p className="text-xs text-slate-500">{formatBytes(stage.file.size)}</p>
          </div>
          <button type="button" disabled={disabled} onClick={onClear} className="btn-secondary px-3 py-1.5">
            <X className="h-4 w-4" aria-hidden="true" /> Remove
          </button>
        </div>
      ) : (
        <div className="flex flex-col items-start gap-2 rounded-lg border-2 border-dashed border-slate-300 bg-white p-4 sm:flex-row sm:items-center">
          <button type="button" disabled={disabled} onClick={() => ref.current?.click()} className="btn-secondary py-2" aria-describedby={error ? errorId : undefined}>
            <UploadCloud className="h-4 w-4" aria-hidden="true" /> Choose screenshot
          </button>
          <p className="text-xs text-slate-500">PNG, JPG or WEBP, up to {formatBytes(LIMITS.imageBytes)}</p>
        </div>
      )}
      <input
        ref={ref}
        type="file"
        accept=".png,.jpg,.jpeg,.webp,image/png,image/jpeg,image/webp"
        className="sr-only"
        tabIndex={-1}
        aria-label={`Screenshot for ${stage.name || 'this stage'}`}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
          e.target.value = '';
        }}
      />
    </div>
  );
}

/** "Same page as…" select: declares that this stage shows an earlier stage's page in another format. */
function FuseSelect({ stage, index, stages, busy, error, onChange, id }) {
  if (index < 1 && !stage.fuseWith) return null;
  const earlier = stages.slice(0, index);
  const valid = earlier.some((s) => s.uid === stage.fuseWith);
  const label = (s, i) => `Stage ${i + 1}${s.name.trim() ? ` — ${s.name.trim()}` : ''}`;
  return (
    <div className="mt-4">
      <label htmlFor={id('fuse')} className="label flex items-center gap-1.5">
        <Copy className="h-3.5 w-3.5 text-slate-500" aria-hidden="true" />
        Same page as… <span className="font-normal text-slate-500">(optional)</span>
      </label>
      <select
        id={id('fuse')}
        value={stage.fuseWith}
        disabled={busy}
        onChange={(e) => onChange({ fuseWith: e.target.value })}
        aria-invalid={!!error}
        aria-describedby={`${id('fuse-help')}${error ? ` ${id('fuse-err')}` : ''}`}
        className={`${inputCls} sm:max-w-sm`}
      >
        <option value="">Not a duplicate — a separate page</option>
        {stage.fuseWith && !valid && <option value={stage.fuseWith}>(not an earlier stage)</option>}
        {earlier.map((s, i) => (
          <option key={s.uid} value={s.uid}>{label(s, i)}</option>
        ))}
      </select>
      <p id={id('fuse-help')} className="mt-1 text-[11px] text-slate-500">
        Use when this stage is the same page captured in another format (text/HTML/screenshot); its findings are merged, not counted twice.
      </p>
      <FieldError id={id('fuse-err')} msg={error} />
    </div>
  );
}

function StageEditor({ stage, index, count, stages, errors, busy, onChange, onMove, onRemove, onFileError }) {
  const id = (f) => `${stage.uid}-${f}`;
  const htmlFileRef = useRef(null);
  const set = (patch) => onChange(stage.uid, patch);

  async function loadHtmlFile(file) {
    const problem = checkHtmlFile(file);
    if (problem) {
      onFileError(stage.uid, problem);
      return;
    }
    const text = await file.text();
    set({ content: text });
  }

  function chooseShot(file) {
    const problem = checkImageFile(file);
    if (problem) {
      onFileError(stage.uid, problem);
      return;
    }
    set({ file });
  }

  const contentLen = stage.inputType === 'text' ? stage.content.length : stage.inputType === 'html' ? byteLength(stage.content) : 0;

  return (
    <li className="card p-4 sm:p-5" aria-labelledby={id('title')}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 id={id('title')} className="flex min-w-0 items-center gap-2 text-sm font-bold text-slate-900">
          <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-teal-700 text-xs text-white">{index + 1}</span>
          <span className="truncate">{stage.name.trim() || `Stage ${index + 1}`}</span>
          {stage.sampleSynthetic && stage.sampleContent === stage.content && (
            <span className="shrink-0 rounded bg-amber-100 px-1.5 py-0.5 text-[10px] font-bold uppercase text-amber-900">Synthetic</span>
          )}
        </h3>
        <div className="flex items-center gap-1">
          <button type="button" disabled={busy || index === 0} onClick={() => onMove(index, -1)} className="rounded-md p-2 text-slate-600 hover:bg-slate-100 disabled:opacity-40" aria-label={`Move stage ${index + 1} up`}>
            <ArrowUp className="h-4 w-4" aria-hidden="true" />
          </button>
          <button type="button" disabled={busy || index === count - 1} onClick={() => onMove(index, 1)} className="rounded-md p-2 text-slate-600 hover:bg-slate-100 disabled:opacity-40" aria-label={`Move stage ${index + 1} down`}>
            <ArrowDown className="h-4 w-4" aria-hidden="true" />
          </button>
          <button type="button" disabled={busy || count === 1} onClick={() => onRemove(stage.uid)} className="rounded-md p-2 text-slate-600 hover:bg-red-50 hover:text-red-700 disabled:opacity-40" aria-label={`Remove stage ${index + 1}`}>
            <Trash2 className="h-4 w-4" aria-hidden="true" />
          </button>
        </div>
      </div>

      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        <div>
          <label htmlFor={id('name')} className="label">Stage name</label>
          <input
            id={id('name')}
            value={stage.name}
            maxLength={NAME_MAX + 20}
            disabled={busy}
            onChange={(e) => set({ name: e.target.value })}
            placeholder="e.g. Product page"
            aria-invalid={!!errors.name}
            aria-describedby={errors.name ? id('name-err') : undefined}
            className={inputCls}
          />
          <FieldError id={id('name-err')} msg={errors.name} />
        </div>
        <div>
          <label htmlFor={id('type')} className="label">Stage type <span className="font-normal text-slate-500">(optional)</span></label>
          <select id={id('type')} value={stage.stageType} disabled={busy} onChange={(e) => set({ stageType: e.target.value })} className={inputCls}>
            <option value="">Not specified</option>
            {STAGE_TYPES.map((t) => (
              <option key={t} value={t}>{t.charAt(0).toUpperCase() + t.slice(1)}</option>
            ))}
          </select>
        </div>
      </div>

      <fieldset className="mt-4">
        <legend className="label">Input</legend>
        <div role="radiogroup" aria-label="Input type" className="grid grid-cols-3 rounded-lg bg-slate-100 p-1 sm:inline-flex">
          {INPUT_TYPES.map(({ id: t, label, icon: Icon }) => (
            <button
              key={t}
              type="button"
              role="radio"
              aria-checked={stage.inputType === t}
              disabled={busy}
              onClick={() => set({ inputType: t })}
              className={`inline-flex items-center justify-center gap-1.5 whitespace-nowrap rounded-md px-3 py-1.5 text-sm font-medium transition-colors disabled:opacity-60 ${
                stage.inputType === t ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Icon className="h-4 w-4" aria-hidden="true" />
              {label}
            </button>
          ))}
        </div>

        <div className="mt-3">
          {stage.inputType === 'screenshot' ? (
            <ShotPicker stage={stage} disabled={busy} onFile={chooseShot} onClear={() => set({ file: null })} error={errors.content} errorId={id('content-err')} />
          ) : (
            <>
              <label htmlFor={id('content')} className="sr-only">{stage.inputType === 'html' ? 'HTML markup' : 'Page text'} for stage {index + 1}</label>
              <textarea
                id={id('content')}
                value={stage.content}
                disabled={busy}
                rows={stage.inputType === 'html' ? 7 : 6}
                spellCheck={stage.inputType !== 'html'}
                onChange={(e) => set({ content: e.target.value })}
                placeholder={stage.inputType === 'html' ? '<div class="banner">Only 2 left!</div>' : 'Paste the visible text of this step.'}
                aria-invalid={!!errors.content}
                aria-describedby={`${id('count')}${errors.content ? ` ${id('content-err')}` : ''}`}
                className={`${inputCls} resize-y font-mono ${stage.inputType === 'html' ? 'text-xs' : 'text-sm'} leading-relaxed`}
              />
              <div className="mt-1 flex flex-wrap items-center justify-between gap-2">
                {stage.inputType === 'html' ? (
                  <>
                    <button type="button" disabled={busy} onClick={() => htmlFileRef.current?.click()} className="text-xs font-medium text-teal-800 hover:underline disabled:opacity-50">
                      Load from .html file
                    </button>
                    <input
                      ref={htmlFileRef}
                      type="file"
                      accept=".html,.htm,text/html"
                      className="sr-only"
                      tabIndex={-1}
                      aria-label={`HTML file for ${stage.name || 'this stage'}`}
                      onChange={(e) => {
                        const f = e.target.files?.[0];
                        if (f) loadHtmlFile(f);
                        e.target.value = '';
                      }}
                    />
                  </>
                ) : <span />}
                <p id={id('count')} className="text-xs tabular-nums text-slate-500">
                  {stage.inputType === 'html'
                    ? `${formatBytes(contentLen)} / ${formatBytes(LIMITS.htmlBytes)}`
                    : `${contentLen.toLocaleString()} / ${LIMITS.textChars.toLocaleString()} characters`}
                </p>
              </div>
            </>
          )}
          <FieldError id={id('content-err')} msg={errors.content} />
        </div>
      </fieldset>

      <FuseSelect
        stage={stage}
        index={index}
        stages={stages}
        busy={busy}
        error={fuseProblem(stage, stages)}
        onChange={set}
        id={id}
      />

      <details className="group mt-4 rounded-lg border border-slate-200" open={!!(stage.capturedAt || stage.choiceOn) || undefined}>
        <summary className="cursor-pointer list-none rounded-lg px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 [&::-webkit-details-marker]:hidden">
          Optional: capture time and declared choice path
        </summary>
        <div className="grid gap-4 border-t border-slate-200 p-3 sm:grid-cols-2">
          <div>
            <label htmlFor={id('time')} className="label">Captured at</label>
            <div className="flex gap-2">
              <input
                id={id('time')}
                type="datetime-local"
                step="1"
                value={stage.capturedAt}
                disabled={busy}
                onChange={(e) => set({ capturedAt: e.target.value })}
                aria-invalid={!!errors.capturedAt}
                className={inputCls}
              />
              {stage.capturedAt && (
                <button type="button" disabled={busy} onClick={() => set({ capturedAt: '' })} className="rounded-md px-2 text-slate-500 hover:bg-slate-100" aria-label="Clear capture time">
                  <X className="h-4 w-4" aria-hidden="true" />
                </button>
              )}
            </div>
            <p className="mt-1 text-[11px] text-slate-500">Sent with your browser’s UTC offset. Enables timestamp ordering and countdown checks.</p>
            <FieldError msg={errors.capturedAt} />
          </div>
          <div>
            <label className="flex items-center gap-2 text-sm font-medium text-slate-700">
              <input type="checkbox" checked={stage.choiceOn} disabled={busy} onChange={(e) => set({ choiceOn: e.target.checked })} className="h-4 w-4 rounded border-slate-300 text-teal-700 focus:ring-teal-600" />
              Declare a choice path
            </label>
            <p className="mt-1 text-[11px] text-slate-500">Steps needed to accept vs decline (e.g. subscribe vs cancel), counted by you. Used for interaction asymmetry.</p>
            {stage.choiceOn && (
              <div className="mt-2 space-y-2">
                <div>
                  <label htmlFor={id('cl')} className="sr-only">Choice label</label>
                  <input id={id('cl')} value={stage.choiceLabel} maxLength={CHOICE_LABEL_MAX + 20} disabled={busy} onChange={(e) => set({ choiceLabel: e.target.value })} placeholder="Choice, e.g. Cancel membership" className={inputCls} aria-invalid={!!errors.choiceLabel} />
                  <FieldError msg={errors.choiceLabel} />
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label htmlFor={id('as')} className="text-xs font-medium text-slate-600">Accept steps</label>
                    <input id={id('as')} type="number" inputMode="numeric" min={STEPS_MIN} max={STEPS_MAX} step="1" value={stage.acceptSteps} disabled={busy} onChange={(e) => set({ acceptSteps: e.target.value })} className={inputCls} aria-invalid={!!errors.acceptSteps} />
                    <FieldError msg={errors.acceptSteps} />
                  </div>
                  <div>
                    <label htmlFor={id('ds')} className="text-xs font-medium text-slate-600">Decline steps</label>
                    <input id={id('ds')} type="number" inputMode="numeric" min={STEPS_MIN} max={STEPS_MAX} step="1" value={stage.declineSteps} disabled={busy} onChange={(e) => set({ declineSteps: e.target.value })} className={inputCls} aria-invalid={!!errors.declineSteps} />
                    <FieldError msg={errors.declineSteps} />
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </details>
    </li>
  );
}

/**
 * Journey builder. `draft` restores a previous builder state ({stages, sample}); `onDraftChange`
 * receives every change so the app can offer "Edit journey" from the results view.
 */
export default function JourneyBuilder({ draft, onDraftChange, onResult }) {
  const [stages, setStages] = useState(() => draft?.stages?.length ? draft.stages : [newStage()]);
  const [sample, setSample] = useState(draft?.sample || null);
  const [samples, setSamples] = useState([]);
  const [samplesStatus, setSamplesStatus] = useState('loading');
  const [errors, setErrors] = useState({});
  const [formError, setFormError] = useState(null);
  const [apiError, setApiError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    onDraftChange?.({ stages, sample });
  }, [stages, sample, onDraftChange]);

  useEffect(() => {
    let cancelled = false;
    getJourneySamples()
      .then((list) => {
        if (cancelled) return;
        setSamples(list.filter((s) => Array.isArray(s?.stages) && s.stages.length > 0));
        setSamplesStatus('ready');
      })
      .catch(() => !cancelled && setSamplesStatus('error'));
    return () => {
      cancelled = true;
    };
  }, []);

  // A loaded demo stays "loaded" (badge + provenance) only while at least one stage still holds its content unchanged.
  const sampleIntact = useMemo(
    () => !!sample && stages.some((s) => s.sampleContent != null && s.sampleContent === s.content),
    [sample, stages],
  );

  function update(uid, patch) {
    setStages((list) => list.map((s) => (s.uid === uid ? { ...s, ...patch } : s)));
    setErrors((e) => {
      if (!e[uid]) return e;
      const next = { ...e[uid] };
      for (const k of Object.keys(patch)) {
        if (k === 'file') delete next.content;
        else delete next[k];
      }
      if (patch.inputType) delete next.content;
      if (patch.choiceOn === false) ['choiceLabel', 'acceptSteps', 'declineSteps'].forEach((k) => delete next[k]);
      return { ...e, [uid]: next };
    });
    setFormError(null);
    setApiError(null);
  }

  function fileError(uid, msg) {
    setErrors((e) => ({ ...e, [uid]: { ...(e[uid] || {}), content: msg } }));
  }

  function move(index, dir) {
    setStages((list) => {
      const j = index + dir;
      if (j < 0 || j >= list.length) return list;
      const next = [...list];
      [next[index], next[j]] = [next[j], next[index]];
      return next;
    });
  }

  function remove(uid) {
    // stages that were marked as the same page as the removed one become separate pages again
    setStages((list) => (list.length > 1 ? list.filter((s) => s.uid !== uid).map((s) => (s.fuseWith === uid ? { ...s, fuseWith: '' } : s)) : list));
    setErrors((e) => {
      const { [uid]: _drop, ...rest } = e;
      return rest;
    });
  }

  function add() {
    setStages((list) => (list.length < MAX_STAGES ? [...list, newStage()] : list));
    setFormError(null);
  }

  function hasUserContent() {
    return stages.some((s) => s.name.trim() || s.content.trim() || s.file);
  }

  function loadSample(id) {
    const s = samples.find((x) => x.id === id);
    if (!s) return;
    if (hasUserContent() && !window.confirm('Replace the current stages with this demo journey?')) return;
    const loaded = fromSampleStages(s.stages.slice(0, MAX_STAGES), s.synthetic);
    setStages(loaded);
    setSample({ id: s.id, title: s.title, description: s.description, synthetic: !!s.synthetic, provenance: s.provenance });
    setErrors({});
    setFormError(null);
    setApiError(null);
  }

  function clearAll() {
    setStages([newStage()]);
    setSample(null);
    setErrors({});
    setFormError(null);
    setApiError(null);
  }

  async function submit() {
    if (stages.length < 1 || stages.length > MAX_STAGES) {
      setFormError(`A journey needs between 1 and ${MAX_STAGES} stages.`);
      return;
    }
    const all = {};
    let bad = 0;
    for (const st of stages) {
      const e = validateStage(st, stages);
      if (Object.keys(e).length) {
        all[st.uid] = e;
        bad += 1;
      }
    }
    setErrors(all);
    if (bad) {
      setFormError(`${bad} stage${bad === 1 ? ' needs' : 's need'} attention before the journey can be audited.`);
      const first = stages.find((s) => all[s.uid]);
      document.getElementById(`${first.uid}-title`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      return;
    }
    setFormError(null);
    setApiError(null);
    setLoading(true);
    try {
      const { stages: payload, files } = toRequest(stages);
      const response = await auditJourney(payload, files);
      // Screenshot previews: the API returns no image URL, so keep local object URLs (owned and revoked by the app).
      const stageImages = {};
      stages.forEach((st, i) => {
        if (st.inputType === 'screenshot' && st.file) stageImages[i] = URL.createObjectURL(st.file);
      });
      // Keep the original screenshot Files (by multipart field) so the result view can offer "Verify by replay".
      onResult({ response, stageImages, files, sample: sampleIntact ? sample : null });
    } catch (err) {
      setApiError(err?.message || 'Something went wrong while running the journey audit. Please try again.');
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <header>
        <p className="text-xs font-semibold uppercase tracking-wider text-teal-700">Journey audit</p>
        <h1 className="mt-1 text-2xl font-bold text-slate-900 sm:text-3xl">Audit a multi-step journey</h1>
        <p className="mt-2 text-sm text-slate-600">
          Add each step of a flow — product page, cart, checkout — in order. ClearCart audits every stage, then measures how pressure
          builds across the journey.
        </p>
      </header>

      <div className="card flex flex-col gap-3 p-4 sm:flex-row sm:items-end sm:justify-between">
        <div className="sm:w-80">
          <label htmlFor="journey-sample" className="label">Load demo journey</label>
          <select
            id="journey-sample"
            className={inputCls}
            disabled={loading || samplesStatus !== 'ready' || samples.length === 0}
            value=""
            onChange={(e) => loadSample(e.target.value)}
          >
            <option value="">
              {samplesStatus === 'loading' ? 'Loading demo journeys…' : samplesStatus === 'error' ? 'Demo journeys unavailable' : samples.length === 0 ? 'No demo journeys available' : 'Choose a demo journey…'}
            </option>
            {samples.map((s) => (
              <option key={s.id} value={s.id}>
                {s.title} ({s.stages.length} stages{s.synthetic ? ', synthetic' : ''})
              </option>
            ))}
          </select>
        </div>
        <p className="text-xs text-slate-500 sm:max-w-[16rem] sm:text-right">
          {stages.length} of {MAX_STAGES} stages · text, HTML or screenshot per stage
        </p>
      </div>

      {sample && sampleIntact && (
        <div className="flex items-start gap-3 rounded-lg bg-slate-50 p-3 text-xs text-slate-600 ring-1 ring-inset ring-slate-200">
          <BookOpen className="mt-0.5 h-4 w-4 shrink-0 text-slate-500" aria-hidden="true" />
          <div className="min-w-0 flex-1 space-y-1">
            <p className="font-semibold text-slate-800">{sample.title}</p>
            {sample.synthetic && <SyntheticBadge />}
            {sample.description && <p>{sample.description}</p>}
            {sample.provenance && <p>Provenance: {sample.provenance}</p>}
          </div>
          <button type="button" onClick={() => setSample(null)} className="rounded p-1 text-slate-500 hover:bg-slate-200 hover:text-slate-800" aria-label="Dismiss demo journey details">
            <X className="h-4 w-4" aria-hidden="true" />
          </button>
        </div>
      )}

      <ol className="space-y-4" aria-label="Journey stages">
        {stages.map((st, i) => (
          <StageEditor
            key={st.uid}
            stage={st}
            index={i}
            count={stages.length}
            stages={stages}
            errors={errors[st.uid] || {}}
            busy={loading}
            onChange={update}
            onMove={move}
            onRemove={remove}
            onFileError={fileError}
          />
        ))}
      </ol>

      <button type="button" onClick={add} disabled={loading || stages.length >= MAX_STAGES} className="btn-secondary w-full border-dashed">
        <Plus className="h-4 w-4" aria-hidden="true" />
        {stages.length >= MAX_STAGES ? `Maximum of ${MAX_STAGES} stages reached` : 'Add stage'}
      </button>

      {formError && (
        <p role="alert" className="flex items-start gap-2 text-sm text-red-800">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
          {formError}
        </p>
      )}
      {apiError && (
        <div role="alert" className="flex items-start gap-3 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-900">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
          <div>
            <p className="font-semibold">The journey audit did not complete.</p>
            <p className="mt-0.5">{apiError}</p>
          </div>
        </div>
      )}
      {loading && (
        <div role="status" aria-live="polite" className="flex items-center gap-3 rounded-xl border border-teal-200 bg-teal-50/60 p-4 text-sm font-semibold text-teal-900">
          <Loader2 className="h-5 w-5 animate-spin" aria-hidden="true" />
          Auditing {stages.length} stage{stages.length === 1 ? '' : 's'} and computing journey indices…
        </div>
      )}

      <div className="flex flex-col-reverse gap-3 sm:flex-row sm:justify-end">
        <button type="button" onClick={clearAll} disabled={loading} className="btn-secondary">
          <Eraser className="h-4 w-4" aria-hidden="true" /> Clear
        </button>
        <button type="button" onClick={submit} disabled={loading} className="btn-primary" aria-busy={loading}>
          {loading ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" /> : <Route className="h-4 w-4" aria-hidden="true" />}
          {loading ? 'Auditing journey…' : 'Audit journey'}
        </button>
      </div>
    </div>
  );
}
