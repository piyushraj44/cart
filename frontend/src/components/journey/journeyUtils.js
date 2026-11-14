// Helpers for the journey builder: limits, validation and timestamp conversion.
import { LIMITS, formatBytes } from '../../format.js';

export const MAX_STAGES = 8;
export const NAME_MAX = 80;
export const CHOICE_LABEL_MAX = 200;
export const STEPS_MIN = 1;
export const STEPS_MAX = 50;

// Mirrors the backend's canonical stage order (backend/intelligence/config.py CANONICAL_STAGES).
export const STAGE_TYPES = ['landing', 'product', 'cart', 'checkout', 'payment', 'confirmation', 'exit'];

const IMAGE_TYPES = ['image/png', 'image/jpeg', 'image/webp'];

let uidCounter = 0;
export function newStage(partial = {}) {
  uidCounter += 1;
  return {
    uid: `st${Date.now().toString(36)}${uidCounter}`,
    name: '',
    stageType: '',
    inputType: 'text',
    content: '',
    file: null,
    capturedAt: '', // datetime-local value (local wall time)
    capturedAtSource: null, // { local, iso } when loaded from a demo journey, to send the original ISO back unchanged
    choiceOn: false,
    choiceLabel: '',
    acceptSteps: '',
    declineSteps: '',
    fuseWith: '', // uid of an EARLIER stage that shows the same page in another modality ('' = none)
    sampleContent: null, // content as loaded from a demo journey; `synthetic` is sent only while unchanged
    sampleSynthetic: false,
    ...partial,
  };
}

export function byteLength(str) {
  return new TextEncoder().encode(str).length;
}

const pad = (n) => String(n).padStart(2, '0');

/** datetime-local value -> ISO-8601 with the browser's UTC offset, e.g. 2026-11-12T09:30:00+05:30. */
export function localToIso(local) {
  if (!local) return null;
  const d = new Date(local);
  if (Number.isNaN(d.getTime())) return null;
  const off = -d.getTimezoneOffset();
  const sign = off >= 0 ? '+' : '-';
  const abs = Math.abs(off);
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` +
    `T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}` +
    `${sign}${pad(Math.floor(abs / 60))}:${pad(abs % 60)}`
  );
}

/** ISO-8601 -> datetime-local value (seconds precision) in the browser's time zone. */
export function isoToLocal(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` +
    `T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
  );
}

export function checkImageFile(file) {
  const typeOk = IMAGE_TYPES.includes(file.type) || /\.(png|jpe?g|webp)$/i.test(file.name);
  if (!typeOk) return 'Only PNG, JPG, JPEG or WEBP screenshots are supported.';
  if (file.size > LIMITS.imageBytes) {
    return `That image is ${formatBytes(file.size)}. The screenshot limit is ${formatBytes(LIMITS.imageBytes)}.`;
  }
  return null;
}

export function checkHtmlFile(file) {
  if (!/\.(html?|htm)$/i.test(file.name)) return 'Only .html or .htm files are supported.';
  if (file.size > LIMITS.htmlBytes) {
    return `That file is ${formatBytes(file.size)}. The HTML limit is ${formatBytes(LIMITS.htmlBytes)}.`;
  }
  return null;
}

function parseSteps(v) {
  const s = String(v).trim();
  if (!/^\d+$/.test(s)) return null;
  const n = Number(s);
  return n >= STEPS_MIN && n <= STEPS_MAX ? n : null;
}

/**
 * Problem with a stage's "same page as" reference, or null. `stages` is the ordered builder list;
 * the referenced stage must come EARLIER in the journey (the API takes a 0-based earlier index).
 */
export function fuseProblem(st, stages) {
  if (!st.fuseWith || !Array.isArray(stages)) return null;
  const own = stages.findIndex((s) => s.uid === st.uid);
  const target = stages.findIndex((s) => s.uid === st.fuseWith);
  if (target < 0) return 'The stage this was marked as the same page as no longer exists. Choose another or clear the selection.';
  if (own <= 0 || target >= own) return 'Must reference an earlier stage. Move this stage below the one it duplicates, or clear the selection.';
  return null;
}

/** Returns an object of field -> message for one stage (empty when valid). `stages` enables cross-stage checks. */
export function validateStage(st, stages) {
  const e = {};
  const fp = fuseProblem(st, stages);
  if (fp) e.fuseWith = fp;
  const name = st.name.trim();
  if (!name) e.name = 'Give the stage a name.';
  else if (name.length > NAME_MAX) e.name = `Keep the name to ${NAME_MAX} characters.`;

  if (st.inputType === 'text') {
    if (!st.content.trim()) e.content = 'Paste the text shown at this stage.';
    else if (st.content.length > LIMITS.textChars) {
      e.content = `The text is ${st.content.length.toLocaleString()} characters. The limit is ${LIMITS.textChars.toLocaleString()}.`;
    }
  } else if (st.inputType === 'html') {
    if (!st.content.trim()) e.content = 'Paste the HTML of this stage or load an .html file.';
    else {
      const bytes = byteLength(st.content);
      if (bytes > LIMITS.htmlBytes) e.content = `The HTML is ${formatBytes(bytes)}. The limit is ${formatBytes(LIMITS.htmlBytes)}.`;
    }
  } else if (!st.file) {
    e.content = 'Choose a screenshot (PNG, JPG or WEBP) for this stage.';
  } else {
    const p = checkImageFile(st.file);
    if (p) e.content = p;
  }

  if (st.capturedAt && !localToIso(st.capturedAt)) e.capturedAt = 'Enter a valid date and time, or clear the field.';

  if (st.choiceOn) {
    const label = st.choiceLabel.trim();
    if (!label) e.choiceLabel = 'Describe the choice, e.g. “Cancel subscription”.';
    else if (label.length > CHOICE_LABEL_MAX) e.choiceLabel = `Keep the label to ${CHOICE_LABEL_MAX} characters.`;
    if (parseSteps(st.acceptSteps) == null) e.acceptSteps = `Whole number from ${STEPS_MIN} to ${STEPS_MAX}.`;
    if (parseSteps(st.declineSteps) == null) e.declineSteps = `Whole number from ${STEPS_MIN} to ${STEPS_MAX}.`;
  }
  return e;
}

/** Builder stages -> { stages: JourneyStageInput[], files: {field: File} } for POST /api/audit/journey. */
export function toRequest(stages) {
  const files = {};
  const out = stages.map((st, i) => {
    const s = { name: st.name.trim(), input_type: st.inputType };
    if (st.stageType) s.stage_type = st.stageType;
    if (st.inputType === 'screenshot') {
      const field = `stage_${i}_file`;
      s.file_field = field;
      files[field] = st.file;
    } else {
      s.content = st.content;
    }
    if (st.capturedAt) {
      s.captured_at =
        st.capturedAtSource && st.capturedAtSource.local === st.capturedAt ? st.capturedAtSource.iso : localToIso(st.capturedAt);
    }
    if (st.choiceOn) {
      s.choice_path = {
        label: st.choiceLabel.trim(),
        accept_steps: parseSteps(st.acceptSteps),
        decline_steps: parseSteps(st.declineSteps),
      };
    }
    if (st.fuseWith) {
      const target = stages.findIndex((x) => x.uid === st.fuseWith);
      if (target >= 0 && target < i) s.fuse_with = target;
    }
    if (st.sampleSynthetic && st.inputType !== 'screenshot' && st.sampleContent === st.content) s.synthetic = true;
    return s;
  });
  return { stages: out, files };
}

/** Demo journey stages -> builder stages, resolving `fuse_with` indices to stage uids. */
export function fromSampleStages(list, sampleSynthetic) {
  const out = list.map((st) => fromSampleStage(st, sampleSynthetic));
  list.forEach((st, i) => {
    const f = st?.fuse_with;
    if (Number.isInteger(f) && f >= 0 && f < i) out[i].fuseWith = out[f].uid;
  });
  return out;
}

/** Demo journey stage (JourneyStageInput) -> builder stage. */
export function fromSampleStage(s, sampleSynthetic) {
  const local = isoToLocal(s.captured_at);
  const cp = s.choice_path;
  return newStage({
    name: s.name || '',
    stageType: STAGE_TYPES.includes(s.stage_type) ? s.stage_type : '',
    inputType: s.input_type === 'html' ? 'html' : 'text',
    content: s.content || '',
    capturedAt: local,
    capturedAtSource: local ? { local, iso: s.captured_at } : null,
    choiceOn: !!cp,
    choiceLabel: cp?.label || '',
    acceptSteps: cp?.accept_steps != null ? String(cp.accept_steps) : '',
    declineSteps: cp?.decline_steps != null ? String(cp.decline_steps) : '',
    sampleContent: s.content || '',
    sampleSynthetic: !!(sampleSynthetic || s.synthetic),
  });
}
