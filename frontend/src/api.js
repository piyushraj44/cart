// All network calls live here. Every error is normalised to ApiError so the
// UI can show `error.message` in a banner without knowing transport details.

export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

const FALLBACK_MESSAGES = {
  413: 'The file is too large for the server to process.',
  415: 'That file type is not supported.',
  422: 'The input was empty or could not be read.',
  503: 'Screenshot analysis (OCR) is not available on this server.',
  500: 'The server hit an unexpected error. Please try again.',
};

function detailToMessage(detail) {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    // FastAPI validation errors: [{ loc, msg, ... }]
    const msgs = detail
      .map((d) => (d && typeof d.msg === 'string' ? d.msg : null))
      .filter(Boolean);
    if (msgs.length) return msgs.join('; ');
  }
  return null;
}

async function normaliseError(res) {
  let message = null;
  try {
    const body = await res.json();
    message = detailToMessage(body?.detail);
  } catch {
    // Non-JSON error body: fall through to status-based message.
  }
  if (!message) message = FALLBACK_MESSAGES[res.status] || `Request failed (HTTP ${res.status}).`;
  return new ApiError(message, res.status);
}

async function send(url, init) {
  let res;
  try {
    res = await fetch(url, init);
  } catch {
    throw new ApiError('Cannot reach the ClearCart server. Check that the backend is running.');
  }
  if (!res.ok) throw await normaliseError(res);
  return res;
}

async function getJson(url) {
  const res = await send(url, { headers: { Accept: 'application/json' } });
  try {
    return await res.json();
  } catch {
    throw new ApiError('The server returned an unreadable response.', res.status);
  }
}

async function postJson(url, body) {
  const res = await send(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(body),
  });
  return res.json();
}

async function postForm(url, formData) {
  const res = await send(url, {
    method: 'POST',
    headers: { Accept: 'application/json' },
    body: formData,
  });
  return res.json();
}

export function getHealth() {
  return getJson('/health');
}

export function getModelInfo() {
  return getJson('/api/model-info');
}

export async function getSamples() {
  const data = await getJson('/api/samples');
  return Array.isArray(data) ? data : [];
}

export function auditText(text) {
  return postJson('/api/audit/text', { text });
}

/** Audit an uploaded .html/.htm file. */
export function auditHtmlFile(file) {
  const fd = new FormData();
  fd.append('file', file, file.name);
  return postForm('/api/audit/html', fd);
}

/** Audit pasted HTML (or an HTML sample) via the `html` form field. */
export function auditHtmlText(html) {
  const fd = new FormData();
  fd.append('html', html);
  return postForm('/api/audit/html', fd);
}

export function auditScreenshot(file) {
  const fd = new FormData();
  fd.append('file', file, file.name);
  return postForm('/api/audit/screenshot', fd);
}

/** Returns a Blob of the PDF report for an AuditResponse. */
export async function exportPdf(auditResponse) {
  const res = await send('/api/export/pdf', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/pdf' },
    body: JSON.stringify(auditResponse),
  });
  return res.blob();
}

/* ------------------------------------------------------------------ journey audits */

/** Demo journeys (text/html stages only). Each: {id, title, description, synthetic, provenance, stages}. */
export async function getJourneySamples() {
  const data = await getJson('/api/samples/journeys');
  return Array.isArray(data) ? data : [];
}

/**
 * Audit an ordered multi-step journey.
 * @param {object[]} stages JourneyStageInput[]; screenshot stages carry `file_field`.
 * @param {Record<string, File>} files multipart field name -> screenshot File.
 */
export async function auditJourney(stages, files = {}) {
  const fd = new FormData();
  fd.append('stages', JSON.stringify(stages));
  for (const [field, file] of Object.entries(files)) fd.append(field, file, file.name);
  try {
    return await postForm('/api/audit/journey', fd);
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) {
      throw new ApiError('This server does not offer journey audits yet (endpoint not found). Update the ClearCart backend and try again.', 404);
    }
    throw err;
  }
}

/** Returns a Blob of the PDF report for a JourneyResponse. */
export async function exportJourneyPdf(journeyResponse) {
  const res = await send('/api/export/journey-pdf', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/pdf' },
    body: JSON.stringify(journeyResponse),
  });
  return res.blob();
}

/**
 * Re-run a journey audit from its ReplayRecord and compare it with the recorded outputs.
 * Screenshot stages are stored only as hashes, so their original Files must be re-sent under the
 * recorded `file_field` names.
 * @param {object} replay JourneyResponse.replay (ReplayRecord)
 * @param {Record<string, File>} files multipart field name -> original screenshot File
 * @returns {Promise<{identical: boolean, differences: string[], journey: object}>}
 */
export async function replayJourney(replay, files = {}) {
  const fd = new FormData();
  fd.append('replay', JSON.stringify(replay));
  for (const [field, file] of Object.entries(files)) fd.append(field, file, file.name);
  try {
    return await postForm('/api/audit/journey/replay', fd);
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) {
      throw new ApiError('This server does not offer journey replay yet (endpoint not found). Update the ClearCart backend and try again.', 404);
    }
    throw err;
  }
}
