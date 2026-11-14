import { useState } from 'react';
import { AlertTriangle, CheckCircle2, ChevronDown, Fingerprint, Loader2, RotateCcw, XCircle } from 'lucide-react';
import { replayJourney } from '../../api.js';
import Collapsible, { Chip } from '../Collapsible.jsx';
import { ReproducibilityBlock, fmtGeneric, humanKeyLabel } from '../intel/shared.jsx';

/** Result of POST /api/audit/journey/replay: {identical, differences[], journey}. */
export function ReplayBanner({ outcome }) {
  if (!outcome) return null;
  const diffs = Array.isArray(outcome.differences) ? outcome.differences : [];
  if (outcome.identical) {
    return (
      <div role="status" className="flex items-start gap-3 rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-950" data-testid="replay-banner">
        <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-emerald-700" aria-hidden="true" />
        <div className="min-w-0">
          <p className="font-semibold">Replay reproduced identical results</p>
          {outcome.journey?.journey_id && (
            <p className="mt-0.5 text-xs">
              Re-run as journey <span className="break-all font-mono">{outcome.journey.journey_id}</span>; every recorded output matched.
            </p>
          )}
        </div>
      </div>
    );
  }
  return (
    <div role="alert" className="flex items-start gap-3 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-950" data-testid="replay-banner">
      <XCircle className="mt-0.5 h-5 w-5 shrink-0 text-amber-700" aria-hidden="true" />
      <div className="min-w-0">
        <p className="font-semibold">Replay did not reproduce the recorded results</p>
        {diffs.length > 0 ? (
          <ul className="mt-1 list-disc space-y-0.5 pl-4 text-xs">
            {diffs.map((d, i) => <li key={i} className="break-words">{d}</li>)}
          </ul>
        ) : (
          <p className="mt-0.5 text-xs">The server reported a mismatch but listed no differences.</p>
        )}
      </div>
    </div>
  );
}

/**
 * Reproducibility block for a journey: versions, input fingerprints and a "Verify by replay" action.
 * `files` maps multipart field names to the original screenshot Files kept from the builder.
 */
export default function ReplayPanel({ response, files, stageName, open, onOpenChange }) {
  const replay = response?.replay || null;
  const versions = response?.versions || replay?.versions || null;
  const [running, setRunning] = useState(false);
  const [outcome, setOutcome] = useState(null);
  const [error, setError] = useState(null);
  if (!replay && !versions) return null;

  const inputs = Array.isArray(replay?.inputs) ? replay.inputs : [];
  const hashes = Array.isArray(replay?.input_sha256) ? replay.input_sha256 : [];
  const shots = inputs.map((inp, i) => ({ inp, i })).filter(({ inp }) => inp?.input_type === 'screenshot');
  const missing = shots.filter(({ inp }) => !inp.file_field || !files?.[inp.file_field]);
  const canReplay = !!replay && missing.length === 0;
  const expected = replay?.expected && typeof replay.expected === 'object' ? Object.entries(replay.expected) : [];

  async function run() {
    setRunning(true);
    setError(null);
    setOutcome(null);
    try {
      const shotFiles = {};
      for (const { inp } of shots) shotFiles[inp.file_field] = files[inp.file_field];
      setOutcome(await replayJourney(replay, shotFiles));
    } catch (err) {
      setError(err?.message || 'The replay could not be run.');
    } finally {
      setRunning(false);
    }
  }

  const nVersions = Object.keys(versions || {}).length;
  return (
    <Collapsible
      title="Reproducibility and replay"
      icon={Fingerprint}
      open={open}
      onOpenChange={onOpenChange}
      summary={
        <>
          {nVersions > 0 && <Chip>{nVersions} version{nVersions === 1 ? '' : 's'} recorded</Chip>}
          {replay && <Chip tone={canReplay ? 'teal' : 'muted'}>{canReplay ? 'replay available' : 'replay needs original files'}</Chip>}
          {outcome && <Chip tone={outcome.identical ? 'green' : 'amber'}>{outcome.identical ? 'reproduced' : 'differences found'}</Chip>}
        </>
      }
    >
      <div className="space-y-3" data-testid="journey-repro">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <p className="min-w-0 text-sm text-slate-600">
          {replay
            ? 'Re-runs the audit from the recorded inputs and versions and checks that every recorded output is reproduced.'
            : 'This result has no replay record; the versions used are listed below.'}
        </p>
        {replay && (
          <button
            type="button"
            onClick={run}
            disabled={!canReplay || running}
            className="btn-secondary shrink-0"
            aria-busy={running}
            aria-describedby={missing.length ? 'replay-missing' : undefined}
          >
            {running ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" /> : <RotateCcw className="h-4 w-4" aria-hidden="true" />}
            {running ? 'Replaying…' : 'Verify by replay'}
          </button>
        )}
      </div>

      {(missing.length > 0 || error || outcome) && (
        <div className="space-y-2">
          {missing.length > 0 && (
            <p id="replay-missing" className="flex items-start gap-2 rounded-lg bg-slate-50 p-2.5 text-xs text-slate-700 ring-1 ring-inset ring-slate-200">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-600" aria-hidden="true" />
              <span>
                Screenshot stages cannot be replayed without the original files ({missing.map(({ i }) => stageName(i)).join(', ')}): only
                their hashes are recorded. Run the journey from the builder in this session to replay it.
              </span>
            </p>
          )}
          {error && (
            <p role="alert" className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 p-2.5 text-xs text-red-900">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
              {error}
            </p>
          )}
          <ReplayBanner outcome={outcome} />
        </div>
      )}

      <div className="rounded-lg border border-slate-200">
        <ReproducibilityBlock versions={versions} id="j-repro" bare title="Versions and recorded inputs">
          {inputs.length > 0 && (
            <details className="group/fp">
              <summary className="flex cursor-pointer list-none items-center gap-1.5 text-xs font-semibold text-slate-700 [&::-webkit-details-marker]:hidden">
                <ChevronDown className="h-3.5 w-3.5 text-slate-500 transition-transform group-open/fp:rotate-180" aria-hidden="true" />
                Input fingerprints (SHA-256, {inputs.length})
              </summary>
              <ul className="mt-1.5 space-y-1 text-xs">
                {inputs.map((inp, i) => (
                  <li key={i} className="flex min-w-0 flex-wrap items-baseline gap-x-2">
                    <span className="text-slate-600">{stageName(i)} · {inp?.input_type}</span>
                    <span className="min-w-0 break-all font-mono text-slate-900">{hashes[i] || '—'}</span>
                  </li>
                ))}
              </ul>
            </details>
          )}
          {expected.length > 0 && (
            <details className="group/ex">
              <summary className="flex cursor-pointer list-none items-center gap-1.5 text-xs font-semibold text-slate-700 [&::-webkit-details-marker]:hidden">
                <ChevronDown className="h-3.5 w-3.5 text-slate-500 transition-transform group-open/ex:rotate-180" aria-hidden="true" />
                Recorded outputs compared on replay ({expected.length})
              </summary>
              <dl className="mt-1.5 space-y-1 text-xs">
                {expected.map(([k, v]) => (
                  <div key={k} className="flex min-w-0 flex-wrap gap-x-2">
                    <dt className="text-slate-600">{humanKeyLabel(k)}</dt>
                    <dd className="min-w-0 break-words font-medium tabular-nums text-slate-900">{fmtGeneric(v, k)}</dd>
                  </div>
                ))}
              </dl>
            </details>
          )}
        </ReproducibilityBlock>
      </div>
      </div>
    </Collapsible>
  );
}
