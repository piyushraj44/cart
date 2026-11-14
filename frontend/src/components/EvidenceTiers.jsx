import { useId } from 'react';
import { CircleHelp, Eye, MessageSquareWarning } from 'lucide-react';
import { Chip } from './Collapsible.jsx';
import { pct } from '../format.js';

// Older responses have no evidence_tier: DOM observations are facts, model findings are claims.
const tierOf = (f) => f.evidence_tier || (f.detection_method === 'dom_observation' ? 'observed' : 'pressure_claim');

const GATE_LABEL = {
  checkable_claim: 'Checkable promotion',
  structured_promotion: 'Structured promotion',
  neutral_choice: 'Neutral choice',
  quoted_mention: 'Quoted example',
  out_of_distribution: 'Unlike training data',
};

const VERDICT = {
  checkable: { tone: 'green', label: 'checkable terms' },
  vague: { tone: 'orange', label: 'vague pressure' },
  mixed: { tone: 'amber', label: 'mixed' },
  none: { tone: 'muted', label: 'no markers' },
};

/** "hurry: Hurry" → "hurry"; markers carry the matched text after the colon, shown as the tooltip. */
function Markers({ check }) {
  if (!check) return null;
  const items = [
    ...(check.checkable || []).map((m) => ['green', m]),
    ...(check.vague || []).map((m) => ['orange', m]),
  ];
  if (!items.length) return null;
  return (
    <span className="flex flex-wrap gap-1">
      {items.map(([tone, m], i) => {
        const [label, ...rest] = m.split(':');
        return <Chip key={i} tone={tone} title={rest.join(':').trim() || undefined}>{label}</Chip>;
      })}
    </span>
  );
}

function Item({ text, meta, check, note }) {
  return (
    <li className="space-y-1 py-2.5 first:pt-0 last:pb-0">
      <p className="break-words text-sm text-slate-800">{text}</p>
      <div className="flex flex-wrap items-center gap-1">
        {meta}
        <Markers check={check} />
      </div>
      {note && <p className="text-xs text-slate-500">{note}</p>}
    </li>
  );
}

function Column({ icon: Icon, title, sub, count, accent, empty, children }) {
  return (
    <div className="min-w-0 bg-white p-4 sm:p-5">
      <div className="flex items-start gap-2.5">
        <span className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${accent}`}>
          <Icon className="h-4 w-4" aria-hidden="true" />
        </span>
        <div className="min-w-0">
          <h3 className="text-sm font-semibold text-slate-900">
            {title} <span className="font-normal tabular-nums text-slate-500">({count})</span>
          </h3>
          <p className="text-xs text-slate-500">{sub}</p>
        </div>
      </div>
      {count > 0 ? <ul className="mt-3 divide-y divide-slate-100">{children}</ul> : <p className="mt-3 text-xs text-slate-400">{empty}</p>}
    </div>
  );
}

/**
 * Three evidence tiers for one audit:
 *   observed facts (read directly from the input) · pressure claims (model, unverified) · held for review (gated).
 * A genuine, specific sale lands in "held for review", not in the red findings.
 */
export default function EvidenceTiers({ response: r }) {
  const headingId = useId();
  const findings = r.findings || [];
  const review = r.uncertain_elements || [];
  if (!findings.length && !review.length) return null;
  const observed = findings.filter((f) => tierOf(f) === 'observed');
  const claims = findings.filter((f) => tierOf(f) === 'pressure_claim');

  return (
    <section aria-labelledby={headingId} className="card overflow-hidden">
      <div className="px-4 pt-4 sm:px-5">
        <h2 id={headingId} className="text-base font-semibold text-slate-900">Evidence tiers</h2>
        <p className="mt-0.5 text-sm text-slate-600">
          What ClearCart saw directly, what the model reads as pressure, and what it held back because the claim looks
          specific or the evidence is weak.
        </p>
      </div>
      <div className="mt-4 grid gap-px border-t border-slate-100 bg-slate-100 md:grid-cols-3">
        <Column icon={Eye} title="Observed facts" sub="Read directly from the input, no model score" count={observed.length}
          accent="bg-red-50 text-red-700" empty="Nothing observed directly (e.g. no pre-ticked boxes).">
          {observed.map((f) => (
            <Item key={f.id} text={f.evidence} note={f.short_reason}
              meta={<Chip tone="red">{f.category_label}</Chip>} />
          ))}
        </Column>
        <Column icon={MessageSquareWarning} title="Pressure claims" sub="Model-detected wording; truth not verified" count={claims.length}
          accent="bg-orange-50 text-orange-700" empty="No pressure wording above the threshold.">
          {claims.map((f) => (
            <Item key={f.id} text={f.evidence} check={f.claim_check}
              meta={<>
                <Chip tone="orange">{f.category_label}</Chip>
                {f.model_score != null && <Chip tone="value">{pct(f.model_score)}</Chip>}
                {f.claim_check && <Chip tone={VERDICT[f.claim_check.verdict]?.tone} title={f.claim_check.note}>{VERDICT[f.claim_check.verdict]?.label}</Chip>}
              </>} />
          ))}
        </Column>
        <Column icon={CircleHelp} title="Held for review" sub="Above the threshold but withheld by a false-positive check" count={review.length}
          accent="bg-slate-100 text-slate-600" empty="Nothing withheld.">
          {review.map((u, i) => (
            <Item key={i} text={u.text} check={u.claim_check} note={u.gate_note}
              meta={<>
                <Chip tone="muted">{GATE_LABEL[u.gate] || 'Withheld'}</Chip>
                <Chip tone="value">{pct(u.p_dark)}</Chip>
              </>} />
          ))}
        </Column>
      </div>
    </section>
  );
}
