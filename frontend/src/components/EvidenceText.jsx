import { useMemo } from 'react';
import { categoryStyle, humanKey } from '../format.js';

/**
 * Splits text into plain and highlighted segments. Offsets from the API are
 * code-point indexes (Python str), so the text is split with Array.from.
 */
function buildSegments(text, findings) {
  const chars = Array.from(text);
  const ranges = findings
    .map((f) => ({ f, start: f.source_location?.char_start, end: f.source_location?.char_end }))
    .filter(
      (r) =>
        Number.isInteger(r.start) &&
        Number.isInteger(r.end) &&
        r.start >= 0 &&
        r.end > r.start &&
        r.end <= chars.length,
    )
    .sort((a, b) => a.start - b.start);

  const out = [];
  let cursor = 0;
  for (const r of ranges) {
    if (r.start < cursor) continue; // overlapping range: first one wins
    if (r.start > cursor) out.push({ text: chars.slice(cursor, r.start).join('') });
    out.push({ text: chars.slice(r.start, r.end).join(''), finding: r.f });
    cursor = r.end;
  }
  if (cursor < chars.length) out.push({ text: chars.slice(cursor).join('') });
  return out;
}

export default function EvidenceText({ text, findings, activeId, onSelect }) {
  const segments = useMemo(() => buildSegments(text, findings), [text, findings]);
  const legend = [...new Map(findings.map((f) => [f.category, f.category_label])).entries()];

  return (
    <div>
      <div
        tabIndex={0}
        aria-label="Submitted text with highlighted findings"
        className="max-h-[28rem] overflow-auto rounded-lg border border-slate-200 bg-slate-50 p-4 text-sm leading-7 text-slate-800 whitespace-pre-wrap break-words"
      >
        {segments.map((seg, i) => {
          if (!seg.finding) return <span key={i}>{seg.text}</span>;
          const f = seg.finding;
          const style = categoryStyle(f.category);
          const active = activeId === f.id;
          return (
            <mark
              key={i}
              role="button"
              tabIndex={0}
              title={`${f.category_label}: jump to finding`}
              aria-label={`${f.category_label}, jump to finding: ${seg.text}`}
              onClick={() => onSelect(f.id)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  onSelect(f.id);
                }
              }}
              className={`cursor-pointer rounded px-0.5 text-slate-900 ${style.mark} ${
                active ? 'ring-2 ring-teal-600' : ''
              }`}
            >
              {seg.text}
            </mark>
          );
        })}
      </div>
      {legend.length > 0 && (
        <ul className="mt-3 flex flex-wrap gap-x-4 gap-y-1.5 text-xs text-slate-700" aria-label="Highlight legend">
          {legend.map(([cat, label]) => (
            <li key={cat} className="flex items-center gap-1.5">
              <span className={`h-3 w-3 rounded-sm ${categoryStyle(cat).dot}`} aria-hidden="true" />
              {label || humanKey(cat)}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
