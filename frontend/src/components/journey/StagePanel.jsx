import { Clock, ListX } from 'lucide-react';
import { AnnotatedPreview } from '../AuditReport.jsx';
import Collapsible, { Chip } from '../Collapsible.jsx';
import { BenignItems } from '../NoConcern.jsx';
import RiskBadge from '../RiskBadge.jsx';
import { WarningsNotice, useJumpToFinding } from '../ReportParts.jsx';
import { formatDate, humanKey } from '../../format.js';
import { fmtShare, isNum } from '../intel/shared.jsx';

const INPUT_LABEL = { text: 'Text', html: 'HTML', screenshot: 'Screenshot' };

/** One stage's annotated preview, with its unflagged elements collapsed below it. */
function StageAudit({ stage, imageUrl, onOpenRedesign }) {
  const r = stage.audit;
  const s = r.summary;
  const benign = r.benign_elements || [];
  const prefix = `s${stage.index}-`;
  const { activeId, jumpTo } = useJumpToFinding(prefix, onOpenRedesign);

  const result = {
    response: r,
    inputType: stage.input_type,
    inputText: stage.input_type === 'text' ? stage.content : undefined,
    htmlSource: stage.input_type === 'html' ? stage.content : undefined,
    imageUrl: stage.input_type === 'screenshot' ? imageUrl : null,
    inputName: stage.name,
  };

  return (
    <div className="space-y-3">
      <WarningsNotice warnings={r.warnings} />
      {s.overall_confidence_statement && <p className="text-sm text-slate-600">{s.overall_confidence_statement}</p>}
      <AnnotatedPreview result={result} onSelect={jumpTo} activeId={activeId} headingLevel={4} />
      {benign.length > 0 && (
        <Collapsible
          variant="plain"
          headingLevel={4}
          title={`${s.no_concern ? 'Elements analysed' : 'Other elements analysed'} (${benign.length})`}
          icon={ListX}
        >
          <BenignItems elements={benign} />
        </Collapsible>
      )}
    </div>
  );
}

/** Collapsible per-stage panel; the stage report is mounted on first open. Controlled when `open` + `onOpenChange` are given. */
export default function StagePanel({ stage, imageUrl, defaultOpen = false, open, onOpenChange, onOpenRedesign }) {
  const s = stage.audit?.summary;
  const n = s?.suspicious_findings;
  return (
    <Collapsible
      lazy
      defaultOpen={defaultOpen}
      open={open}
      onOpenChange={onOpenChange}
      headingLevel={3}
      bodyClassName="bg-slate-50/60"
      leading={
        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-teal-700 text-xs font-bold text-white" aria-hidden="true">
          {stage.index + 1}
        </span>
      }
      title={
        <span className="flex min-w-0 flex-col">
          <span className="truncate">{stage.name}</span>
          <span className="flex flex-wrap items-center gap-x-1.5 text-xs font-normal text-slate-500">
            <span>{INPUT_LABEL[stage.input_type] || stage.input_type}</span>
            {stage.stage_type && <span>· {humanKey(stage.stage_type)}</span>}
            <span>· position {stage.position}</span>
            {stage.captured_at && (
              <span className="inline-flex items-center gap-1">
                · <Clock className="h-3 w-3" aria-hidden="true" /> {formatDate(stage.captured_at)}
              </span>
            )}
            {stage.synthetic && <span className="font-semibold text-amber-800">· synthetic</span>}
          </span>
        </span>
      }
      summary={
        <>
          {Number.isInteger(stage.fused_into) && (
            <span title="Same page as an earlier stage in another format; its findings are merged into that stage and not counted twice">
              <Chip tone="teal">merged into stage {stage.fused_into + 1}</Chip>
            </span>
          )}
          {isNum(stage.pressure) && <Chip tone="value" title="Stage pressure">pressure {fmtShare(stage.pressure)}</Chip>}
          {isNum(n) && <Chip>{n} finding{n === 1 ? '' : 's'}</Chip>}
          {s && <RiskBadge level={s.overall_risk} />}
        </>
      }
    >
      <StageAudit stage={stage} imageUrl={imageUrl} onOpenRedesign={onOpenRedesign} />
    </Collapsible>
  );
}
