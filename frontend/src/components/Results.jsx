import { useCallback, useState } from 'react';
import { Download, FileJson, Fingerprint, ListX, Loader2, Plus, EyeOff } from 'lucide-react';
import { Notice, WarningsNotice, useJumpToFinding } from './ReportParts.jsx';
import { AnnotatedPreview, AuditDashboard, ReportTitle } from './AuditReport.jsx';
import Collapsible, { Chip } from './Collapsible.jsx';
import { BenignItems } from './NoConcern.jsx';
import EvidenceTiers from './EvidenceTiers.jsx';
import VisualSignals from './VisualSignals.jsx';
import IntelligencePanel, { TimelineCard } from './intel/IntelligencePanel.jsx';
import { VersionsList } from './intel/shared.jsx';
import { exportPdf } from '../api.js';
import { downloadBlob } from '../format.js';

export default function Results({ result, model, onNewAudit, onRunJourney }) {
  const r = result.response;
  const s = r.summary;
  const findings = r.findings || [];
  const benign = r.benign_elements || [];
  const warnings = r.warnings || [];
  const versions = Object.entries(r.versions || {});
  const hasEdits = (r.intelligence?.counterfactual?.edits || []).length > 0;
  // One accordion group: opening a section closes the others (all may be closed).
  const [openId, setOpenId] = useState(hasEdits ? 'redesign' : null);
  const accordion = (key) => ({ open: openId === key, onOpenChange: (o) => setOpenId(o ? key : null) });
  const openRedesign = useCallback(() => setOpenId('redesign'), []);
  const { activeId, jumpTo } = useJumpToFinding('', openRedesign);
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState(null);

  function downloadJson() {
    const blob = new Blob([JSON.stringify(r, null, 2)], { type: 'application/json' });
    downloadBlob(blob, `clearcart-audit-${r.audit_id}.json`);
  }

  async function downloadPdf() {
    setExporting(true);
    setExportError(null);
    try {
      const blob = await exportPdf(r);
      downloadBlob(blob, `clearcart-audit-${r.audit_id}.pdf`);
    } catch (err) {
      setExportError(err?.message || 'The PDF could not be created.');
    } finally {
      setExporting(false);
    }
  }

  return (
    <div className="space-y-4 sm:space-y-5">
      <p role="status" aria-live="polite" className="sr-only">
        Audit complete. {s.suspicious_findings} suspicious {s.suspicious_findings === 1 ? 'finding' : 'findings'}.
      </p>

      <ReportTitle
        result={result}
        actions={
          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={downloadJson} className="btn-secondary px-3 py-2">
              <FileJson className="h-4 w-4" aria-hidden="true" />
              Download JSON
            </button>
            <button type="button" onClick={downloadPdf} disabled={exporting} className="btn-secondary px-3 py-2" aria-busy={exporting}>
              {exporting ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" /> : <Download className="h-4 w-4" aria-hidden="true" />}
              {exporting ? 'Preparing PDF…' : 'Download PDF'}
            </button>
            <button type="button" onClick={onNewAudit} className="btn-primary px-3 py-2">
              <Plus className="h-4 w-4" aria-hidden="true" />
              New audit
            </button>
          </div>
        }
      />

      {result.isMock && (
        <Notice tone="muted">
          <p className="text-xs">Offline demo: a saved ClearCart audit of a held-out dataset sample, shown because the page was opened with ?mock.</p>
        </Notice>
      )}
      {exportError && (
        <Notice tone="error" role="alert">
          <p>PDF export failed: {exportError}</p>
        </Notice>
      )}
      <WarningsNotice warnings={warnings} />

      {/* 1. dashboard → 2. annotated preview → 3. redesign, MCAI, FSPM, CDI → 4. details (one accordion group) */}
      <AuditDashboard response={r} />

      <EvidenceTiers response={r} />

      <AnnotatedPreview result={result} onSelect={jumpTo} activeId={activeId} />

      <IntelligencePanel intelligence={r.intelligence} findings={findings} onRunJourney={onRunJourney} accordion={accordion} />

      <TimelineCard temporal={r.intelligence?.temporal} {...accordion('timeline')} />

      <VisualSignals signals={r.visual_signals} {...accordion('visual')} />

      {benign.length > 0 && (
        <Collapsible
          title={s.no_concern ? `Elements analysed (${benign.length})` : `Other elements analysed (${benign.length})`}
          icon={ListX}
          summary={<Chip tone="muted">below the flagging threshold</Chip>}
          {...accordion('benign')}
        >
          <BenignItems elements={benign} />
        </Collapsible>
      )}

      {versions.length > 0 && (
        <Collapsible
          title="Reproducibility"
          icon={Fingerprint}
          summary={<Chip>{versions.length} version{versions.length === 1 ? '' : 's'} recorded</Chip>}
          {...accordion('repro')}
        >
          <VersionsList versions={r.versions} />
        </Collapsible>
      )}

      {r.disclaimer && (
        <footer className="pt-2">
          <p className="flex items-start gap-2 text-xs leading-relaxed text-slate-500">
            <EyeOff className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
            {r.disclaimer}
          </p>
        </footer>
      )}
    </div>
  );
}
