import { useEffect, useRef, useState, useCallback } from 'react';
import Header from './components/Header.jsx';
import Landing from './components/Landing.jsx';
import Workspace from './components/Workspace.jsx';
import Results from './components/Results.jsx';
import ModelInfo from './components/ModelInfo.jsx';
import JourneyBuilder from './components/journey/JourneyBuilder.jsx';
import JourneyResults from './components/journey/JourneyResults.jsx';
import { getHealth, getModelInfo } from './api.js';
import mockAudit from './mock/sampleAudit.json';
import mockInput from './mock/sampleInput.txt?raw';
import benignAudit from './mock/benignAudit.json';
import benignInput from './mock/benignInput.txt?raw';

// Offline demo: ?mock=1 (deceptive) or ?mock=benign renders a SAVED real ClearCart audit of a
// dataset sample (generated from the API, see README) without needing the backend.
// Without that query parameter this code path is never taken.
const MOCK_PARAM =
  typeof window !== 'undefined' ? new URLSearchParams(window.location.search).get('mock') : null;
const MOCK_MODE = MOCK_PARAM === '1' || MOCK_PARAM === 'benign';
// ?mock=journey renders src/mock/sampleJourney.json: a SAVED real response of POST /api/audit/journey
// for the synthetic demo journey "journey-subscription" (data/samples/journeys.json). Loaded on demand
// so it never ships in the main bundle path.
const MOCK_JOURNEY = MOCK_PARAM === 'journey';

const INITIAL_RESULT = MOCK_MODE
  ? MOCK_PARAM === 'benign'
    ? { response: benignAudit, inputType: 'text', inputText: benignInput, inputName: 'Benign storefront text (test split)', imageUrl: null, isMock: true }
    : { response: mockAudit, inputType: 'text', inputText: mockInput, inputName: 'Product page text with dark patterns (test split)', imageUrl: null, isMock: true }
  : null;

export default function App() {
  const [view, setView] = useState(MOCK_MODE ? 'results' : MOCK_JOURNEY ? 'journeyResults' : 'workspace');
  const [result, setResult] = useState(INITIAL_RESULT);
  const [journeyResult, setJourneyResult] = useState(null);
  const [journeyPending, setJourneyPending] = useState(MOCK_JOURNEY);
  const journeyDraft = useRef(null);
  const [model, setModel] = useState({ status: 'loading', data: null, error: null });
  const [health, setHealth] = useState('checking');

  const loadModel = useCallback(() => {
    setModel({ status: 'loading', data: null, error: null });
    getModelInfo()
      .then((data) => setModel({ status: 'ready', data, error: null }))
      .catch((err) => setModel({ status: 'error', data: null, error: err.message }));
  }, []);

  useEffect(() => {
    loadModel();
    getHealth()
      .then((h) => setHealth(h?.status === 'ok' ? 'online' : 'offline'))
      .catch(() => setHealth('offline'));
  }, [loadModel]);

  useEffect(() => {
    if (!MOCK_JOURNEY) return;
    import('./mock/sampleJourney.json')
      .then((m) => setJourneyResult({ response: m.default, stageImages: {}, sample: null, isMock: true }))
      .catch(() => setView('journey'))
      .finally(() => setJourneyPending(false));
  }, []);

  // Revoke journey screenshot object URLs once the journey result is replaced or the app unmounts.
  useEffect(() => {
    const urls = Object.values(journeyResult?.stageImages || {});
    return () => urls.forEach((u) => URL.revokeObjectURL(u));
  }, [journeyResult]);

  // Revoke the screenshot object URL once a result is replaced or the app unmounts.
  useEffect(() => {
    const url = result?.imageUrl;
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [result]);

  const navigate = useCallback((next) => {
    setView(next);
    window.scrollTo({ top: 0 });
  }, []);

  const handleResult = useCallback((next) => {
    setResult(next);
    setView('results');
    window.scrollTo({ top: 0 });
  }, []);

  const startNewAudit = useCallback(() => {
    setResult(null);
    navigate('workspace');
  }, [navigate]);

  const handleJourneyResult = useCallback((next) => {
    setJourneyResult(next);
    setView('journeyResults');
    window.scrollTo({ top: 0 });
  }, []);

  const saveJourneyDraft = useCallback((d) => {
    journeyDraft.current = d;
  }, []);

  const startNewJourney = useCallback(() => {
    journeyDraft.current = null;
    setJourneyResult(null);
    navigate('journey');
  }, [navigate]);

  const editJourney = useCallback(() => navigate('journey'), [navigate]);

  // If someone lands on a results view without a result, send them to the matching workspace.
  useEffect(() => {
    if (view === 'results' && !result) setView('workspace');
    if (view === 'journeyResults' && !journeyResult && !journeyPending) setView('journey');
  }, [view, result, journeyResult, journeyPending]);

  return (
    <div className="flex min-h-screen flex-col">
      <Header view={view} onNavigate={navigate} health={health} />

      <main id="main" tabIndex={-1} className="mx-auto w-full max-w-6xl flex-1 px-4 py-8 sm:px-6 sm:py-10 lg:px-8">
        {view === 'landing' && <Landing onStart={() => navigate('workspace')} model={model} />}
        {view === 'workspace' && <Workspace onResult={handleResult} onJourney={() => navigate('journey')} />}
        {view === 'results' && result && (
          <Results result={result} model={model} onNewAudit={startNewAudit} onRunJourney={() => navigate('journey')} />
        )}
        {view === 'journey' && (
          <JourneyBuilder draft={journeyDraft.current} onDraftChange={saveJourneyDraft} onResult={handleJourneyResult} />
        )}
        {view === 'journeyResults' && journeyPending && (
          <p role="status" className="text-sm text-slate-600">Loading saved journey audit…</p>
        )}
        {view === 'journeyResults' && journeyResult && (
          <JourneyResults
            result={journeyResult}
            model={model}
            onNewJourney={startNewJourney}
            onEditJourney={journeyResult.isMock || !journeyDraft.current ? null : editJourney}
          />
        )}
        {view === 'model' && <ModelInfo state={model} onRetry={loadModel} />}
      </main>

    </div>
  );
}
