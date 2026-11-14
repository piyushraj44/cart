// Renders ClearCart's own pages (About us / Model / Audit workspace) to static HTML with react-dom/server.
// Used ONLY to build a qualitative false-positive probe set of known-good text (ml/eval_false_positives.py).
// Run: node scripts/render_static_pages.mjs <model-info.json> <out-dir> <tailwind.css>
import fs from 'node:fs';
import path from 'node:path';
import { renderToStaticMarkup } from 'react-dom/server';
import Header from '../src/components/Header.jsx';
import Landing from '../src/components/Landing.jsx';
import ModelInfo from '../src/components/ModelInfo.jsx';
import Workspace from '../src/components/Workspace.jsx';

const [modelPath, outDir, cssPath] = process.argv.slice(2);
const model = { status: 'ready', data: JSON.parse(fs.readFileSync(modelPath, 'utf8')), error: null };
const css = cssPath ? fs.readFileSync(cssPath, 'utf8') : '';
const noop = () => {};
const pages = {
  about: ['landing', <Landing onStart={noop} model={model} />],
  model: ['model', <ModelInfo state={model} onRetry={noop} />],
  workspace: ['workspace', <Workspace onResult={noop} onJourney={noop} />],
};
fs.mkdirSync(outDir, { recursive: true });
for (const [name, [view, el]] of Object.entries(pages)) {
  const body = renderToStaticMarkup(
    <div className="flex min-h-screen flex-col">
      <Header view={view} onNavigate={noop} health="online" />
      <main id="main" className="mx-auto w-full max-w-6xl flex-1 px-4 py-8 sm:px-6 sm:py-10 lg:px-8">{el}</main>
    </div>,
  );
  const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><title>ClearCart ${name}</title>` +
    `<meta name="viewport" content="width=device-width, initial-scale=1"><style>${css}</style></head><body>${body}</body></html>`;
  fs.writeFileSync(path.join(outDir, `${name}.html`), html);
  console.log(`wrote ${name}.html (${html.length} bytes)`);
}
