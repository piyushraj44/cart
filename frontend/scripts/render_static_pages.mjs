// Bundles scripts/static_pages.jsx with esbuild (installed with Vite) and runs it in Node. See that file.
import { build } from 'esbuild';
import { pathToFileURL, fileURLToPath } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const out = path.join(here, '..', 'tests', '.out', 'static_pages.mjs');
await build({
  entryPoints: [path.join(here, 'static_pages.jsx')], bundle: true, platform: 'node', format: 'esm', outfile: out,
  jsx: 'automatic', loader: { '.js': 'jsx', '.jsx': 'jsx', '.txt': 'text' }, logLevel: 'error',
  define: { 'process.env.NODE_ENV': '"production"', 'import.meta.env': '{}' },
  banner: { js: "import { createRequire } from 'module'; const require = createRequire(import.meta.url);" },
});
await import(pathToFileURL(out).href);
