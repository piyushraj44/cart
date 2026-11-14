// Bundles tests/render.test.jsx with esbuild (already installed by Vite) and runs it in Node.
import { build } from 'esbuild';
import { pathToFileURL } from 'node:url';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const out = path.join(here, '.out', 'render.test.mjs');
await build({
  entryPoints: [path.join(here, 'render.test.jsx')], bundle: true, platform: 'node', format: 'esm', outfile: out,
  jsx: 'automatic', loader: { '.js': 'jsx', '.jsx': 'jsx' }, logLevel: 'error',
  define: { 'process.env.NODE_ENV': '"production"' },
  banner: { js: "import { createRequire } from 'module'; const require = createRequire(import.meta.url);" },
});
await import(pathToFileURL(out).href);
