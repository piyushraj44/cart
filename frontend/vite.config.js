import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Backend for the dev proxy; override with CLEARCART_API=http://127.0.0.1:8001 npm run dev
const api = process.env.CLEARCART_API || 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': api,
      '/health': api,
    },
  },
});
