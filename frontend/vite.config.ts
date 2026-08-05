import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8765',
        // Old Macs can take >25s when market-ops is saturating CPU; don't
        // drop the proxy socket before the browser's own AbortController.
        timeout: 120_000,
        proxyTimeout: 120_000,
      },
      '/reports': 'http://127.0.0.1:8766',
      '/evidence': 'http://127.0.0.1:8766',
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
})
