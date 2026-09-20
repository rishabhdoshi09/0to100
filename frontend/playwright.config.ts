import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  timeout: 15 * 60 * 1000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [['line'], ['json', { outputFile: '../logs/product/frontend_burnin.json' }]],
  use: {
    baseURL: process.env.QT_DESK_URL || 'http://127.0.0.1:5173',
    headless: true,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
})
