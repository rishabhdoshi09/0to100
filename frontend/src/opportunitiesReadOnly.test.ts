import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

// 9 Nov P0: opening Opportunities fired RUN_SCAN_NOW whenever the dashboard
// read cache was still warming, rescanning an already-scanned NSE session.
const source = readFileSync(fileURLToPath(new URL('./recommendationsViews.tsx', import.meta.url)), 'utf8')

describe('reading Opportunities never starts a job', () => {
  it('starts a market scan only from an explicit click', () => {
    const starts = [...source.matchAll(/marketScan\.start\(\)/g)].map((m) => m.index ?? 0)
    expect(starts.length).toBeGreaterThan(0)
    for (const at of starts) {
      expect(source.slice(Math.max(0, at - 40), at)).toContain('onClick')
    }
  })
  it('does not queue product bootstrap jobs on page open', () => {
    expect(source).not.toMatch(/bootstrapProduct\s*\(/)
  })
})
