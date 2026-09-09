import { describe, expect, it } from 'vitest'
import { durableReadFallback } from './readFallback'
import { keepRicherMemory } from './sessionMemory'

describe('durableReadFallback', () => {
  it('returns the saved recommendation workspace on a transient GET failure', () => {
    const saved = {
      categories: [{ id: 'wealth_builders', count: 1, cards: [{ symbol: 'ABC' }] }],
      scan_scanned_at: '2026-09-09T12:00:00Z',
    }
    keepRicherMemory('reco-workspace', saved, () => false)

    expect(durableReadFallback<typeof saved>('/api/recommendations-workspace')).toEqual(saved)
  })

  it('never falls back for writes or unrelated endpoints', () => {
    const saved = { categories: [{ id: 'wealth_builders', count: 1 }] }
    keepRicherMemory('reco-workspace', saved, () => false)

    expect(durableReadFallback('/api/recommendations-workspace', 'POST')).toBeUndefined()
    expect(durableReadFallback('/api/operations', 'GET')).toBeUndefined()
  })
})
