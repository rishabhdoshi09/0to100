import { describe, expect, it, beforeEach } from 'vitest'
import { keepRicher, markInvestigate, recall, remember, wantsInvestigate, writeSessionJson, readSessionJson } from './sessionMemory'

class MemoryStorage {
  store = new Map<string, string>()
  getItem(key: string) { return this.store.has(key) ? this.store.get(key)! : null }
  setItem(key: string, value: string) { this.store.set(key, value) }
  removeItem(key: string) { this.store.delete(key) }
}

describe('sessionMemory', () => {
  beforeEach(() => {
    const storage = new MemoryStorage()
    Object.assign(globalThis, { window: { sessionStorage: storage } })
  })

  it('recalls a value after remember, including from sessionStorage', () => {
    remember('radar-home', { counts: { breakouts: 3 } })
    expect(recall('radar-home')).toEqual({ counts: { breakouts: 3 } })
    expect(readSessionJson('qt:radar-home')).toEqual({ counts: { breakouts: 3 } })
  })

  it('does not replace a populated snapshot with an empty one', () => {
    remember('radar-home', { counts: { breakouts: 4 } })
    const kept = keepRicher(
      'radar-home',
      { counts: { breakouts: 0 } },
      (value) => (value.counts?.breakouts || 0) === 0,
    )
    expect(kept).toEqual({ counts: { breakouts: 4 } })
    expect(recall('radar-home')).toEqual({ counts: { breakouts: 4 } })
  })

  it('marks Investigate for a selected symbol', () => {
    markInvestigate('ofss')
    expect(wantsInvestigate('OFSS')).toBe(true)
    expect(wantsInvestigate('TCS')).toBe(false)
  })

  it('writes nav snapshots', () => {
    expect(writeSessionJson('quantterm-nav', { active: 'Recommendations', selected: 'TCS' })).toBe(true)
    expect(readSessionJson('quantterm-nav')).toEqual({ active: 'Recommendations', selected: 'TCS' })
  })
})
