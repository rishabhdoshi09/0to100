import { afterEach, describe, expect, it } from 'vitest'
import {
  cacheKey,
  evictCaches,
  isNseSymbol,
  loadCachedJson,
  loadDeskSession,
  patchDeskSession,
  pinnedSymbol,
  saveCachedJson,
  SESSION_KEY,
  stashDashboard,
} from './deskSession'

class MemoryStorage implements Storage {
  private data = new Map<string, string>()
  get length() { return this.data.size }
  clear() { this.data.clear() }
  getItem(key: string) { return this.data.has(key) ? this.data.get(key)! : null }
  key(index: number) { return [...this.data.keys()][index] ?? null }
  removeItem(key: string) { this.data.delete(key) }
  setItem(key: string, value: string) { this.data.set(key, value) }
}

const memory = new MemoryStorage()

Object.defineProperty(globalThis, 'localStorage', {
  configurable: true,
  value: memory,
})

afterEach(() => {
  memory.clear()
})

describe('desk session', () => {
  it('accepts NSE symbols and rejects junk', () => {
    expect(isNseSymbol('M&M')).toBe(true)
    expect(isNseSymbol('BAJAJ-AUTO')).toBe(true)
    expect(isNseSymbol('<script>')).toBe(false)
    expect(isNseSymbol('')).toBe(false)
  })

  it('round-trips page, compare list and Ideas tab without pinning a name on Ideas', () => {
    patchDeskSession({
      active: 'Recommendations',
      selected: 'ABCCO',
      compareSymbols: ['TCS', 'not a symbol!!!', 'INFY'],
      ideasCategory: 'super_trends',
      ideasLifecycle: 'Active',
    })
    const session = loadDeskSession()
    expect(session?.active).toBe('Recommendations')
    expect(session?.selected).toBe('')
    expect(session?.compareSymbols).toEqual(['TCS', 'INFY'])
    expect(session?.ideasCategory).toBe('super_trends')
    expect('query' in (session || {})).toBe(false)
  })

  it('keeps a name only on stock-focus pages', () => {
    patchDeskSession({ active: 'Stock Intelligence', selected: 'm&m' })
    expect(loadDeskSession()?.selected).toBe('M&M')
    patchDeskSession({ active: 'Recommendations', selected: 'M&M' })
    expect(loadDeskSession()?.selected).toBe('')
    expect(pinnedSymbol('System Health', 'M&M')).toBe('')
    expect(pinnedSymbol('Home', 'INFY')).toBe('')
    expect(pinnedSymbol('F&O Desk', 'INFY')).toBe('INFY')
  })

  it('does not restore a leftover search-box ticker from older sessions', () => {
    memory.setItem(SESSION_KEY, JSON.stringify({
      active: 'System Health',
      selected: 'ABCCO',
      query: 'ABCCO',
      compareSymbols: [],
    }))
    const session = loadDeskSession()
    expect(session?.active).toBe('System Health')
    expect(session?.selected).toBe('')
    expect((session as { query?: string } | null)?.query).toBeUndefined()
  })

  it('does not restore Stock Intelligence without a selected name', () => {
    memory.setItem(SESSION_KEY, JSON.stringify({
      active: 'Stock Intelligence',
      selected: '',
      compareSymbols: [],
    }))
    expect(loadDeskSession()?.active).toBe('Recommendations')
  })

  it('ignores unknown pages and does not pin a name on Home', () => {
    memory.setItem(SESSION_KEY, JSON.stringify({
      active: 'NotAPage',
      selected: 'TCS',
      compareSymbols: [],
    }))
    expect(loadDeskSession()?.active).toBe('Home')
    expect(loadDeskSession()?.selected).toBe('')
  })

  it('hydrates last payloads and strips phantom running jobs from dashboard cache', () => {
    expect(saveCachedJson('reco-workspace', { categories: [{ id: 'best_setups', count: 2 }] })).toBe(true)
    expect(loadCachedJson<{ categories: Array<{ count: number }> }>('reco-workspace')?.categories[0].count).toBe(2)
    stashDashboard({
      scan: { records: [{ symbol: 'TCS' }] },
      operations: { active: [{ kind: 'MARKET_SCAN', status: 'RUNNING' }], recent: [] },
    })
    const dash = loadCachedJson<{ operations: { active: unknown[] }; scan: { records: unknown[] } }>('dashboard')
    expect(dash?.scan.records).toHaveLength(1)
    expect(dash?.operations.active).toEqual([])
  })

  it('evicts payload caches without dropping the session', () => {
    patchDeskSession({ active: 'Watchlist', selected: 'TCS' })
    saveCachedJson('radar-home', { counts: { breakouts: 3 } })
    evictCaches()
    expect(loadCachedJson('radar-home')).toBeNull()
    expect(memory.getItem(cacheKey('radar-home'))).toBeNull()
    expect(loadDeskSession()?.active).toBe('Watchlist')
    expect(loadDeskSession()?.selected).toBe('')
  })
})
