import { describe, expect, it } from 'vitest'
import { classifyPageError, pageHealth, pageRequestStatus, pageStatusLabel } from './pageRequest'

describe('pageRequestStatus', () => {
  it('never treats a timeout as infinite loading', () => {
    expect(pageRequestStatus({ loading: true, data: null, error: 'Request timed out. The page is not waiting forever — retry.' })).toBe('timeout')
    expect(pageStatusLabel('timeout')).toBe('Failed')
  })

  it('can be ready while a later lane is still loading', () => {
    expect(pageRequestStatus({ loading: true, data: { symbol: 'TCS' } })).toBe('ready')
  })

  it('uses a valid empty state instead of a blank screen', () => {
    expect(pageRequestStatus({
      loading: false,
      data: { cards: [] },
      isEmpty: (row) => Array.isArray((row as { cards?: unknown[] }).cards) && (row as { cards: unknown[] }).cards.length === 0,
    })).toBe('empty')
    expect(pageStatusLabel('empty')).toBe('Empty')
  })
})

describe('pageHealth', () => {
  it('records request completion and last error', () => {
    const health = pageHealth({
      page: 'Recommendations',
      loading: false,
      data: null,
      error: 'Market API is not running on :8765. Start with bash scripts/run_quantterm_complete.sh, then retry.',
    })
    expect(health.status).toBe('error')
    expect(health.apiReachable).toBe(false)
    expect(health.requestCompleted).toBe(true)
    expect(health.lastError).toContain('Market API is not running')
  })

  it('marks a qualified-empty payload as valid empty', () => {
    const health = pageHealth({
      page: 'Recommendations',
      loading: false,
      data: { cards: [] },
      isEmpty: () => true,
    })
    expect(health.status).toBe('empty')
    expect(health.validEmpty).toBe(true)
    expect(health.payloadValid).toBe(true)
    expect(health.dataPresent).toBe(false)
  })
})

describe('classifyPageError', () => {
  it('separates timeout from other failures', () => {
    expect(classifyPageError('Request timed out')).toBe('timeout')
    expect(classifyPageError('scan failed')).toBe('error')
  })
})
