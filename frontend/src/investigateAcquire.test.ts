import { describe, expect, it } from 'vitest'
import { investigateIsAcquiring } from './investigateAcquire'

describe('investigateIsAcquiring', () => {
  it('is idle when nothing is running', () => {
    expect(investigateIsAcquiring('', null)).toBe(false)
    expect(investigateIsAcquiring('REFRESH_STOCK_FUNDAMENTALS', null)).toBe(false)
  })

  it('locks from local busy state', () => {
    expect(investigateIsAcquiring('ACQUIRE_DUE_DILIGENCE', null)).toBe(true)
    expect(investigateIsAcquiring('ACQUIRE_DUE_DILIGENCE_ALL', null)).toBe(true)
  })

  it('locks from an in-flight backend job', () => {
    expect(investigateIsAcquiring('', { status: 'RUNNING', failed: false })).toBe(true)
    expect(investigateIsAcquiring('', { status: 'QUEUED', failed: false })).toBe(true)
  })

  it('unlocks after a finished or failed job', () => {
    expect(investigateIsAcquiring('', { status: 'SUCCEEDED', failed: false })).toBe(false)
    expect(investigateIsAcquiring('', { status: 'FAILED', failed: true })).toBe(false)
    expect(investigateIsAcquiring('', { status: 'BLOCKED', failed: true })).toBe(false)
  })
})
