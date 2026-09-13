import { describe, expect, it } from 'vitest'

import { formatExecutionTime, formatMarketDate, scanTruthDetail, scanTruthRows } from './scanTruth'


describe('scan truth display', () => {
  it('renders market-session dates without timezone conversion', () => {
    expect(formatMarketDate('2026-09-11')).toBe('11 Sep 2026')
    expect(formatMarketDate('')).toBe('Unavailable')
    expect(formatMarketDate('not-a-date')).toBe('Unavailable')
  })

  it('renders execution instants in Asia/Kolkata, not as a market date', () => {
    const rendered = formatExecutionTime('2026-09-13T12:50:00+00:00')
    expect(rendered).toContain('2026')
    expect(rendered).toContain('18:20')
  })

  it('keeps scan execution, market session and price date separate', () => {
    const rows = scanTruthRows({
      available: true,
      scan_completed_at: '2026-09-13T12:50:00+00:00',
      market_session_date: '2026-09-11',
      price_data_as_of: '2026-09-11',
      freshness_state: 'CURRENT',
    })

    expect(rows.map(row => row.label)).toEqual([
      'Scan executed',
      'Market session',
      'Price data as of',
    ])
    expect(rows[0].value).toContain('18:20')
    expect(rows[1].value).toBe('11 Sep 2026')
    expect(rows[2].value).toBe('11 Sep 2026')
    expect(rows[0].value).not.toBe(rows[1].value)
  })

  it('surfaces unavailable provenance and invalid timing reasons', () => {
    expect(scanTruthDetail({ available: false, reason: 'NO_SAVED_SCAN' })).toBe('NO_SAVED_SCAN')
    expect(scanTruthDetail({
      available: true,
      freshness_state: 'CURRENT',
      scan_duration_status: 'UNAVAILABLE',
      scan_duration_reason: 'SCAN_TIMESTAMP_ORDER_INVALID',
    })).toBe('Timing unavailable · SCAN_TIMESTAMP_ORDER_INVALID')
  })
})
