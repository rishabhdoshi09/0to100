import { describe, expect, it } from 'vitest'
import { recommendationsAreCurrent } from './recommendationAuthority'

describe('NSE stock recommendation authority', () => {
  const scan = { available: true, scanned_at: '2026-11-09T11:00:00+05:30', provenance: { data_current: true } }
  const saved = { scan_scanned_at: scan.scanned_at, long_term_scanned_at: '2026-11-08', records_status: 'SUCCEEDED' }

  it('allows the exact verified current scan and long-term generation', () => {
    expect(recommendationsAreCurrent(saved, scan, '2026-11-08', true)).toBe(true)
  })
  it('blocks cached BUY cards across an NSE scan or funds rotation', () => {
    expect(recommendationsAreCurrent(saved, { ...scan, scanned_at: '2026-11-09T11:05:00+05:30' }, '2026-11-08', true)).toBe(false)
    expect(recommendationsAreCurrent(saved, scan, '2026-11-09', true)).toBe(false)
  })
  it('does not call an incomplete/failed rebuild a current stock recommendation', () => {
    expect(recommendationsAreCurrent({ ...saved, records_status: 'REFRESHING' }, scan, '2026-11-08', true)).toBe(false)
    expect(recommendationsAreCurrent({ ...saved, records_status: 'FAILED' }, scan, '2026-11-08', true)).toBe(false)
  })
  it('blocks unknown source/readiness instead of inventing a current BUY', () => {
    expect(recommendationsAreCurrent(saved, { ...scan, available: false }, '2026-11-08', true)).toBe(false)
    expect(recommendationsAreCurrent(saved, scan, '2026-11-08', false)).toBe(false)
    expect(recommendationsAreCurrent(null, scan, '2026-11-08', true)).toBe(false)
  })
})
