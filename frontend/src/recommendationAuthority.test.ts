import { describe, expect, it } from 'vitest'
import { recommendationStateMessage, recommendationsAreCurrent } from './recommendationAuthority'

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

describe('Opportunities currency from the server display authority (9 Nov P0)', () => {
  const now = Date.parse('2026-11-09T11:00:30+05:30')
  const scanAt = '2026-11-09T11:00:00+05:30'
  const authority = (over: Record<string, unknown> = {}) => ({
    status: 'CURRENT', scan_current: true, scan_scanned_at: scanAt,
    evaluated_at: '2026-11-09T11:00:20+05:30', ...over,
  })
  const workspace = (over: Record<string, unknown> = {}) => ({
    scan_scanned_at: scanAt, long_term_scanned_at: 'L', records_status: '',
    display_authority: authority(over),
  })
  // What the desk saw during the incident: the 15 s dashboard cache was stale,
  // so data.ready=false, scan.available=false and data_current=false.
  const staleDesk = { available: false, scanned_at: '', provenance: { data_current: false } }

  it('shows the published projection while the dashboard read cache is stale', () => {
    expect(recommendationsAreCurrent(workspace(), staleDesk, undefined, false, now)).toBe(true)
  })
  it('never treats a rebuilding, failed or non-current scan as current', () => {
    for (const status of ['REFRESHING', 'FAILED', 'SCAN_UNAVAILABLE', 'SCAN_NOT_CURRENT']) {
      expect(recommendationsAreCurrent(workspace({ status }), staleDesk, undefined, false, now)).toBe(false)
    }
    expect(recommendationsAreCurrent(workspace({ scan_current: false }), staleDesk, undefined, false, now)).toBe(false)
  })
  it('does not let a recalled workspace from session memory revive old cards', () => {
    const recalled = workspace({ evaluated_at: '2026-11-09T10:55:00+05:30' })
    expect(recommendationsAreCurrent(recalled, staleDesk, undefined, false, now)).toBe(false)
    expect(recommendationsAreCurrent(workspace({ evaluated_at: 'garbage' }), staleDesk, undefined, false, now)).toBe(false)
  })
  it('requires the authority to describe the cards it ships with', () => {
    expect(recommendationsAreCurrent(
      { ...workspace(), scan_scanned_at: '2026-11-09T10:00:00+05:30' }, staleDesk, undefined, false, now,
    )).toBe(false)
  })
  it('lets only a fresh, ready dashboard veto it by proving a newer scan', () => {
    const freshNewer = { available: true, scanned_at: '2026-11-09T11:15:00+05:30', provenance: { data_current: true } }
    expect(recommendationsAreCurrent(workspace(), freshNewer, undefined, true, now)).toBe(false)
    expect(recommendationsAreCurrent(workspace(), freshNewer, undefined, false, now)).toBe(true)
    const freshSame = { ...freshNewer, scanned_at: scanAt }
    expect(recommendationsAreCurrent(workspace(), freshSame, undefined, true, now)).toBe(true)
  })
})

describe('bounded, truthful Opportunities states', () => {
  const ws = (status: string, extra: Record<string, unknown> = {}) => ({
    scan_scanned_at: 'S', long_term_scanned_at: 'L', records_status: status,
    display_authority: { status, ...extra }, ...extra,
  })
  it('names each non-current state instead of loading forever', () => {
    expect(recommendationStateMessage(ws('REFRESHING')).state).toBe('REBUILDING')
    expect(recommendationStateMessage(ws('FAILED', { rebuild_error: 'OSError: disk full' })).detail).toContain('disk full')
    expect(recommendationStateMessage(ws('SCAN_UNAVAILABLE')).state).toBe('NO_SCAN')
    expect(recommendationStateMessage(ws('SCAN_NOT_CURRENT')).state).toBe('STALE')
    // A server that withheld old-day cards without display_authority still reads as STALE.
    expect(recommendationStateMessage({ scan_scanned_at: '', long_term_scanned_at: '', records_status: 'STALE' }).state).toBe('STALE')
    expect(recommendationStateMessage(null).state).toBe('UNAVAILABLE')
  })
  it('explains a failed scan publication', () => {
    const failed = ws('REFRESHING', { publication: { recommendations: 'error', error: { error_code: 'DESK_PERSIST_FAILED' } } })
    expect(recommendationStateMessage(failed).detail).toContain('DESK_PERSIST_FAILED')
  })
})
