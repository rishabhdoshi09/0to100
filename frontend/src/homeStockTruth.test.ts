import { describe, expect, it } from 'vitest'
import { verifiedHomeStockScan } from './homeStockTruth'

describe('Home NSE stock recommendation freshness', () => {
  const scan = { available: true, scanned_at: '2026-11-09T10:01:00+05:30', provenance: { data_current: true } }

  it('shows current Best 5 or a genuine no-trade only after scan + Home match', () => {
    expect(verifiedHomeStockScan(scan, true, scan.scanned_at)).toBe(true)
  })
  it('shows refreshing rather than no-trade while historical data is incomplete', () => {
    expect(verifiedHomeStockScan(scan, false, scan.scanned_at)).toBe(false)
  })
  it('never promotes old cached BUY cards after the latest scan changes', () => {
    expect(verifiedHomeStockScan(scan, true, '2026-11-06T15:30:00+05:30')).toBe(false)
  })
  it('blocks stale/invalid source and missing scan identity', () => {
    expect(verifiedHomeStockScan({ ...scan, available: false }, true, scan.scanned_at)).toBe(false)
    expect(verifiedHomeStockScan({ ...scan, provenance: { data_current: false } }, true, scan.scanned_at)).toBe(false)
    expect(verifiedHomeStockScan({ available: true }, true, scan.scanned_at)).toBe(false)
  })
})
