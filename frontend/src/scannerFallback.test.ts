import { describe, expect, it } from 'vitest'
import {
  bestSetupsFromRadar,
  dashCell,
  projectScanRecord,
  scannerEmptyHint,
  scannerFallbackRows,
  scannerMetaFromDashboard,
} from './scannerFallback'
import type { RadarHome } from './productApi'
import type { DashboardPayload } from './types'

const dashboard = {
  scan: {
    scanned_at: '2026-08-24T16:49:23+00:00',
    universe_size: 2559,
    records: [
      { symbol: 'AAA', signals: ['BREAKOUT_52W'], status: 'Ready to trade', score: 70, chase_risk: false, momentum_5d: 4.2 },
      { symbol: 'BBB', signals: ['MOMENTUM'], verdict: 'BUY', score: 88, chase_risk: true },
      { symbol: 'CCC', signals: ['MOMENTUM', 'BREAKOUT_RES'], score: 91, sepa_score: 62 },
    ],
  },
  long_term: {
    scanned_at: '2026-08-24T16:50:00+00:00',
    records: [{ symbol: 'LT1', combined_score: 80 }],
  },
  conviction: [{ symbol: 'CV1', conviction_score: 77 }],
} as unknown as DashboardPayload

describe('scannerFallbackRows', () => {
  it('projects saved scan lanes without inventing symbols', () => {
    expect(scannerFallbackRows('Breakouts', dashboard).map((row) => row.symbol)).toEqual(['AAA', 'CCC'])
    expect(scannerFallbackRows('Momentum', dashboard).map((row) => row.symbol)).toEqual(['BBB', 'CCC'])
    expect(scannerFallbackRows('Best Setups', dashboard).map((row) => row.symbol)).toEqual(['CCC', 'BBB', 'AAA'])
    expect(scannerFallbackRows('Long-Term', dashboard).map((row) => row.symbol)).toEqual(['LT1'])
  })
})

describe('bestSetupsFromRadar', () => {
  it('prefers SEPA cards and falls back to the saved scan', () => {
    const home = {
      best_setups: [{ symbol: 'SEPA1', score: 80 }],
      lanes: { breakouts: [{ symbol: 'AAA' }], momentum: [] },
    } as unknown as RadarHome
    expect(bestSetupsFromRadar(home, dashboard).map((row) => row.symbol)).toEqual(['SEPA1'])
    expect(bestSetupsFromRadar({ best_setups: [], lanes: { breakouts: [], momentum: [] } } as unknown as RadarHome, dashboard)[0].symbol).toBe('CCC')
  })
})

describe('scanner meta and empty copy', () => {
  it('reads universe from the dashboard even before the workspace fetch', () => {
    expect(scannerMetaFromDashboard('Best Setups', dashboard)).toEqual({
      scanned_at: '2026-08-24T16:49:23+00:00',
      universe: 2559,
    })
    expect(scannerEmptyHint(0, 0, false)).toBe('No matches in saved scan data. Run Scan Now.')
    expect(scannerEmptyHint(10, 0, true)).toBe('No matches for these filters.')
    expect(scannerEmptyHint(0, 0, true)).toBe('This lane is empty in the saved scan.')
  })
})

describe('projectScanRecord', () => {
  it('maps raw scan fields so the table does not render Undefined', () => {
    const projected = projectScanRecord({
      symbol: 'AAA',
      signals: ['BREAKOUT_52W'],
      status: 'Ready to trade',
      verdict: 'BUY',
      momentum_5d: 4.2,
      score: 70,
    })
    expect(projected.change_5d_pct).toBe(4.2)
    expect(projected.breakout_state).toBe('confirmed_breakout')
    expect(projected.momentum_state).toBeNull()
    expect(projected.setup_label).toBe('Ready to trade')
    expect(dashCell(undefined)).toBe('—')
    expect(dashCell('undefined')).toBe('—')
    expect(dashCell('confirmed_breakout')).toBe('confirmed_breakout')
  })
})
