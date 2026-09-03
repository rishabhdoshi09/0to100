import { describe, expect, it } from 'vitest'
import {
  bestSetupsFromRadar,
  dashCell,
  projectScanRecord,
  recoCanonicalDecision,
  scannerDecision,
  scannerEmptyHint,
  scannerFallbackRows,
  scannerMetaFromDashboard,
  scannerWhy,
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
  it('maps raw scan fields without pretending the scanner made an investment decision', () => {
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
    expect(projected.decision).toBe('SETUP READY')
  })
})

describe('scannerDecision', () => {
  it('uses pre-decision language and reserves BUY/AVOID for the committee', () => {
    expect(scannerDecision({ status: 'Ready to trade', verdict: 'BUY' })).toBe('SETUP READY')
    expect(scannerDecision({ verdict: 'BUY' })).toBe('WATCH')
    expect(scannerDecision({ status: 'Watch for breakout' })).toBe('WATCH')
    expect(scannerDecision({ chase_risk: true })).toBe('EXTENDED')
    expect(scannerDecision({ verdict: 'AVOID' })).toBe('SETUP FAILED')
    expect(scannerDecision({ symbol: 'TCS', exit_reason: 'TARGET', decision: 'ENTER' }, ['TCS'])).toBe('EXIT')
    expect(scannerDecision({ symbol: 'TCS', exit_reason: 'TARGET' }, [])).toBe('WATCH')
  })

  it('keeps scanner why-copy descriptive instead of pretending to decide', () => {
    expect(scannerWhy({ why: 'Too extended', chase_risk: true })).toBe('Too extended')
    expect(scannerWhy({ chase_risk: true })).toBe('Setup is stretched; wait for a lower-risk entry.')
    expect(scannerWhy({ status: 'Ready to trade', verdict: 'BUY' })).toContain('Qualified setup')
  })
})

describe('recoCanonicalDecision', () => {
  it('does not promote recommendation badges or tiers into final BUY', () => {
    expect(recoCanonicalDecision({ action_badge: 'Buy' })).toBe('CANDIDATE')
    expect(recoCanonicalDecision({ reco_tier: 'high_conviction' })).toBe('CANDIDATE')
    expect(recoCanonicalDecision({ action_badge: 'Watch', entry_state: 'extended' })).toBe('WAIT')
    expect(recoCanonicalDecision({ action_badge: 'Hold / Research' })).toBe('WATCH')
    expect(recoCanonicalDecision({ blockers: ['DD_GATE_FAILED'] })).toBe('WATCH')
  })

  it('uses the persisted committee judgment as the only final decision truth', () => {
    expect(recoCanonicalDecision({ committee_decision: 'BUY', execution_state: 'BLOCKED_BROKER_AUTH' })).toBe('BUY')
    expect(recoCanonicalDecision({ committee_decision: 'WAIT' })).toBe('WAIT')
    expect(recoCanonicalDecision({ committee_decision: 'AVOID' })).toBe('AVOID')
    expect(recoCanonicalDecision({ committee_decision: 'NO_JUDGMENT' })).toBe('NO JUDGMENT')
  })
})
