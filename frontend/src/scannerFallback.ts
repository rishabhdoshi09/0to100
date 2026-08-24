import type { RadarHome, ScannerWorkspaceRow } from './productApi'
import type { DashboardPayload, ScanRecord } from './types'

export type ScannerMeta = {
  scanned_at: string
  universe: number
}

export function scannerMetaFromDashboard(mode: string, dashboard: DashboardPayload): ScannerMeta {
  const universe = Number(dashboard.scan.universe_size || 0)
  if (mode === 'Long-Term') {
    return {
      scanned_at: dashboard.long_term.scanned_at || dashboard.scan.scanned_at || '',
      universe,
    }
  }
  return { scanned_at: dashboard.scan.scanned_at || '', universe }
}

export function scannerFallbackRows(mode: string, dashboard: DashboardPayload): ScannerWorkspaceRow[] {
  const scan = dashboard.scan.records
  if (mode === 'Best Setups') {
    return [...scan].sort((a, b) => {
      const aSepa = Number(a.sepa_score || 0)
      const bSepa = Number(b.sepa_score || 0)
      if (aSepa !== bSepa) return bSepa - aSepa
      return Number(b.score || 0) - Number(a.score || 0)
    })
  }
  if (mode === 'Conviction') return [...dashboard.conviction]
  if (mode === 'Long-Term') return [...dashboard.long_term.records] as ScannerWorkspaceRow[]
  if (mode === 'Breakouts') {
    return scan.filter((row) => row.signals?.some((signal) => signal.includes('BREAKOUT')) || row.status === 'Ready to trade')
  }
  if (mode === 'Pre-Breakout') {
    return scan.filter((row) => row.signals?.includes('PRE_BREAKOUT') || row.status === 'Watch for breakout')
  }
  if (mode === 'Avoid') {
    return scan.filter((row) => row.chase_risk || row.status === 'Wait for pullback')
  }
  if (mode === 'F&O Coverage' || mode === 'F&O') {
    return scan.filter((row) => Boolean((row as ScanRecord & { fno_available?: boolean }).fno_available))
  }
  return scan.filter((row) => row.signals?.includes('MOMENTUM') || row.verdict === 'BUY')
}

export function bestSetupsFromRadar(home: RadarHome, dashboard: DashboardPayload): ScannerWorkspaceRow[] {
  if ((home.best_setups || []).length) return home.best_setups as ScannerWorkspaceRow[]
  const lanes = [...(home.lanes?.breakouts || []), ...(home.lanes?.momentum || [])]
  if (lanes.length) {
    const seen = new Set<string>()
    return lanes.filter((row) => {
      if (seen.has(row.symbol)) return false
      seen.add(row.symbol)
      return true
    })
  }
  return scannerFallbackRows('Best Setups', dashboard)
}

export function scannerEmptyHint(rows: number, filtered: number, hasScan: boolean): string {
  if (filtered > 0) return ''
  if (rows > 0) return 'No matches for these filters.'
  if (hasScan) return 'This lane is empty in the saved scan.'
  return 'No matches in saved scan data. Run Scan Now.'
}
