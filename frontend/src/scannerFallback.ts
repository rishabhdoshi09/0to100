import type { RadarHome, ScannerWorkspaceRow } from './productApi'
import type { DashboardPayload, ScanRecord } from './types'

export type ScannerMeta = {
  scanned_at: string
  universe: number
}

const BREAKOUT_TAGS = ['BREAKOUT_52W', 'BREAKOUT_RES', 'GOLDEN_CROSS', 'VOL_SQUEEZE']

function signalsOf(row: Record<string, unknown>): string[] {
  return Array.isArray(row.signals) ? row.signals.map(String) : []
}

function missing(value: unknown): boolean {
  return value == null || value === '' || value === 'undefined' || value === 'null'
}

export function projectScanRecord(row: Record<string, unknown>): ScannerWorkspaceRow {
  const signals = signalsOf(row)
  const status = String(row.status || '')
  const chase = Boolean(row.chase_risk)
  const verdict = String(row.verdict || '').toUpperCase()
  const vol = Number(row.volume_ratio)
  const volKnown = Number.isFinite(vol) && vol > 0
  const volOk = volKnown && vol >= 0.7

  let breakout_state = row.breakout_state
  if (missing(breakout_state)) {
    if (chase) breakout_state = 'extended_after_breakout'
    else if (signals.includes('PRE_BREAKOUT') || status === 'Watch for breakout') breakout_state = 'near_breakout'
    else if (signals.some((item) => BREAKOUT_TAGS.some((tag) => item.includes(tag)))) {
      if (verdict === 'BUY' && status === 'Ready to trade') {
        breakout_state = volKnown && !volOk ? 'breakout_without_volume' : 'confirmed_breakout'
      } else {
        breakout_state = 'breakout_under_observation'
      }
    } else {
      breakout_state = null
    }
  }

  let momentum_state = row.momentum_state
  if (missing(momentum_state)) {
    if (!signals.some((item) => item.includes('MOMENTUM'))) momentum_state = null
    else if (chase) momentum_state = 'strong_but_extended'
    else if (status === 'Ready to trade') momentum_state = 'strong_actionable'
    else momentum_state = 'watch_momentum'
  }

  return {
    ...(row as ScannerWorkspaceRow),
    change_5d_pct: (row.change_5d_pct ?? row.momentum_5d ?? null) as number | null,
    setup_label: (row.setup_label ?? status ?? row.verdict ?? null) as string | null,
    sector: (row.sector ?? null) as string | null,
    relative_strength: (row.relative_strength ?? row.score ?? null) as number | null,
    breakout_state: breakout_state as string | null,
    momentum_state: momentum_state as string | null,
  }
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
  let rows: Array<Record<string, unknown>>
  if (mode === 'Best Setups') {
    rows = [...scan].sort((a, b) => {
      const aSepa = Number(a.sepa_score || 0)
      const bSepa = Number(b.sepa_score || 0)
      if (aSepa !== bSepa) return bSepa - aSepa
      return Number(b.score || 0) - Number(a.score || 0)
    })
  } else if (mode === 'Conviction') {
    rows = [...dashboard.conviction]
  } else if (mode === 'Long-Term') {
    rows = [...dashboard.long_term.records] as Array<Record<string, unknown>>
  } else if (mode === 'Breakouts') {
    rows = scan.filter((row) => row.signals?.some((signal) => signal.includes('BREAKOUT')) || row.status === 'Ready to trade')
  } else if (mode === 'Pre-Breakout') {
    rows = scan.filter((row) => row.signals?.includes('PRE_BREAKOUT') || row.status === 'Watch for breakout')
  } else if (mode === 'Avoid') {
    rows = scan.filter((row) => row.chase_risk || row.status === 'Wait for pullback')
  } else if (mode === 'F&O Coverage' || mode === 'F&O') {
    rows = scan.filter((row) => Boolean((row as ScanRecord & { fno_available?: boolean }).fno_available))
  } else {
    rows = scan.filter((row) => row.signals?.includes('MOMENTUM') || row.verdict === 'BUY')
  }
  return rows.map((row) => projectScanRecord(row as Record<string, unknown>))
}

export function bestSetupsFromRadar(home: RadarHome, dashboard: DashboardPayload): ScannerWorkspaceRow[] {
  if ((home.best_setups || []).length) {
    return (home.best_setups as Array<Record<string, unknown>>).map((row) => projectScanRecord(row))
  }
  const lanes = [...(home.lanes?.breakouts || []), ...(home.lanes?.momentum || [])]
  if (lanes.length) {
    const seen = new Set<string>()
    return lanes.filter((row) => {
      if (seen.has(row.symbol)) return false
      seen.add(row.symbol)
      return true
    }).map((row) => projectScanRecord(row as Record<string, unknown>))
  }
  return scannerFallbackRows('Best Setups', dashboard)
}

export function scannerEmptyHint(rows: number, filtered: number, hasScan: boolean): string {
  if (filtered > 0) return ''
  if (rows > 0) return 'No matches for these filters.'
  if (hasScan) return 'This lane is empty in the saved scan.'
  return 'No matches in saved scan data. Run Scan Now.'
}

export function dashCell(value: unknown): string {
  if (missing(value)) return '—'
  return String(value)
}
