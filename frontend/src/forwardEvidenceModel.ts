/**
 * Presentation logic for the forward-evidence board.
 *
 * The screen's hardest job is to stay honest when there is nothing to show. A
 * row of zeros reads like a measurement, so an empty board must say in words
 * that nothing has settled — and the tone must not be the same green as a
 * board with three hundred trades behind it.
 */

export type EvidenceCell = {
  context_key: string
  count: number
  wins: number
  losses: number
  win_rate: number | null
  wilson_lower_bound: number | null
  expectancy_R: number | null
  median_R: number | null
  calibration_gap: number | null
  usable_for_ranking: boolean
}

export type GroupRow = {
  count: number
  wins: number
  losses: number
  cells: number
  expectancy_R: number | null
  usable_for_ranking: boolean
  setup?: string
  regime?: string
  sector?: string
}

export type UnresolvedRow = {
  symbol: string
  entry_date: string
  entry_price: number | null
  stop_price: number | null
  target_price: number | null
  bars_held: number
  decision_id: string
  context_key: string
  attributable: boolean
}

export type ForwardEvidenceBoard = {
  schema_version: number
  state: 'NO_MARKET_EVIDENCE' | 'ACCUMULATING' | 'MEASURED' | string
  headline: string
  evidence_class: string
  min_sample: number
  settled_trades: number
  cells: EvidenceCell[]
  cells_usable_for_ranking: number
  by_setup: GroupRow[]
  by_regime: GroupRow[]
  by_sector: GroupRow[]
  r_distribution: Record<string, number>
  unresolved: UnresolvedRow[]
  unresolved_count: number
  unattributable_open: number
  non_market_evidence: {
    historical_replay_cells: number
    test_fixture_cells: number
    note: string
  }
}

export const R_BUCKET_ORDER = [
  '<= -2R', '-2R..-1R', '-1R..0', '0..1R', '1R..2R', '>= 2R',
] as const

/** Never green for an empty board: absence of evidence is not a healthy state. */
export function stateTone(state: string): string {
  if (state === 'MEASURED') return 'forward-evidence__state--measured'
  if (state === 'ACCUMULATING') return 'forward-evidence__state--accumulating'
  return 'forward-evidence__state--none'
}

export function hasSettledEvidence(board: ForwardEvidenceBoard | null): boolean {
  return Boolean(board && board.settled_trades > 0)
}

/** How far a context is from being allowed to affect ranking. */
export function progressToFloor(count: number, minSample: number): string {
  if (minSample <= 0) return ''
  if (count >= minSample) return 'counts toward ranking'
  return `${count}/${minSample} to count`
}

export function formatR(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  const sign = value > 0 ? '+' : ''
  return `${sign}${value.toFixed(2)}R`
}

export function formatPct(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  return `${(value * 100).toFixed(0)}%`
}

/** Bars for the R distribution, scaled to the largest bucket. */
export function distributionBars(
  distribution: Record<string, number> | undefined,
): Array<{ bucket: string; count: number; share: number }> {
  const rows = R_BUCKET_ORDER.map((bucket) => ({
    bucket,
    count: Number(distribution?.[bucket] ?? 0),
    share: 0,
  }))
  const peak = Math.max(...rows.map((r) => r.count), 0)
  if (peak <= 0) return rows
  return rows.map((r) => ({ ...r, share: r.count / peak }))
}

/** One line about trades that cannot name the decision that opened them. */
export function attributionWarning(board: ForwardEvidenceBoard | null): string {
  const orphans = board?.unattributable_open ?? 0
  if (orphans <= 0) return ''
  const noun = orphans === 1 ? 'open trade' : 'open trades'
  return `${orphans} ${noun} cannot name the decision behind them, so their outcomes will not become evidence.`
}
