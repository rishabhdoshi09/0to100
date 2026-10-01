/**
 * Presentation logic for the Evolution Lab: Champion vs Challenger.
 *
 * The screen must stay honest when there is no Champion yet (fresh install,
 * before the first PAPER_CYCLE) or no Challengers at all -- an empty board
 * reads as "nothing to report", never as a broken page.
 */

export type EvolutionScorecard = {
  policy_id: string
  decision_snapshots: number
  selected_trades: number
  rejects: number
  missed_winners: number
  avoided_losers: number
  correct_rejections: number
  winner_taken: number
  loser_taken: number
  win_rate: number | null
  wilson_lb: number | null
  expectancy_R: number | null
  median_R: number | null
  avg_win_R: number | null
  avg_loss_R: number | null
  profit_factor: number | null
  max_drawdown_R: number | null
  selection_rate: number | null
  opportunity_capture_rate: number | null
  false_positive_rate: number | null
}

export type PairedComparison = {
  champion_policy_id: string
  challenger_policy_id: string
  paired_snapshots: number
  incremental_expectancy_R: number | null
  agreement: {
    both_selected: number
    both_rejected: number
    champion_only: number
    challenger_only: number
  }
}

export type EvolutionPolicyRow = {
  policy_id: string
  version: number
  parent_policy_id: string | null
  status: string
  hypothesis: string
  created_at: string
  scorecard: EvolutionScorecard
  paired_vs_champion: PairedComparison | null
  controls_paper_decisions: boolean
  manifest_fingerprint?: string
  promotion_evaluation?: {
    status?: string
    reason?: string
    paired_snapshots?: number
    incremental_expectancy_R?: number | null
    fdr_rejected?: boolean
    harness_stats?: Record<string, number>
    regime_breadth?: Record<string, unknown>
  }
  latest_promotion_proof?: Record<string, unknown> | null
}

export type EvolutionEvent = {
  policy_id: string
  at: string
  status: string
  reason: string
}

export type EvolutionLabBoard = {
  domain: string
  champion: EvolutionPolicyRow | null
  challenger_leaderboard: EvolutionPolicyRow[]
  retired_count: number
  recent_events: EvolutionEvent[]
  auto_promotion_enabled: boolean
  live_locked: boolean
  live_execution_authorized: boolean
}

export function formatR(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  const sign = value > 0 ? '+' : ''
  return `${sign}${value.toFixed(2)}R`
}

export function formatPct(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return `${(value * 100).toFixed(0)}%`
}

export function incrementalTone(value: number | null | undefined): 'positive' | 'negative' | 'neutral' {
  if (value === null || value === undefined || Number.isNaN(value)) return 'neutral'
  if (value > 0) return 'positive'
  if (value < 0) return 'negative'
  return 'neutral'
}

export function sampleFloorNote(n: number, floor = 30): string {
  if (n >= floor) return `${n} paired observations`
  return `${n}/${floor} paired observations — below the promotion floor`
}
