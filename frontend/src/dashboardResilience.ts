import type { DashboardPayload } from './types'

function hasScanRows(payload: DashboardPayload['scan']): boolean {
  return Boolean(payload.records?.length || payload.scanned_at)
}

function hasLongTermRows(payload: DashboardPayload['long_term']): boolean {
  return Boolean(payload.records?.length || payload.scanned_at)
}

function hasPaperHistory(payload: DashboardPayload['paper']): boolean {
  return Boolean(
    payload.open_positions?.length
    || payload.closed_trades?.length
    || payload.equity_curve?.length
    || payload.refusals?.length
    || Object.keys(payload.last_cycle || {}).length
    || Object.keys(payload.learning || {}).length,
  )
}

function hasAutonomyHistory(payload: DashboardPayload['autonomy']): boolean {
  return Boolean(
    payload.recent_dialogue?.length
    || payload.recent_transitions?.length
    || payload.jobs_recent?.length
    || Object.keys(payload.last_cycle || {}).length,
  )
}

/**
 * Merge a degraded dashboard read with the last durable read snapshot.
 *
 * Current operational/safety truth always comes from `incoming`. Only durable
 * read history is carried forward when the new projection is unavailable or
 * empty. This keeps the desk useful during a refresh/provider failure without
 * turning stale observations into current execution authority.
 */
export function reconcileDashboard(
  previous: DashboardPayload,
  incoming: DashboardPayload,
): DashboardPayload {
  const scan = !hasScanRows(incoming.scan) && hasScanRows(previous.scan)
    ? {
        ...previous.scan,
        available: incoming.scan.available,
      }
    : incoming.scan

  const longTerm = !hasLongTermRows(incoming.long_term) && hasLongTermRows(previous.long_term)
    ? {
        ...previous.long_term,
        available: incoming.long_term.available,
        job: incoming.long_term.job || previous.long_term.job,
      }
    : incoming.long_term

  const conviction = incoming.conviction.length === 0 && previous.conviction.length > 0
    ? previous.conviction
    : incoming.conviction

  const preservePaper = !hasPaperHistory(incoming.paper) && hasPaperHistory(previous.paper)
  const paper: DashboardPayload['paper'] = preservePaper
    ? {
        ...incoming.paper,
        // `available` means the paper book is readable. Execution truth remains
        // fail-closed through enabled/supervisor/autonomy fields from incoming.
        available: true,
        capital: previous.paper.capital,
        equity: previous.paper.equity,
        equity_curve: previous.paper.equity_curve,
        open_risk: previous.paper.open_risk,
        risk_per_trade_pct: previous.paper.risk_per_trade_pct,
        max_positions: previous.paper.max_positions,
        open_positions: previous.paper.open_positions,
        closed_trades: previous.paper.closed_trades,
        refusals: previous.paper.refusals,
        last_cycle: previous.paper.last_cycle,
        learning: previous.paper.learning,
        last_error: incoming.paper.last_error || previous.paper.last_error,
      }
    : incoming.paper

  const preserveAutonomy = !hasAutonomyHistory(incoming.autonomy) && hasAutonomyHistory(previous.autonomy)
  const autonomy: DashboardPayload['autonomy'] = preserveAutonomy
    ? {
        ...incoming.autonomy,
        // These are historical/read-only surfaces. Current running state,
        // capabilities, failures, broker state and write controls stay incoming.
        recent_dialogue: previous.autonomy.recent_dialogue,
        recent_transitions: previous.autonomy.recent_transitions,
        jobs_recent: previous.autonomy.jobs_recent,
        last_cycle: previous.autonomy.last_cycle,
      }
    : incoming.autonomy

  const news = !incoming.news.available
    && incoming.news.articles.length === 0
    && previous.news.articles.length > 0
    ? {
        ...previous.news,
        available: false,
        latest_refresh: incoming.news.latest_refresh || previous.news.latest_refresh,
        error: incoming.news.error || previous.news.error,
      }
    : incoming.news

  const fno = !incoming.fno.available
    && incoming.fno.underlyings.length === 0
    && previous.fno.underlyings.length > 0
    ? {
        ...previous.fno,
        available: false,
        error: incoming.fno.error || previous.fno.error,
      }
    : incoming.fno

  const market = !incoming.market.available && previous.market.available
    ? {
        ...previous.market,
        available: false,
      }
    : incoming.market

  return {
    ...incoming,
    market,
    scan,
    long_term: longTerm,
    conviction,
    paper,
    autonomy,
    news,
    fno,
    data: {
      ...incoming.data,
      // Counts describe durable artifacts. Readiness/blockers remain current.
      scan_saved: incoming.data.scan_saved || previous.data.scan_saved || hasScanRows(scan),
      scan_records: Math.max(incoming.data.scan_records || 0, previous.data.scan_records || 0, scan.records.length),
      long_term_saved: incoming.data.long_term_saved || previous.data.long_term_saved || hasLongTermRows(longTerm),
      long_term_records: Math.max(
        incoming.data.long_term_records || 0,
        previous.data.long_term_records || 0,
        longTerm.records.length,
      ),
    },
  }
}
