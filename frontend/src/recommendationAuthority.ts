/** Stale or rebuilding stock recommendations are never actionable. */

export type DisplayAuthority = {
  status?: string
  projection?: string
  scan_current?: boolean
  scan_scanned_at?: string
  scan_id?: string
  scan_market_session?: string
  projection_scan_scanned_at?: string
  evaluated_at?: string
  publication?: {
    recommendations?: string
    recommendation_cards?: number | null
    error?: { error_code?: string; error_message?: string } | null
    scan_scanned_at?: string
  }
}

type Workspace = {
  scan_scanned_at: string
  long_term_scanned_at: string
  records_status: string
  display_authority?: DisplayAuthority
  rebuild_error?: string
  error?: string
} | null

type DeskScan = { available: boolean; scanned_at?: string; provenance?: { data_current?: boolean } }

/** A server authority older than this cannot certify the cards on screen
 * (for example a workspace recalled from session memory after a reload). */
export const AUTHORITY_MAX_AGE_MS = 90_000

export function recommendationsAreCurrent(
  workspace: Workspace,
  scan: DeskScan,
  longTermScannedAt: string | undefined,
  dataReady: boolean,
  nowMs: number = Date.now(),
): boolean {
  if (!workspace) return false
  const authority = workspace.display_authority
  if (authority) {
    // The Opportunities API judged currency itself, at read time, from the
    // canonical scan identity and freshness. A 15 s UI dashboard cache cannot.
    if (String(authority.status || '').toUpperCase() !== 'CURRENT' || authority.scan_current !== true) return false
    if (!authority.scan_scanned_at || authority.scan_scanned_at !== workspace.scan_scanned_at) return false
    const evaluated = Date.parse(String(authority.evaluated_at || ''))
    if (!Number.isFinite(evaluated) || nowMs - evaluated > AUTHORITY_MAX_AGE_MS || evaluated - nowMs > 60_000) return false
    // Only a fresh, ready dashboard may veto it, and only by proving a different scan.
    if (dataReady && scan.available && scan.scanned_at && scan.scanned_at !== authority.scan_scanned_at) return false
    return true
  }
  // Older backends without display_authority: keep the conservative rule.
  if (!dataReady || !scan.available || scan.provenance?.data_current === false) return false
  const status = String(workspace.records_status || '').toUpperCase()
  if (['FAILED', 'REFRESHING', 'STALE', 'PRIOR_DAY_SNAPSHOT', 'UNAVAILABLE'].includes(status)) return false
  return Boolean(
    scan.scanned_at && workspace.scan_scanned_at === scan.scanned_at &&
    workspace.long_term_scanned_at === String(longTermScannedAt || ''),
  )
}

export type RecommendationState = { state: string; title: string; detail: string }

/** One bounded, truthful sentence for every non-current state. Never "loading" forever. */
export function recommendationStateMessage(workspace: Workspace): RecommendationState {
  const authority = workspace?.display_authority
  const status = String(authority?.status || workspace?.records_status || '').toUpperCase()
  const publication = authority?.publication
  if (!workspace) {
    return { state: 'UNAVAILABLE', title: 'Recommendations are not available yet', detail: 'The Opportunities API has not answered.' }
  }
  if (status === 'REFRESHING') {
    const failed = String(publication?.recommendations || '').toLowerCase() === 'error'
    return {
      state: 'REBUILDING',
      title: 'Rebuilding recommendations for the latest scan',
      detail: failed
        ? `The scan finished, but its recommendation publication failed (${publication?.error?.error_code || 'error'}). QuantTerm is rebuilding them in the background; previous cards stay withheld.`
        : 'A newer scan was published. Its recommendations are being built in the background; previous cards stay withheld.',
    }
  }
  if (status === 'FAILED') {
    return {
      state: 'FAILED',
      title: 'Recommendations could not be built',
      detail: String(workspace.rebuild_error || workspace.error || 'The background rebuild failed; it retries automatically.'),
    }
  }
  if (status === 'SCAN_UNAVAILABLE') {
    return { state: 'NO_SCAN', title: 'No verified market scan', detail: 'Recommendations appear after the scheduler publishes a verified scan.' }
  }
  if (status === 'SCAN_NOT_CURRENT' || status === 'STALE') {
    return {
      state: 'STALE',
      title: 'Saved scan is not current for this session',
      detail: 'No verified NSE scan for the current session yet. Its recommendation cards are withheld (nothing is rebuilt from an old scan) until the scheduler publishes a verified scan.',
    }
  }
  return { state: 'VERIFYING', title: 'Verifying current NSE stock recommendations', detail: '' }
}
