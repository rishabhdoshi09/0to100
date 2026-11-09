/** Stale or rebuilding stock recommendations are never actionable. */
export function recommendationsAreCurrent(
  workspace: { scan_scanned_at: string; long_term_scanned_at: string; records_status: string } | null,
  scan: { available: boolean; scanned_at?: string; provenance?: { data_current?: boolean } },
  longTermScannedAt: string | undefined,
  dataReady: boolean,
): boolean {
  if (!workspace || !dataReady || !scan.available || scan.provenance?.data_current === false) return false
  const status = String(workspace.records_status || '').toUpperCase()
  if (['FAILED', 'REFRESHING', 'STALE', 'PRIOR_DAY_SNAPSHOT', 'UNAVAILABLE'].includes(status)) return false
  return Boolean(
    scan.scanned_at && workspace.scan_scanned_at === scan.scanned_at &&
    workspace.long_term_scanned_at === String(longTermScannedAt || ''),
  )
}
