export const DESK_STATES = [
  'PREPARING_DATA',
  'WAITING_FOR_PROVIDER',
  'OPERATION_STUCK',
  'API_UNRESPONSIVE',
  'RESOURCE_EXHAUSTED',
  'HISTORY_STALE',
  'READY',
] as const

export type DeskStartupState = (typeof DESK_STATES)[number]

export type DeskStartupInput = {
  apiUnresponsive?: boolean
  resourceState?: string
  operationStuck?: boolean
  waitingForProvider?: boolean
  historyStale?: boolean
  dataReady?: boolean
  hasSavedData?: boolean
}

export function deskStartupState(input: DeskStartupInput): DeskStartupState {
  const resource = String(input.resourceState || '').toUpperCase()
  if (resource === 'RESOURCE_EXHAUSTED') return 'RESOURCE_EXHAUSTED'
  if (input.apiUnresponsive) return 'API_UNRESPONSIVE'
  if (input.operationStuck) return 'OPERATION_STUCK'
  if (input.waitingForProvider) return 'WAITING_FOR_PROVIDER'
  if (input.historyStale && input.hasSavedData) return 'HISTORY_STALE'
  if (input.dataReady) return 'READY'
  return 'PREPARING_DATA'
}

export function deskStartupLabel(state: DeskStartupState): string {
  switch (state) {
    case 'READY':
      return 'DATA READY'
    case 'WAITING_FOR_PROVIDER':
      return 'WAITING FOR PROVIDER'
    case 'OPERATION_STUCK':
      return 'OPERATION STUCK'
    case 'API_UNRESPONSIVE':
      return 'API UNRESPONSIVE'
    case 'RESOURCE_EXHAUSTED':
      return 'RESOURCE EXHAUSTED'
    case 'HISTORY_STALE':
      return 'HISTORY STALE'
    default:
      return 'PREPARING DATA'
  }
}

export function deskStartupRecovery(state: DeskStartupState): string {
  switch (state) {
    case 'RESOURCE_EXHAUSTED':
      return 'Restart the terminal API and market-ops worker. Do not keep polling until file descriptors drop.'
    case 'API_UNRESPONSIVE':
      return 'Two consecutive health probes missed. Check the installed host and /api/health before considering a restart; do not start a duplicate service.'
    case 'OPERATION_STUCK':
      return 'A desk operation exceeded its deadline. Wait for recovery or restart market-ops; saved pages stay usable.'
    case 'WAITING_FOR_PROVIDER':
      return 'A research provider is cooling down. Saved company files remain usable.'
    case 'HISTORY_STALE':
      return 'Official history is behind the expected session. Saved scans remain on screen.'
    case 'READY':
      return ''
    default:
      return 'Official prices or the market scan are still being prepared.'
  }
}

/** Data readiness for the header, from the authority that actually knows.
 *
 * The dashboard read cache marks its own projection data.ready=false whenever
 * the snapshot is older than its short freshness window. That is a property of
 * a UI cache, not of official NSE history. While the dashboard projection is
 * stale or warming, use /api/health's official-history verdict instead, so a
 * current, verified history is never shown as "PREPARING DATA".
 */
export function effectiveDataReady(input: {
  dashboardReady?: boolean
  dashboardStale?: boolean
  officialHistoryCurrent?: boolean | null
}): boolean {
  if (input.dashboardStale) return input.officialHistoryCurrent === true
  return input.dashboardReady === true
}
