export const DESK_STATES = [
  'PREPARING_DATA',
  'WAITING_FOR_PROVIDER',
  'OPERATION_STUCK',
  'API_UNRESPONSIVE',
  'RESOURCE_EXHAUSTED',
  'RESOURCE_UNKNOWN',
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
  if (resource === 'RESOURCE_UNKNOWN') return 'RESOURCE_UNKNOWN'
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
    case 'RESOURCE_UNKNOWN':
      return 'RESOURCE UNKNOWN'
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
    case 'RESOURCE_UNKNOWN':
      return 'File-descriptor usage could not be measured. Health cannot call this READY.'
    case 'API_UNRESPONSIVE':
      return 'The market API is not answering. Start it with bash scripts/run_quantterm_complete.sh.'
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
