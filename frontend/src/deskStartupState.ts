export const DESK_STATES = [
  'PREPARING_DATA',
  'WAITING_FOR_PROVIDER',
  'OPERATION_STUCK',
  'API_UNRESPONSIVE',
  'RESOURCE_EXHAUSTED',
  'RESOURCE_UNKNOWN',
  'HISTORY_STALE',
  'STARTING',
  'DEGRADED',
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
  lifecycle?: string
}

export type DeskHealthReasonInput = {
  lifecycle?: string
  reason?: string
  reasons?: string[]
  components?: Array<{ name?: string; status?: string; detail?: string }>
  resourceReason?: string
  state: DeskStartupState
}

export function deskStartupState(input: DeskStartupInput): DeskStartupState {
  const resource = String(input.resourceState || '').toUpperCase()
  const life = String(input.lifecycle || '').toUpperCase()
  if (resource === 'RESOURCE_EXHAUSTED') return 'RESOURCE_EXHAUSTED'
  if (resource === 'RESOURCE_UNKNOWN') return 'RESOURCE_UNKNOWN'
  if (input.apiUnresponsive) return 'API_UNRESPONSIVE'
  if (input.operationStuck) return 'OPERATION_STUCK'
  if (input.waitingForProvider) return 'WAITING_FOR_PROVIDER'
  if (input.historyStale && input.hasSavedData) return 'HISTORY_STALE'
  if (life === 'FAILED') return 'API_UNRESPONSIVE'
  if (life === 'DEGRADED' || life === 'RECOVERING') return 'DEGRADED'
  if (life === 'STARTING') {
    return input.hasSavedData || input.dataReady ? 'STARTING' : 'PREPARING_DATA'
  }
  if (input.dataReady || input.hasSavedData) return 'READY'
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
    case 'STARTING':
      return 'STARTING'
    case 'DEGRADED':
      return 'DEGRADED'
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
    case 'STARTING':
      return 'A required desk component is not READY yet. Saved scans and pages stay on screen.'
    case 'DEGRADED':
      return 'Health is not READY. The saved desk stays visible; fix the listed blocker.'
    case 'READY':
      return ''
    default:
      return 'Official prices or the market scan are still being prepared.'
  }
}

function firstNonReadyComponent(
  components?: Array<{ name?: string; status?: string; detail?: string }>,
): string {
  for (const row of components || []) {
    const status = String(row.status || '').toUpperCase()
    if (!status || status === 'READY') continue
    const name = String(row.name || 'component')
    const detail = String(row.detail || '').trim()
    return detail ? `${name} is ${status}: ${detail}` : `${name} is ${status}`
  }
  return ''
}

export function deskStartupReason(input: DeskHealthReasonInput): string {
  const reasons = (input.reasons || []).map((row) => String(row || '').trim()).filter(Boolean)
  if (reasons.length) return reasons[0]
  const life = String(input.lifecycle || '').toUpperCase()
  const generic = /^desk is still coming up$/i.test(String(input.reason || '').trim())
  if (life && life !== 'READY') {
    const fromComponent = firstNonReadyComponent(input.components)
    if (fromComponent) return fromComponent
    if (input.reason && !generic) return String(input.reason)
  }
  if (input.resourceReason) return String(input.resourceReason)
  if (input.reason) return String(input.reason)
  return deskStartupRecovery(input.state)
}
