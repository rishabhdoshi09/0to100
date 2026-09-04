import type { HomeAction } from './productApi'

export const SYSTEM_LANE_ORDER = ['data', 'zerodha', 'automation', 'paper_bot', 'learning'] as const
export type SystemLaneId = (typeof SYSTEM_LANE_ORDER)[number]

export const SAFE_HOME_CONTROLS = new Set([
  'REFRESH_DATA_NOW',
  'RUN_SCAN_NOW',
  'PAUSE_NEW_PAPER_ENTRIES',
  'RESUME_NEW_PAPER_ENTRIES',
  'RUN_CYCLE_NOW',
  'VERIFY_FORWARD_SOAK',
  'CHECK_SYSTEM',
  'SIMULATE_PAST_DECISIONS',
  'OBSERVE_ONLY_TODAY',
  'CLEAR_OBSERVE_ONLY',
])

export const FORBIDDEN_HOME_CONTROLS = new Set([
  'KILL_PID',
  'RUN_SHELL',
  'DELETE_QUEUE',
  'WIPE_JOBS',
  'UNLOCK_LIVE_MONEY',
  'LIVE_BUY',
  'LIVE_SELL',
  'BROKER_BUY',
  'BROKER_SELL',
  'PROMOTE_STRATEGY',
  'DISABLE_RISK',
  'DD_BYPASS',
  'CHASE_BYPASS',
])

const SECRET_PARTS = ['token', 'secret', 'password', 'authorization', 'bearer', 'api_key']

export type SystemLane = {
  id?: string
  label?: string
  status?: string
  status_code?: string
  summary?: string
  detail?: string
  what?: string
  meaning?: string
  waiting_for?: string
  current?: string
  next?: string
  after_that?: string
  last_success_at?: string
  last_failure_at?: string
  last_failure_reason?: string
  progress?: { kind?: string; label?: string; current?: number | null; total?: number | null; status?: string; stage?: string; message?: string } | null
  current_job?: string
  current_job_id?: string
  current_job_started_at?: string | number
  next_check_at?: string | number
  freshness?: string
  source?: string
  dependencies?: string[]
  needs_user?: boolean
  recovering?: boolean
  degraded?: boolean
  optional_capability?: boolean
  blocks_autonomy?: boolean
  login_required?: boolean
  primary_action?: HomeAction | null
  secondary_actions?: HomeAction[]
  full_details_page?: string
  full_details_label?: string
  technical?: Record<string, unknown>
  live_locked?: boolean
  positions?: Array<{
    symbol?: string
    entry?: number
    status?: string
    stop?: number
    target?: number
    risk_used?: number
  }>
  on?: boolean
  paused?: boolean
  positions_open?: number
  todays_entries?: number
  last_decision?: string
  why?: string
  real_forward_observations?: number
  settled_trades?: number
  rejected_candidates_settled?: number
  execution_adjusted_coverage_pct?: number | null
  insufficient_evidence?: boolean
  forward_soak_status?: string
}

export type CheckSystemSnapshot = {
  read_only?: boolean
  source?: string
  lanes?: Array<{ id?: string; label?: string; status?: string; detail?: string }>
  action?: HomeAction | null
}

export const LANE_TITLE: Record<string, string> = {
  data: 'DATA',
  zerodha: 'ZERODHA',
  automation: 'AUTOMATION',
  paper_bot: 'PAPER BOT',
  learning: 'LEARNING',
  check_system: 'SYSTEM CHECK',
}

export function laneTitle(id: string, lane?: SystemLane): string {
  return lane?.label || LANE_TITLE[id] || id.replace('_', ' ').toUpperCase()
}

export function isActivatingKey(key: string): boolean {
  return key === 'Enter' || key === ' '
}

export function isSafeHomeControl(control: string | undefined | null): boolean {
  const name = String(control || '')
  if (!name) return true
  if (FORBIDDEN_HOME_CONTROLS.has(name)) return false
  if (name.includes('BUY') || name.includes('SELL') || name.includes('UNLOCK_LIVE')) return false
  return SAFE_HOME_CONTROLS.has(name)
}

export function filterSafeActions(actions: Array<HomeAction | null | undefined>): HomeAction[] {
  return actions.filter((action): action is HomeAction => {
    if (!action || !action.label) return false
    if (action.kind === 'instruction' || action.kind === 'refresh') return true
    return isSafeHomeControl(action.control)
  })
}

export function lanePrimaryAction(lane: SystemLane | undefined): HomeAction | null {
  if (!lane) return null
  const status = String(lane.status || '')
  const action = lane.primary_action || null
  if (!action) return null
  if (!isSafeHomeControl(action.control) && action.kind !== 'instruction') return null
  if (status === 'Ready' || status === 'Working' || lane.optional_capability) return null
  return action
}

export function laneSecondaryActions(lane: SystemLane | undefined): HomeAction[] {
  if (!lane) return []
  const status = String(lane.status || '')
  const actions = filterSafeActions(lane.secondary_actions || [])
  if (status === 'Working') {
    return actions.filter((action) => action.control !== 'REFRESH_DATA_NOW' && action.control !== 'RUN_SCAN_NOW')
  }
  if (status === 'Ready') {
    return actions.filter((action) => action.control !== lane.primary_action?.control)
  }
  return actions
}

export function nothingNeeded(lane: SystemLane | undefined): boolean {
  if (!lane || lane.needs_user) return false
  const status = String(lane.status || '')
  if (lane.optional_capability) return true
  return status === 'Working' || status === 'Ready'
}

export function hasSecretKey(key: string): boolean {
  const lowered = key.toLowerCase()
  return SECRET_PARTS.some((part) => lowered.includes(part))
}

export function scrubTechnical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(scrubTechnical)
  if (!value || typeof value !== 'object') return value
  const out: Record<string, unknown> = {}
  for (const [key, item] of Object.entries(value as Record<string, unknown>)) {
    if (hasSecretKey(key)) continue
    if (typeof item === 'string' && SECRET_PARTS.some((part) => item.toLowerCase().includes(`${part}=`))) continue
    out[key] = scrubTechnical(item)
  }
  return out
}

export function technicalLines(technical: Record<string, unknown> | undefined, limit = 24): string[] {
  const clean = scrubTechnical(technical || {}) as Record<string, unknown>
  const lines: string[] = []
  for (const [key, value] of Object.entries(clean)) {
    if (value == null || value === '') continue
    if (typeof value === 'object') {
      lines.push(`${key}=${JSON.stringify(value)}`)
    } else {
      lines.push(`${key}=${String(value)}`)
    }
    if (lines.length >= limit) break
  }
  return lines
}

export function laneAriaLabel(id: string, lane?: SystemLane): string {
  const title = laneTitle(id, lane)
  const status = lane?.status || 'Waiting'
  const suffix = lane?.optional_capability && !lane?.needs_user ? ' Optional capability; no action required.' : ' View details'
  return `${title}: ${status}.${suffix}`
}

export function checkSystemRows(snapshot: CheckSystemSnapshot | undefined, fallback: Record<string, SystemLane>): Array<{ id: string; label: string; status: string }> {
  if (snapshot?.lanes?.length) {
    return snapshot.lanes.map((row) => ({
      id: String(row.id || ''),
      label: String(row.label || row.id || ''),
      status: String(row.status || 'Waiting'),
    }))
  }
  return [
    ...SYSTEM_LANE_ORDER.map((id) => ({
      id,
      label: LANE_TITLE[id],
      status: fallback[id]?.status || 'Waiting',
    })),
    { id: 'live_money', label: 'Live Money', status: 'Locked' },
  ]
}

export function liveMoneyStillLocked(osLiveLocked: boolean | undefined, lane?: SystemLane): boolean {
  if (osLiveLocked === false) return false
  if (lane?.live_locked === false) return false
  return true
}
