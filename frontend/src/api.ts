import { DASHBOARD_FETCH_TIMEOUT_MS, fetchJson } from './http'
import type { ChartBar, ControlName, DashboardPayload, OperationRecord } from './types'

function request<T>(url: string, init?: RequestInit & { timeoutMs?: number }): Promise<T> {
  return fetchJson<T>(url, { headers: { Accept: 'application/json' }, ...init })
}

export const fetchDashboard = (): Promise<DashboardPayload> =>
  request<DashboardPayload>('/api/dashboard', { timeoutMs: DASHBOARD_FETCH_TIMEOUT_MS })

export const fetchHealth = (): Promise<{
  ok: boolean
  service?: string
  lifecycle?: string
  reason?: string
  reasons?: string[]
  components?: Array<{ name?: string; status?: string; detail?: string }>
  history?: { current?: boolean }
  integrity?: { state?: string; detail?: string }
  resources?: {
    state?: string
    reason?: string
    api?: { pid?: number; fd_count?: number; fd_soft_limit?: number; fd_used_pct?: number }
    market_ops?: { pid?: number; fd_count?: number }
    active_operation_age_s?: number | null
    oldest_running_operation?: { kind?: string; age_s?: number; operation_id?: string } | null
  }
}> =>
  request('/api/health', { timeoutMs: 4_000 })

export const fetchChart = (symbol: string): Promise<{ symbol: string; bars: ChartBar[] }> =>
  request<{ symbol: string; bars: ChartBar[] }>(`/api/chart/${encodeURIComponent(symbol)}`)

export const fetchOperation = (operationId: string): Promise<OperationRecord> =>
  request<OperationRecord>(`/api/operations/${encodeURIComponent(operationId)}`)

export const fetchOperationsPayload = (): Promise<DashboardPayload['operations']> =>
  request<DashboardPayload['operations']>('/api/operations')

export const sendControl = (
  control: ControlName,
): Promise<{
  accepted: boolean
  control: string
  control_id?: string
  operation_id?: string
  operation_status?: string
  created?: boolean
}> =>
  request(`/api/controls/${control}`, { method: 'POST' })
