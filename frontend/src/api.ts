import { DASHBOARD_FETCH_TIMEOUT_MS, fetchJson } from './http'
import type { ChartBar, ControlName, DashboardPayload, OperationRecord } from './types'

function request<T>(url: string, init?: RequestInit & { timeoutMs?: number }): Promise<T> {
  return fetchJson<T>(url, { headers: { Accept: 'application/json' }, ...init })
}

export const fetchDashboard = (): Promise<DashboardPayload> =>
  request<DashboardPayload>('/api/dashboard', { timeoutMs: DASHBOARD_FETCH_TIMEOUT_MS })

export const fetchHealth = (): Promise<{ ok: boolean; service?: string }> =>
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
