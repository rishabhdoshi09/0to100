import { readJson } from './http'
import type { ChartBar, ControlName, DashboardPayload, OperationRecord } from './types'

const json = readJson

export const fetchDashboard = (): Promise<DashboardPayload> =>
  fetch('/api/dashboard', { headers: { Accept: 'application/json' } })
    .then((response) => json<DashboardPayload>(response))

export const fetchChart = (symbol: string): Promise<{ symbol: string; bars: ChartBar[] }> =>
  fetch(`/api/chart/${encodeURIComponent(symbol)}`, {
    headers: { Accept: 'application/json' },
  }).then((response) => json<{ symbol: string; bars: ChartBar[] }>(response))

export const fetchOperation = (operationId: string): Promise<OperationRecord> =>
  fetch(`/api/operations/${encodeURIComponent(operationId)}`, {
    headers: { Accept: 'application/json' },
  }).then((response) => json<OperationRecord>(response))

export const fetchOperationsPayload = (): Promise<DashboardPayload['operations']> =>
  fetch('/api/operations', { headers: { Accept: 'application/json' } })
    .then((response) => json<DashboardPayload['operations']>(response))

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
  fetch(`/api/controls/${control}`, {
    method: 'POST',
    headers: { Accept: 'application/json' },
  }).then((response) => json<{
    accepted: boolean
    control: string
    control_id?: string
    operation_id?: string
    operation_status?: string
    created?: boolean
  }>(response))
