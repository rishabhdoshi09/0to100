import type { ChartBar, ControlName, DashboardPayload } from './types'

const json = async <T>(response: Response): Promise<T> => {
  if (!response.ok) {
    const body = await response.text()
    throw new Error(body || `Request failed with ${response.status}`)
  }
  return response.json() as Promise<T>
}

export const fetchDashboard = (): Promise<DashboardPayload> =>
  fetch('/api/dashboard', { headers: { Accept: 'application/json' } })
    .then((response) => json<DashboardPayload>(response))

export const fetchChart = (symbol: string): Promise<{ symbol: string; bars: ChartBar[] }> =>
  fetch(`/api/chart/${encodeURIComponent(symbol)}`, {
    headers: { Accept: 'application/json' },
  }).then((response) => json<{ symbol: string; bars: ChartBar[] }>(response))

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
