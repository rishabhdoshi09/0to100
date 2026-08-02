import type {
  ChartBar,
  ControlName,
  DashboardPayload,
  OperationRecord,
  OptionsChainPayload,
  OptionsEodHistoryPayload,
} from './types'

const json = async <T>(response: Response): Promise<T> => {
  if (!response.ok) {
    const body = await response.text()
    throw new Error(body || `Request failed with ${response.status}`)
  }
  return response.json() as Promise<T>
}

export const fetchDashboard = (): Promise<DashboardPayload> => {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), 60_000)
  return fetch('/api/dashboard', {
    headers: { Accept: 'application/json' },
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        const body = await response.text()
        throw new Error(
          body?.trim()
            || `Dashboard HTTP ${response.status} — is http://127.0.0.1:8765 up?`,
        )
      }
      return response.json() as Promise<DashboardPayload>
    })
    .catch((reason) => {
      if (reason instanceof DOMException && reason.name === 'AbortError') {
        throw new Error(
          'Dashboard timed out after 60s. Check Terminal API on :8765 and market-ops worker; then Retry.',
        )
      }
      if (reason instanceof TypeError) {
        throw new Error(
          'Cannot reach /api/dashboard (proxy → :8765). Run: bash scripts/stop_quantterm.sh && bash scripts/run_quantterm_complete.sh',
        )
      }
      throw reason
    })
    .finally(() => window.clearTimeout(timer))
}

export const fetchChart = (symbol: string): Promise<{ symbol: string; bars: ChartBar[] }> =>
  fetch(`/api/chart/${encodeURIComponent(symbol)}`, {
    headers: { Accept: 'application/json' },
  }).then((response) => json<{ symbol: string; bars: ChartBar[] }>(response))

export const fetchOperation = (operationId: string): Promise<OperationRecord> =>
  fetch(`/api/operations/${encodeURIComponent(operationId)}`, {
    headers: { Accept: 'application/json' },
  }).then((response) => json<OperationRecord>(response))

export const fetchMarketOptions = (symbol: string, force = false): Promise<OptionsChainPayload> =>
  fetch(`/api/market/options/${encodeURIComponent(symbol)}?force=${force ? 'true' : 'false'}`, {
    headers: { Accept: 'application/json' },
  }).then((response) => json<OptionsChainPayload>(response))

export const fetchOptionsEodHistory = (
  symbol: string,
  days = 14,
): Promise<OptionsEodHistoryPayload> =>
  fetch(`/api/market/options/${encodeURIComponent(symbol)}/history?days=${days}`, {
    headers: { Accept: 'application/json' },
  }).then((response) => json<OptionsEodHistoryPayload>(response))

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
