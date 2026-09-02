export type PageRequestStatus = 'idle' | 'loading' | 'ready' | 'empty' | 'error' | 'timeout'

export type PageHealth = {
  page: string
  apiReachable: boolean
  requestCompleted: boolean
  payloadValid: boolean
  dataPresent: boolean
  validEmpty: boolean
  lastSuccessAt: string
  lastError: string
  loadingMs: number
  status: PageRequestStatus
}

const TIMEOUT_HINT = 'timed out'

export function classifyPageError(message: string): PageRequestStatus {
  return String(message || '').toLowerCase().includes(TIMEOUT_HINT) ? 'timeout' : 'error'
}

export function pageRequestStatus(opts: {
  loading: boolean
  data: unknown
  error?: string
  isEmpty?: (data: unknown) => boolean
}): PageRequestStatus {
  if (opts.error) return classifyPageError(opts.error)
  if (opts.loading && opts.data == null) return 'loading'
  if (opts.data == null) return 'idle'
  if (opts.isEmpty?.(opts.data)) return 'empty'
  return 'ready'
}

export function pageHealth(opts: {
  page: string
  loading: boolean
  data: unknown
  error?: string
  startedAt?: number
  isEmpty?: (data: unknown) => boolean
}): PageHealth {
  const status = pageRequestStatus(opts)
  const empty = Boolean(opts.data != null && opts.isEmpty?.(opts.data))
  return {
    page: opts.page,
    apiReachable: status !== 'timeout' && !String(opts.error || '').includes('Market API is not running'),
    requestCompleted: status !== 'loading' && status !== 'idle',
    payloadValid: status === 'ready' || status === 'empty',
    dataPresent: status === 'ready',
    validEmpty: empty,
    lastSuccessAt: status === 'ready' || status === 'empty' ? new Date().toISOString() : '',
    lastError: opts.error || '',
    loadingMs: opts.startedAt ? Math.max(0, Date.now() - opts.startedAt) : 0,
    status,
  }
}

export function pageStatusLabel(status: PageRequestStatus): string {
  if (status === 'loading') return 'Loading'
  if (status === 'empty') return 'Empty'
  if (status === 'timeout') return 'Failed'
  if (status === 'error') return 'Failed'
  if (status === 'ready') return 'Ready'
  return 'Waiting'
}
