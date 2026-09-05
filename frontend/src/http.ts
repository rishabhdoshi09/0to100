import { dedupeInFlight } from './pollGate'

export const API_DOWN_MESSAGE =
  'Market API is not running on :8765. Start with bash scripts/run_quantterm_complete.sh, then retry.'

export const REQUEST_TIMEOUT_MESSAGE =
  'Request timed out. The backend did not respond in time.'

export const DEFAULT_FETCH_TIMEOUT_MS = 30_000
export const DASHBOARD_FETCH_TIMEOUT_MS = 20_000

const PROXY_STATUSES = new Set([500, 502, 503, 504])

export async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = (await response.text()).trim()
    if (!body && PROXY_STATUSES.has(response.status)) {
      throw new Error(API_DOWN_MESSAGE)
    }
    throw new Error(body || `Request failed with ${response.status}`)
  }
  return response.json() as Promise<T>
}

type TimeoutHandle = {
  signal: AbortSignal
  cleanup: () => void
}

function createTimeoutHandle(ms: number, parent?: AbortSignal): TimeoutHandle {
  const controller = new AbortController()
  let cleaned = false
  const timeout = setTimeout(() => controller.abort(), Math.max(1, ms))
  const onParentAbort = () => {
    if (!controller.signal.aborted) controller.abort()
  }
  if (parent) {
    if (parent.aborted) onParentAbort()
    else parent.addEventListener('abort', onParentAbort, { once: true })
  }
  const cleanup = () => {
    if (cleaned) return
    cleaned = true
    clearTimeout(timeout)
    if (parent) parent.removeEventListener('abort', onParentAbort)
  }
  controller.signal.addEventListener('abort', cleanup, { once: true })
  return { signal: controller.signal, cleanup }
}

export function withTimeout(ms: number, parent?: AbortSignal): AbortSignal {
  return createTimeoutHandle(ms, parent).signal
}

export async function fetchJson<T>(
  input: RequestInfo | URL,
  init?: RequestInit & { timeoutMs?: number; dedupe?: boolean },
): Promise<T> {
  const timeoutMs = init?.timeoutMs ?? DEFAULT_FETCH_TIMEOUT_MS
  const { timeoutMs: _ignored, signal, dedupe, ...rest } = (init || {}) as RequestInit & {
    timeoutMs?: number
    dedupe?: boolean
  }
  const run = async () => {
    const timed = createTimeoutHandle(timeoutMs, signal || undefined)
    try {
      const response = await fetch(input, { ...rest, signal: timed.signal })
      return await readJson<T>(response)
    } catch (reason) {
      const name = reason instanceof Error ? reason.name : ''
      if (name === 'AbortError' || (reason instanceof DOMException && reason.name === 'AbortError')) {
        throw new Error(REQUEST_TIMEOUT_MESSAGE)
      }
      throw reason
    } finally {
      // A successful fetch used to leave its timeout and parent abort listener alive
      // until the full timeout window elapsed. With several desk pollers that created
      // avoidable timer/listener pressure. Always release them when the request ends.
      timed.cleanup()
    }
  }
  if (dedupe === false) return run()
  const key = `${String(rest.method || 'GET').toUpperCase()} ${String(input)}`
  return dedupeInFlight(key, run)
}
