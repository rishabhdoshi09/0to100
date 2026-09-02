export const API_DOWN_MESSAGE =
  'Market API is not running on :8765. Start with bash scripts/run_quantterm_complete.sh, then retry.'

export const REQUEST_TIMEOUT_MESSAGE =
  'Request timed out. The page is not waiting forever — retry.'

export const DEFAULT_FETCH_TIMEOUT_MS = 30_000

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

export function withTimeout(ms: number, parent?: AbortSignal): AbortSignal {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), Math.max(1, ms))
  const abort = () => {
    clearTimeout(timer)
    if (!controller.signal.aborted) controller.abort()
  }
  if (parent) {
    if (parent.aborted) abort()
    else parent.addEventListener('abort', abort, { once: true })
  }
  controller.signal.addEventListener('abort', () => clearTimeout(timer), { once: true })
  return controller.signal
}

export async function fetchJson<T>(
  input: RequestInfo | URL,
  init?: RequestInit & { timeoutMs?: number },
): Promise<T> {
  const timeoutMs = init?.timeoutMs ?? DEFAULT_FETCH_TIMEOUT_MS
  const { timeoutMs: _ignored, signal, ...rest } = (init || {}) as RequestInit & { timeoutMs?: number }
  const combined = withTimeout(timeoutMs, signal || undefined)
  try {
    const response = await fetch(input, { ...rest, signal: combined })
    return await readJson<T>(response)
  } catch (reason) {
    const name = reason instanceof Error ? reason.name : ''
    if (name === 'AbortError' || (reason instanceof DOMException && reason.name === 'AbortError')) {
      throw new Error(REQUEST_TIMEOUT_MESSAGE)
    }
    throw reason
  }
}
