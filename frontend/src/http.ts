export const API_DOWN_MESSAGE =
  'Market API is not running on :8765. Start with bash scripts/run_quantterm_complete.sh, then retry.'

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
