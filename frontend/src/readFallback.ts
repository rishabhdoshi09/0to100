/**
 * Network failures must NOT resurrect old actionable BUY cards.
 *
 * The read-only Recommendations backend intentionally serves REFRESHING or
 * FAILED with zero cards when scan identity changes. Returning an opaque
 * remembered workspace after a 503 defeated that safety policy.
 * Saved research is retained in the view's memory but not offered as current.
 */
export function durableReadFallback<T>(_input: RequestInfo | URL, _method = 'GET'): T | undefined {
  return undefined
}
