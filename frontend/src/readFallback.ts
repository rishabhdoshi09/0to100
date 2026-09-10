import { recallMemory } from './sessionMemory'

/**
 * Return last-known durable read data only for GET surfaces where stale
 * visibility is safer than a blank page. This never services writes and never
 * upgrades stale data into current decision authority.
 */
export function durableReadFallback<T>(input: RequestInfo | URL, method = 'GET'): T | undefined {
  if (String(method || 'GET').toUpperCase() !== 'GET') return undefined
  const url = String(input)
  if (url.includes('/api/recommendations-workspace')) {
    return recallMemory<T>('reco-workspace')
  }
  return undefined
}
