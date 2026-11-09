import type { BrokerReadiness } from './types'

export type StatusTone = 'live-pill' | 'work-pill' | 'offline-pill'

export type BrokerIndicator = {
  label: string
  tone: StatusTone
  title: string
}

/** Unknown or optional Kite capability is not an NSE desk outage.
 * Only an explicitly unavailable provider gets an offline/error badge.
 * The paper-only system still has separate operational and data readiness.
 */
export function brokerIndicator(broker: BrokerReadiness | null | undefined): BrokerIndicator {
  const state = String(broker?.state || 'UNKNOWN').toUpperCase()
  const title = broker?.detail || 'Zerodha live-data readiness has not been reported yet.'

  if (broker?.live_data_ready === true) {
    return { label: 'ZERODHA OK', tone: 'live-pill', title }
  }
  if (state === 'UNAVAILABLE') {
    return { label: 'ZERODHA OFFLINE', tone: 'offline-pill', title }
  }
  if (state === 'LOGIN_REQUIRED') {
    return { label: 'ZERODHA LOGIN', tone: 'work-pill', title }
  }
  if (state === 'SNAPSHOT_REQUIRED') {
    return { label: 'ZERODHA DATA', tone: 'work-pill', title }
  }
  if (state === 'CONFIG_REQUIRED') {
    return { label: 'ZERODHA CONFIG', tone: 'work-pill', title }
  }
  return { label: 'ZERODHA CHECK', tone: 'work-pill', title }
}

/** A single slow health read is not proof that the market API is down.
 * The independent host watchdog still owns actual process recovery.
 */
export function apiUnresponsiveAfter(consecutiveFailures: number): boolean {
  return consecutiveFailures >= 2
}
