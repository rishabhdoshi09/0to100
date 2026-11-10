import { describe, expect, it } from 'vitest'
import { apiUnresponsiveAfter, brokerIndicator } from './deskStatusIndicators'
import type { BrokerReadiness } from './types'

function broker(state: string, live_data_ready = false): BrokerReadiness {
  return {
    state, live_data_ready, ready: live_data_ready, execution_ready: false,
    auth_ready: false, login_required: false, auth_status: '', reason_code: '',
    detail: 'Source-specific readiness message', snapshot_id: '',
  }
}

describe('desk status indicators', () => {
  it('does not present unknown broker readiness as red/offline', () => {
    expect(brokerIndicator(undefined)).toMatchObject({ label: 'ZERODHA CHECK', tone: 'work-pill' })
    expect(brokerIndicator(broker('UNKNOWN'))).toMatchObject({ label: 'ZERODHA CHECK', tone: 'work-pill' })
    expect(brokerIndicator(broker('NOT_READY'))).toMatchObject({ label: 'ZERODHA CHECK', tone: 'work-pill' })
  })

  it('separates optional broker login, snapshot, config and real unavailability', () => {
    expect(brokerIndicator(broker('LOGIN_REQUIRED'))).toMatchObject({ label: 'ZERODHA LOGIN', tone: 'work-pill' })
    expect(brokerIndicator(broker('SNAPSHOT_REQUIRED'))).toMatchObject({ label: 'ZERODHA DATA', tone: 'work-pill' })
    expect(brokerIndicator(broker('CONFIG_REQUIRED'))).toMatchObject({ label: 'ZERODHA CONFIG', tone: 'work-pill' })
    expect(brokerIndicator(broker('UNAVAILABLE'))).toMatchObject({ label: 'ZERODHA OFFLINE', tone: 'offline-pill' })
    expect(brokerIndicator(broker('READY', true))).toMatchObject({ label: 'ZERODHA OK', tone: 'live-pill' })
  })

  it('retains the backend reason for the tooltip', () => {
    expect(brokerIndicator(broker('LOGIN_REQUIRED')).title).toBe('Source-specific readiness message')
  })

  it('keeps an isolated health timeout from being a definitive API outage', () => {
    expect(apiUnresponsiveAfter(0)).toBe(false)
    expect(apiUnresponsiveAfter(1)).toBe(false)
    expect(apiUnresponsiveAfter(2)).toBe(true)
    expect(apiUnresponsiveAfter(3)).toBe(true)
  })
})
