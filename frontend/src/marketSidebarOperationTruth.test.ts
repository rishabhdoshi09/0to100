import { describe, expect, it } from 'vitest'

import { scanOperationState } from './MarketSidebar'
import type { DashboardPayload, OperationRecord } from './types'

function op(status: string): OperationRecord {
  return {
    operation_id: `scan-${status}`,
    kind: 'MARKET_SCAN',
    lane: 'market_scan',
    status,
    requested_by: 'test',
    requested_at: 1,
    updated_at: 1,
    attempt: 1,
    stage: '',
    message: '',
    progress_current: 0,
    progress_total: 0,
  }
}

function dashboard({
  workerRunning = true,
  active = [],
  latest = {},
  scannedAt = '',
}: {
  workerRunning?: boolean
  active?: OperationRecord[]
  latest?: Record<string, OperationRecord>
  scannedAt?: string
} = {}): DashboardPayload {
  return {
    scan: { scanned_at: scannedAt },
    operations: { running: workerRunning, active, latest },
  } as unknown as DashboardPayload
}

describe('scan operation truth', () => {
  it('does not report WORKING merely because the operations worker is alive', () => {
    expect(scanOperationState(dashboard({ workerRunning: true }))).toEqual({
      label: 'WAITING',
      healthy: false,
    })
  })

  it('shows a durable active scan as running', () => {
    expect(scanOperationState(dashboard({ active: [op('RUNNING')] }))).toEqual({
      label: 'RUNNING',
      healthy: true,
    })
  })

  it('marks only successful latest scans healthy', () => {
    expect(scanOperationState(dashboard({ latest: { MARKET_SCAN: op('SUCCEEDED') } }))).toEqual({
      label: 'SUCCEEDED',
      healthy: true,
    })
    expect(scanOperationState(dashboard({ latest: { MARKET_SCAN: op('FAILED') } }))).toEqual({
      label: 'FAILED',
      healthy: false,
    })
    expect(scanOperationState(dashboard({ latest: { MARKET_SCAN: op('CANCELLED') } }))).toEqual({
      label: 'CANCELLED',
      healthy: false,
    })
  })

  it('uses RECORDED only when a persisted scan exists and no durable operation is available', () => {
    expect(scanOperationState(dashboard({ scannedAt: '2026-09-13T12:00:00+00:00' }))).toEqual({
      label: 'RECORDED',
      healthy: true,
    })
  })
})
