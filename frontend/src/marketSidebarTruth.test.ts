import { describe, expect, it } from 'vitest'

import { scanOperationState } from './MarketSidebar'
import type { DashboardPayload, OperationRecord } from './types'

function dashboard(active: OperationRecord[] = [], latest: Record<string, OperationRecord> = {}, running = true) {
  return {
    operations: { active, latest, running },
  } as unknown as DashboardPayload
}

function operation(status: string): OperationRecord {
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

describe('scanner operation truth', () => {
  it('does not call the scanner running merely because the worker process is alive', () => {
    expect(scanOperationState(dashboard([], {}, true), null)).toEqual({
      label: 'WAITING',
      active: false,
      healthy: false,
    })
  })

  it('shows a durable running market-scan operation as RUNNING', () => {
    expect(scanOperationState(dashboard([operation('RUNNING')]), null)).toEqual({
      label: 'RUNNING',
      active: true,
      healthy: true,
    })
  })

  it('does not mark a cancelled scan healthy', () => {
    expect(scanOperationState(dashboard([], { MARKET_SCAN: operation('CANCELLED') }), null)).toEqual({
      label: 'CANCELLED',
      active: false,
      healthy: false,
    })
  })

  it('marks only a persisted successful latest scan healthy', () => {
    expect(scanOperationState(dashboard([], { MARKET_SCAN: operation('SUCCEEDED') }), null)).toEqual({
      label: 'SUCCEEDED',
      active: false,
      healthy: true,
    })
  })
})
