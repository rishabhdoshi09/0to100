import { useState } from 'react'

import { sendControl } from './api'
import type { ControlName, DashboardPayload, OperationRecord } from './types'

export type OperatorState = 'RUNNING' | 'REFRESHING' | 'ATTENTION'

export const OPERATOR_ACTIONS: Array<{
  control: ControlName
  label: string
  shortLabel: string
  detail: string
  operationKind?: string
}> = [
  {
    control: 'REFRESH_DATA_NOW',
    label: 'Refresh market data',
    shortLabel: 'Refresh data',
    detail: 'Prepare the latest official NSE history used by QuantTerm.',
    operationKind: 'DATA_PREPARE',
  },
  {
    control: 'RUN_SCAN_NOW',
    label: 'Run market scan',
    shortLabel: 'Run scan',
    detail: 'Run the canonical whole-market scan now.',
    operationKind: 'MARKET_SCAN',
  },
  {
    control: 'REFRESH_NEWS_NOW',
    label: 'Refresh news',
    shortLabel: 'Refresh news',
    detail: 'Refresh the dated market-news evidence lane.',
    operationKind: 'NEWS_REFRESH',
  },
  {
    control: 'REFRESH_LONG_TERM_NOW',
    label: 'Refresh long-term analysis',
    shortLabel: 'Long-term',
    detail: 'Refresh the long-term overlay without creating a second scanner.',
    operationKind: 'LONG_TERM_REFRESH',
  },
  {
    control: 'REFRESH_FNO_NOW',
    label: 'Refresh F&O map',
    shortLabel: 'Refresh F&O',
    detail: 'Refresh the current futures-and-options universe mapping.',
    operationKind: 'FNO_REFRESH',
  },
  {
    control: 'REFRESH_MARKET_REPORT_NOW',
    label: 'Refresh market report',
    shortLabel: 'Market report',
    detail: 'Rebuild the market report from current stored evidence.',
    operationKind: 'MARKET_REPORT',
  },
  {
    control: 'RUN_CYCLE_NOW',
    label: 'Run paper cycle',
    shortLabel: 'Paper cycle',
    detail: 'Request one paper/autonomy cycle. This does not unlock live money.',
  },
]

const ACTIVE_STATUSES = new Set(['PENDING', 'RUNNING'])
const BAD_STATUSES = new Set(['FAILED', 'BLOCKED', 'CANCELLED'])

export function activeOperation(dashboard: DashboardPayload, kind: string): OperationRecord | undefined {
  return (dashboard.operations.active || []).find((row) => (
    row.kind === kind && ACTIVE_STATUSES.has(String(row.status || '').toUpperCase())
  ))
}

export function operatorState(dashboard: DashboardPayload): OperatorState {
  const runtimeOnline = Boolean(
    dashboard.operations.running
    || dashboard.autonomy.running
    || dashboard.autonomy.process_running,
  )
  if (!runtimeOnline) return 'ATTENTION'

  const failed = Object.values(dashboard.operations.latest || {}).some((row) => (
    BAD_STATUSES.has(String(row?.status || '').toUpperCase())
  ))
  if (failed && !(dashboard.operations.active || []).length) return 'ATTENTION'

  if ((dashboard.operations.active || []).length > 0) return 'REFRESHING'
  return 'RUNNING'
}

export function operatorActionDisabled(
  dashboard: DashboardPayload,
  action: (typeof OPERATOR_ACTIONS)[number],
): boolean {
  if (!action.operationKind) return false
  return Boolean(activeOperation(dashboard, action.operationKind))
}

export function OperatorQuickControls({
  dashboard,
  openControlCenter,
}: {
  dashboard: DashboardPayload
  openControlCenter: () => void
}) {
  const [message, setMessage] = useState('')
  const [busyControl, setBusyControl] = useState<ControlName | ''>('')

  const request = async (control: ControlName) => {
    setBusyControl(control)
    setMessage('Sending request…')
    try {
      const result = await sendControl(control)
      if (!result.accepted) {
        setMessage('Request was not accepted')
      } else if (result.operation_id) {
        setMessage(`Queued · ${result.operation_status || 'PENDING'}`)
      } else {
        setMessage('Queued successfully')
      }
    } catch (reason) {
      setMessage(reason instanceof Error ? reason.message : 'Control request failed')
    } finally {
      setBusyControl('')
    }
  }

  return (
    <details className="nav-advanced operator-quick-controls">
      <summary>Manual controls</summary>
      <div className="operator-quick-control-buttons">
        {OPERATOR_ACTIONS.map((action) => {
          const operationActive = operatorActionDisabled(dashboard, action)
          const busy = busyControl === action.control
          return (
            <button
              key={action.control}
              type="button"
              title={action.detail}
              disabled={operationActive || busy}
              onClick={() => void request(action.control)}
            >
              {operationActive ? `${action.shortLabel} · running` : (busy ? 'Sending…' : action.shortLabel)}
            </button>
          )
        })}
        <button type="button" onClick={openControlCenter}>Open Control Center</button>
      </div>
      {message ? <small>{message}</small> : null}
      <small>Safe data/research/PAPER controls only. Live-money controls are not exposed.</small>
    </details>
  )
}
