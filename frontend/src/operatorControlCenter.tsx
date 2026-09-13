import { useEffect, useMemo, useState } from 'react'

import { Panel } from './components'
import { fetchRadarHome, type RadarHome } from './productApi'
import type { ControlName, DashboardPayload, OperationRecord } from './types'

export type OperatorState = 'RUNNING' | 'REFRESHING' | 'ATTENTION'

export const OPERATOR_ACTIONS: Array<{
  control: ControlName
  label: string
  detail: string
  operationKind?: string
}> = [
  {
    control: 'REFRESH_DATA_NOW',
    label: 'Refresh market data',
    detail: 'Prepare the latest official NSE history used by QuantTerm.',
    operationKind: 'DATA_PREPARE',
  },
  {
    control: 'RUN_SCAN_NOW',
    label: 'Run market scan',
    detail: 'Run the canonical whole-market scan now.',
    operationKind: 'MARKET_SCAN',
  },
  {
    control: 'REFRESH_NEWS_NOW',
    label: 'Refresh news',
    detail: 'Refresh the dated market-news evidence lane.',
    operationKind: 'NEWS_REFRESH',
  },
  {
    control: 'REFRESH_LONG_TERM_NOW',
    label: 'Refresh long-term analysis',
    detail: 'Refresh the long-term overlay without creating a second scanner.',
    operationKind: 'LONG_TERM_REFRESH',
  },
  {
    control: 'REFRESH_FNO_NOW',
    label: 'Refresh F&O map',
    detail: 'Refresh the current futures-and-options universe mapping.',
    operationKind: 'FNO_REFRESH',
  },
  {
    control: 'REFRESH_MARKET_REPORT_NOW',
    label: 'Refresh market report',
    detail: 'Rebuild the market report from current stored evidence.',
    operationKind: 'MARKET_REPORT',
  },
  {
    control: 'RUN_CYCLE_NOW',
    label: 'Run paper cycle',
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

export function liveLockState(home: RadarHome | null): 'LOCKED' | 'UNVERIFIED' {
  const os = home?.home_os
  if (
    os?.live_locked === true
    || os?.broker?.live_locked === true
    || os?.learning?.live_locked === true
  ) return 'LOCKED'
  return 'UNVERIFIED'
}

export function operatorActionDisabled(
  dashboard: DashboardPayload,
  action: (typeof OPERATOR_ACTIONS)[number],
): boolean {
  if (!action.operationKind) return false
  return Boolean(activeOperation(dashboard, action.operationKind))
}

function operationProgress(row?: OperationRecord): string {
  if (!row) return 'Idle'
  const current = Number(row.progress_current || 0)
  const total = Number(row.progress_total || 0)
  const progress = total > 0 ? `${current.toLocaleString('en-IN')}/${total.toLocaleString('en-IN')}` : row.stage || row.status
  return `${progress}${row.message ? ` · ${row.message}` : ''}`
}

function latestOperation(dashboard: DashboardPayload, kind: string): OperationRecord | undefined {
  return activeOperation(dashboard, kind) || dashboard.operations.latest?.[kind]
}

function statusClass(ok: boolean): string {
  return ok ? 'positive' : 'negative'
}

export function OperatorControlCenter({
  dashboard,
  runControl,
  onRefresh,
}: {
  dashboard: DashboardPayload
  runControl: (control: ControlName) => Promise<void>
  onRefresh: () => Promise<void> | void
}) {
  const [home, setHome] = useState<RadarHome | null>(null)
  const [homeError, setHomeError] = useState('')

  const loadHome = async () => {
    try {
      const payload = await fetchRadarHome()
      setHome(payload)
      setHomeError('')
    } catch (reason) {
      setHomeError(reason instanceof Error ? reason.message : 'Operator status unavailable')
    }
  }

  useEffect(() => {
    void loadHome()
    const timer = window.setInterval(() => void loadHome(), 15_000)
    return () => window.clearInterval(timer)
  }, [])

  const state = operatorState(dashboard)
  const liveLock = liveLockState(home)
  const scan = latestOperation(dashboard, 'MARKET_SCAN')
  const data = latestOperation(dashboard, 'DATA_PREPARE')
  const news = latestOperation(dashboard, 'NEWS_REFRESH')
  const longTerm = latestOperation(dashboard, 'LONG_TERM_REFRESH')
  const fno = latestOperation(dashboard, 'FNO_REFRESH')
  const report = latestOperation(dashboard, 'MARKET_REPORT')
  const operations = useMemo(() => [
    ['Market data', data],
    ['Market scan', scan],
    ['News', news],
    ['Long-term', longTerm],
    ['F&O', fno],
    ['Market report', report],
  ] as Array<[string, OperationRecord | undefined]>, [data, scan, news, longTerm, fno, report])

  const paperPaused = dashboard.autonomy.new_paper_entries === false
  const runtimeOnline = Boolean(dashboard.operations.running || dashboard.autonomy.running || dashboard.autonomy.process_running)

  return (
    <section className="workspace-view operator-control-center">
      <div className="reco-how">
        <div className="qt-eyebrow">Operator Control Center</div>
        <p>
          This is the manual control surface for QuantTerm. Automation continues by default; these buttons
          let you safely request or retry data, scan, evidence and paper workflows without using Terminal.
          Live-money execution cannot be enabled from this page.
        </p>
      </div>

      <Panel title="SYSTEM NOW" subtitle="Backend truth — not an optimistic UI badge">
        <div className="fact-grid">
          <div><span>QuantTerm</span><strong className={state === 'ATTENTION' ? 'negative' : 'positive'}>{state}</strong></div>
          <div><span>Market operations</span><strong className={statusClass(dashboard.operations.running)}>{dashboard.operations.running ? 'RUNNING' : 'OFFLINE'}</strong></div>
          <div><span>Autonomy</span><strong className={statusClass(Boolean(dashboard.autonomy.running || dashboard.autonomy.process_running))}>{dashboard.autonomy.running || dashboard.autonomy.process_running ? 'RUNNING' : 'OFFLINE'}</strong></div>
          <div><span>Official data</span><strong className={statusClass(dashboard.data.ready)}>{dashboard.data.ready ? `READY · ${dashboard.data.bhavcopy.latest_date || 'current'}` : 'NOT READY'}</strong></div>
          <div><span>Paper bot</span><strong>{dashboard.paper.supervisor_running ? (paperPaused ? 'RUNNING · ENTRIES PAUSED' : 'RUNNING') : 'NOT RUNNING'}</strong></div>
          <div><span>Zerodha data</span><strong>{dashboard.autonomy.broker?.live_data_ready ? 'READY' : (dashboard.autonomy.broker?.state || 'CHECKING')}</strong></div>
          <div><span>Live money</span><strong className={liveLock === 'LOCKED' ? 'positive' : 'negative'}>{liveLock === 'LOCKED' ? 'VERIFIED LOCKED' : 'LOCK NOT VERIFIED IN THIS SNAPSHOT'}</strong></div>
          <div><span>Worker PID</span><strong>{dashboard.operations.worker_pid || '—'}</strong></div>
        </div>
        <p className="panel-copy">
          Heartbeat: {dashboard.operations.heartbeat || 'not reported'} · Last saved scan: {dashboard.scan.scanned_at ? new Date(dashboard.scan.scanned_at).toLocaleString('en-IN') : 'none'}
        </p>
        {homeError ? <div className="api-warning">Live-lock/home status refresh: {homeError}</div> : null}
        <div className="inline-actions">
          <button type="button" onClick={() => { void onRefresh(); void loadHome() }}>Refresh status</button>
          <button
            type="button"
            onClick={() => void runControl(paperPaused ? 'RESUME_NEW_PAPER_ENTRIES' : 'PAUSE_NEW_PAPER_ENTRIES')}
          >
            {paperPaused ? 'Resume paper entries' : 'Pause new paper entries'}
          </button>
        </div>
      </Panel>

      <Panel title="SAFE MANUAL CONTROLS" subtitle="Requests use the same durable queues as automation">
        <div className="operator-action-grid">
          {OPERATOR_ACTIONS.map((action) => {
            const disabled = operatorActionDisabled(dashboard, action)
            const row = action.operationKind ? latestOperation(dashboard, action.operationKind) : undefined
            return (
              <div className="insight" key={action.control}>
                <i className={disabled ? 'cyan' : 'green'} />
                <div>
                  <strong>{action.label}</strong>
                  <span>{action.detail}</span>
                  {row ? <span>{operationProgress(row)}</span> : null}
                  <button
                    type="button"
                    disabled={disabled}
                    onClick={() => void runControl(action.control)}
                  >
                    {disabled ? 'Already running' : action.label}
                  </button>
                </div>
              </div>
            )
          })}
        </div>
      </Panel>

      <Panel title="ACTIVE / RECENT OPERATIONS" subtitle="What QuantTerm is actually doing">
        {operations.map(([label, row]) => (
          <div className="insight" key={label}>
            <i className={row && ACTIVE_STATUSES.has(String(row.status).toUpperCase()) ? 'cyan' : (row?.status === 'SUCCEEDED' ? 'green' : 'amber')} />
            <div>
              <strong>{label} · {row?.status || 'IDLE'}</strong>
              <span>{operationProgress(row)}</span>
              {row?.error_message ? <span>{row.error_code || 'ERROR'} · {row.error_message}</span> : null}
            </div>
          </div>
        ))}
      </Panel>

      <Panel title="SAFETY BOUNDARY" subtitle="Intentionally unavailable from the UI">
        <p className="panel-copy">
          There is no Unlock Live Money, Live Buy, Live Sell, broker-order, shell, process-kill or risk-bypass control here.
          Manual control is limited to safe data, research, scan and PAPER workflows.
        </p>
        <div className="fact-grid">
          <div><span>Runtime API reachable</span><strong className={statusClass(runtimeOnline)}>{runtimeOnline ? 'YES' : 'NO'}</strong></div>
          <div><span>Live-control button</span><strong>NOT EXPOSED</strong></div>
        </div>
      </Panel>
    </section>
  )
}
