import './truthfulStatus.css'
import type { DashboardPayload, OperationRecord } from './types'

export type TruthStatus =
  | 'FRESH'
  | 'STALE'
  | 'REFRESHING'
  | 'FAILED'
  | 'PARTIAL'
  | 'MISSING'
  | 'RUNNING'
  | 'READY'
  | 'DEGRADED'
  | 'OFFLINE'

export type TruthLane = {
  id: string
  label: string
  status: TruthStatus
  asOf: string
  source: string
  detail: string
}

function operationMatches(op: OperationRecord, kinds: string[]): boolean {
  return kinds.includes(String(op.kind || '').toUpperCase())
}

function activeOperation(dashboard: DashboardPayload, kinds: string[]): OperationRecord | undefined {
  return (dashboard.operations.active || []).find((op) => operationMatches(op, kinds))
}

function latestOperation(dashboard: DashboardPayload, kinds: string[]): OperationRecord | undefined {
  const rows = kinds
    .map((kind) => dashboard.operations.latest?.[kind])
    .filter(Boolean) as OperationRecord[]
  return rows.sort((a, b) => Number(b.updated_at || 0) - Number(a.updated_at || 0))[0]
}

function operationFailed(op?: OperationRecord): boolean {
  const status = String(op?.status || '').toUpperCase()
  return status === 'FAILED' || status === 'RETRYABLE_FAILED' || status === 'BLOCKED'
}

function sourceDate(value?: string | null): string {
  return String(value || '').trim()
}

export function deriveTruthLanes(dashboard: DashboardPayload): TruthLane[] {
  const dataPrep = activeOperation(dashboard, ['DATA_PREPARE'])
  const marketScan = activeOperation(dashboard, ['MARKET_SCAN'])
  const latestMarketScan = latestOperation(dashboard, ['MARKET_SCAN'])
  const fundsRefresh = activeOperation(dashboard, ['LONG_TERM_REFRESH', 'LONG_TERM_SCAN'])
  const latestFundsRefresh = latestOperation(dashboard, ['LONG_TERM_REFRESH', 'LONG_TERM_SCAN'])

  const historyCurrent = dashboard.data.bhavcopy.current
  const historyError = dashboard.data.bhavcopy.error
  let historyStatus: TruthStatus = 'MISSING'
  if (dataPrep) historyStatus = 'REFRESHING'
  else if (historyError) historyStatus = dashboard.data.bhavcopy.ready ? 'STALE' : 'FAILED'
  else if (dashboard.data.bhavcopy.ready && historyCurrent === false) historyStatus = 'STALE'
  else if (dashboard.data.bhavcopy.ready) historyStatus = 'FRESH'

  let scanStatus: TruthStatus = 'MISSING'
  if (marketScan) scanStatus = 'REFRESHING'
  else if (dashboard.scan.scanned_at && operationFailed(latestMarketScan)) scanStatus = 'STALE'
  else if (dashboard.scan.scanned_at) scanStatus = historyCurrent === false ? 'STALE' : 'FRESH'
  else if (operationFailed(latestMarketScan)) scanStatus = 'FAILED'

  let longTermStatus: TruthStatus = 'MISSING'
  if (fundsRefresh) longTermStatus = 'REFRESHING'
  else if (dashboard.long_term.scanned_at && operationFailed(latestFundsRefresh)) longTermStatus = 'STALE'
  else if (dashboard.long_term.scanned_at) longTermStatus = 'FRESH'
  else if (operationFailed(latestFundsRefresh)) longTermStatus = 'FAILED'

  const activeFailures = dashboard.autonomy.active_failures || []
  let autonomyStatus: TruthStatus = 'OFFLINE'
  if (activeFailures.length > 0) autonomyStatus = 'DEGRADED'
  else if (dashboard.autonomy.process_running || dashboard.autonomy.running) autonomyStatus = 'RUNNING'
  else if (dashboard.autonomy.available) autonomyStatus = 'READY'

  const activeJobs = (dashboard.operations.active || []).length
  const recentAutonomyFailures = (dashboard.autonomy.jobs_recent || []).filter((job) => {
    const status = String(job.status || '').toUpperCase()
    return status === 'FAILED' || status === 'RETRYABLE_FAILED' || status === 'BLOCKED'
  }).length
  const jobsStatus: TruthStatus = activeJobs > 0
    ? 'RUNNING'
    : recentAutonomyFailures > 0
      ? 'DEGRADED'
      : 'READY'

  return [
    {
      id: 'prices',
      label: 'Official prices',
      status: historyStatus,
      asOf: sourceDate(dashboard.data.bhavcopy.available_session || dashboard.data.bhavcopy.latest_date),
      source: dashboard.data.bhavcopy.source || 'NSE bhavcopy',
      detail: dataPrep?.message
        || dashboard.data.bhavcopy.error
        || (historyCurrent === false
          ? `Expected ${dashboard.data.bhavcopy.expected_latest_completed_session || 'latest completed NSE session'}`
          : `${dashboard.data.bhavcopy.sessions || 0} official sessions loaded`),
    },
    {
      id: 'scan',
      label: 'Market scan',
      status: scanStatus,
      asOf: sourceDate(dashboard.scan.scanned_at),
      source: 'Persisted scan artifact',
      detail: marketScan?.message
        || (operationFailed(latestMarketScan) ? `Last refresh failed: ${latestMarketScan?.error_message || latestMarketScan?.message || 'unknown error'}` : '')
        || `${dashboard.scan.records.length || dashboard.data.scan_records || 0} saved rows`,
    },
    {
      id: 'fundamentals',
      label: 'Long-term / funds',
      status: longTermStatus,
      asOf: sourceDate(dashboard.long_term.scanned_at),
      source: dashboard.long_term.fundamentals_source || 'Persisted fundamentals overlay',
      detail: fundsRefresh?.message
        || (operationFailed(latestFundsRefresh) ? `Last refresh failed: ${latestFundsRefresh?.error_message || latestFundsRefresh?.message || 'unknown error'}` : '')
        || `${dashboard.long_term.records.length || dashboard.data.long_term_records || 0} saved rows`,
    },
    {
      id: 'autonomy',
      label: 'Autonomy supervisor',
      status: autonomyStatus,
      asOf: sourceDate(dashboard.autonomy.heartbeat_ist),
      source: 'SQLite JobStore + supervisor heartbeat',
      detail: activeFailures[0]
        || dashboard.autonomy.plain_state
        || dashboard.autonomy.explanation
        || 'No supervisor detail reported',
    },
    {
      id: 'jobs',
      label: 'Background jobs',
      status: jobsStatus,
      asOf: sourceDate(dashboard.operations.heartbeat),
      source: 'Durable market operations + autonomy jobs',
      detail: activeJobs > 0
        ? `${activeJobs} active · ${(dashboard.operations.active || []).map((op) => `${op.kind}:${op.stage || op.status}`).slice(0, 2).join(' · ')}`
        : recentAutonomyFailures > 0
          ? `${recentAutonomyFailures} recent failed/retryable job${recentAutonomyFailures === 1 ? '' : 's'}`
          : 'No active work',
    },
  ]
}

export function formatTruthTime(value: string): string {
  if (!value) return 'not recorded'
  const n = Number(value)
  const date = Number.isFinite(n) && n > 0
    ? new Date(n > 10_000_000_000 ? n : n * 1000)
    : new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString('en-IN', {
    timeZone: 'Asia/Kolkata',
    day: '2-digit',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  }) + ' IST'
}

export function TruthBadge({ status }: { status: TruthStatus | string }) {
  const clean = String(status || 'MISSING').toUpperCase().replace(/[^A-Z_]/g, '_')
  return <span className={`truth-badge truth-${clean.toLowerCase()}`}>{clean.replace(/_/g, ' ')}</span>
}

function JobsPeek({ dashboard }: { dashboard: DashboardPayload }) {
  const market = (dashboard.operations.active || []).slice(0, 4)
  const autonomy = (dashboard.autonomy.jobs_recent || []).slice(0, 5)
  if (market.length === 0 && autonomy.length === 0) {
    return <p className="truth-empty">No durable job records surfaced yet.</p>
  }
  return (
    <div className="truth-job-list">
      {market.map((job) => (
        <div className="truth-job" key={`op-${job.operation_id}`}>
          <div><strong>{job.kind}</strong><TruthBadge status={job.status} /></div>
          <small>{job.stage || '—'} · attempt {job.attempt || 1}{job.message ? ` · ${job.message}` : ''}</small>
          {job.error_message ? <em>{job.error_message}</em> : null}
        </div>
      ))}
      {autonomy.map((job) => (
        <div className="truth-job" key={`job-${job.job_id}`}>
          <div><strong>{job.job_type}</strong><TruthBadge status={job.status} /></div>
          <small>attempt {job.attempt || 1}{job.result_summary ? ` · ${job.result_summary}` : ''}</small>
          {job.error_message ? <em>{job.error_message}</em> : null}
        </div>
      ))}
    </div>
  )
}

export function SystemTruthMonitor({
  dashboard,
  onOpenHealth,
}: {
  dashboard: DashboardPayload
  onOpenHealth?: () => void
}) {
  const lanes = deriveTruthLanes(dashboard)
  const unhealthy = lanes.filter((lane) => ['FAILED', 'STALE', 'DEGRADED', 'OFFLINE'].includes(lane.status)).length
  const working = lanes.filter((lane) => ['RUNNING', 'REFRESHING'].includes(lane.status)).length
  return (
    <section className="truth-monitor" aria-label="Persisted system truth">
      <header>
        <div>
          <span className="truth-kicker">SYSTEM TRUTH</span>
          <strong>{working > 0 ? `${working} working` : unhealthy > 0 ? `${unhealthy} need attention` : 'Persisted state ready'}</strong>
        </div>
        {onOpenHealth ? <button type="button" onClick={onOpenHealth}>Details</button> : null}
      </header>
      <div className="truth-lanes">
        {lanes.map((lane) => (
          <div className="truth-lane" key={lane.id} title={`${lane.source} · ${lane.detail}`}>
            <div><span>{lane.label}</span><TruthBadge status={lane.status} /></div>
            <small>{lane.asOf ? formatTruthTime(lane.asOf) : lane.detail}</small>
          </div>
        ))}
      </div>
      <details className="truth-details">
        <summary>Sources & jobs</summary>
        <div className="truth-source-list">
          {lanes.slice(0, 4).map((lane) => (
            <div key={`src-${lane.id}`}>
              <span>{lane.label}</span>
              <strong>{lane.source}</strong>
              <small>{lane.detail}</small>
            </div>
          ))}
        </div>
        <JobsPeek dashboard={dashboard} />
      </details>
    </section>
  )
}
