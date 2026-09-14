import './marketSidebar.css'
import { OperatorQuickControls, operatorState } from './operatorControlCenter'
import type { DashboardPayload } from './types'

const PRIMARY_NAV = [
  ['⌂', 'Home', 'Desk'],
  ['⚙', 'System Health', 'Control Center'],
  ['▣', 'Recommendations', 'Opportunities'],
  ['◉', 'Stock Intelligence', 'Stock Intelligence'],
  ['?', 'Why This Decision', 'Why This Decision'],
  ['▣', 'Paper Portfolio', 'Portfolio'],
  ['✎', 'Learning', 'Learning'],
  ['∑', 'Forward Evidence', 'Forward Evidence'],
] as const

const ADVANCED_NAV = [
  ['◎', 'Market Scanner', 'Scanner'],
  ['▤', 'Market Reports', 'Market Reports'],
  ['★', 'Watchlist', 'Watchlist'],
  ['⇔', 'Compare', 'Compare'],
  ['⌬', 'Strategies', 'Strategies'],
  ['🧪', 'Backtest', 'Backtests'],
  ['▤', 'Research Data', 'Research Data'],
  ['◎', 'Coverage', 'Coverage'],
] as const

const ROUTE_ALIAS: Record<string, string> = {
  'Command Center': 'Home',
  Market: 'Home',
  Scanner: 'Market Scanner',
  Reports: 'Market Reports',
  'Long-Term': 'Long-Term Picks',
  Portfolio: 'Paper Portfolio',
  'Market Internals': 'Market Overview',
  Automation: 'System Health',
  Today: 'Home',
  Setups: 'Market Scanner',
  Desk: 'Home',
  'Stock Investigator': 'Stock Intelligence',
  'Company Intelligence': 'Stock Intelligence',
  Backtests: 'Backtest',
  Health: 'System Health',
  Data: 'Research Data',
  Why: 'Why This Decision',
  Evidence: 'Forward Evidence',
}

function NavigationRows({
  rows,
  active,
  setActive,
}: {
  rows: ReadonlyArray<readonly [string, string, string]>
  active: string
  setActive: (value: string) => void
}) {
  return (
    <>
      {rows.map(([icon, route, display]) => (
        <button
          key={route}
          className={active === route ? 'nav-item active' : 'nav-item'}
          type="button"
          onClick={() => setActive(route)}
        >
          <span className="reco-ico" aria-hidden="true">{icon}</span>
          {display}
        </button>
      ))}
    </>
  )
}

/**
 * A scan's execution instant is not a data date. Rendering `scanned_at` with
 * toLocaleDateString() previously made a Sunday job run look like Sunday
 * prices for a market that last traded on Friday. These helpers keep the two
 * facts visually and semantically separate, and refuse to invent either one.
 */
function scanRanAt(dashboard: DashboardPayload): string {
  const stamp = dashboard.scan.provenance?.scan_completed_at || dashboard.scan.scanned_at
  if (!stamp) return 'not yet run'
  const when = new Date(stamp)
  if (Number.isNaN(when.getTime())) return 'not yet run'
  return when.toLocaleString('en-IN', {
    day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit', hour12: false,
  })
}

function marketSessionLabel(dashboard: DashboardPayload): string {
  const session = dashboard.scan.provenance?.market_session_date
  if (session) return new Date(`${session}T00:00:00`).toLocaleDateString('en-IN', {
    day: '2-digit', month: 'short',
  })
  return 'unavailable'
}

function sessionClass(dashboard: DashboardPayload): string {
  const prov = dashboard.scan.provenance
  if (!prov?.market_session_date) return 'scan-session scan-session-unknown'
  return prov.data_current ? 'scan-session scan-session-current' : 'scan-session scan-session-stale'
}

function freshnessNote(dashboard: DashboardPayload): string {
  const prov = dashboard.scan.provenance
  if (!prov) return ''
  if (!prov.market_session_date) return prov.provenance_reason || 'market session unavailable'
  if (prov.data_current) return ''
  const behind = prov.sessions_behind
  const gap = typeof behind === 'number' && behind > 0 ? `${behind} session${behind === 1 ? '' : 's'} behind` : 'stale'
  const expected = prov.expected_session_date ? ` · expected ${prov.expected_session_date}` : ''
  return `${gap}${expected}`
}

function dataCopy(dashboard: DashboardPayload): string {
  if (dashboard.data.ready) {
    return `READY · ${dashboard.data.bhavcopy.latest_date || '—'}`
  }
  const busy = dashboard.operations.running || (dashboard.operations.active || []).length > 0
  return busy ? 'Preparing official history…' : 'Starting official prices…'
}

export function scanOperationState(dashboard: DashboardPayload): {
  label: string
  healthy: boolean
} {
  const active = (dashboard.operations.active || []).find(
    operation => operation.kind === 'MARKET_SCAN' && ['PENDING', 'RUNNING'].includes(operation.status),
  )
  if (active) {
    return {
      label: active.status === 'PENDING' ? 'QUEUED' : 'RUNNING',
      healthy: true,
    }
  }

  const latest = dashboard.operations.latest?.MARKET_SCAN
  if (latest?.status) {
    return {
      label: latest.status,
      healthy: latest.status === 'SUCCEEDED',
    }
  }

  if (dashboard.scan.scanned_at) {
    return { label: 'RECORDED', healthy: true }
  }
  return { label: 'WAITING', healthy: false }
}

export function MarketSidebar({
  active,
  setActive,
  dashboard,
}: {
  active: string
  setActive: (value: string) => void
  dashboard: DashboardPayload
}) {
  const scanState = scanOperationState(dashboard)
  const current = ROUTE_ALIAS[active] || active
  const advancedActive = ADVANCED_NAV.some(([, route]) => route === current)
  const runtimeState = operatorState(dashboard)
  const scanOperation = (dashboard.operations.active || []).find((row) => (
    row.kind === 'MARKET_SCAN' && ['PENDING', 'RUNNING'].includes(String(row.status || '').toUpperCase())
  ))
  const scanCurrent = Number(scanOperation?.progress_current || dashboard.scan_progress?.current || 0)
  const scanTotal = Number(scanOperation?.progress_total || dashboard.scan_progress?.total || 0)
  const runtimeOnline = runtimeState !== 'ATTENTION'

  return (
    <aside className="sidebar reco-sidebar">
      <div className="reco-brand">
        <div className="reco-mark" aria-hidden="true">QT</div>
        <div className="reco-brand-copy">
          <strong>QUANTTERM</strong>
          <small>AUTONOMOUS MARKET INTELLIGENCE</small>
        </div>
      </div>
      <nav aria-label="Primary navigation">
        <div className="nav-section-label">OPERATE</div>
        <NavigationRows rows={PRIMARY_NAV} active={current} setActive={setActive} />
        <p className="nav-primary-note">Daily use and safe manual controls stay here. Research plumbing remains under Advanced.</p>
        <OperatorQuickControls dashboard={dashboard} openControlCenter={() => setActive('System Health')} />
        <details className="nav-advanced" open={advancedActive || undefined}>
          <summary>Advanced</summary>
          <NavigationRows rows={ADVANCED_NAV} active={current} setActive={setActive} />
        </details>
      </nav>
      <div className="sidebar-spacer" />
      <div className="reco-telemetry broker-card compact-service-card">
        <div className="broker-row">
          <strong>SYSTEM</strong>
          <span className={runtimeOnline ? 'status-dot' : 'status-dot status-dot-off'} />
        </div>
        <small>{runtimeState}</small>
        <div className="broker-stats">
          <div>
            <span>Worker</span>
            <strong>{dashboard.operations.worker_pid || '—'}</strong>
          </div>
          <div>
            <span>Jobs</span>
            <strong>{(dashboard.operations.active || []).length}</strong>
          </div>
        </div>
        {scanOperation ? (
          <small>
            Scan {scanTotal > 0 ? `${scanCurrent.toLocaleString('en-IN')}/${scanTotal.toLocaleString('en-IN')}` : (scanOperation.stage || scanOperation.status)}
          </small>
        ) : null}
        <button type="button" onClick={() => setActive('System Health')}>Open Control Center</button>
      </div>
      <div className="reco-telemetry broker-card">
        <div className="broker-row">
          <strong>MARKET DATA</strong>
          <span className={dashboard.data.ready ? 'status-dot' : 'status-dot status-dot-off'} />
        </div>
        <small>{dataCopy(dashboard)}</small>
        <div className="broker-stats">
          <div>
            <span>Sessions</span>
            <strong>{dashboard.data.bhavcopy.sessions || 0}</strong>
          </div>
          <div>
            <span>Universe</span>
            <strong>{dashboard.scan.universe_size.toLocaleString('en-IN')}</strong>
          </div>
        </div>
      </div>
      <div className="reco-telemetry broker-card compact-service-card">
        <div className="broker-row">
          <strong>AUTONOMOUS SCAN</strong>
          <span className={scanState.healthy ? 'status-dot' : 'status-dot status-dot-off'} />
        </div>
        <small>{scanState.label}</small>
        <dl className="scan-provenance">
          <div>
            <dt>Scan ran</dt>
            <dd>{scanRanAt(dashboard)}</dd>
          </div>
          <div>
            <dt>Market session</dt>
            <dd className={sessionClass(dashboard)}>{marketSessionLabel(dashboard)}</dd>
          </div>
        </dl>
        {freshnessNote(dashboard) ? (
          <small className="scan-provenance-note">{freshnessNote(dashboard)}</small>
        ) : null}
      </div>
    </aside>
  )
}
