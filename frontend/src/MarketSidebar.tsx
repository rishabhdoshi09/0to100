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

function dataCopy(dashboard: DashboardPayload): string {
  if (dashboard.data.ready) {
    return `READY · ${dashboard.data.bhavcopy.latest_date || '—'}`
  }
  const busy = dashboard.operations.running || (dashboard.operations.active || []).length > 0
  return busy ? 'Preparing official history…' : 'Starting official prices…'
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
  const operations = dashboard.operations.running
  const current = ROUTE_ALIAS[active] || active
  const advancedActive = ADVANCED_NAV.some(([, route]) => route === current)
  const runtimeState = operatorState(dashboard)
  const scanOperation = (dashboard.operations.active || []).find((row) => row.kind === 'MARKET_SCAN')
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
            Scan {scanTotal > 0 ? `${scanCurrent.toLocaleString('en-IN')}/${scanTotal.toLocaleString('en-IN')}` : scanOperation.stage || scanOperation.status}
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
          <strong>MARKET SCAN</strong>
          <span className={operations ? 'status-dot' : 'status-dot status-dot-off'} />
        </div>
        <small>
          {scanOperation ? `RUNNING · ${scanTotal > 0 ? `${scanCurrent}/${scanTotal}` : scanOperation.stage}` : 'IDLE'} · last saved{' '}
          {dashboard.scan.scanned_at
            ? new Date(dashboard.scan.scanned_at).toLocaleDateString('en-IN')
            : 'none'}
        </small>
      </div>
    </aside>
  )
}
