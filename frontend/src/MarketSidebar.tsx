import type { DashboardPayload } from './types'

const MARKET_NAV = [
  ['⌂', 'Home', 'Market'],
  ['◎', 'Market Scanner', 'Scanner'],
  ['▣', 'Recommendations', 'Recommendations'],
  ['▤', 'Market Reports', 'Reports'],
  ['▣', 'Paper Portfolio', 'Portfolio'],
] as const

const INTELLIGENCE_NAV = [
  ['◉', 'Stock Intelligence', 'Company Intelligence'],
  ['⇔', 'Compare', 'Compare'],
  ['★', 'Watchlist', 'Watchlist'],
] as const

const RESEARCH_NAV = [
  ['⌬', 'Strategies', 'Strategies'],
  ['🧪', 'Backtest', 'Backtests'],
  ['✎', 'Learning', 'Learning'],
] as const

const SYSTEM_NAV = [
  ['▤', 'Research Data', 'Data'],
  ['◎', 'Coverage', 'Coverage'],
  ['◌', 'System Health', 'Health'],
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
  Desk: 'System Health',
  'Stock Investigator': 'Stock Intelligence',
  'Company Intelligence': 'Stock Intelligence',
  Backtests: 'Backtest',
  Health: 'System Health',
  Data: 'Research Data',
}

function NavigationGroup({
  label,
  rows,
  active,
  setActive,
}: {
  label: string
  rows: ReadonlyArray<readonly [string, string, string]>
  active: string
  setActive: (value: string) => void
}) {
  return (
    <>
      <div className="nav-section-label">{label}</div>
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
  return (
    <aside className="sidebar reco-sidebar">
      <div className="reco-brand">
        <div className="reco-mark" aria-hidden="true">QT</div>
        <div className="reco-brand-copy">
          <strong>QUANTTERM</strong>
          <small>MARKETS + INTELLIGENCE</small>
        </div>
      </div>
      <nav aria-label="Primary navigation">
        <NavigationGroup label="MARKETS" rows={MARKET_NAV} active={current} setActive={setActive} />
        <NavigationGroup label="INTELLIGENCE" rows={INTELLIGENCE_NAV} active={current} setActive={setActive} />
        <NavigationGroup label="RESEARCH" rows={RESEARCH_NAV} active={current} setActive={setActive} />
        <NavigationGroup label="SYSTEM" rows={SYSTEM_NAV} active={current} setActive={setActive} />
      </nav>
      <div className="sidebar-spacer" />
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
          <strong>SCAN ENGINE</strong>
          <span className={operations ? 'status-dot' : 'status-dot status-dot-off'} />
        </div>
        <small>
          {operations ? 'WORKING' : 'READY'} · last scan{' '}
          {dashboard.scan.scanned_at
            ? new Date(dashboard.scan.scanned_at).toLocaleDateString('en-IN')
            : 'queued'}
        </small>
      </div>
    </aside>
  )
}
