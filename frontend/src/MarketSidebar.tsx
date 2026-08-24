import type { DashboardPayload } from './types'

const PRIMARY_NAV = [
  ['⌂', 'Home', 'Home'],
  ['◎', 'Market Scanner', 'Market Scanner'],
  ['▣', 'Recommendations', 'Recommendations'],
  ['▤', 'Market Reports', 'Market Reports'],
  ['◉', 'Stock Intelligence', 'Stock Intelligence'],
  ['◇', 'Long-Term Picks', 'Long-Term Picks'],
  ['⇔', 'Compare', 'Compare'],
  ['★', 'Watchlist', 'Watchlist'],
] as const

const SECONDARY_NAV = [
  ['↗', 'Market Overview', 'Market Overview'],
  ['◈', 'News & Events', 'News & Events'],
  ['✎', 'Education', 'Education'],
  ['▤', 'Research Data', 'Research Data'],
  ['🧪', 'Backtest', 'Backtest'],
  ['⬡', 'F&O Desk', 'F&O Desk'],
  ['▣', 'My Holdings', 'Paper Portfolio'],
  ['◌', 'System Health', 'System Health'],
] as const

const ROUTE_ALIAS: Record<string, string> = {
  'Command Center': 'Home',
  Scanner: 'Market Scanner',
  'Long-Term': 'Long-Term Picks',
  Portfolio: 'Paper Portfolio',
  'Market Internals': 'Market Overview',
  Automation: 'System Health',
  Today: 'Home',
  Setups: 'Market Scanner',
  Desk: 'System Health',
}

function ArcReactor() {
  return <div className="hud-arc" aria-hidden="true" />
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
          <span className="hud-ico" aria-hidden="true">{icon}</span>
          {display}
        </button>
      ))}
    </>
  )
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
    <aside className="sidebar hud-sidebar">
      <div className="hud-brand">
        <ArcReactor />
        <div className="hud-brand-copy">
          <strong>QUANTTERM</strong>
          <small>JARVIS DESK</small>
        </div>
      </div>
      <nav aria-label="Primary navigation">
        <NavigationGroup label="DISCOVERY" rows={PRIMARY_NAV} active={current} setActive={setActive} />
        <NavigationGroup label="TOOLS & EVIDENCE" rows={SECONDARY_NAV} active={current} setActive={setActive} />
      </nav>
      <div className="sidebar-spacer" />
      <div className="hud-telemetry broker-card">
        <div className="broker-row">
          <strong>MARKET DATA</strong>
          <span className={dashboard.data.ready ? 'status-dot' : 'status-dot status-dot-off'} />
        </div>
        <small>
          {dashboard.data.ready
            ? `READY · ${dashboard.data.bhavcopy.latest_date || '—'}`
            : 'INCOMPLETE'}
        </small>
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
      <div className="hud-telemetry broker-card compact-service-card">
        <div className="broker-row">
          <strong>SCAN ENGINE</strong>
          <span className={operations ? 'status-dot' : 'status-dot status-dot-off'} />
        </div>
        <small>
          {operations ? 'ONLINE' : 'OFFLINE'} · last scan{' '}
          {dashboard.scan.scanned_at
            ? new Date(dashboard.scan.scanned_at).toLocaleDateString('en-IN')
            : '—'}
        </small>
      </div>
    </aside>
  )
}
