import type { DashboardPayload } from './types'
import { money } from './format'

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
  ['⬡', 'F&O Desk', 'F&O Desk'],
  ['▣', 'My Holdings', 'Paper Portfolio'],
  ['◌', 'System Health', 'System Health'],
] as const

function Logo() {
  return <div className="brand-mark" aria-hidden="true"><span /><span /><span /></div>
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
          <span>{icon}</span>{display}
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
  return (
    <aside className="sidebar">
      <div className="brand"><Logo /><div><strong>QUANTTERM</strong><small>MARKET RADAR</small></div></div>
      <nav>
        <NavigationGroup label="DISCOVERY" rows={PRIMARY_NAV} active={active} setActive={setActive} />
        <NavigationGroup label="TOOLS & EVIDENCE" rows={SECONDARY_NAV} active={active} setActive={setActive} />
      </nav>
      <div className="sidebar-spacer" />
      <div className="broker-card">
        <div className="broker-row">
          <strong>MARKET DATA</strong>
          <span className={dashboard.data.ready ? 'status-dot' : 'status-dot status-dot-off'} />
        </div>
        <small>{dashboard.data.ready ? `READY · ${dashboard.data.bhavcopy.latest_date || '—'}` : 'INCOMPLETE'}</small>
        <div className="broker-stats">
          <div><span>Sessions</span><strong>{dashboard.data.bhavcopy.sessions || 0}</strong></div>
          <div><span>Universe</span><strong>{dashboard.scan.universe_size.toLocaleString('en-IN')}</strong></div>
        </div>
      </div>
      <div className="broker-card compact-service-card">
        <div className="broker-row"><strong>SCAN ENGINE</strong><span className={operations ? 'status-dot' : 'status-dot status-dot-off'} /></div>
        <small>{operations ? 'ONLINE' : 'OFFLINE'} · last scan {dashboard.scan.scanned_at ? new Date(dashboard.scan.scanned_at).toLocaleDateString('en-IN') : '—'}</small>
      </div>
    </aside>
  )
}
