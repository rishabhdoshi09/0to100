import type { DashboardPayload } from './types'
import { money } from './format'

const PRIMARY_NAV = [
  ['⌂', 'Command Center', 'Home'],
  ['◉', 'Scanner', 'Discover'],
  ['◎', 'Stock Intelligence', 'Stock Intelligence'],
  ['◇', 'Long-Term', 'Long-Term Research'],
  ['▤', 'Research Data', 'Research Data'],
] as const

const OPERATIONS_NAV = [
  ['↗', 'Market Internals', 'Market & Breadth'],
  ['◈', 'News & Events', 'News & Events'],
  ['ƒ', 'F&O Desk', 'F&O Coverage'],
  ['◌', 'Automation', 'System Health'],
  ['▣', 'Portfolio', 'Paper Portfolio'],
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
  const autonomy = dashboard.autonomy.running
  const operations = dashboard.operations.running
  return (
    <aside className="sidebar">
      <div className="brand"><Logo /><div><strong>QUANTTERM</strong><small>RETAIL QUANT OS</small></div></div>
      <nav>
        <NavigationGroup label="RESEARCH" rows={PRIMARY_NAV} active={active} setActive={setActive} />
        <NavigationGroup label="OPERATIONS & EVIDENCE" rows={OPERATIONS_NAV} active={active} setActive={setActive} />
      </nav>
      <div className="sidebar-spacer" />
      <div className="broker-card">
        <div className="broker-row">
          <strong>MARKET DATA</strong>
          <span className={dashboard.data.ready ? 'status-dot' : 'status-dot status-dot-off'} />
        </div>
        <small>{dashboard.data.ready ? `READY · ${dashboard.data.bhavcopy.latest_date || 'date unknown'}` : 'INCOMPLETE · inspect Home'}</small>
        <div className="broker-stats">
          <div><span>Sessions</span><strong>{dashboard.data.bhavcopy.sessions || 0}</strong></div>
          <div><span>Symbols</span><strong>{(dashboard.data.bhavcopy.symbols || 0).toLocaleString('en-IN')}</strong></div>
        </div>
      </div>
      <div className="broker-card compact-service-card">
        <div className="broker-row"><strong>MARKET OPS</strong><span className={operations ? 'status-dot' : 'status-dot status-dot-off'} /></div>
        <small>{operations ? `ONLINE · PID ${dashboard.operations.worker_pid || '—'}` : 'OFFLINE · scans unavailable'}</small>
      </div>
      <div className="broker-card compact-service-card secondary-service">
        <div className="broker-row"><strong>PAPER AUTONOMY</strong><span className={autonomy ? 'status-dot' : 'status-dot status-dot-off'} /></div>
        <small>{dashboard.autonomy.state || 'UNKNOWN'} · equity {money(dashboard.paper.equity)}</small>
      </div>
    </aside>
  )
}
