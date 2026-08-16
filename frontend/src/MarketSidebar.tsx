import type { DashboardPayload } from './types'
import { hubOf, routeForHub, type NavHub } from './hubs'

const FIND: Array<[string, Exclude<NavHub, ''>, string]> = [
  ['⌂', 'Home', 'Home'],
  ['◎', 'Ideas', 'Ideas'],
  ['◈', 'Context', 'Context'],
]

const BOOK: Array<[string, Exclude<NavHub, ''>, string]> = [
  ['★', 'Watchlist', 'Watchlist'],
  ['▣', 'Holdings', 'Holdings'],
]

const RUN: Array<[string, Exclude<NavHub, ''>, string]> = [
  ['◌', 'System', 'System'],
]

function Logo() {
  return <div className="brand-mark" aria-hidden="true"><span /><span /><span /></div>
}

function NavigationGroup({
  label,
  rows,
  hub,
  setActive,
}: {
  label: string
  rows: Array<[string, Exclude<NavHub, ''>, string]>
  hub: NavHub
  setActive: (value: string) => void
}) {
  return (
    <>
      <div className="nav-section-label">{label}</div>
      {rows.map(([icon, id, display]) => (
        <button
          key={id}
          className={hub === id ? 'nav-item active' : 'nav-item'}
          type="button"
          onClick={() => setActive(routeForHub(id))}
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
  const stale = Boolean(dashboard.data.bhavcopy.is_stale)
  const hub = hubOf(active)
  return (
    <aside className="sidebar">
      <div className="brand"><Logo /><div><strong>QUANTTERM</strong><small>MARKET RADAR</small></div></div>
      <nav>
        <NavigationGroup label="Find" rows={FIND} hub={hub} setActive={setActive} />
        <NavigationGroup label="Your book" rows={BOOK} hub={hub} setActive={setActive} />
        <NavigationGroup label="Keep honest" rows={RUN} hub={hub} setActive={setActive} />
      </nav>
      <div className="sidebar-spacer" />
      <div className="broker-card">
        <div className="broker-row">
          <strong>MARKET DATA</strong>
          <span className={dashboard.data.ready && !stale ? 'status-dot' : 'status-dot status-dot-off'} />
        </div>
        <small>
          {stale
            ? `STALE · ${dashboard.data.bhavcopy.latest_date || '—'} · need ${dashboard.data.bhavcopy.required_session || 'latest session'}`
            : dashboard.data.ready
              ? `READY · ${dashboard.data.bhavcopy.latest_date || '—'}`
              : 'INCOMPLETE'}
        </small>
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
