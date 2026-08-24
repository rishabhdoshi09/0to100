import type { DashboardPayload } from './types'
import { money } from './format'

const DESK_NAV = [
  ['⌂', 'Today', 'Home'],
  ['↗', 'Setups', 'Recommendations'],
  ['⚡', 'Paper Desk', 'Momentum'],
  ['🧪', 'Backtest', 'Backtest'],
  ['$', 'Portfolio', 'Wealth Builders'],
  ['☰', 'Desk', 'Market Reports'],
] as const

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
  const current = (
    {
      Home: 'Today',
      'Command Center': 'Today',
      'Market Scanner': 'Setups',
      Scanner: 'Setups',
      'Long-Term Picks': 'Setups',
      'Long-Term': 'Setups',
      'Paper Portfolio': 'Paper Desk',
      Automation: 'Desk',
      'System Health': 'Desk',
      'Market Overview': 'Desk',
      'Market Internals': 'Desk',
      'News & Events': 'Desk',
      'Research Data': 'Desk',
      'Stock Intelligence': 'Desk',
      Compare: 'Desk',
      Watchlist: 'Desk',
    } as Record<string, string>
  )[active] || active

  return (
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark" aria-hidden="true">R</div><div><strong>Reco Wealth</strong><small>Recommendations</small></div></div>
      <nav>
        <div className="nav-section-label">Menu</div>
        {DESK_NAV.map(([icon, route, display]) => (
          <button
            key={route}
            className={current === route ? 'nav-item active' : 'nav-item'}
            type="button"
            onClick={() => setActive(route)}
          >
            <span>{icon}</span>{display}
          </button>
        ))}
      </nav>
      <div className="sidebar-spacer" />
      <div className="broker-card">
        <div className="broker-row">
          <strong>MARKET DATA</strong>
          <span className={dashboard.data.ready ? 'status-dot' : 'status-dot status-dot-off'} />
        </div>
        <small>{dashboard.data.ready ? `READY · ${dashboard.data.bhavcopy.latest_date || '—'}` : 'INCOMPLETE'}</small>
        <div className="broker-stats">
          <div><span>Paper equity</span><strong>{money(dashboard.paper.equity)}</strong></div>
          <div><span>Universe</span><strong>{dashboard.scan.universe_size.toLocaleString('en-IN')}</strong></div>
        </div>
      </div>
      <div className="broker-card compact-service-card">
        <div className="broker-row"><strong>AUTONOMY</strong><span className={operations || dashboard.autonomy.running ? 'status-dot' : 'status-dot status-dot-off'} /></div>
        <small>{dashboard.autonomy.running ? 'ONLINE' : operations ? 'OPS ONLINE' : 'OFFLINE'} · live locked</small>
      </div>
    </aside>
  )
}
