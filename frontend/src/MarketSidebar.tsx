import type { DashboardPayload } from './types'
import { money } from './format'

const NAV_ITEMS = [
  ['⌘', 'Command Center', 'Command Center'],
  ['◉', 'Scanner', 'Scanner'],
  ['◎', 'Stock Intelligence', 'Stock Intelligence'],
  ['◇', 'Long-Term Research', 'Long-Term'],
  ['◈', 'News & Events', 'News & Events'],
  ['▤', 'Research Data', 'Research Data'],
  ['↗', 'Market & Breadth', 'Market Internals'],
  ['ƒ', 'F&O Coverage', 'F&O Desk'],
  ['◌', 'System Health', 'Automation'],
  ['▣', 'Paper Portfolio', 'Portfolio'],
] as const

function Logo() {
  return <div className="brand-mark" aria-hidden="true"><span /><span /><span /></div>
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
  const dataReady = dashboard.data.ready
  return (
    <aside className="sidebar">
      <div className="brand"><Logo /><div><strong>QUANTTERM</strong><small>RETAIL QUANT RESEARCH</small></div></div>
      <nav>
        {NAV_ITEMS.map(([icon, label, route]) => (
          <button
            key={route}
            className={active === route ? 'nav-item active' : 'nav-item'}
            type="button"
            onClick={() => setActive(route)}
          >
            <span>{icon}</span>{label}
          </button>
        ))}
      </nav>
      <div className="sidebar-spacer" />
      <div className="broker-card">
        <div className="broker-row">
          <strong>RESEARCH ENGINE</strong>
          <span className={operations ? 'status-dot' : 'status-dot status-dot-off'} />
        </div>
        <small>{operations ? `ONLINE · PID ${dashboard.operations.worker_pid || '—'}` : 'OFFLINE · direct scans unavailable'}</small>
        <div className="broker-stats">
          <div><span>History</span><strong>{dashboard.data.bhavcopy.ready ? `${dashboard.data.bhavcopy.sessions}d` : 'MISSING'}</strong></div>
          <div><span>Scanner rows</span><strong>{dashboard.data.scan_records || 0}</strong></div>
        </div>
      </div>
      <div className="broker-card">
        <div className="broker-row">
          <strong>DATA QUALITY</strong>
          <span className={dataReady ? 'status-dot' : 'status-dot status-dot-off'} />
        </div>
        <small>{dataReady ? `READY · ${dashboard.data.bhavcopy.latest_date || 'dated source'}` : `${dashboard.data.blockers.length} blocker(s)`}</small>
        <div className="broker-stats">
          <div><span>Long-term</span><strong>{dashboard.data.long_term_records || 0}</strong></div>
          <div><span>Paper equity</span><strong>{money(dashboard.paper.equity)}</strong></div>
        </div>
      </div>
    </aside>
  )
}
