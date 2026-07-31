import type { DashboardPayload } from './types'
import { money } from './format'

const NAV_ITEMS = [
  ['⌘', 'Command Center'],
  ['◉', 'Scanner'],
  ['◎', 'Stock Intelligence'],
  ['▣', 'Portfolio'],
  ['↗', 'Market Internals'],
  ['◇', 'Long-Term'],
  ['◈', 'News & Events'],
  ['ƒ', 'F&O Desk'],
  ['◌', 'Automation'],
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
  const autonomy = dashboard.autonomy.running
  const operations = dashboard.operations.running
  return (
    <aside className="sidebar">
      <div className="brand"><Logo /><div><strong>QUANTTERM</strong><small>PROFESSIONAL</small></div></div>
      <nav>
        {NAV_ITEMS.map(([icon, label]) => (
          <button
            key={label}
            className={active === label ? 'nav-item active' : 'nav-item'}
            type="button"
            onClick={() => setActive(label)}
          >
            <span>{icon}</span>{label}
          </button>
        ))}
      </nav>
      <div className="sidebar-spacer" />
      <div className="broker-card">
        <div className="broker-row">
          <strong>MARKET OPS</strong>
          <span className={operations ? 'status-dot' : 'status-dot status-dot-off'} />
        </div>
        <small>{operations ? `ONLINE · PID ${dashboard.operations.worker_pid || '—'}` : 'OFFLINE · scans unavailable'}</small>
        <div className="broker-stats">
          <div><span>Active work</span><strong>{dashboard.operations.active.length}</strong></div>
          <div><span>F&O mapped</span><strong>{dashboard.fno.mapped_underlyings || 0}</strong></div>
        </div>
      </div>
      <div className="broker-card">
        <div className="broker-row">
          <strong>AUTONOMY</strong>
          <span className={autonomy ? 'status-dot' : 'status-dot status-dot-off'} />
        </div>
        <small>{dashboard.autonomy.state || 'UNKNOWN'} · PID {dashboard.autonomy.scheduler_owner_pid || '—'}</small>
        <div className="broker-stats">
          <div><span>Paper equity</span><strong>{money(dashboard.paper.equity)}</strong></div>
          <div><span>New entries</span><strong>{dashboard.autonomy.new_paper_entries ? 'ALLOWED' : 'BLOCKED'}</strong></div>
        </div>
      </div>
    </aside>
  )
}
