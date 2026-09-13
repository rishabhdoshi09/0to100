import './marketSidebar.css'
import type { DashboardPayload } from './types'
import { SystemTruthMonitor } from './truthfulStatus'

const PRIMARY_NAV = [
  ['⌂', 'Home', 'Desk'],
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
  ['◌', 'System Health', 'System Health'],
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

export function MarketSidebar({
  active,
  setActive,
  dashboard,
}: {
  active: string
  setActive: (value: string) => void
  dashboard: DashboardPayload
}) {
  const current = ROUTE_ALIAS[active] || active
  const advancedActive = ADVANCED_NAV.some(([, route]) => route === current)
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
        <p className="nav-primary-note">Daily use stays here. Scanner, backtests and system plumbing are secondary tools.</p>
        <details className="nav-advanced" open={advancedActive || undefined}>
          <summary>Advanced</summary>
          <NavigationRows rows={ADVANCED_NAV} active={current} setActive={setActive} />
        </details>
      </nav>
      <div className="sidebar-spacer" />
      <SystemTruthMonitor
        dashboard={dashboard}
        onOpenHealth={() => setActive('System Health')}
      />
    </aside>
  )
}
