import { useEffect, useState } from 'react'
import './marketSidebar.css'
import { fetchScanProvenance } from './api'
import { scanTruthDetail, scanTruthRows, type ScanProvenance } from './scanTruth'
import type { DashboardPayload } from './types'

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

function dataCopy(dashboard: DashboardPayload): string {
  if (dashboard.data.ready) {
    return `READY · ${dashboard.data.bhavcopy.latest_date || '—'}`
  }
  const busy = dashboard.operations.running || (dashboard.operations.active || []).length > 0
  return busy ? 'Preparing official history…' : 'Starting official prices…'
}

function scanOperationState(dashboard: DashboardPayload, provenance: ScanProvenance | null): {
  label: string
  active: boolean
  healthy: boolean
} {
  const activeScan = (dashboard.operations.active || []).find(
    operation => operation.kind === 'MARKET_SCAN' && ['PENDING', 'RUNNING'].includes(operation.status),
  )
  if (activeScan) {
    return { label: activeScan.status === 'PENDING' ? 'QUEUED' : 'RUNNING', active: true, healthy: true }
  }
  const latest = dashboard.operations.latest?.MARKET_SCAN
  if (latest?.status) {
    return {
      label: latest.status,
      active: false,
      healthy: ['SUCCEEDED', 'CANCELLED'].includes(latest.status),
    }
  }
  if (provenance?.available) return { label: 'RECORDED', active: false, healthy: true }
  return { label: 'WAITING', active: false, healthy: false }
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
  const [scanProvenance, setScanProvenance] = useState<ScanProvenance | null>(null)
  const current = ROUTE_ALIAS[active] || active
  const advancedActive = ADVANCED_NAV.some(([, route]) => route === current)

  useEffect(() => {
    let cancelled = false
    fetchScanProvenance()
      .then(payload => {
        if (!cancelled) setScanProvenance(payload)
      })
      .catch(() => {
        if (!cancelled) {
          setScanProvenance({
            available: false,
            reason: 'SCAN_PROVENANCE_API_UNAVAILABLE',
          })
        }
      })
    return () => {
      cancelled = true
    }
  }, [dashboard.scan.scanned_at])

  const scanState = scanOperationState(dashboard, scanProvenance)
  const scanRows = scanTruthRows(scanProvenance)

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
      <div className="reco-telemetry broker-card compact-service-card scan-truth-card">
        <div className="broker-row">
          <strong>AUTONOMOUS SCAN</strong>
          <span className={scanState.healthy ? 'status-dot' : 'status-dot status-dot-off'} />
        </div>
        <small>{scanState.label} · {scanTruthDetail(scanProvenance)}</small>
        <div className="scan-truth-list" aria-label="Scan provenance">
          {scanRows.map(row => (
            <div className="scan-truth-row" key={row.label}>
              <span>{row.label}</span>
              <strong>{row.value}</strong>
            </div>
          ))}
        </div>
      </div>
    </aside>
  )
}
