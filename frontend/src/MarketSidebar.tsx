import './marketSidebar.css'
import { operatorState } from './operatorControlCenter'
import {
  TOOL_GROUPS,
  WORKSPACE_NAV,
  canonicalNavRoute,
  isToolRoute,
  type NavRow,
} from './navigation'
import type { DashboardPayload } from './types'

function NavigationRows({
  rows,
  active,
  setActive,
}: {
  rows: readonly NavRow[]
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

/**
 * A scan's execution instant is not a data date. Rendering `scanned_at` with
 * toLocaleDateString() previously made a Sunday job run look like Sunday
 * prices for a market that last traded on Friday. These helpers keep the two
 * facts visually and semantically separate, and refuse to invent either one.
 */
function scanRanAt(dashboard: DashboardPayload): string {
  const stamp = dashboard.scan.provenance?.scan_completed_at || dashboard.scan.scanned_at
  if (!stamp) return 'not yet run'
  const when = new Date(stamp)
  if (Number.isNaN(when.getTime())) return 'not yet run'
  return when.toLocaleString('en-IN', {
    day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit', hour12: false,
  })
}

function marketSessionLabel(dashboard: DashboardPayload): string {
  const session = dashboard.scan.provenance?.market_session_date
  if (session) return new Date(`${session}T00:00:00`).toLocaleDateString('en-IN', {
    day: '2-digit', month: 'short',
  })
  return 'unavailable'
}

function sessionClass(dashboard: DashboardPayload): string {
  const prov = dashboard.scan.provenance
  if (!prov?.market_session_date) return 'scan-session scan-session-unknown'
  return prov.data_current ? 'scan-session scan-session-current' : 'scan-session scan-session-stale'
}

function freshnessNote(dashboard: DashboardPayload): string {
  const prov = dashboard.scan.provenance
  if (!prov) return ''
  if (!prov.market_session_date) return prov.provenance_reason || 'market session unavailable'
  if (prov.data_current) return ''
  const behind = prov.sessions_behind
  const gap = typeof behind === 'number' && behind > 0 ? `${behind} session${behind === 1 ? '' : 's'} behind` : 'stale'
  const expected = prov.expected_session_date ? ` · expected ${prov.expected_session_date}` : ''
  return `${gap}${expected}`
}

function dataCopy(dashboard: DashboardPayload): string {
  if (dashboard.data.ready) {
    return `READY · ${dashboard.data.bhavcopy.latest_date || '—'}`
  }
  const busy = dashboard.operations.running || (dashboard.operations.active || []).length > 0
  return busy ? 'Preparing official history…' : 'Starting official prices…'
}

export function scanOperationState(dashboard: DashboardPayload): {
  label: string
  healthy: boolean
} {
  const active = (dashboard.operations.active || []).find(
    operation => operation.kind === 'MARKET_SCAN' && ['PENDING', 'RUNNING'].includes(operation.status),
  )
  if (active) {
    return {
      label: active.status === 'PENDING' ? 'QUEUED' : 'RUNNING',
      healthy: true,
    }
  }

  const latest = dashboard.operations.latest?.MARKET_SCAN
  if (latest?.status) {
    return {
      label: latest.status,
      healthy: latest.status === 'SUCCEEDED',
    }
  }

  if (dashboard.scan.scanned_at) {
    return { label: 'RECORDED', healthy: true }
  }
  return { label: 'WAITING', healthy: false }
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
  const scanState = scanOperationState(dashboard)
  const current = canonicalNavRoute(active)
  const toolsActive = isToolRoute(current)
  const runtimeState = operatorState(dashboard)
  const runtimeOnline = runtimeState !== 'ATTENTION'

  return (
    <aside className="sidebar reco-sidebar">
      <div className="reco-brand">
        <div className="reco-mark" aria-hidden="true">QT</div>
        <div className="reco-brand-copy">
          <strong>QUANTTERM</strong>
          <small>MARKET INTELLIGENCE</small>
        </div>
      </div>

      <nav aria-label="Primary navigation">
        <div className="nav-section-label">WORKSPACES</div>
        <NavigationRows rows={WORKSPACE_NAV} active={current} setActive={setActive} />
        <p className="nav-primary-note">
          Five places for daily work. Focused diagnostics stay under Tools instead of becoming separate products.
        </p>

        <details className="nav-advanced" open={toolsActive || undefined}>
          <summary>Tools</summary>
          {TOOL_GROUPS.map((group) => (
            <div className="nav-tool-group" key={group.label}>
              <div className="nav-tool-group-label">{group.label}</div>
              <NavigationRows rows={group.items} active={current} setActive={setActive} />
            </div>
          ))}
        </details>
      </nav>

      <div className="sidebar-spacer" />

      <div className="sidebar-summary">
        <button type="button" className="sidebar-summary-head" onClick={() => setActive('System Health')}>
          <span>
            <strong>{runtimeState === 'RUNNING' ? 'SYSTEM READY' : runtimeState}</strong>
            <small>Open System</small>
          </span>
          <i className={runtimeOnline ? 'status-dot' : 'status-dot status-dot-off'} />
        </button>

        <dl>
          <div>
            <dt>Data</dt>
            <dd>{dataCopy(dashboard)}</dd>
          </div>
          <div>
            <dt>Scan</dt>
            <dd>{scanState.label} · {scanRanAt(dashboard)}</dd>
          </div>
          <div>
            <dt>Market session</dt>
            <dd className={sessionClass(dashboard)}>{marketSessionLabel(dashboard)}</dd>
          </div>
          <div>
            <dt>Active jobs</dt>
            <dd>{(dashboard.operations.active || []).length}</dd>
          </div>
        </dl>

        {freshnessNote(dashboard) ? (
          <small className="scan-provenance-note">{freshnessNote(dashboard)}</small>
        ) : null}
      </div>
    </aside>
  )
}
