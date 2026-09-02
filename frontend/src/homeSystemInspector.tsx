import type { HomeAction } from './productApi'
import {
  LANE_TITLE,
  SYSTEM_LANE_ORDER,
  checkSystemRows,
  laneAriaLabel,
  lanePrimaryAction,
  laneSecondaryActions,
  laneTitle,
  liveMoneyStillLocked,
  nothingNeeded,
  technicalLines,
  type CheckSystemSnapshot,
  type SystemLane,
} from './backendControlPlane'

function Field({ label, value }: { label: string; value?: string | number | null }) {
  if (value == null || value === '') return null
  return (
    <div className="home-os-inspect-field">
      <span>{label}</span>
      <strong>{String(value)}</strong>
    </div>
  )
}

function ActionRow({
  actions,
  busy,
  onAction,
}: {
  actions: HomeAction[]
  busy: boolean
  onAction: (action: HomeAction) => void
}) {
  if (!actions.length) return null
  return (
    <div className="home-os-inspect-actions">
      {actions.map((action) => (
        <button
          key={`${action.control || action.kind}-${action.label}`}
          type="button"
          className={action === actions[0] && action.kind !== 'refresh' ? undefined : 'secondary'}
          disabled={busy}
          onClick={() => onAction(action)}
        >
          {action.label}
        </button>
      ))}
    </div>
  )
}

export function SystemLaneInspector({
  laneId,
  lane,
  depth,
  busy,
  liveLocked,
  checkSystem,
  system,
  onAction,
  onOpenPage,
  onClose,
}: {
  laneId: string
  lane?: SystemLane
  depth: string
  busy: boolean
  liveLocked: boolean
  checkSystem?: CheckSystemSnapshot
  system: Record<string, SystemLane>
  onAction: (action: HomeAction) => void
  onOpenPage?: (page: string) => void
  onClose: () => void
}) {
  if (laneId === 'check_system') {
    const rows = checkSystemRows(checkSystem, system)
    return (
      <aside className="home-os-inspector" role="region" aria-label="System check">
        <header>
          <div>
            <span>SYSTEM CHECK</span>
            <h3>Read-only snapshot</h3>
          </div>
          <button type="button" className="secondary" onClick={onClose} aria-label="Close inspector">Close</button>
        </header>
        <p>This uses the same Home state. It does not start a second health engine.</p>
        <div className="home-os-check-grid">
          {rows.map((row) => (
            <div key={row.id}>
              <span>{row.label}</span>
              <strong>{row.status}</strong>
            </div>
          ))}
        </div>
        <p className="home-os-inspect-lock">Live money: Locked. Paper only.</p>
      </aside>
    )
  }

  const title = laneTitle(laneId, lane)
  const primary = lanePrimaryAction(lane)
  const secondary = laneSecondaryActions(lane)
  const actions = [...(primary ? [primary] : []), ...secondary.filter((item) => item.label !== primary?.label)]
  const page = lane?.full_details_page
  const pageLabel = lane?.full_details_label || (page ? `Open ${page}` : '')
  const tech = depth === 'professional' ? technicalLines(lane?.technical) : []

  return (
    <aside className="home-os-inspector" role="region" aria-label={`${title} details`}>
      <header>
        <div>
          <span>{title}</span>
          <h3>{lane?.status || 'Waiting'}</h3>
        </div>
        <button type="button" className="secondary" onClick={onClose} aria-label="Close inspector">Close</button>
      </header>
      <p className="home-os-inspect-what">{lane?.what || `${title} is part of QuantTerm.`}</p>
      <p>{lane?.meaning || lane?.summary || lane?.detail || 'Status is known from the current Home projection.'}</p>
      {nothingNeeded(lane) ? <p className="home-os-inspect-ok">Nothing needed from you.</p> : null}
      {lane?.recovering ? <p>QuantTerm is attempting recovery.</p> : null}
      <div className="home-os-inspect-grid">
        <Field label="Now" value={lane?.current} />
        <Field label="Waiting for" value={lane?.waiting_for} />
        <Field label="Next" value={lane?.next} />
        <Field label="After that" value={lane?.after_that} />
        <Field label="Last good" value={lane?.last_success_at} />
        <Field label="Last problem" value={lane?.last_failure_reason || lane?.last_failure_at} />
      </div>
      {laneId === 'paper_bot' && (lane?.positions || []).length ? (
        <ul className="home-os-inspect-positions">
          {(lane?.positions || []).map((pos) => (
            <li key={pos.symbol}>
              <b>{pos.symbol}</b>
              <span>
                {pos.status || 'Open'}
                {pos.entry != null ? ` · in ${pos.entry}` : ''}
                {pos.stop != null ? ` · stop ${pos.stop}` : ''}
                {pos.target != null ? ` · target ${pos.target}` : ''}
                {pos.risk_used != null ? ` · risk ${pos.risk_used}` : ''}
              </span>
            </li>
          ))}
        </ul>
      ) : null}
      {laneId === 'paper_bot' ? (
        <p className="panel-copy">
          {lane?.last_decision ? `Last decision: ${lane.last_decision}` : null}
          {lane?.why ? ` Why: ${lane.why}` : ''}
        </p>
      ) : null}
      {lane?.primary_action?.kind === 'instruction' && lane.primary_action.instruction ? (
        <p className="panel-copy">{lane.primary_action.instruction}</p>
      ) : null}
      <ActionRow actions={actions} busy={busy} onAction={onAction} />
      {page && onOpenPage ? (
        <button type="button" className="home-os-inspect-link" onClick={() => onOpenPage(page)}>
          {pageLabel || 'Open full details'}
        </button>
      ) : null}
      {tech.length ? (
        <details className="home-os-why">
          <summary>Technical details</summary>
          {tech.map((line) => <p key={line}>{line}</p>)}
        </details>
      ) : null}
      <p className="home-os-inspect-lock">
        Live money: {liveMoneyStillLocked(liveLocked, lane) ? 'Locked' : 'Must stay locked'}. No live buy button.
      </p>
    </aside>
  )
}

export function SystemLaneStrip({
  system,
  selected,
  onSelect,
}: {
  system: Record<string, SystemLane>
  selected: string | null
  onSelect: (id: string) => void
}) {
  return (
    <div className="home-os-system">
      {SYSTEM_LANE_ORDER.map((key) => {
        const lane = system[key] || {}
        const open = selected === key
        return (
          <button
            key={key}
            type="button"
            className={[
              'home-os-lane',
              `lane-${String(lane.status || 'Waiting').toLowerCase().replace(/\s+/g, '-')}`,
              open ? 'selected' : '',
            ].filter(Boolean).join(' ')}
            aria-label={laneAriaLabel(key, lane)}
            aria-expanded={open}
            aria-pressed={open}
            onClick={() => onSelect(open ? '' : key)}
          >
            <span>{LANE_TITLE[key]}</span>
            <strong>{lane.status || 'Waiting'}</strong>
            <em>{open ? 'Hide' : 'View'}</em>
          </button>
        )
      })}
    </div>
  )
}
