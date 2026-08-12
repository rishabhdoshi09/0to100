import type { ReactNode } from 'react'
import { words } from './format'

export function SectionTabs({
  tabs,
  active,
  onChange,
}: {
  tabs: string[]
  active: string
  onChange: (tab: string) => void
}) {
  return (
    <nav className="section-tabs" aria-label="Section navigation">
      {tabs.map((tab) => (
        <button
          key={tab}
          type="button"
          className={active === tab ? 'active' : ''}
          onClick={() => onChange(tab)}
        >
          {tab}
        </button>
      ))}
    </nav>
  )
}

export function StatusBadge({ status }: { status: string }) {
  const tone = status.toLowerCase()
  const cls = ['fresh', 'stale', 'missing', 'error', 'partial'].includes(tone) ? tone : 'missing'
  return <span className={`status-badge ${cls}`}>{words(status)}</span>
}

export function FreshnessBadge({ label, asOf }: { label: string; asOf?: string }) {
  return (
    <span className={`status-badge ${label.toLowerCase()}`}>
      {words(label)}{asOf ? ` · ${asOf}` : ''}
    </span>
  )
}

export function EmptyState({ title, detail }: { title: string; detail?: string }) {
  return (
    <div className="large-empty">
      <strong>{title}</strong>
      {detail && <p>{detail}</p>}
    </div>
  )
}

export function MetricCell({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <div className="fact-grid-cell">
      <span>{label}</span>
      <strong>{value}</strong>
      {hint && <small>{hint}</small>}
    </div>
  )
}
