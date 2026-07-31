import type {
  AutonomyJob,
  ChartBar,
  ConvictionRecord,
  DashboardPayload,
  LongTermRecord,
  PaperPosition,
  ScanRecord,
} from './types'
import { PriceChart } from './PriceChart'
import { compactDateTime, money, pct, score, words } from './format'

export const NAV_ITEMS = [
  ['⌘', 'Command Center'],
  ['◉', 'Scanner'],
  ['◎', 'Stock Intelligence'],
  ['▣', 'Portfolio'],
  ['↗', 'Market Internals'],
  ['◇', 'Long-Term'],
  ['◌', 'Automation'],
] as const

function Logo() {
  return (
    <div className="brand-mark" aria-hidden="true">
      <span /><span /><span />
    </div>
  )
}

export function Sidebar({
  active,
  setActive,
  dashboard,
}: {
  active: string
  setActive: (value: string) => void
  dashboard: DashboardPayload
}) {
  const running = dashboard.autonomy.running
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
          <strong>AUTONOMY</strong>
          <span className={running ? 'status-dot' : 'status-dot status-dot-off'} />
        </div>
        <small>{dashboard.autonomy.state || 'UNKNOWN'} · PID {dashboard.autonomy.scheduler_owner_pid || '—'}</small>
        <div className="broker-stats">
          <div><span>Paper equity</span><strong>{money(dashboard.paper.equity)}</strong></div>
          <div><span>New entries</span><strong>{dashboard.autonomy.new_paper_entries ? 'ALLOWED' : 'BLOCKED'}</strong></div>
        </div>
      </div>
      <div className="system-mini">
        <span>Last heartbeat</span>
        <strong className={running ? '' : 'negative'}>{dashboard.autonomy.heartbeat_ist || 'No heartbeat'}</strong>
        <span>{dashboard.autonomy.plain_state}</span>
      </div>
    </aside>
  )
}

export function MetricCard({
  label,
  value,
  detail,
  tone = 'cyan',
}: {
  label: string
  value: string
  detail?: string
  tone?: 'cyan' | 'green' | 'purple' | 'amber'
}) {
  return (
    <article className={`metric metric-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail || 'No additional reading'}</small>
    </article>
  )
}

export function Panel({
  title,
  subtitle,
  action,
  children,
  className = '',
}: {
  title: string
  subtitle?: string
  action?: React.ReactNode
  children: React.ReactNode
  className?: string
}) {
  return (
    <article className={`panel ${className}`}>
      <div className="panel-title">
        <div><strong>{title}</strong>{subtitle && <small>{subtitle}</small>}</div>
        {action}
      </div>
      {children}
    </article>
  )
}

const rowScore = (row: ScanRecord | ConvictionRecord) =>
  Number((row as ConvictionRecord).conviction_score ?? row.score ?? 0)

export function SecurityTable({
  rows,
  selected,
  onSelect,
  empty = 'No securities match this view.',
  limit,
}: {
  rows: Array<ScanRecord | ConvictionRecord>
  selected?: string
  onSelect: (symbol: string) => void
  empty?: string
  limit?: number
}) {
  const visible = typeof limit === 'number' ? rows.slice(0, limit) : rows
  return (
    <div className="table-shell">
      <div className="table-head"><span>#</span><span>STOCK</span><span>SCORE</span><span>PRICE</span><span>5D</span><span>SETUP</span></div>
      {visible.length === 0 && <div className="empty-row">{empty}</div>}
      {visible.map((row, index) => (
        <button
          key={`${row.symbol}-${index}`}
          type="button"
          className={selected === row.symbol ? 'table-row selected' : 'table-row'}
          onClick={() => onSelect(row.symbol)}
        >
          <span>{index + 1}</span>
          <strong>{row.symbol}</strong>
          <span className="score-cell">{score(rowScore(row))}</span>
          <span>{money(row.price)}</span>
          <span className={(row.momentum_5d || 0) >= 0 ? 'positive' : 'negative'}>{pct(row.momentum_5d)}</span>
          <span>{words((row as ConvictionRecord).classification || row.signals?.[0] || row.status || row.verdict)}</span>
        </button>
      ))}
    </div>
  )
}

export function LongTermTable({
  rows,
  selected,
  onSelect,
  limit,
}: {
  rows: LongTermRecord[]
  selected?: string
  onSelect: (symbol: string) => void
  limit?: number
}) {
  const visible = typeof limit === 'number' ? rows.slice(0, limit) : rows
  return (
    <div className="lt-table">
      <div className="lt-head"><span>#</span><span>STOCK</span><span>CLASS</span><span>FUND.</span><span>TECH.</span><span>COMBINED</span></div>
      {visible.length === 0 && <div className="empty-row">No long-term records match this view.</div>}
      {visible.map((row, index) => (
        <button
          type="button"
          className={selected === row.symbol ? 'lt-row selected' : 'lt-row'}
          key={`${row.symbol}-${index}`}
          onClick={() => onSelect(row.symbol)}
        >
          <span>{index + 1}</span>
          <strong>{row.symbol}</strong>
          <span>{words(row.classification)}</span>
          <span>{score(row.fundamental_score)}</span>
          <span>{score(row.technical_score)}</span>
          <b>{score(row.combined_score)}</b>
        </button>
      ))}
    </div>
  )
}

export function ChartWorkspace({
  symbol,
  bars,
  row,
}: {
  symbol: string
  bars: ChartBar[]
  row?: ScanRecord | ConvictionRecord | LongTermRecord
}) {
  const scan = row as ScanRecord | undefined
  return (
    <div>
      {bars.length > 0 ? (
        <PriceChart symbol={symbol} bars={bars} />
      ) : (
        <div className="chart-empty">
          <div className="chart-grid" />
          <strong>No chart data for {symbol || 'the selected stock'}</strong>
          <span>Saved bhavcopy history is required. Missing history is not simulated.</span>
        </div>
      )}
      <div className="ohlc-strip">
        <div><span>PRICE</span><strong>{money(row?.price)}</strong></div>
        <div><span>ENTRY</span><strong>{money(scan?.entry)}</strong></div>
        <div><span>STOP</span><strong className="negative">{money(scan?.stop)}</strong></div>
        <div><span>TARGET</span><strong className="positive">{money(scan?.target)}</strong></div>
        <div><span>RSI / SCORE</span><strong>{Number.isFinite(scan?.rsi) ? Number(scan?.rsi).toFixed(0) : score((row as LongTermRecord)?.combined_score || scan?.score)}</strong></div>
      </div>
    </div>
  )
}

export function PositionsTable({ rows, closed = false }: { rows: PaperPosition[]; closed?: boolean }) {
  return (
    <div className="positions-table wide-table">
      <div className="positions-head">
        <span>STOCK</span><span>STRATEGY</span><span>ENTRY</span><span>{closed ? 'EXIT / CURRENT' : 'CURRENT'}</span><span>QTY</span><span>P&L</span><span>R / REASON</span>
      </div>
      {rows.length === 0 && <div className="empty-row">No {closed ? 'closed trades' : 'open paper positions'}.</div>}
      {rows.map((row, index) => {
        const pnl = Number(row.pnl || 0)
        return (
          <div className="position-row" key={`${row.symbol || 'position'}-${index}`}>
            <strong>{String(row.symbol || '—')}</strong>
            <span>{String(row.strategy || '—')}</span>
            <span>{money(Number(row.entry_price))}</span>
            <span>{money(Number(row.current_price))}</span>
            <span>{Number(row.quantity || 0)}</span>
            <span className={pnl >= 0 ? 'positive' : 'negative'}>{money(pnl)} {Number.isFinite(row.pnl_pct) ? `(${pct(Number(row.pnl_pct))})` : ''}</span>
            <span>{closed ? String(row.exit_reason || row.result_r || '—') : `${money(Number(row.stop))} / ${money(Number(row.target))}`}</span>
          </div>
        )
      })}
    </div>
  )
}

export function JobLedger({ jobs }: { jobs: AutonomyJob[] }) {
  return (
    <div className="job-ledger wide-table">
      <div className="job-head"><span>JOB</span><span>STATUS</span><span>ATTEMPT</span><span>WHEN</span><span>RESULT / BLOCKER</span></div>
      {jobs.length === 0 && <div className="empty-row">No durable jobs recorded yet.</div>}
      {jobs.map((job) => (
        <div className="job-row" key={job.job_id}>
          <strong>{words(job.job_type)}</strong>
          <span className={`job-status job-${job.status.toLowerCase()}`}>{job.status}</span>
          <span>{job.attempt}</span>
          <span>{compactDateTime(job.finished_at || job.started_at || job.scheduled_for)}</span>
          <span>{job.result_summary || job.blocked_reason || job.error_message || job.error_code || 'No summary'}</span>
        </div>
      ))}
    </div>
  )
}

export function EvidenceList({ title, items, tone = 'cyan' }: { title: string; items?: string[]; tone?: 'cyan' | 'red' | 'green' }) {
  return (
    <div className="evidence-list">
      <strong>{title}</strong>
      {(!items || items.length === 0) && <span>No recorded items.</span>}
      {(items || []).map((item, index) => <div key={`${item}-${index}`}><i className={tone} /> <span>{item}</span></div>)}
    </div>
  )
}
