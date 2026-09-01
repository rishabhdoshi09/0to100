import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { SectionTabs, StatusBadge, EmptyState } from './designSystem'
import {
  ChartWorkspace,
  EvidenceList,
  LongTermTable,
  MetricCard,
  Panel,
  SecurityTable,
} from './components'
import { compactDateTime, money, words } from './format'
// `money` reused by RiskLensCard below; risk % is rendered inline (fractions, not the +/- pct helper)
import {
  bootstrapProduct,
  fetchProductReadiness,
  fetchStockIntelligence,
  fetchDueDiligence,
  acquireDueDiligence,
  fetchInvestigatorSuggest,
  fetchTradePlan,
  refreshStockFundamentals,
  fetchSymbolRatios,
  type DueDiligenceKpi,
  type DueDiligenceReport,
  type InvestigatorMatch,
  type IntelligenceMetric,
  type OptionChainSnapshot,
  type ProductReadiness,
  type StockWorkspace,
  type TradePlan,
} from './productApi'
import { keepRicher, markInvestigate, recall } from './sessionMemory'
import { fetchOperation } from './api'
import type { ChartBar, ControlName, DashboardPayload } from './types'

type AcquireJobState = {
  operationId: string
  status: string
  stage: string
  message: string
  failed: boolean
  error?: string
}

const ACQUIRE_POLL_MS = 400

async function pollAcquireJob(
  operationId: string,
  onTick: (job: AcquireJobState) => void,
): Promise<AcquireJobState> {
  for (;;) {
    const op = await fetchOperation(operationId)
    const failed = op.status === 'FAILED' || op.status === 'BLOCKED' || op.status === 'CANCELLED'
    const job: AcquireJobState = {
      operationId,
      status: op.status,
      stage: op.stage || '',
      message: op.message || '',
      failed,
      error: op.error_code
        ? `${op.error_code}: ${op.error_message || op.message || 'blocked'}`
        : (op.error_message || undefined),
    }
    onTick(job)
    if (op.status === 'SUCCEEDED' || failed) return job
    await new Promise((resolve) => window.setTimeout(resolve, ACQUIRE_POLL_MS))
  }
}

function AcquireBanner({
  job,
  busy,
  onRetry,
}: {
  job?: AcquireJobState | null
  busy: string
  onRetry: () => void
}) {
  const acquiring = busy === 'ACQUIRE_DUE_DILIGENCE' || busy === 'ACQUIRE_DUE_DILIGENCE_ALL' || Boolean(job && !job.failed && job.status !== 'SUCCEEDED')
  if (!acquiring && !job) return null
  return (
    <aside className={`dd-acquire-banner ${job?.failed ? 'is-failed' : ''}`} aria-live="polite">
      {job?.failed ? (
        <>
          <p>{job.error || job.message || 'Research acquire failed.'}</p>
          <button type="button" onClick={onRetry}>Retry</button>
        </>
      ) : (
        <p>
          {job?.stage || 'Downloading filings and fundamentals…'}
          {job?.operationId ? ` · Job ${job.operationId.slice(0, 8)}` : ''}
          {job?.message ? ` · ${job.message}` : ''}
        </p>
      )}
    </aside>
  )
}

// Read-only risk-first "R lens" — exact shares, rupee risk, reward:risk, book heat. No orders.
export function RiskLensCard({ plan }: { plan: TradePlan | null }) {
  if (!plan) return null
  if (!plan.available) {
    return (
      <section className="risk-lens">
        <h3>Risk lens</h3>
        <p className="risk-lens-empty">{plan.message || 'No risk plan available for this symbol.'}</p>
      </section>
    )
  }
  if (plan.tradeable === false) {
    return (
      <section className="risk-lens">
        <h3>Risk lens</h3>
        <p className="risk-lens-empty">Not tradeable: {plan.reason}</p>
      </section>
    )
  }
  const verdict = (plan.heat_verdict || 'OK').toLowerCase()
  return (
    <section className="risk-lens">
      <h3>Risk lens <small>risk before reward · read-only · no orders</small></h3>
      <div className="risk-lens-grid">
        <div><span>Position</span><strong>{plan.qty} sh</strong><small>{money(plan.invested)} ({plan.pct_of_capital}% of capital)</small></div>
        <div><span>Risk if stopped</span><strong>{money(plan.rupee_risk)}</strong><small>{plan.risk_pct_of_capital}% of capital</small></div>
        <div><span>Reward : risk</span><strong>{plan.reward_risk != null ? `${plan.reward_risk}R` : '—'}</strong><small>invalidation −{plan.invalidation_pct}%</small></div>
        <div className={`risk-lens-heat risk-lens-${verdict}`}><span>Book open-risk</span><strong>{plan.open_risk_pct_before != null ? `${plan.open_risk_pct_before}%→${plan.open_risk_pct_after}%` : '—'}</strong><small>{plan.heat_verdict}</small></div>
      </div>
      {plan.market_risk_factor != null && plan.market_risk_factor < 1 && (
        <p className="risk-lens-note">Risk throttled to {((plan.suggested_risk_pct ?? 0) * 100).toFixed(2)}% for a {plan.market_health} market.</p>
      )}
      {plan.correlation_status === 'adds_to_bet' && (plan.correlated_with || []).length > 0 && (
        <p className="risk-lens-note">Not a new bet — moves with {(plan.correlated_with || []).join(', ')}.</p>
      )}
      <p className="risk-lens-summary">{plan.summary}</p>
    </section>
  )
}

type ViewProps = {
  dashboard: DashboardPayload
  selected: string
  setSelected: (symbol: string) => void
  bars: ChartBar[]
  setActive: (page: string) => void
  runControl: (control: ControlName) => Promise<void>
  depth?: import('./productLanguage').DisplayDepth
  onCompare?: (symbol: string) => void
  onWatchlist?: (symbol: string) => void
}

const laneTone = (status: string) => {
  if (status === 'FRESH') return 'fresh'
  if (status === 'STALE') return 'stale'
  if (status === 'UNKNOWN_DATE') return 'unknown'
  return 'missing'
}

const ageLabel = (seconds: number | null) => {
  if (seconds == null) return 'age unknown'
  if (seconds < 60) return `${Math.round(seconds)} sec old`
  if (seconds < 3600) return `${Math.round(seconds / 60)} min old`
  if (seconds < 86400) return `${Math.round(seconds / 3600)} hr old`
  return `${Math.round(seconds / 86400)} day old`
}

function ReadinessHero({ readiness, busy, onBootstrap }: {
  readiness: ProductReadiness | null
  busy: boolean
  onBootstrap: () => void
}) {
  const scoreValue = readiness?.score ?? 0
  return (
    <section className={`product-readiness-hero ${String(readiness?.state || 'EMPTY').toLowerCase()}`}>
      <div className="readiness-score"><span>PRODUCT READINESS</span><strong>{scoreValue}</strong><b>/100</b></div>
      <div className="readiness-copy">
        <span>{readiness?.state || 'CHECKING'}</span>
        <h2>{readiness?.summary || 'Inspecting QuantTerm’s real data lanes…'}</h2>
        <p>Freshness is measured from source dates. Empty cards do not count as working features.</p>
      </div>
      <button type="button" disabled={busy} onClick={onBootstrap}>
        {busy ? 'Preparing QuantTerm…' : scoreValue >= 90 ? 'Refresh complete product' : 'Make QuantTerm ready'}
      </button>
    </section>
  )
}

function LaneGrid({ readiness }: { readiness: ProductReadiness | null }) {
  return (
    <div className="readiness-lanes">
      {(readiness?.lanes || []).map((lane) => (
        <article key={lane.key}>
          <header><strong>{lane.label}</strong><span className={`lane-status ${laneTone(lane.status)}`}>{lane.status}</span></header>
          <p>{lane.meaning}</p>
          <b>{lane.details}</b>
          <small>As of {lane.as_of || 'unknown'} · {ageLabel(lane.age_seconds)}</small>
        </article>
      ))}
    </div>
  )
}

export function ProductCommandCenterView(props: ViewProps) {
  const { dashboard, selected, setSelected, bars, setActive, runControl } = props
  const [readiness, setReadiness] = useState<ProductReadiness | null>(null)
  const [bootstrapBusy, setBootstrapBusy] = useState(false)
  const [message, setMessage] = useState('')

  const loadReadiness = async () => {
    try {
      setReadiness(await fetchProductReadiness())
    } catch (reason) {
      setMessage(reason instanceof Error ? reason.message : 'Readiness check is still starting')
    }
  }

  useEffect(() => {
    void loadReadiness()
    const timer = window.setInterval(() => void loadReadiness(), 10_000)
    return () => window.clearInterval(timer)
  }, [])

  const bootstrap = async () => {
    setBootstrapBusy(true)
    setMessage('Starting the next desk download…')
    try {
      const result = await bootstrapProduct()
      setReadiness(result.readiness)
      setMessage(result.message || (result.queued_kind
        ? `Queued ${result.queued_kind}. The next download waits until this one finishes.`
        : 'Desk data is current.'))
    } catch (reason) {
      setMessage(reason instanceof Error ? reason.message : 'Product preparation failed')
    } finally {
      setBootstrapBusy(false)
    }
  }

  const momentum = useMemo(() => [...dashboard.scan.records]
    .filter((row) => row.signals?.includes('MOMENTUM') || row.verdict === 'BUY')
    .sort((a, b) => (b.score || 0) - (a.score || 0)), [dashboard.scan.records])
  const quality = useMemo(() => [...dashboard.long_term.records]
    .filter((row) => ['QUALITY_COMPOUNDER', 'GARP_CANDIDATE', 'QUALITY_BUT_EXPENSIVE'].includes(row.classification || ''))
    .sort((a, b) => (b.combined_score || 0) - (a.combined_score || 0)), [dashboard.long_term.records])
  const selectedRow = dashboard.conviction.find((row) => row.symbol === selected)
    || dashboard.scan.records.find((row) => row.symbol === selected)
    || dashboard.long_term.records.find((row) => row.symbol === selected)
  const latestNews = dashboard.news.articles.slice(0, 5)

  return (
    <section className="product-command-center">
      <ReadinessHero readiness={readiness} busy={bootstrapBusy} onBootstrap={() => void bootstrap()} />
      {message && <div className="product-message">{message}</div>}
      <LaneGrid readiness={readiness} />

      <div className="product-quick-actions">
        <div><strong>Start with a job, not a menu.</strong><span>Every action below produces a visible operation and a dated saved result.</span></div>
        <button type="button" onClick={() => void runControl('RUN_SCAN_NOW')}>Scan market</button>
        <button type="button" onClick={() => void runControl('REFRESH_LONG_TERM_NOW')}>Refresh funds</button>
        <button type="button" onClick={() => void runControl('REFRESH_NEWS_NOW')}>Refresh context</button>
      </div>

      <section className="metric-grid product-metrics">
        <MetricCard label="MARKET" value={dashboard.market.health.toUpperCase()} detail={dashboard.market.breadth || dashboard.market.summary} tone={dashboard.market.available ? 'green' : 'amber'} />
        <MetricCard label="MOMENTUM MATCHES" value={String(momentum.length)} detail={`${dashboard.scan.universe_size.toLocaleString('en-IN')} stocks evaluated`} />
        <MetricCard label="LONG-TERM QUALITY" value={String(quality.length)} detail={`${dashboard.long_term.summary.coverage_pct ?? 0}% reported coverage`} tone="purple" />
        <MetricCard label="NEWS CONTEXT" value={String(dashboard.news.stats.total || 0)} detail={`${dashboard.news.stats.important || 0} high-impact in 24h`} tone="cyan" />
      </section>

      <div className="product-decision-grid">
        <Panel title="TOP TECHNICAL OPPORTUNITIES" subtitle={dashboard.scan.scanned_at ? `Scan as of ${dashboard.scan.scanned_at}` : 'No scan has completed'} action={<button type="button" onClick={() => setActive('Scanner')}>Open scanner</button>}>
          <SecurityTable rows={momentum} selected={selected} onSelect={setSelected} limit={8} />
        </Panel>
        <Panel title={`SELECTED STOCK · ${selected || 'NONE'}`} subtitle="Official daily history with the saved research record">
          <ChartWorkspace symbol={selected} bars={bars} row={selectedRow} />
          <footer className="product-panel-footer"><button type="button" disabled={!selected} onClick={() => setActive('Stock Intelligence')}>Explain this stock</button></footer>
        </Panel>
        <Panel title="LONG-TERM RESEARCH" subtitle={dashboard.long_term.scanned_at ? `As of ${dashboard.long_term.scanned_at}` : 'No long-term run has completed'} action={<button type="button" onClick={() => setActive('Long-Term')}>Open research</button>}>
          <LongTermTable rows={quality} selected={selected} onSelect={setSelected} limit={7} />
        </Panel>
        <Panel title="LATEST MARKET CONTEXT" subtitle="Open-source context; never a standalone trading signal" action={<button type="button" onClick={() => setActive('News & Events')}>Open all</button>}>
          <div className="command-news-list">
            {latestNews.length === 0 && <div className="large-empty">No news is loaded. Use Refresh context and inspect source health.</div>}
            {latestNews.map((item) => (
              <article key={item.article_id}><div><strong>{item.headline}</strong><span>{item.source} · {compactDateTime(item.published_at)}</span></div><b>{item.impact_score}</b></article>
            ))}
          </div>
        </Panel>
      </div>
      <div className="secondary-layer-note">Paper trading and automation remain available under System Health and Paper Portfolio, but they are not the heart of the research product.</div>
    </section>
  )
}

const metricValue = (metric: IntelligenceMetric) => {
  if (metric.value == null || metric.value === '') return 'Not available'
  if (typeof metric.value === 'number') {
    const value = Math.abs(metric.value) >= 1000 ? metric.value.toLocaleString('en-IN', { maximumFractionDigits: 2 }) : metric.value.toFixed(2).replace(/\.00$/, '')
    return `${value}${metric.unit === '%' ? '%' : metric.unit ? ` ${metric.unit}` : ''}`
  }
  return `${metric.value}${metric.unit ? ` ${metric.unit}` : ''}`
}

function MetricExplanation({ metric }: { metric: IntelligenceMetric }) {
  return (
    <article className={`explain-metric ${metric.value == null ? 'unavailable' : ''}`}>
      <span>{metric.label}</span>
      <strong>{metricValue(metric)}</strong>
      <p>{metric.interpretation}</p>
      <small>{metric.meaning}</small>
    </article>
  )
}

function sparkHeights(points: { period: string; value: number }[] | undefined): number[] {
  const values = (points || []).map((item) => item.value)
  if (!values.length) return []
  const min = Math.min(...values)
  const max = Math.max(...values)
  const span = max - min || 1
  return values.map((value) => 18 + ((value - min) / span) * 82)
}

function verdictTone(value: string): string {
  const v = (value || '').toUpperCase()
  if (v.includes('STRONGLY SUPPORT') || v.includes('STRONG SUPPORT') || v === 'SUPPORTS' || v === 'SUPPORT') return 'is-supports'
  if (v.includes('CONTRADICTS')) return 'is-contradicts'
  if (v.includes('CAUTION')) return 'is-neutral'
  return 'is-neutral'
}

function OptionChainFacts({ chain }: { chain?: OptionChainSnapshot | null }) {
  if (!chain?.available) {
    return (
      <EmptyState
        title="Data unavailable"
        detail={chain?.reason || 'No nearest-expiry option-chain snapshot on file. Acquire on an F&O name, or this stays empty.'}
      />
    )
  }
  const calls = (chain.top_call_oi || []).map((row) => `${row.strike} (${row.oi})`).join(', ')
  const puts = (chain.top_put_oi || []).map((row) => `${row.strike} (${row.oi})`).join(', ')
  return (
    <ul className="dd-watch">
      <li>Expiry: {chain.expiry || 'unavailable'}</li>
      <li>Spot: {chain.spot == null ? 'unavailable' : chain.spot}</li>
      <li>Call OI: {chain.call_oi ?? 'unavailable'} · Put OI: {chain.put_oi ?? 'unavailable'} · PCR: {chain.pcr ?? 'unavailable'}</li>
      <li>Max pain: {chain.max_pain ?? 'unavailable'} · ATM strike: {chain.atm_strike ?? 'unavailable'} · ATM IV: {chain.atm_iv == null ? 'unavailable' : `${chain.atm_iv}%`}</li>
      {calls ? <li>Highest call OI: {calls}</li> : null}
      {puts ? <li>Highest put OI: {puts}</li> : null}
      <li>{chain.note || 'Nearest-expiry snapshot. Not a buy/sell signal.'}</li>
    </ul>
  )
}

function InvestigatePanel({
  report,
  loading,
  error,
  caseMemory,
  decision,
  plan,
  acquireJob,
  onResearchData,
  onRefresh,
  onAcquire,
  busy,
}: {
  report: DueDiligenceReport | null
  loading: boolean
  error: string
  caseMemory?: StockWorkspace['case']
  decision?: StockWorkspace['decision_memory']
  plan?: TradePlan | null
  acquireJob?: AcquireJobState | null
  onResearchData: () => void
  onRefresh: () => void
  onAcquire: (mode?: 'missing_or_stale' | 'all') => void
  busy: string
}) {
  const [openId, setOpenId] = useState<string | null>(null)
  const [section, setSection] = useState('Overview')
  useEffect(() => { setOpenId(null); setSection('Overview') }, [report?.symbol])
  const banner = (
    <AcquireBanner job={acquireJob} busy={busy} onRetry={() => onAcquire('missing_or_stale')} />
  )
  if (loading && !report) {
    return (
      <div className="dd-root">
        {banner}
        <div className="large-empty">Loading sector-framework due diligence from files on disk…</div>
      </div>
    )
  }
  if (error && !report) {
    return (
      <div className="dd-root">
        {banner}
        <EmptyState title="Due diligence did not load" detail={error} />
      </div>
    )
  }
  if (!report) {
    return (
      <div className="dd-root">
        {banner}
        <EmptyState title="Investigate is empty" detail="No due-diligence report is on file for this symbol." />
      </div>
    )
  }
  const screen = report.first_screen
  const confirmation = report.fundamental_confirmation || report.vs_technical_setup
  const scoreLabel = report.fundamental_quality.score == null
    ? 'Unmeasured'
    : `${report.fundamental_quality.score} / 100 — ${report.fundamental_quality.label}`
  const snap = report.company_snapshot || {}
  const kpisFor = (want: string) => {
    if (want === 'Sector KPIs') return report.kpis
    if (want === 'Quarterly') return report.kpis.filter((k) => (k.table || 'quarterly_results') === 'quarterly_results')
    if (want === 'Annual') return report.kpis.filter((k) => k.table === 'profit_loss' || k.table === 'cash_flow')
    if (want === 'Shareholding') return report.kpis.filter((k) => k.table === 'shareholding' || ['promoter', 'pledge', 'fii', 'dii', 'public'].includes(k.id))
    if (want === 'Fundamentals') return report.kpis.filter((k) => ['growth', 'profitability', 'cash', 'leverage'].includes(k.pillar))
    return report.kpis
  }
  const is = (name: string) => section === name
  const kpiSection = ['Fundamentals', 'Sector KPIs', 'Quarterly', 'Annual', 'Shareholding'].includes(section)
  const kpiTable = (rows: DueDiligenceKpi[]) => (
    <div className="dd-kpi-table">
      {rows.map((kpi) => {
        const open = openId === kpi.id
        const heights = sparkHeights(kpi.snapshot?.points)
        return (
          <article key={kpi.id} className={`dd-kpi ${kpi.available ? '' : 'unavailable'} ${open ? 'open' : ''}`}>
            <button type="button" onClick={() => setOpenId(open ? null : kpi.id)}>
              <div>
                <span>{kpi.label}</span>
                <strong>{kpi.available ? kpi.fact : 'Data unavailable'}</strong>
                <small>{kpi.available ? `${kpi.trend} · ${kpi.pillar}${kpi.importance ? ` · ${kpi.importance}` : ''}` : `${kpi.availability_label || (kpi.implemented === false ? 'No validated acquisition path' : 'Not yet acquired')} — not estimated.`}</small>
              </div>
              {heights.length > 1 ? (
                <div className="dd-spark" aria-hidden="true">
                  {heights.map((height, index) => (
                    <i key={`${kpi.id}-${index}`} style={{ height: `${height}%` }} />
                  ))}
                </div>
              ) : null}
            </button>
            {open ? (
              <dl className="dd-kpi-detail">
                <div><dt>Fact</dt><dd>{kpi.fact}</dd></div>
                {kpi.snapshot?.current != null && kpi.snapshot.year_ago != null ? (
                  <div><dt>Prints</dt><dd>{kpi.snapshot.current} vs {kpi.snapshot.year_ago} ({kpi.snapshot.current_period} vs {kpi.snapshot.year_ago_period})</dd></div>
                ) : null}
                <div><dt>Formula</dt><dd>{kpi.formula || 'Calculation not possible'}</dd></div>
                <div><dt>Interpretation</dt><dd>{kpi.interpretation}</dd></div>
                <div><dt>Implication</dt><dd>{kpi.implication}</dd></div>
                <div>
                  <dt>Source</dt>
                  <dd>
                    {kpi.source_url
                      ? <a href={kpi.source_url} target="_blank" rel="noreferrer">{kpi.source}</a>
                      : kpi.source}
                    {kpi.source_date ? ` · ${kpi.source_date}` : ''} · confidence {kpi.confidence}
                    {kpi.provenance?.source_type_label ? ` · ${kpi.provenance.source_type_label}` : ''}
                    {kpi.provenance?.retrieved_at ? ` · retrieved ${kpi.provenance.retrieved_at}` : ''}
                    {kpi.period_type ? ` · ${kpi.period_type}` : ''}
                    {kpi.reporting_basis ? ` · ${kpi.reporting_basis}` : ''}
                    {kpi.source_consensus === 'confirmed' && (kpi.source_count || 0) >= 2 ? ` · Confirmed by ${kpi.source_count} sources` : ''}
                    {kpi.source_consensus === 'conflict' ? ' · Source Conflict' : ''}
                  </dd>
                </div>
                {kpi.definition ? <div><dt>Definition</dt><dd>{kpi.definition}</dd></div> : null}
                {kpi.period_policy ? <div><dt>Period</dt><dd>{kpi.period_policy}</dd></div> : null}
                {kpi.reliability_label ? <div><dt>Acquisition</dt><dd>{kpi.implemented ? kpi.reliability_label : 'Not implemented — listed in the framework only'}</dd></div> : null}
              </dl>
            ) : null}
          </article>
        )
      })}
    </div>
  )
  const coverage = report.research_coverage
  const coveragePct = coverage?.coverage_pct
  const decisionPct = screen?.decision_coverage_pct ?? report.decision_coverage_pct
  const implPct = screen?.implementation_coverage_pct ?? report.implementation_coverage_pct
  const audit = report.framework_audit
  const auditMetrics = screen?.framework_audit_metrics || audit?.decision_metrics || []
  const scoreCov = screen?.score_coverage_pct ?? report.fundamental_quality.score_coverage_pct ?? report.fundamental_quality.coverage_pct
  const missingCritical = screen?.critical_metrics_missing || report.critical_metrics_missing || []
  const missingEvidence = screen?.missing_evidence || report.missing_evidence || []
  const confirmationReason = screen?.confirmation_reason || report.confirmation_reason
  return (
    <div className="dd-root">
      <p className="dd-question">{report.question}</p>
      {banner}
      {error && report ? <div className="api-warning">{error}</div> : null}
      <header className="dd-hero">
        <div>
          <span>{String(snap.sector || report.profile.sector || 'Unclassified')}</span>
          <h2>{report.company}</h2>
          <p>{report.symbol} · {String(snap.selected_by || screen?.selected_by || 'Manual investigator')}</p>
          {(screen?.business_model || report.profile.business_model) && (screen?.business_model || report.profile.business_model) !== 'Data unavailable' ? (
            <p>{String(screen?.sub_sector || report.profile.sub_sector || report.framework.label)} · {String(screen?.business_model || report.profile.business_model)}</p>
          ) : null}
          {report.profile.classification_note ? <p>{report.profile.classification_note}</p> : null}
        </div>
        <aside className={`dd-verdict ${verdictTone(confirmation)}`} aria-label="Fundamental confirmation">
          <span>Fundamental confirmation</span>
          <strong>{confirmation}</strong>
          <p>{confirmationReason ? `${confirmationReason}. ` : ''}{report.vs_detail}</p>
        </aside>
      </header>
      <aside className="dd-conclusion" aria-label="Practical conclusion">
        <p>
          {report.thesis?.text
            || report.first_screen?.technical_reason?.[0]
            || report.vs_detail
            || 'Workspace loaded from files on disk. Missing research stays missing until acquire finishes.'}
        </p>
        <div className="dd-score-grid">
          <article>
            <span>Entry</span>
            <strong>{plan?.available && plan.entry != null ? money(plan.entry, 2) : '—'}</strong>
            <small>{plan?.available ? 'From the saved trade plan' : 'No trade plan on file — levels are not invented'}</small>
          </article>
          <article>
            <span>Invalidation / stop</span>
            <strong>{plan?.available && plan.stop != null ? money(plan.stop, 2) : '—'}</strong>
          </article>
          <article>
            <span>Target</span>
            <strong>{plan?.available && plan.target != null ? money(plan.target, 2) : '—'}</strong>
          </article>
          <article>
            <span>Freshness</span>
            <strong>{report.as_of.fundamentals_freshness || 'MISSING'}</strong>
            <small>{report.as_of.fundamentals_fetched_at || 'Not acquired'}</small>
          </article>
        </div>
      </aside>
      <div className="dd-score-grid dd-first-grid">
        <article><span>Technical score</span><strong>{screen?.technical_score != null ? `${screen.technical_score}` : (report.technical_context.scanner_score != null ? `${report.technical_context.scanner_score}` : 'Data unavailable')}</strong></article>
        <article><span>Fundamental quality</span><strong>{scoreLabel}</strong><small>{report.fundamental_quality.explain}</small></article>
        <article><span>Score coverage</span><strong>{scoreCov == null ? 'Unmeasured' : `${scoreCov}%`}</strong><small>Share of the scoring framework that had enough data to evaluate. Missing is unknown, not zero.</small></article>
        <article><span>Data coverage</span><strong>{coveragePct == null ? 'Unmeasured' : `${coveragePct}%`}</strong><small>{coverage?.summary || 'Datasets acquired — not decision confidence.'}</small></article>
        <article><span>Implementation coverage</span><strong>{implPct == null ? 'Unmeasured' : `${implPct}%`}</strong><small>{screen?.implementation_coverage_summary || audit?.summary || 'Validated acquisition paths for this framework — not this company\'s Decision Coverage.'}</small></article>
        <article><span>Decision coverage</span><strong>{decisionPct == null ? 'Unmeasured' : `${decisionPct}%`}</strong><small>Important sector evidence actually available to judge the company.</small></article>
        <article><span>Business trend</span><strong>{report.business_trend}</strong></article>
        <article><span>{report.framework.label || 'Sector'} KPIs</span><strong>{screen?.sector_kpis || report.sector_kpi_label || 'Unmeasured'}</strong><small>{screen?.sector_kpi_detail || report.sector_kpi_detail || ''}</small></article>
        <article><span>Critical metrics missing</span><strong>{missingCritical.length ? missingCritical.join(', ') : 'None'}</strong></article>
        <article><span>Critical red flags</span><strong>{report.flag_groups?.n_critical ?? 0}</strong></article>
        <article><span>Warnings</span><strong>{report.flag_groups?.n_warnings ?? report.red_flags.length}</strong></article>
        <article><span>Latest results</span><strong>{screen?.latest_financial_quarter || report.as_of.latest_financial_period || 'Data unavailable'}</strong><small>Refresh: {screen?.latest_data_refresh || report.as_of.latest_data_refresh || report.as_of.fundamentals_fetched_at || 'Data unavailable'}</small></article>
      </div>
      {(report.named_quality_scores?.scores || []).length ? (
        <div className="dd-score-grid" aria-label="Named quality scores">
          {(report.named_quality_scores?.scores || []).map((row) => (
            <article key={row.id}>
              <span>{row.label}</span>
              <strong>{row.label_text || 'Unmeasured'}</strong>
              <small>{row.detail || (row.available ? '' : 'Missing evidence stays Unmeasured, not a weak company.')}</small>
            </article>
          ))}
        </div>
      ) : null}
      {missingEvidence.length ? (
        <aside className="dd-missing-evidence" aria-label="Important missing evidence">
          <header>
            <strong>Important missing evidence</strong>
            <span>The system likes what it has seen only if those prints are actually on file. Missing stays missing.</span>
          </header>
          <ul>
            {missingEvidence.map((row) => (
              <li key={String(row.id || row.label)}>
                <strong>{row.label || row.id}</strong>
                <span>{row.reason || 'Metric not reliably extracted'}{row.importance ? ` · ${row.importance}` : ''}{row.availability_state === 'not_implemented' ? ' · listed in the framework, not implemented' : ''}</span>
              </li>
            ))}
          </ul>
          {(screen?.deeper_acquire_available || report.deeper_acquire_available) ? (
            <p>Try deeper source acquisition — additional official providers remain unqueried. Metrics with no validated acquisition path are not re-scraped.</p>
          ) : (
            <p>Already-queried sources that reported no value are not re-scraped. A listed framework metric is implemented only when QuantTerm has a validated path, definition, period handling, provenance and tests.</p>
          )}
        </aside>
      ) : null}
      {auditMetrics.length ? (
        <div className="dd-audit-strip" aria-label="Framework coverage audit">
          <header>
            <strong>{audit?.label || report.framework.label} framework coverage</strong>
            <span>A metric is implemented only with a validated acquisition path, canonical definition, period handling, provenance and tests. This is system capability, not this company&apos;s Decision Coverage.</span>
          </header>
          <ul>
            {auditMetrics.map((row) => (
              <li key={String(row.id || row.label)} className={`dd-ds dd-ds-${row.implemented ? (row.company_state === 'populated' ? 'current' : 'stale') : 'not_implemented'}`}>
                <span>{row.label || row.id}</span>
                <strong>{row.company_state && row.company_state !== 'not evaluated' ? row.company_state : (row.reliability_label || (row.implemented ? 'obtainable' : 'no acquisition path'))}</strong>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      <div className="dd-coverage-strip" aria-label="Research dataset inventory">
        <header>
          <strong>Research data: {coverage?.summary || '0/0 datasets available'}</strong>
          <span>Acquire only what is missing or stale. Failures stay Data unavailable.</span>
        </header>
        <ul>
          {(coverage?.datasets || []).map((row) => (
            <li key={row.id} className={`dd-ds dd-ds-${row.status}`}>
              <span>{row.label}</span>
              <strong>{row.age_label || row.status}</strong>
            </li>
          ))}
        </ul>
        <div className="dd-actions">
          <button type="button" className="reco-primary" disabled={acquiring || !coverage?.needs_acquire} onClick={() => onAcquire('missing_or_stale')}>
            {busy === 'ACQUIRE_DUE_DILIGENCE' ? 'Refreshing missing/stale data…' : 'Refresh Missing/Stale Data'}
          </button>
          <button type="button" className="reco-ghost" disabled={acquiring} onClick={() => onAcquire('all')}>
            {busy === 'ACQUIRE_DUE_DILIGENCE_ALL' ? 'Re-downloading all sources…' : 'Re-download all sources'}
          </button>
        </div>
      </div>
      <div className="dd-snapshot-grid">
        <div><span>Market cap</span><strong>{String(snap.market_cap_display || 'Data unavailable')}</strong></div>
        <div><span>Price</span><strong>{String(snap.current_price_display || 'Data unavailable')}</strong></div>
        <div><span>52w high</span><strong>{String(snap.high_52w_display || 'Data unavailable')}</strong></div>
        <div><span>52w low</span><strong>{String(snap.low_52w_display || 'Data unavailable')}</strong></div>
        <div><span>Promoter</span><strong>{snap.promoter_holding == null ? 'Data unavailable' : `${snap.promoter_holding}%`}</strong></div>
        <div><span>FII</span><strong>{snap.fii_holding == null ? 'Data unavailable' : `${snap.fii_holding}%`}</strong></div>
        <div><span>DII</span><strong>{snap.dii_holding == null ? 'Data unavailable' : `${snap.dii_holding}%`}</strong></div>
        <div><span>Pledge</span><strong>{snap.promoter_pledge == null ? 'Data unavailable' : `${snap.promoter_pledge}%`}</strong></div>
      </div>
      {(typeof snap.about === 'string' && snap.about && snap.about !== 'Data unavailable') ? (
        <p className="dd-framework">{snap.about}</p>
      ) : null}
      {report.thesis?.text ? (
        <aside className="dd-thesis" aria-label="Rule-based desk synthesis">
          <span>Desk synthesis · rules, not a language model</span>
          <p>{report.thesis.text}</p>
        </aside>
      ) : null}
      <SectionTabs
        tabs={screen?.sections || ['Overview', 'Fundamentals', 'Sector KPIs', 'Quarterly', 'Cash Flow', 'Peers', 'Shareholding', 'News', 'Filings', 'Red Flags', 'Sources']}
        active={section}
        onChange={setSection}
      />
      {is('Overview') ? (
        <div className="stock-overview-grid">
          <Panel title="IMPROVING" subtitle="Measured numerical trends only">
            {(screen?.improving || report.strengths || []).length
              ? <ul className="dd-watch">{(screen?.improving || report.strengths).map((item) => <li key={item}>{item}</li>)}</ul>
              : <EmptyState title="No measured improvement" detail="Improving KPIs will appear here when values exist." />}
          </Panel>
          <Panel title="DETERIORATING" subtitle="Measured numerical trends only">
            {(screen?.deteriorating || report.concerns || []).length
              ? <ul className="dd-watch">{(screen?.deteriorating || report.concerns).map((item) => <li key={item}>{item}</li>)}</ul>
              : <EmptyState title="No measured deterioration" />}
          </Panel>
        </div>
      ) : null}
      {is('Overview') ? (
        <div className="stock-overview-grid">
          <Panel title="RECENT MATERIAL EVENTS" subtitle="Taxonomy + materiality — no LLM sentiment">
            {(screen?.recent_material_events || []).length
              ? (
                <ul className="dd-events">
                  {screen!.recent_material_events!.map((event) => (
                    <li key={`${event.date}-${event.headline}`}>
                      <strong>{event.headline}</strong>
                      <span>{event.date} · {event.category} · {event.materiality} · {event.source || 'source unavailable'}</span>
                      {event.url ? <a href={event.url} target="_blank" rel="noreferrer">Original source</a> : null}
                    </li>
                  ))}
                </ul>
              )
              : <EmptyState title="No material company-tagged development on file" />}
          </Panel>
          <Panel title="TECHNICAL REASON FOR SELECTION" subtitle="From the saved scan — this engine does not rescan">
            {(screen?.technical_reason || []).length
              ? <ul className="dd-watch">{screen!.technical_reason!.map((item) => <li key={item}>{item}</li>)}</ul>
              : <EmptyState title="Not on the current scanner shortlist" detail={report.technical_context.detail} />}
          </Panel>
        </div>
      ) : null}
      <p className="dd-framework">{report.framework.label}. {report.framework.blurb} Sector: {report.profile.sector || 'Unclassified'}{report.profile.sub_sector ? ` · ${report.profile.sub_sector}` : ''}. Business model: {report.profile.business_model || 'Data unavailable'}.</p>
      {(report.profile.revenue_drivers && report.profile.revenue_drivers !== 'Data unavailable — no segment table on file.') ? (
        <p className="dd-framework">Revenue drivers: {report.profile.revenue_drivers}</p>
      ) : null}
      {is('Overview') && (caseMemory || decision || report.long_term_overlay?.classification) ? (
        <aside className="dd-overlay" aria-label="Already-wired QuantTerm layers">
          {report.long_term_overlay?.classification ? (
            <p><strong>Long-term overlay.</strong> {report.long_term_overlay.classification}. {report.long_term_overlay.note}</p>
          ) : null}
          {decision?.setup_quality?.score != null ? (
            <p><strong>Setup quality.</strong> {decision.setup_quality.score}/100 — from decision memory, not the Investigate score.</p>
          ) : null}
          {caseMemory?.memory_line ? <p><strong>Case memory.</strong> {caseMemory.memory_line}</p> : null}
        </aside>
      ) : null}
      {kpiSection ? (
        <Panel title={section.toUpperCase()} subtitle="Click a row for formula, prints and source. Missing stays Data unavailable.">
          {kpiTable(kpisFor(section))}
          {(report.unavailable || []).length > 0 && section === 'Sector KPIs' && (
            <p className="dd-missing">Data unavailable: {report.unavailable.join(', ')}.</p>
          )}
        </Panel>
      ) : null}
      {is('Fundamentals') ? (
        <Panel title="SCORE BREAKDOWN" subtitle="Every point is inspectable. Missing buckets are skipped.">
          {(report.fundamental_quality.breakdown?.pillars || []).length
            ? (
              <ul className="dd-flag-list">
                {report.fundamental_quality.breakdown!.pillars!.map((pillar) => (
                  <li key={pillar.id}>
                    <strong>{pillar.label} · {pillar.display}</strong>
                    <span>{pillar.explain}</span>
                    {pillar.formula ? <small>{pillar.formula}</small> : null}
                  </li>
                ))}
              </ul>
            )
            : <EmptyState title="Unmeasured" detail={report.fundamental_quality.explain} />}
        </Panel>
      ) : null}
      {is('Red Flags') ? (
        <Panel title="RED FLAGS" subtitle="Critical / Warnings / Monitor — each with rule, threshold and evidence">
          {(report.red_flags || []).length
            ? (
              <ul className="dd-flag-list">
                {report.red_flags.map((flag) => (
                  <li key={flag.id}>
                    <strong>{(flag.severity || 'monitor').toUpperCase()} · {flag.title}</strong>
                    <span>{flag.evidence || flag.fact}</span>
                    <small>
                      Rule: {flag.rule || flag.kind}
                      {flag.triggered_value != null ? ` · triggered ${typeof flag.triggered_value === 'object' ? JSON.stringify(flag.triggered_value) : String(flag.triggered_value)}` : ''}
                      {flag.threshold != null ? ` · threshold ${typeof flag.threshold === 'object' ? JSON.stringify(flag.threshold) : String(flag.threshold)}` : ''}
                      {' · '}{flag.source || 'Source unavailable'} · {flag.source_date || 'date unavailable'}
                    </small>
                  </li>
                ))}
              </ul>
            )
            : <EmptyState title="No red flag on file" />}
        </Panel>
      ) : null}
      {is('Overview') ? (
        <Panel title="WHAT CHANGED RECENTLY" subtitle="Meaningful items only">
          {(report.what_changed || []).length
            ? <ul className="dd-watch">{report.what_changed.map((item) => <li key={item}>{item}</li>)}</ul>
            : <EmptyState title="No material quarter-to-quarter change measured" />}
        </Panel>
      ) : null}
      {is('News') ? (
        <Panel title="NEWS & EVENTS" subtitle="Broker roundups are dropped; empty stays empty">
          {(report.events || []).length
            ? (
              <ul className="dd-events">
                {report.events.map((event) => (
                  <li key={`${event.published_at}-${event.headline}`}>
                    <strong>{event.headline}</strong>
                    <span>{event.published_at || 'date unavailable'} · {event.category || event.event_type} · {event.materiality || 'Unmeasured'} · {event.source || 'source unavailable'}{event.verified ? ' · verified' : ''}</span>
                    {event.materiality_basis ? <small>{event.materiality_basis}</small> : null}
                    {event.url ? <a href={event.url} target="_blank" rel="noreferrer">Original source</a> : <small>No URL on file</small>}
                  </li>
                ))}
              </ul>
            )
            : <EmptyState title="No material company-tagged development on file" />}
        </Panel>
      ) : null}
      {is('Cash Flow') ? (
        <Panel title="CASH FLOW QUALITY" subtitle={report.cash_flow_quality?.detail || 'Rule-based. Missing stays missing.'}>
          {(report.cash_flow_quality?.metrics || []).length
            ? (
              <ul className="dd-watch">
                {report.cash_flow_quality!.metrics!.map((item) => (
                  <li key={item.id}>{item.available ? item.fact : `${item.label}: Data unavailable`}{item.formula ? ` · ${item.formula}` : ''}</li>
                ))}
              </ul>
            )
            : <EmptyState title={report.cash_flow_quality?.label || 'Data unavailable'} />}
        </Panel>
      ) : null}
      {is('Peers') ? (
        <Panel title="PEERS" subtitle={report.peers?.detail || 'Peer table on file only — no inferred comparables'}>
          {(report.peers?.ranks || []).length
            ? <ul className="dd-watch">{report.peers!.ranks!.map((row) => <li key={row.metric}>{row.metric}: {row.quartile} ({row.formula})</li>)}</ul>
            : (report.evidence_pack?.peers || []).length
              ? <ul className="dd-watch">{report.evidence_pack!.peers.map((item) => <li key={item.name}>{item.fact}</li>)}</ul>
              : <EmptyState title="Data unavailable" detail="Peer comparison table is empty in this cache." />}
        </Panel>
      ) : null}
      {is('Valuation') ? (
        <Panel title="CURRENT SNAPSHOT (NOT SCORED)" subtitle="Same extractor Stock Intelligence already uses — not a quarterly trend">
          <div className="dd-kpi-table">
            {(report.evidence_pack?.snapshot_metrics || []).map((item) => (
              <article key={item.id} className={`dd-kpi ${item.available ? '' : 'unavailable'}`}>
                <div className="dd-kpi-static">
                  <span>{item.label}</span>
                  <strong>{item.available ? item.fact : 'Data unavailable'}</strong>
                  <small>{item.interpretation}</small>
                </div>
              </article>
            ))}
          </div>
        </Panel>
      ) : null}
      {is('Filings') ? (
        <Panel title="EXCHANGE FILINGS" subtitle="Acquire archive + official curator items. No LLM summary.">
          {(report.filings || []).length
            ? (
              <ul className="dd-events">
                {report.filings!.map((item, index) => (
                  <li key={`${item.title}-${index}`}>
                    <strong>{item.title}</strong>
                    <span>{item.category} · {item.source || 'source unavailable'}{item.published_at ? ` · ${item.published_at}` : ''}</span>
                    {item.url ? <a href={item.url} target="_blank" rel="noreferrer">Open original</a> : <small>No URL on file</small>}
                  </li>
                ))}
              </ul>
            )
            : <EmptyState title="No filing on file yet" detail="Acquire from the internet or wait for the desk pipeline." />}
        </Panel>
      ) : null}
      {is('Sources') ? (
        <Panel title="SOURCES" subtitle="Provenance that survived the pipeline">
          {(report.source_conflicts || []).length
            ? (
              <ul className="dd-flag-list">
                {report.source_conflicts!.map((item, index) => (
                  <li key={`${item.field}-${index}`}>
                    <strong>Source conflict · {item.field}</strong>
                    <span>{item.status}. Preferred: {String(item.preferred?.value)} ({item.preferred?.source}). Other: {String(item.other?.value)} ({item.other?.source}).</span>
                    <small>{item.note}</small>
                  </li>
                ))}
              </ul>
            )
            : null}
          {(report.sources || []).length
            ? <ul className="dd-watch">{report.sources!.map((item, index) => <li key={`${item.source}-${index}`}>{item.source} · {item.source_type_label || ''} · {item.period || ''}{item.source_url ? ` · ${item.source_url}` : ''}</li>)}</ul>
            : <EmptyState title="No sourced print on file" />}
        </Panel>
      ) : null}
      {is('Overview') ? (
        <Panel title="WHAT TO WATCH NEXT" subtitle="Measurable follow-ups, not forecasts">
          <ul className="dd-watch">{(report.watch_next || []).map((item) => <li key={item}>{item}</li>)}</ul>
        </Panel>
      ) : null}
      {is('Overview') ? (
        <Panel title="FILING / COMMENTARY TONE" subtitle="Rule-extracted from files on disk — never invented">
          {(report.extracted_guidance || []).length
            ? (
              <ul className="dd-events">
                {report.extracted_guidance!.map((item) => (
                  <li key={`${item.source}-${item.excerpt.slice(0, 24)}`}>
                    <strong>{item.tone}</strong>
                    <span>{item.excerpt}</span>
                    <small>{item.source}{item.source_date ? ` · ${item.source_date}` : ''}</small>
                  </li>
                ))}
              </ul>
            )
            : <EmptyState title="Data unavailable" detail="No guidance tokens in a concall, filing or commentary file yet. Run Acquire or upload a transcript." />}
        </Panel>
      ) : null}
      {is('Filings') ? (
        <Panel title="AUTONOMY DOWNLOADS" subtitle="Internet fetch writes files here; Investigate GET only reads them">
          {report.autonomy?.acquired_at ? (
            <ul className="dd-watch">
              <li>Last acquire: {report.autonomy.acquired_at}</li>
              {(report.autonomy.steps || []).map((step) => (
                <li key={step.id}>{step.id}: {step.ok ? 'downloaded' : (step.error || 'not downloaded')}</li>
              ))}
              {(report.autonomy.still_missing || []).length
                ? <li>Still missing after acquire: {report.autonomy.still_missing!.join(', ')}</li>
                : null}
            </ul>
          ) : (
            <EmptyState title="No autonomous download on file yet" detail="Acquire from the internet fills Screener and NSE filings, then this page reloads from disk." />
          )}
        </Panel>
      ) : null}
      {is('Overview') ? (
        <Panel title="MANAGEMENT COMMENTARY" subtitle="Structured uploads first; Acquire fills holes from filing / annual-report text">
          {(report.evidence_pack?.management_commentary || []).length
            ? (
              <ul className="dd-events">
                {report.evidence_pack!.management_commentary.map((item) => (
                  <li key={`${item.event_date}-${item.commentary.slice(0, 24)}`}>
                    <strong>{item.speaker}{item.topic ? ` · ${item.topic}` : ''}</strong>
                    <span>{item.commentary}</span>
                    <small>{item.event_date || 'date unavailable'}</small>
                  </li>
                ))}
              </ul>
            )
            : <EmptyState title="Data unavailable" detail="No concall / guidance wording on file yet. Run Acquire or upload a transcript in Research Data." />}
        </Panel>
      ) : null}
      {is('Overview') ? (
        <Panel title="ORDER BOOK / GUIDANCE" subtitle="Structured uploads first; Acquire fills holes from filing text">
          {(report.evidence_pack?.order_book || []).length
            ? <ul className="dd-watch">{report.evidence_pack!.order_book.map((item) => <li key={item.fact}>{item.fact}</li>)}</ul>
            : <EmptyState title="Data unavailable" detail="No order-book or forward-guidance figure on file." />}
        </Panel>
      ) : null}
      {is('Overview') ? (
        <Panel title="OPTION CHAIN SNAPSHOT" subtitle="Nearest expiry from last Acquire — not live depth, not Greeks, not a signal">
          <OptionChainFacts chain={report.evidence_pack?.option_chain || report.autonomy?.option_chain} />
        </Panel>
      ) : null}
      <Panel title="EVIDENCE PACK GAPS" subtitle={`Research Data coverage ${report.evidence_pack?.coverage_pct ?? 0}% — complete here, not by guessing`}>
        {(report.evidence_pack?.gaps || []).length
          ? (
            <ul className="dd-flag-list">
              {report.evidence_pack!.gaps.map((gap) => (
                <li key={gap.key}>
                  <strong>{gap.label} · {gap.status}</strong>
                  <span>{gap.why || gap.instructions}</span>
                  {gap.link_url ? <a href={gap.link_url} target="_blank" rel="noreferrer">{gap.link_label || 'Source'}</a> : null}
                </li>
              ))}
            </ul>
          )
          : <EmptyState title="Required evidence rows that this desk already tracks are present" />}
        <div className="dd-actions">
          <button type="button" className="reco-primary" disabled={acquiring || !coverage?.needs_acquire} onClick={() => onAcquire('missing_or_stale')}>
            {busy === 'ACQUIRE_DUE_DILIGENCE' ? 'Refreshing missing/stale data…' : 'Refresh Missing/Stale Data'}
          </button>
          <button type="button" className="reco-primary" onClick={onResearchData}>Complete missing research data</button>
          <button type="button" className="reco-ghost" disabled={busy === 'REFRESH_STOCK_FUNDAMENTALS'} onClick={onRefresh}>
            {busy === 'REFRESH_STOCK_FUNDAMENTALS' ? 'Refreshing…' : 'Refresh this stock’s fundamentals'}
          </button>
          <button type="button" className="reco-ghost" disabled={acquiring} onClick={() => onAcquire('all')}>
            {busy === 'ACQUIRE_DUE_DILIGENCE_ALL' ? 'Re-downloading all sources…' : 'Re-download all sources'}
          </button>
        </div>
      </Panel>
      <p className="dd-fresh">
        Latest financial period: {report.as_of.latest_financial_period || 'Data unavailable'}
        {' · '}fundamentals fetched: {report.as_of.fundamentals_fetched_at || 'Data unavailable'}
        {' · '}freshness: {report.as_of.fundamentals_freshness || 'MISSING'}
        {' · '}latest material news: {report.as_of.latest_material_news || 'Data unavailable'}
      </p>
      <p className="dd-disclaimer">{report.disclaimer}</p>
    </div>
  )
}

export function ProductStockIntelligenceView(props: ViewProps) {
  const { dashboard, selected, bars, runControl, setActive, onCompare, onWatchlist, depth } = props
  const [workspace, setWorkspace] = useState<StockWorkspace | null>(() => (
    selected ? recall<StockWorkspace>(`stock:${selected}`) ?? null : null
  ))
  const [plan, setPlan] = useState<TradePlan | null>(null)
  const [ratios, setRatios] = useState<import('./productApi').SymbolRatioRow[]>([])
  const [tab, setTab] = useState('Investigate')
  const [loading, setLoading] = useState(false)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [dd, setDd] = useState<DueDiligenceReport | null>(null)
  const [ddLoading, setDdLoading] = useState(false)
  const [ddError, setDdError] = useState('')
  const [acquireJob, setAcquireJob] = useState<AcquireJobState | null>(null)
  const autoAcquired = useRef(new Set<string>())
  const acquirePollRef = useRef(0)

  const intelTabs = ['Investigate', 'Overview', 'Chart', 'Financials', 'Ratios', 'Ownership', 'Events', 'Peers', 'Evidence']

  const load = async () => {
    if (!selected) {
      setWorkspace(null)
      setPlan(null)
      setRatios([])
      return
    }
    setLoading(!recall(`stock:${selected}`))
    try {
      const next = await fetchStockIntelligence(selected)
      const kept = keepRicher(`stock:${selected}`, next, (row) => !row.company && !row.summary)
      setWorkspace(kept)
      try { setPlan(await fetchTradePlan(selected)) } catch { setPlan(null) }
      try {
        const ratioPayload = await fetchSymbolRatios(selected)
        setRatios(ratioPayload.ratios || [])
      } catch {
        setRatios([])
      }
      setError('')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Stock intelligence is still loading')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    const cached = selected ? recall<StockWorkspace>(`stock:${selected}`) : undefined
    if (cached) setWorkspace(cached)
    else if (!selected) setWorkspace(null)
    void load()
  }, [selected, dashboard.scan.scanned_at])

  useEffect(() => {
    setTab('Investigate')
    setAcquireJob(null)
  }, [selected])

  const loadDd = async () => {
    if (!selected) {
      setDd(null)
      return
    }
    setDdLoading(true)
    try {
      const next = await fetchDueDiligence(selected)
      setDd(next)
      setDdError('')
    } catch (reason) {
      setDdError(reason instanceof Error ? reason.message : 'Due diligence failed')
    } finally {
      setDdLoading(false)
    }
  }

  const startAcquire = useCallback(async (mode: 'missing_or_stale' | 'all' = 'missing_or_stale') => {
    if (!selected) return
    const token = ++acquirePollRef.current
    setBusy(mode === 'all' ? 'ACQUIRE_DUE_DILIGENCE_ALL' : 'ACQUIRE_DUE_DILIGENCE')
    setDdError('')
    try {
      const result = await acquireDueDiligence(selected, mode, { asyncJob: true })
      if (token !== acquirePollRef.current) return
      if (result.report) {
        setDd(result.report)
        setAcquireJob(null)
        await load()
        return
      }
      const operationId = result.operation_id
      if (!operationId) {
        setDdError('Acquire queued without an operation id')
        return
      }
      const final = await pollAcquireJob(operationId, (job) => {
        if (token === acquirePollRef.current) setAcquireJob(job)
      })
      if (token !== acquirePollRef.current) return
      if (final.failed) {
        setDdError(final.error || final.message || 'Research acquire failed')
        return
      }
      await load()
      await loadDd()
    } catch (reason) {
      if (token !== acquirePollRef.current) return
      setDdError(reason instanceof Error ? reason.message : 'Acquire failed — showing files on disk.')
      setAcquireJob((prev) => prev ? { ...prev, failed: true, error: reason instanceof Error ? reason.message : 'Acquire failed' } : {
        operationId: '',
        status: 'FAILED',
        stage: '',
        message: '',
        failed: true,
        error: reason instanceof Error ? reason.message : 'Acquire failed',
      })
    } finally {
      if (token === acquirePollRef.current) setBusy('')
    }
  }, [selected])

  useEffect(() => {
    if (selected) void loadDd()
  }, [selected, dashboard.scan.scanned_at])

  useEffect(() => {
    if (!selected) return
    const needs = Boolean(dd?.research_coverage?.needs_acquire)
      || Boolean(workspace && workspace.fundamentals && !workspace.fundamentals.available)
    if (!needs || autoAcquired.current.has(selected)) return
    autoAcquired.current.add(selected)
    void startAcquire('missing_or_stale')
  }, [selected, dd?.research_coverage?.needs_acquire, workspace?.fundamentals.available, startAcquire])

  const runAction = async (control: ControlName | 'REFRESH_STOCK_FUNDAMENTALS' | 'ACQUIRE_DUE_DILIGENCE' | 'ACQUIRE_DUE_DILIGENCE_ALL') => {
    if (!selected) return
    setBusy(control)
    setError('')
    try {
      if (control === 'REFRESH_STOCK_FUNDAMENTALS') {
        const result = await refreshStockFundamentals(selected)
        setWorkspace(result.workspace)
        await loadDd()
      } else if (control === 'ACQUIRE_DUE_DILIGENCE' || control === 'ACQUIRE_DUE_DILIGENCE_ALL') {
        await startAcquire(control === 'ACQUIRE_DUE_DILIGENCE_ALL' ? 'all' : 'missing_or_stale')
      } else {
        await runControl(control)
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Action failed')
    } finally {
      if (control === 'REFRESH_STOCK_FUNDAMENTALS') setBusy('')
    }
  }

  if (!selected) return <section className="workspace-view"><div className="large-empty">Search or select an NSE symbol. QuantTerm can open the workspace even when the stock is not already in a saved shortlist.</div></section>
  if (loading && !workspace) return <section className="workspace-view"><div className="large-empty">Loading verified price, technical, fundamental and source data for {selected}…</div></section>

  return (
    <section className="stock-workspace-v2 reco-light">
      {error && <div className="api-warning">{error}</div>}
      <header className="stock-workspace-hero">
        <div><span>{workspace?.sector || 'Sector not classified'}</span><h2>{workspace?.company || selected}</h2><p>{selected} · {workspace?.summary || 'Verified research is still loading.'}</p></div>
        <div className="stock-workspace-actions">
          <button
            type="button"
            onClick={() => {
              markInvestigate(selected)
              setTab('Investigate')
            }}
          >
            Investigate
          </button>
          {onWatchlist && (
            <button type="button" onClick={() => onWatchlist(selected)}>★ Watchlist</button>
          )}
          {onCompare && (
            <button type="button" onClick={() => onCompare(selected)}>⇔ Compare</button>
          )}
        </div>
        <div className="stock-workspace-state"><span>{words(workspace?.state || 'LOADING')}</span><strong>{workspace?.confidence_pct ?? 0}%</strong><small>data confidence</small></div>
      </header>

      <RiskLensCard plan={plan} />

      <AcquireBanner
        job={acquireJob}
        busy={busy}
        onRetry={() => void startAcquire('missing_or_stale')}
      />

      <div className="stock-action-row">
        {(workspace?.next_actions || []).map((item) => (
          <button type="button" key={item.control} disabled={busy === item.control} onClick={() => void runAction(item.control)}>{busy === item.control ? 'Working…' : item.label}</button>
        ))}
        <button type="button" onClick={() => setActive('Research Data')}>Complete missing research data</button>
      </div>

      <SectionTabs tabs={intelTabs} active={tab} onChange={setTab} />

      {tab === 'Investigate' && (
        <InvestigatePanel
          report={dd}
          loading={ddLoading}
          error={ddError}
          caseMemory={workspace?.case}
          decision={workspace?.decision_memory}
          plan={plan}
          acquireJob={acquireJob}
          onResearchData={() => setActive('Research Data')}
          onRefresh={() => void runAction('REFRESH_STOCK_FUNDAMENTALS')}
          onAcquire={(mode) => void runAction(mode === 'all' ? 'ACQUIRE_DUE_DILIGENCE_ALL' : 'ACQUIRE_DUE_DILIGENCE')}
          busy={busy}
        />
      )}

      {tab === 'Overview' && (
        <>
          <div className="stock-overview-grid">
            <Panel title="COMPANY SNAPSHOT" subtitle={workspace?.sector || 'Sector unknown'}>
              {workspace?.fundamentals.company_about
                ? <div className="company-about"><p>{workspace.fundamentals.company_about}</p></div>
                : <EmptyState title="Company description not in cache" detail="Refresh fundamentals or import company profile." />}
              <div className="fact-grid">
                <div><span>State</span><strong>{words(workspace?.state || '—')}</strong></div>
                <div><span>Coverage</span><strong>{workspace?.fundamentals.coverage_pct ?? 0}%</strong></div>
                <div><span>Trend</span><strong>{workspace?.technical.trend || '—'}</strong></div>
                <div><span>Close</span><strong>{money(workspace?.technical.close)}</strong></div>
              </div>
            </Panel>
            <Panel title="DECISION SUMMARY" subtitle="Deterministic scan evidence — not investment advice">
              {workspace?.case ? (
                <aside className={`reco-case is-${workspace.case.verdict || 'unmeasured'}`} aria-label="Case memory">
                  <span>Case memory · {workspace.case.n_similar ?? 0} similar · {(workspace.case.verdict || 'unmeasured').replace(/_/g, ' ')}</span>
                  <p>{workspace.case.memory_line || workspace.case.idea}</p>
                  {workspace.case.invalidation?.[0] ? (
                    <p className="reco-case-invalid">What proves it wrong: {workspace.case.invalidation?.[0]}</p>
                  ) : null}
                  {workspace.case.proven ? null : (
                    <em>{(workspace.case.n_similar ?? 0) > 0 ? 'Not proven yet — fewer than 30 comparable outcomes.' : 'Not remembered yet. Tonight’s check writes the first outcome.'}</em>
                  )}
                </aside>
              ) : null}
              {workspace?.decision_memory ? (
                <aside className="reco-memory" aria-label="Decision memory">
                  <span>Decision memory · {workspace.decision_memory.stance || 'WAIT'}</span>
                  {workspace.decision_memory.setup_quality?.score != null ? (
                    <p>Setup Quality: {workspace.decision_memory.setup_quality.score}/100 — not a win probability.</p>
                  ) : null}
                  {workspace.decision_memory.why_not?.line ? <p>{workspace.decision_memory.why_not.line}</p> : null}
                  {workspace.decision_memory.similar?.line ? <p>{workspace.decision_memory.similar.line}</p> : null}
                  {workspace.decision_memory.trust?.line ? <p>{workspace.decision_memory.trust.line}</p> : null}
                  {workspace.decision_memory.edge?.line && workspace.decision_memory.edge.profile !== 'UNKNOWN' ? (
                    <p>{workspace.decision_memory.edge.line}</p>
                  ) : null}
                </aside>
              ) : null}
              <EvidenceList title="Why it qualified" items={[...((workspace?.scanner.reasons as string[] | undefined) || []), ...(workspace?.fundamentals.quality_factors || [])]} tone="green" />
              <EvidenceList title="What can go wrong" items={[...(workspace?.fundamentals.risk_flags || []), ...(workspace?.gaps || []).map((item) => `${item} is missing or stale.`)]} tone="red" />
              <p className="panel-copy"><strong>Monitor:</strong> invalidation levels in Trade Plan, breadth and sector context on Home.</p>
            </Panel>
          </div>
        </>
      )}

      {tab === 'Chart' && (
        <div className="stock-overview-grid">
          <Panel title={`PRICE STRUCTURE · ${selected}`} subtitle={`History as of ${workspace?.technical.latest_date || 'unknown'}`}>
            <ChartWorkspace symbol={selected} bars={bars} row={workspace?.scanner} />
          </Panel>
          <Panel title="WHAT THE CHART CURRENTLY SAYS">
            <div className="trend-callout"><span>{workspace?.technical.trend || 'UNAVAILABLE'}</span><p>{workspace?.technical.trend_explanation || 'No verified trend calculation.'}</p></div>
            <div className="fact-grid">
              <div><span>Close</span><strong>{money(workspace?.technical.close)}</strong></div>
              <div><span>EMA 20</span><strong>{money(workspace?.technical.ema20)}</strong></div>
              <div><span>EMA 50</span><strong>{money(workspace?.technical.ema50)}</strong></div>
              <div><span>EMA 200</span><strong>{money(workspace?.technical.ema200)}</strong></div>
            </div>
          </Panel>
        </div>
      )}

      {tab === 'Financials' && (
        <Panel title="FUNDAMENTALS — CURRENT SNAPSHOT" subtitle={`${workspace?.fundamentals.coverage_pct ?? 0}% coverage · fetched ${workspace?.fundamentals.fetched_at || 'unknown'}`}>
          {(workspace?.fundamentals.metrics || []).length === 0
            ? <EmptyState title="No fundamental snapshot" detail="Run fundamentals refresh or import financial data." />
            : <div className="explain-metric-grid fundamentals">{(workspace?.fundamentals.metrics || []).map((metric) => <MetricExplanation metric={metric} key={metric.key} />)}</div>}
        </Panel>
      )}

      {tab === 'Ratios' && (
        <Panel title="KEY RATIOS" subtitle="Computed centrally from cached fundamentals — missing inputs stay empty">
          {ratios.length === 0
            ? <EmptyState title="Ratios still loading" detail="Fundamentals cache is filling in. Stay on this page — QuantTerm will not invent missing ratios." />
            : <div className="explain-metric-grid">
              {ratios.map((row) => (
                <article className={`explain-metric ${row.value == null ? 'unavailable' : ''}`} key={row.key}>
                  <span>{row.label}</span>
                  <strong>{row.value != null ? row.value : 'Not available'}</strong>
                  {depth === 'professional' && row.formula && <small>{row.formula} · {row.period || '—'} · {row.scope || '—'}</small>}
                  {row.value == null && row.missing_reason && <small>{row.missing_reason}</small>}
                </article>
              ))}
            </div>}
        </Panel>
      )}

      {tab === 'Ownership' && (
        <Panel title="OWNERSHIP" subtitle="From fundamentals provider when available">
          <div className="explain-metric-grid fundamentals">
            {(workspace?.fundamentals.metrics || []).filter((m) => /promoter|fii|dii|pledge|holding/i.test(m.label)).map((metric) => (
              <MetricExplanation metric={metric} key={metric.key} />
            ))}
          </div>
          {!workspace?.fundamentals.metrics?.some((m) => /promoter|fii|dii/i.test(m.label))
            && <EmptyState title="Ownership not in cache" detail="Refresh fundamentals or import shareholding data." />}
        </Panel>
      )}

      {tab === 'Events' && (
        <div className="stock-context-grid">
          <Panel title="COMPANY-LINKED NEWS">
            <div className="command-news-list">
              {workspace?.news.length
                ? workspace.news.slice(0, 12).map((item) => (
                  <article key={item.article_id}><div><strong>{item.headline}</strong><span>{item.source} · {compactDateTime(item.published_at)}</span></div><b>{item.impact_score}</b></article>
                ))
                : <EmptyState title="No company-linked news" />}
            </div>
          </Panel>
          <Panel title="F&O ELIGIBILITY — NOT A SIGNAL">
            <div className="panel-copy">
              {Object.keys(workspace?.fno || {}).filter((key) => key !== 'option_chain').length
                ? `Maps to ${(workspace?.fno.future_symbol as string) || 'stock future'} · lot ${(workspace?.fno.lot_size as number) || '—'} · expiry ${(workspace?.fno.expiry as string) || '—'}`
                : 'Not in current F&O universe or master file missing.'}
            </div>
            <OptionChainFacts chain={(workspace?.fno?.option_chain as OptionChainSnapshot | undefined) || null} />
          </Panel>
        </div>
      )}

      {tab === 'Peers' && (
        <Panel title="PEERS" subtitle="Sector context from scanner universe">
          <p className="panel-copy">Sector: <strong>{workspace?.sector || '—'}</strong>. Use Compare to evaluate peers side-by-side.</p>
          {onCompare && <button type="button" onClick={() => onCompare(selected)}>Open Compare with {selected}</button>}
        </Panel>
      )}

      {tab === 'Evidence' && (
        <>
          <Panel title="TECHNICALS" subtitle="Value, meaning and interpretation">
            <div className="explain-metric-grid">{(workspace?.technical.metrics || []).map((metric) => <MetricExplanation metric={metric} key={metric.key} />)}</div>
          </Panel>
          <Panel title="SOURCE DATES AND FRESHNESS">
            <div className="stock-source-grid">
              {(workspace?.sources || []).map((source) => (
                <article key={source.name}>
                  <header><strong>{source.name}</strong><StatusBadge status={source.status} /></header>
                  <p>{source.meaning}</p>
                  <small>As of {source.as_of || 'unknown'} · {source.age_days == null ? 'age unknown' : `${source.age_days} day(s) old`}</small>
                </article>
              ))}
            </div>
          </Panel>
        </>
      )}

    </section>
  )
}

export function StockInvestigatorView(props: ViewProps) {
  const { selected, setSelected, setActive, dashboard } = props
  const [query, setQuery] = useState('')
  const [matches, setMatches] = useState<InvestigatorMatch[]>([])
  const [report, setReport] = useState<DueDiligenceReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState('')
  const [acquireJob, setAcquireJob] = useState<AcquireJobState | null>(null)
  const autoAcquired = useRef(new Set<string>())
  const acquirePollRef = useRef(0)

  useEffect(() => {
    const needle = query.trim()
    if (needle.length < 2) {
      setMatches([])
      return
    }
    const timer = window.setTimeout(() => {
      void fetchInvestigatorSuggest(needle)
        .then((payload) => setMatches(payload.matches || []))
        .catch(() => setMatches([]))
    }, 180)
    return () => window.clearTimeout(timer)
  }, [query])

  const load = async (symbol: string) => {
    setLoading(true)
    try {
      const next = await fetchDueDiligence(symbol)
      setReport(next)
      setError('')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Due diligence failed')
      setReport(null)
    } finally {
      setLoading(false)
    }
  }

  const runAcquire = async (mode: 'missing_or_stale' | 'all' = 'missing_or_stale') => {
    if (!selected) return
    const token = ++acquirePollRef.current
    setBusy(mode === 'all' ? 'ACQUIRE_DUE_DILIGENCE_ALL' : 'ACQUIRE_DUE_DILIGENCE')
    setError('')
    try {
      const result = await acquireDueDiligence(selected, mode, { asyncJob: true })
      if (token !== acquirePollRef.current) return
      if (result.report) {
        setReport(result.report)
        setAcquireJob(null)
        return
      }
      const operationId = result.operation_id
      if (!operationId) {
        setError('Acquire queued without an operation id')
        return
      }
      const final = await pollAcquireJob(operationId, (job) => {
        if (token === acquirePollRef.current) setAcquireJob(job)
      })
      if (token !== acquirePollRef.current) return
      if (final.failed) {
        setError(final.error || final.message || 'Research acquire failed')
        return
      }
      await load(selected)
    } catch (reason) {
      if (token !== acquirePollRef.current) return
      setError(reason instanceof Error ? reason.message : 'Acquire failed')
      setAcquireJob((prev) => (
        prev
          ? { ...prev, failed: true, error: reason instanceof Error ? reason.message : 'Acquire failed' }
          : {
            operationId: '',
            status: 'FAILED',
            stage: '',
            message: '',
            failed: true,
            error: reason instanceof Error ? reason.message : 'Acquire failed',
          }
      ))
    } finally {
      if (token === acquirePollRef.current) setBusy('')
    }
  }

  useEffect(() => {
    if (selected) void load(selected)
  }, [selected, dashboard.scan.scanned_at])

  useEffect(() => {
    if (!selected || !report?.research_coverage?.needs_acquire) return
    if (autoAcquired.current.has(selected)) return
    autoAcquired.current.add(selected)
    void runAcquire('missing_or_stale')
  }, [selected, report?.research_coverage?.needs_acquire])

  const pick = (symbol: string) => {
    const clean = symbol.toUpperCase()
    setSelected(clean)
    markInvestigate(clean)
    setQuery(clean)
    setMatches([])
  }

  return (
    <section className="workspace-view reco-light">
      <header className="dd-investigator-search">
        <label htmlFor="stock-investigator-q">Stock Investigator</label>
        <input
          id="stock-investigator-q"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && (matches[0]?.symbol || query.trim())) {
              pick(matches[0]?.symbol || query.trim().toUpperCase())
            }
          }}
          placeholder="Enter ticker or company name — ICICIBANK, DIXON, TRENT, CDSL"
          autoComplete="off"
        />
        {matches.length > 0 ? (
          <ul className="dd-suggest">
            {matches.map((item) => (
              <li key={item.symbol}>
                <button type="button" onClick={() => pick(item.symbol)}>{item.label}</button>
              </li>
            ))}
          </ul>
        ) : null}
        <p>Same research engine as scanner Investigate. This is not a new scanner. Empty stays empty. No language model.</p>
      </header>
      <InvestigatePanel
        report={report}
        loading={loading}
        error={error}
        acquireJob={acquireJob}
        onResearchData={() => setActive('Research Data')}
        onRefresh={() => { if (selected) void load(selected) }}
        onAcquire={(mode) => void runAcquire(mode)}
        busy={busy}
      />
    </section>
  )
}
