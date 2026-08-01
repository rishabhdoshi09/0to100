import { useEffect, useMemo, useState } from 'react'
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
  fetchTradePlan,
  refreshStockFundamentals,
  type IntelligenceMetric,
  type ProductReadiness,
  type StockWorkspace,
  type TradePlan,
} from './productApi'
import type { ChartBar, ControlName, DashboardPayload } from './types'

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
      setMessage(reason instanceof Error ? reason.message : 'Readiness API unavailable')
    }
  }

  useEffect(() => {
    void loadReadiness()
    const timer = window.setInterval(() => void loadReadiness(), 10_000)
    return () => window.clearInterval(timer)
  }, [])

  const bootstrap = async () => {
    setBootstrapBusy(true)
    setMessage('Queueing data, news, scanner and long-term lanes…')
    try {
      const result = await bootstrapProduct()
      setReadiness(result.readiness)
      const created = result.operations.filter((item) => item.created).length
      setMessage(`${created} preparation operation(s) queued. Progress is visible below.`)
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
        <button type="button" onClick={() => void runControl('RUN_LONG_TERM_SCAN_NOW')}>Find long-term candidates</button>
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

export function ProductStockIntelligenceView(props: ViewProps) {
  const { selected, bars, runControl, setActive } = props
  const [workspace, setWorkspace] = useState<StockWorkspace | null>(null)
  const [plan, setPlan] = useState<TradePlan | null>(null)
  const [loading, setLoading] = useState(false)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')

  const load = async () => {
    if (!selected) {
      setWorkspace(null)
      setPlan(null)
      return
    }
    setLoading(true)
    try {
      setWorkspace(await fetchStockIntelligence(selected))
      try { setPlan(await fetchTradePlan(selected)) } catch { setPlan(null) }
      setError('')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Stock intelligence unavailable')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load()
  }, [selected])

  const runAction = async (control: ControlName | 'REFRESH_STOCK_FUNDAMENTALS') => {
    if (!selected) return
    setBusy(control)
    setError('')
    try {
      if (control === 'REFRESH_STOCK_FUNDAMENTALS') {
        const result = await refreshStockFundamentals(selected)
        setWorkspace(result.workspace)
      } else {
        await runControl(control)
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Action failed')
    } finally {
      setBusy('')
    }
  }

  if (!selected) return <section className="workspace-view"><div className="large-empty">Search or select an NSE symbol. QuantTerm can open the workspace even when the stock is not already in a saved shortlist.</div></section>
  if (loading && !workspace) return <section className="workspace-view"><div className="large-empty">Loading verified price, technical, fundamental and source data for {selected}…</div></section>

  return (
    <section className="stock-workspace-v2">
      {error && <div className="api-warning">{error}</div>}
      <header className="stock-workspace-hero">
        <div><span>{workspace?.sector || 'Sector not classified'}</span><h2>{workspace?.company || selected}</h2><p>{selected} · {workspace?.summary || 'Verified research is still loading.'}</p></div>
        <div className="stock-workspace-state"><span>{words(workspace?.state || 'LOADING')}</span><strong>{workspace?.confidence_pct ?? 0}%</strong><small>data confidence</small></div>
      </header>

      <RiskLensCard plan={plan} />

      <div className="stock-action-row">
        {(workspace?.next_actions || []).map((item) => (
          <button type="button" key={item.control} disabled={busy === item.control} onClick={() => void runAction(item.control)}>{busy === item.control ? 'Working…' : item.label}</button>
        ))}
        <button type="button" onClick={() => setActive('Research Data')}>Complete missing research data</button>
      </div>

      <div className="stock-overview-grid">
        <Panel title={`PRICE STRUCTURE · ${selected}`} subtitle={`History as of ${workspace?.technical.latest_date || 'unknown'}`}><ChartWorkspace symbol={selected} bars={bars} row={workspace?.scanner} /></Panel>
        <Panel title="WHAT THE CHART CURRENTLY SAYS"><div className="trend-callout"><span>{workspace?.technical.trend || 'UNAVAILABLE'}</span><p>{workspace?.technical.trend_explanation || 'No verified trend calculation.'}</p></div><div className="fact-grid"><div><span>Close</span><strong>{money(workspace?.technical.close)}</strong></div><div><span>EMA 20</span><strong>{money(workspace?.technical.ema20)}</strong></div><div><span>EMA 50</span><strong>{money(workspace?.technical.ema50)}</strong></div><div><span>EMA 200</span><strong>{money(workspace?.technical.ema200)}</strong></div></div></Panel>
      </div>

      <Panel title="TECHNICALS — VALUE, MEANING AND INTERPRETATION" subtitle="The system explains the metric instead of assuming the user already knows it.">
        <div className="explain-metric-grid">{(workspace?.technical.metrics || []).map((metric) => <MetricExplanation metric={metric} key={metric.key} />)}</div>
      </Panel>

      <Panel title="FUNDAMENTALS — CURRENT SNAPSHOT" subtitle={`${workspace?.fundamentals.coverage_pct ?? 0}% of the supported metric set is available · fetched ${workspace?.fundamentals.fetched_at || 'unknown'}`}>
        {workspace?.fundamentals.company_about && <div className="company-about"><strong>What the company does</strong><p>{workspace.fundamentals.company_about}</p></div>}
        <div className="explain-metric-grid fundamentals">{(workspace?.fundamentals.metrics || []).map((metric) => <MetricExplanation metric={metric} key={metric.key} />)}</div>
      </Panel>

      <div className="stock-evidence-grid">
        <Panel title="WHY IT QUALIFIED"><EvidenceList title="Technical and quality evidence" items={[...((workspace?.scanner.reasons as string[] | undefined) || []), ...(workspace?.fundamentals.quality_factors || [])]} tone="green" /></Panel>
        <Panel title="WHAT CAN GO WRONG"><EvidenceList title="Recorded risks and missing evidence" items={[...(workspace?.fundamentals.risk_flags || []), ...(workspace?.gaps || []).map((item) => `${item} is missing, stale or undated.`)]} tone="red" /></Panel>
      </div>

      <Panel title="SOURCE DATES AND FRESHNESS" subtitle="A number without a source date is treated as incomplete.">
        <div className="stock-source-grid">{(workspace?.sources || []).map((source) => <article key={source.name}><header><strong>{source.name}</strong><span className={`lane-status ${laneTone(source.status)}`}>{source.status}</span></header><p>{source.meaning}</p><small>As of {source.as_of || 'unknown'} · {source.age_days == null ? 'age unknown' : `${source.age_days} day(s) old`}</small></article>)}</div>
      </Panel>

      <div className="stock-context-grid">
        <Panel title="COMPANY-LINKED NEWS"><div className="command-news-list">{workspace?.news.length ? workspace.news.slice(0, 6).map((item) => <article key={item.article_id}><div><strong>{item.headline}</strong><span>{item.source} · {compactDateTime(item.published_at)}</span></div><b>{item.impact_score}</b></article>) : <div className="large-empty">No company-linked news in the current store.</div>}</div></Panel>
        <Panel title="F&O COVERAGE — NOT A SIGNAL"><div className="panel-copy">{Object.keys(workspace?.fno || {}).length ? `Current instrument master maps ${selected} to ${(workspace?.fno.future_symbol as string) || 'a stock future'} with lot size ${(workspace?.fno.lot_size as number) || 'unknown'} and expiry ${(workspace?.fno.expiry as string) || 'unknown'}. This tells you contract eligibility, not direction.` : `${selected} is not mapped in the current stock-derivatives universe, or the instrument master is missing.`}</div></Panel>
      </div>
    </section>
  )
}
