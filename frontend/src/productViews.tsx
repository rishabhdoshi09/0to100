import { useEffect, useMemo, useState } from 'react'
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
  fetchPreTrade,
  fetchStockFundamentals,
  fetchSymbolRatios,
  type IntelligenceMetric,
  type PreTrade,
  type ProductReadiness,
  type StockWorkspace,
  type TradePlan,
} from './productApi'
import type { ChartBar, ControlName, DashboardPayload, OptionsChainPayload } from './types'
import { longTermPicks } from './longTermPicks'
import { fetchMarketOptions } from './api'
import { PositioningReadCard } from './marketViews'

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
      {plan.effective_bets_before != null && plan.effective_bets_after != null && (
        <p className="risk-lens-note">
          Effective bets {plan.effective_bets_before}→{plan.effective_bets_after}
          {plan.cost_drag_r != null ? ` · round-trip cost ≈ ${plan.cost_drag_r.toFixed(2)}R` : ''}
        </p>
      )}
      {plan.effective_bets_before == null && plan.cost_drag_r != null && (
        <p className="risk-lens-note">Round-trip cost ≈ {plan.cost_drag_r.toFixed(2)}R of the stop distance.</p>
      )}
      <p className="risk-lens-summary">{plan.summary}</p>
    </section>
  )
}

/** Pre-trade cockpit: GO / CAUTION / NO_GO over the risk lens. Never a buy signal. */
export function PreTradeCockpit({ cockpit }: { cockpit: PreTrade | null }) {
  if (!cockpit) return null
  const tone = String(cockpit.verdict || 'NO_GO').toLowerCase().replace('_', '-')
  const plan = cockpit.plan || null
  const edge = cockpit.measured_edge_r ?? cockpit.scan?.edge_r
  const learning = cockpit.learning
  return (
    <section className={`pre-trade-cockpit pre-trade-${tone}`}>
      <header className="pre-trade-verdict">
        <div>
          <span>PRE-TRADE</span>
          <strong>{cockpit.verdict}</strong>
        </div>
        <p>{cockpit.meaning}</p>
      </header>
      <div className="key-value-list" style={{ marginBottom: '10px' }}>
        <div>
          <span>Measured edge</span>
          <strong>{edge == null ? '—' : `${edge >= 0 ? '+' : ''}${Number(edge).toFixed(2)}R`}</strong>
        </div>
        <div>
          <span>Learning</span>
          <strong>{learning?.evidence_note || (learning?.signal_backtest_actionable ? 'actionable' : 'unproven')}</strong>
        </div>
      </div>
      {(cockpit.blockers || []).length > 0 && (
        <ul className="pre-trade-blockers">
          {cockpit.blockers.map((item) => <li key={item}>{item}</li>)}
        </ul>
      )}
      {(cockpit.warnings || []).length > 0 && (
        <ul className="pre-trade-warnings">
          {cockpit.warnings.map((item) => <li key={item}>{item}</li>)}
        </ul>
      )}
      <RiskLensCard plan={plan} />
      <p className="pre-trade-honesty">{cockpit.honesty}</p>
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

function RetailChecklist({ readiness }: { readiness: ProductReadiness | null }) {
  const checklist = readiness?.retail_research_checklist
  if (!checklist) return null
  return (
    <section className="retail-research-checklist">
      <header>
        <strong>Research checklist for cash traders</strong>
        <span>{checklist.ready_count} ready · {checklist.gap_count} gap(s)</span>
      </header>
      <p>{checklist.summary}</p>
      <div className="retail-checklist-items">
        {(checklist.items || []).map((item) => (
          <article key={item.key} className={`retail-check ${laneTone(item.status === 'READY' ? 'FRESH' : item.status === 'PARTIAL' ? 'STALE' : 'MISSING')}`}>
            <header><strong>{item.label}</strong><span>{item.status}</span></header>
            <p>{item.why_it_matters}</p>
            <b>{item.evidence}</b>
            {item.next_action && item.next_action !== 'NONE' && (
              <small>Next: {item.next_action}</small>
            )}
          </article>
        ))}
      </div>
    </section>
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
  const quality = useMemo(() => longTermPicks(dashboard.long_term.records)
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
      <RetailChecklist readiness={readiness} />

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
  const { selected, bars, runControl, setActive, onCompare, onWatchlist, depth } = props
  const [workspace, setWorkspace] = useState<StockWorkspace | null>(null)
  const [preTrade, setPreTrade] = useState<PreTrade | null>(null)
  const [ratios, setRatios] = useState<import('./productApi').SymbolRatioRow[]>([])
  const [tab, setTab] = useState('Overview')
  const [loading, setLoading] = useState(false)
  const [busy, setBusy] = useState('')
  const [fundamentalsError, setFundamentalsError] = useState('')
  const [error, setError] = useState('')

  const [optionsChain, setOptionsChain] = useState<OptionsChainPayload | null>(null)
  const [optionsLoading, setOptionsLoading] = useState(false)
  const [optionsForce, setOptionsForce] = useState(0)

  const intelTabs = [
    'Overview',
    'Chart',
    'Financials',
    'Ratios',
    'Ownership',
    'Options',
    'Events',
    'Peers',
    'Evidence',
  ]

  const fundamentalsBusy = busy === 'FETCH_FUNDAMENTALS' || busy === 'REFRESH_STOCK_FUNDAMENTALS'

  const loadRatios = async () => {
    if (!selected) {
      setRatios([])
      return
    }
    try {
      const ratioPayload = await fetchSymbolRatios(selected)
      setRatios(ratioPayload.ratios || [])
    } catch {
      setRatios([])
    }
  }

  const loadFundamentals = async (force: boolean) => {
    if (!selected) return
    const token = force ? 'REFRESH_STOCK_FUNDAMENTALS' : 'FETCH_FUNDAMENTALS'
    setBusy(token)
    setFundamentalsError('')
    try {
      const result = await fetchStockFundamentals(selected, force)
      setWorkspace(result.workspace)
      await loadRatios()
    } catch (reason) {
      setFundamentalsError(
        reason instanceof Error ? reason.message : 'Fundamentals fetch failed — try Retry',
      )
    } finally {
      setBusy('')
    }
  }

  const load = async () => {
    if (!selected) {
      setWorkspace(null)
      setPreTrade(null)
      setRatios([])
      setFundamentalsError('')
      return
    }
    setLoading(true)
    setFundamentalsError('')
    setRatios([])
    try {
      const ws = await fetchStockIntelligence(selected)
      setWorkspace(ws)
      setError('')
      // Clear the full-page loader as soon as the workspace lands. Fundamentals /
      // pre-trade are slower secondary fetches and have their own busy UI.
      setLoading(false)
      try {
        setPreTrade(await fetchPreTrade(selected))
      } catch {
        setPreTrade(null)
      }
      if (!ws.fundamentals?.available || (ws.fundamentals.coverage_pct ?? 0) < 40) {
        void loadFundamentals(false)
      } else {
        void loadRatios()
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Stock intelligence unavailable')
      setLoading(false)
    }
  }

  useEffect(() => {
    void load()
  }, [selected])

  useEffect(() => {
    setTab('Overview')
    setOptionsChain(null)
  }, [selected])

  useEffect(() => {
    if (tab !== 'Options' || !selected) return
    setOptionsLoading(true)
    // Force network on desk/tab loads so a background warm miss cannot blank the chain.
    fetchMarketOptions(selected, true)
      .then((payload) => setOptionsChain(payload))
      .catch(() => setOptionsChain({ available: false, message: 'Option chain fetch failed' }))
      .finally(() => setOptionsLoading(false))
  }, [tab, selected, optionsForce])

  const runAction = async (control: ControlName | 'REFRESH_STOCK_FUNDAMENTALS') => {
    if (!selected) return
    if (control === 'REFRESH_STOCK_FUNDAMENTALS') {
      await loadFundamentals(true)
      return
    }
    setBusy(control)
    setError('')
    try {
      await runControl(control)
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
      {fundamentalsError && (
        <div className="api-warning">
          {fundamentalsError}
          <button type="button" className="mode-action" disabled={fundamentalsBusy} onClick={() => void loadFundamentals(true)}>
            {fundamentalsBusy ? 'Retrying…' : 'Retry fundamentals'}
          </button>
        </div>
      )}
      {fundamentalsBusy && !fundamentalsError && (
        <div className="api-warning" style={{ borderColor: 'var(--accent-cyan, #26d7ff)' }}>
          Fetching fundamentals from Screener.in for {selected}… (~1s)
        </div>
      )}
      <header className="stock-workspace-hero">
        <div><span>{workspace?.sector || 'Sector not classified'}</span><h2>{workspace?.company || selected}</h2><p>{selected} · {workspace?.summary || 'Verified research is still loading.'}</p></div>
        <div className="stock-workspace-actions">
          {onWatchlist && (
            <button type="button" onClick={() => onWatchlist(selected)}>★ Watchlist</button>
          )}
          {onCompare && (
            <button type="button" onClick={() => onCompare(selected)}>⇔ Compare</button>
          )}
        </div>
        <div className="stock-workspace-state"><span>{words(workspace?.state || 'LOADING')}</span><strong>{workspace?.confidence_pct ?? 0}%</strong><small>data confidence</small></div>
      </header>

      <PreTradeCockpit cockpit={preTrade} />

      <div className="stock-action-row">
        {(workspace?.next_actions || []).map((item) => (
          <button
            type="button"
            key={item.control}
            disabled={busy === item.control || (item.control === 'REFRESH_STOCK_FUNDAMENTALS' && fundamentalsBusy)}
            onClick={() => void runAction(item.control)}
          >
            {busy === item.control || (item.control === 'REFRESH_STOCK_FUNDAMENTALS' && fundamentalsBusy) ? 'Working…' : item.label}
          </button>
        ))}
        <button type="button" disabled={fundamentalsBusy || !selected} onClick={() => void loadFundamentals(true)}>
          {fundamentalsBusy ? 'Loading…' : 'Retry fundamentals'}
        </button>
        <button type="button" onClick={() => setActive('Research Data')}>Complete missing research data</button>
      </div>

      <SectionTabs tabs={intelTabs} active={tab} onChange={setTab} />

      {tab === 'Overview' && (
        <>
          <div className="stock-overview-grid">
            <Panel title="COMPANY SNAPSHOT" subtitle={workspace?.sector || 'Sector unknown'}>
              {workspace?.fundamentals.company_about
                ? <div className="company-about"><p>{workspace.fundamentals.company_about}</p></div>
                : <EmptyState title="Company description not loading" detail={fundamentalsBusy ? 'Fetching from Screener.in…' : 'Use Retry fundamentals if the fetch failed.'} />}
              <div className="fact-grid">
                <div><span>State</span><strong>{words(workspace?.state || '—')}</strong></div>
                <div><span>Coverage</span><strong>{workspace?.fundamentals.coverage_pct ?? 0}%</strong></div>
                <div><span>Trend</span><strong>{workspace?.technical.trend || '—'}</strong></div>
                <div><span>Close</span><strong>{money(workspace?.technical.close)}</strong></div>
              </div>
            </Panel>
            <Panel title="DECISION SUMMARY" subtitle="Deterministic scan evidence — not investment advice">
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
        <>
          {(workspace?.fundamentals.key_ratios?.length ?? 0) > 0 && (
            <Panel title="SCREENER KEY RATIOS" subtitle="Top-of-page ratios from Screener.in (includes P/E when published)">
              <div className="explain-metric-grid fundamentals">
                {workspace?.fundamentals.key_ratios?.map((row) => (
                  <article className="explain-metric" key={row.name}>
                    <span>{row.name}</span>
                    <strong>{row.value}</strong>
                  </article>
                ))}
              </div>
            </Panel>
          )}
          <Panel title="FUNDAMENTALS — CURRENT SNAPSHOT" subtitle={`${workspace?.fundamentals.coverage_pct ?? 0}% coverage · fetched ${workspace?.fundamentals.fetched_at || 'unknown'}`}>
            {(workspace?.fundamentals.metrics || []).length === 0
              ? <EmptyState title="No fundamental snapshot" detail={fundamentalsBusy ? 'Loading from Screener.in…' : 'Tap Retry fundamentals above.'} />
              : <div className="explain-metric-grid fundamentals">{(workspace?.fundamentals.metrics || []).map((metric) => <MetricExplanation metric={metric} key={metric.key} />)}</div>}
          </Panel>
        </>
      )}

      {tab === 'Ratios' && (
        <Panel title="KEY RATIOS" subtitle="From Screener.in cache — computed where inputs exist; top ratios used when tables are thin">
          {fundamentalsBusy
            ? <EmptyState title="Loading ratios" detail={`Fetching fundamentals for ${selected}…`} />
            : ratios.length === 0
            ? <EmptyState title="Ratios unavailable" detail="Tap Retry fundamentals above. Screener.in must respond (~1s per symbol)." />
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
            && <EmptyState title="Ownership not in cache" detail="Shareholding loads with fundamentals when you open this stock." />}
        </Panel>
      )}

      {tab === 'Options' && (
        <Panel title="OPTION CHAIN" subtitle="Nearest expiry · NSE then Yahoo fallback · context only">
          {(!workspace?.fno || !Object.keys(workspace.fno).length) && (
            <p className="panel-copy">This symbol may not be in the current F&O universe — chain fetch still attempts NSE/Yahoo if contracts exist.</p>
          )}
          <div className="inline-actions">
            <button type="button" disabled={optionsLoading} onClick={() => setOptionsForce((n) => n + 1)}>
              {optionsLoading ? 'Loading…' : 'Retry option chain'}
            </button>
          </div>
          {optionsLoading && <p className="panel-copy">Loading option chain for {selected}…</p>}
          {!optionsLoading && !optionsChain?.available && (
            <EmptyState title="Options unavailable" detail={optionsChain?.message || 'NSE often blocks off-hours — retry or check on a trading day.'} />
          )}
          {optionsChain?.available && (
            <>
              <PositioningReadCard read={optionsChain.positioning_read} />
              <div className="fact-grid">
                <div><span>Expiry</span><strong>{optionsChain.expiry || '—'}</strong></div>
                <div><span>PCR (OI)</span><strong>{optionsChain.pcr ?? '—'}</strong></div>
                <div><span>Max pain</span><strong>{optionsChain.max_pain ?? '—'}</strong></div>
                <div><span>Bias</span><strong>{optionsChain.bias || '—'}</strong></div>
                <div><span>ATM IV</span><strong>{optionsChain.atm_iv != null ? `${optionsChain.atm_iv}%` : '—'}</strong></div>
              </div>
            </>
          )}
          {optionsChain?.note && <p className="panel-copy">{optionsChain.note}</p>}
          {optionsChain?.honesty && <p className="panel-copy">{optionsChain.honesty}</p>}
          {optionsChain?.top_call_oi?.length && (
            <EvidenceList title="Top call OI strikes" items={optionsChain.top_call_oi.map((r) => `${r.strike}: OI ${r.ce_oi}`)} tone="cyan" />
          )}
          {optionsChain?.top_put_oi?.length && (
            <EvidenceList title="Top put OI strikes" items={optionsChain.top_put_oi.map((r) => `${r.strike}: OI ${r.pe_oi}`)} tone="green" />
          )}
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
              {Object.keys(workspace?.fno || {}).length
                ? `Maps to ${(workspace?.fno.future_symbol as string) || 'stock future'} · lot ${(workspace?.fno.lot_size as number) || '—'} · expiry ${(workspace?.fno.expiry as string) || '—'}`
                : 'Not in current F&O universe or master file missing.'}
            </div>
          </Panel>
        </div>
      )}

      {tab === 'Peers' && (
        <div className="stock-context-grid">
          <Panel title="PEER VALUATION" subtitle="Screener peer table + cached peer fundamentals (no fabricated P/E)">
            <div className="fact-grid">
              <div><span>Average peer P/E</span><strong>{workspace?.peers?.average_pe != null ? `${workspace.peers.average_pe}x` : '—'}</strong></div>
              <div><span>P/E vs peer avg</span><strong>{workspace?.peers?.pe_vs_peer_avg != null ? `${workspace.peers.pe_vs_peer_avg}x` : '—'}</strong></div>
              <div><span>Peer samples</span><strong>{workspace?.peers?.peer_pe_sample_count ?? 0}</strong></div>
              <div><span>Stock P/E</span><strong>{workspace?.peers?.stock_pe != null ? `${workspace.peers.stock_pe}x` : '—'}</strong></div>
            </div>
            {workspace?.peers?.peer_rank != null && (
              <p className="panel-copy">
                <strong>Sector rank:</strong> {workspace.peers.peer_rank}/{workspace.peers.total_peers ?? '—'}
                {workspace.peers.peer_rank_verdict ? ` · ${workspace.peers.peer_rank_verdict}` : ''}
                {workspace.peers.sector_leader ? ' · sector leader' : ''}
                {workspace.peers.peer_rank_note ? ` — ${workspace.peers.peer_rank_note}` : ''}
              </p>
            )}
            {workspace?.peers?.peer_pe_note && <p className="panel-copy">{workspace.peers.peer_pe_note}</p>}
          </Panel>
          <Panel title="PEERS" subtitle={workspace?.peers?.note || 'Sector context from scan + Screener'}>
            <p className="panel-copy">Sector: <strong>{workspace?.peers?.sector || workspace?.sector || '—'}</strong></p>
            {workspace?.peers?.sector_peers?.length
              ? (
                <div className="fno-table wide-table">
                  <div className="fno-head"><span>SYMBOL</span><span>SCORE</span><span>STATUS</span></div>
                  {workspace.peers.sector_peers.map((peer) => (
                    <div className="fno-row" key={peer.symbol} style={{ display: 'grid', cursor: 'pointer' }} onClick={() => onCompare?.(peer.symbol)}>
                      <strong>{peer.symbol}</strong>
                      <span>{peer.score}</span>
                      <span>{peer.status || '—'}</span>
                    </div>
                  ))}
                </div>
              )
              : <EmptyState title="No sector peers in scan" detail="Run a whole-market scan, or open a symbol in a mapped sector (e.g. banking, IT)." />}
            {onCompare && <button type="button" onClick={() => onCompare(selected)}>Compare {selected} with another symbol</button>}
          </Panel>
          <Panel title="SCREENER PEER TABLE" subtitle="From fundamentals cache when Screener publishes it">
            {workspace?.peers?.screener_table?.length
              ? (
                <div className="fno-table wide-table">
                  {workspace.peers.screener_table.slice(0, 12).map((row, idx) => (
                    <div className="fno-row" key={idx} style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr', gap: '8px' }}>
                      <span>{String(row[''] || row.Company || row.company || row.name || '—')}</span>
                      <span>{String(row['P/E'] || row.PE || row.pe || '—')}</span>
                      <span>{String(row.CMP || row.cmp || row.Price || '—')}</span>
                    </div>
                  ))}
                </div>
              )
              : <EmptyState title="Screener peer table not loaded" detail="Retry fundamentals — peer comparison is scraped from Screener.in with the company page." />}
          </Panel>
        </div>
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
