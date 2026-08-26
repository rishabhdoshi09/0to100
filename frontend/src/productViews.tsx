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
  fetchDueDiligence,
  acquireDueDiligence,
  fetchTradePlan,
  refreshStockFundamentals,
  fetchSymbolRatios,
  type DueDiligenceKpi,
  type DueDiligenceReport,
  type IntelligenceMetric,
  type ProductReadiness,
  type StockWorkspace,
  type TradePlan,
} from './productApi'
import { keepRicher, markInvestigate, recall, wantsInvestigate } from './sessionMemory'
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
  if (v.includes('STRONGLY SUPPORTS') || v === 'SUPPORTS') return 'is-supports'
  if (v.includes('CONTRADICTS')) return 'is-contradicts'
  return 'is-neutral'
}

function InvestigatePanel({
  report,
  loading,
  error,
  caseMemory,
  decision,
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
  onResearchData: () => void
  onRefresh: () => void
  onAcquire: () => void
  busy: string
}) {
  const [openId, setOpenId] = useState<string | null>(null)
  useEffect(() => { setOpenId(null) }, [report?.symbol])
  if (loading && !report) {
    return <div className="large-empty">Loading sector-framework due diligence from files on disk…</div>
  }
  if (error && !report) {
    return <EmptyState title="Due diligence did not load" detail={error} />
  }
  if (!report) {
    return <EmptyState title="Investigate is empty" detail="No due-diligence report is on file for this symbol." />
  }
  const scoreLabel = report.fundamental_quality.score == null
    ? 'Unmeasured'
    : `${report.fundamental_quality.score} / 100 — ${report.fundamental_quality.label}`
  return (
    <div className="dd-root">
      <p className="dd-question">{report.question}</p>
      <aside className={`dd-verdict ${verdictTone(report.vs_technical_setup)}`} aria-label="Fundamental versus technical setup">
        <span>Fundamental vs technical setup</span>
        <strong>{report.vs_technical_setup}</strong>
        <p>{report.vs_detail}</p>
      </aside>
      {report.thesis?.text ? (
        <aside className="dd-thesis" aria-label="Rule-based desk synthesis">
          <span>Desk synthesis · rules, not a language model</span>
          <p>{report.thesis.text}</p>
        </aside>
      ) : null}
      <div className="dd-score-grid">
        <article><span>Fundamental quality</span><strong>{scoreLabel}</strong><small>{report.fundamental_quality.explain}</small></article>
        <article><span>Business trend</span><strong>{report.business_trend}</strong></article>
        <article><span>Financial strength</span><strong>{report.financial_strength}</strong></article>
        <article><span>Earnings quality</span><strong>{report.earnings_quality}</strong></article>
        <article><span>Balance-sheet quality</span><strong>{report.balance_sheet_quality}</strong></article>
        <article><span>Governance risk</span><strong>{report.governance_risk}</strong></article>
        <article><span>News / event impact</span><strong>{report.news_event_impact}</strong></article>
      </div>
      <p className="dd-framework">{report.framework.label}. {report.framework.blurb} Sector: {report.profile.sector || 'Unclassified'}.</p>
      {(report.profile.revenue_drivers && report.profile.revenue_drivers !== 'Data unavailable — no segment table on file.') ? (
        <p className="dd-framework">Revenue drivers: {report.profile.revenue_drivers}</p>
      ) : null}
      {(caseMemory || decision || report.long_term_overlay?.classification) ? (
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
      <div className="stock-overview-grid">
        <Panel title="KEY STRENGTHS" subtitle="Measured improving KPIs only">
          {(report.strengths || []).length
            ? <ul className="dd-watch">{report.strengths.map((item) => <li key={item}>{item}</li>)}</ul>
            : <EmptyState title="No measured strength" detail="Improving KPIs will appear here when values exist." />}
        </Panel>
        <Panel title="CONCERNS" subtitle="Measured deteriorating KPIs — not red flags">
          {(report.concerns || []).length
            ? <ul className="dd-watch">{report.concerns.map((item) => <li key={item}>{item}</li>)}</ul>
            : <EmptyState title="No measured concern" />}
        </Panel>
      </div>
      <Panel title="RED FLAGS" subtitle="True flags only — not ordinary weakness">
        {(report.red_flags || []).length
          ? (
            <ul className="dd-flag-list">
              {report.red_flags.map((flag) => (
                <li key={flag.id}>
                  <strong>{flag.title}</strong>
                  <span>{flag.fact}</span>
                  <small>{flag.source || 'Source unavailable'} · {flag.source_date || 'date unavailable'}</small>
                </li>
              ))}
            </ul>
          )
          : <EmptyState title="No red flag on file" />}
      </Panel>
      <Panel title="WHAT CHANGED RECENTLY" subtitle="Meaningful items only">
        {(report.what_changed || []).length
          ? <ul className="dd-watch">{report.what_changed.map((item) => <li key={item}>{item}</li>)}</ul>
          : <EmptyState title="No material quarter-to-quarter change measured" />}
      </Panel>
      <Panel title={`${report.framework.label.toUpperCase()} KPIs`} subtitle="Click a row for fact, interpretation, implication and source">
        <div className="dd-kpi-table">
          {report.kpis.map((kpi: DueDiligenceKpi) => {
            const open = openId === kpi.id
            const heights = sparkHeights(kpi.snapshot?.points)
            return (
              <article key={kpi.id} className={`dd-kpi ${kpi.available ? '' : 'unavailable'} ${open ? 'open' : ''}`}>
                <button type="button" onClick={() => setOpenId(open ? null : kpi.id)}>
                  <div>
                    <span>{kpi.label}</span>
                    <strong>{kpi.available ? kpi.fact : 'Data unavailable'}</strong>
                    <small>{kpi.available ? `${kpi.trend} · ${kpi.pillar}` : 'Missing from cache — not estimated.'}</small>
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
                    <div><dt>Interpretation</dt><dd>{kpi.interpretation}</dd></div>
                    <div><dt>Implication</dt><dd>{kpi.implication}</dd></div>
                    <div>
                      <dt>Source</dt>
                      <dd>
                        {kpi.source_url
                          ? <a href={kpi.source_url} target="_blank" rel="noreferrer">{kpi.source}</a>
                          : kpi.source}
                        {kpi.source_date ? ` · ${kpi.source_date}` : ''} · confidence {kpi.confidence}
                      </dd>
                    </div>
                  </dl>
                ) : null}
              </article>
            )
          })}
        </div>
        {(report.unavailable || []).length > 0 && (
          <p className="dd-missing">Data unavailable: {report.unavailable.join(', ')}.</p>
        )}
      </Panel>
      <Panel title="MATERIAL NEWS AND FILINGS" subtitle="Broker roundups are dropped; empty stays empty">
        {(report.events || []).length
          ? (
            <ul className="dd-events">
              {report.events.map((event) => (
                <li key={`${event.published_at}-${event.headline}`}>
                  <strong>{event.headline}</strong>
                  <span>{event.event_type} · {event.impact} · {event.source || 'source unavailable'} · {event.published_at || 'date unavailable'}{event.verified ? ' · verified' : ''}</span>
                  {event.url ? <a href={event.url} target="_blank" rel="noreferrer">Open source</a> : <small>No URL on file</small>}
                </li>
              ))}
            </ul>
          )
          : <EmptyState title="No material company-tagged development on file" />}
      </Panel>
      <Panel title="WHAT TO WATCH NEXT" subtitle="Measurable follow-ups, not forecasts">
        <ul className="dd-watch">{(report.watch_next || []).map((item) => <li key={item}>{item}</li>)}</ul>
      </Panel>
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
      <Panel title="MANAGEMENT COMMENTARY" subtitle="Structured Research Data uploads, plus rule extracts from those files">
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
          : <EmptyState title="Data unavailable" detail="No concall / guidance rows on file. Upload them in Research Data." />}
      </Panel>
      <Panel title="ORDER BOOK / GUIDANCE" subtitle="Only from uploaded structured rows">
        {(report.evidence_pack?.order_book || []).length
          ? <ul className="dd-watch">{report.evidence_pack!.order_book.map((item) => <li key={item.fact}>{item.fact}</li>)}</ul>
          : <EmptyState title="Data unavailable" detail="No order-book or forward-guidance table on file." />}
      </Panel>
      <Panel title="PEERS ON FILE" subtitle="Screener peer table if present — no estimated relative scores">
        {(report.evidence_pack?.peers || []).length
          ? <ul className="dd-watch">{report.evidence_pack!.peers.map((item) => <li key={item.name}>{item.fact}</li>)}</ul>
          : <EmptyState title="Data unavailable" detail="Peer comparison table is empty in this cache." />}
      </Panel>
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
          <button type="button" className="reco-primary" disabled={busy === 'ACQUIRE_DUE_DILIGENCE'} onClick={onAcquire}>
            {busy === 'ACQUIRE_DUE_DILIGENCE' ? 'Acquiring from the internet…' : 'Acquire from the internet'}
          </button>
          <button type="button" className="reco-primary" onClick={onResearchData}>Complete missing research data</button>
          <button type="button" className="reco-ghost" disabled={busy === 'REFRESH_STOCK_FUNDAMENTALS'} onClick={onRefresh}>
            {busy === 'REFRESH_STOCK_FUNDAMENTALS' ? 'Refreshing…' : 'Refresh this stock’s fundamentals'}
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
  const [tab, setTab] = useState(() => (selected && wantsInvestigate(selected) ? 'Investigate' : 'Overview'))
  const [loading, setLoading] = useState(false)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [dd, setDd] = useState<DueDiligenceReport | null>(null)
  const [ddLoading, setDdLoading] = useState(false)
  const [ddError, setDdError] = useState('')

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
    if (selected && wantsInvestigate(selected)) setTab('Investigate')
    else setTab('Overview')
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

  useEffect(() => {
    if (tab === 'Investigate' && selected) void loadDd()
  }, [tab, selected, dashboard.scan.scanned_at])

  const runAction = async (control: ControlName | 'REFRESH_STOCK_FUNDAMENTALS' | 'ACQUIRE_DUE_DILIGENCE') => {
    if (!selected) return
    setBusy(control)
    setError('')
    try {
      if (control === 'REFRESH_STOCK_FUNDAMENTALS') {
        const result = await refreshStockFundamentals(selected)
        setWorkspace(result.workspace)
        if (tab === 'Investigate') await loadDd()
      } else if (control === 'ACQUIRE_DUE_DILIGENCE') {
        const result = await acquireDueDiligence(selected)
        if (result.report) setDd(result.report)
        else await loadDd()
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
          onResearchData={() => setActive('Research Data')}
          onRefresh={() => void runAction('REFRESH_STOCK_FUNDAMENTALS')}
          onAcquire={() => void runAction('ACQUIRE_DUE_DILIGENCE')}
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
              {Object.keys(workspace?.fno || {}).length
                ? `Maps to ${(workspace?.fno.future_symbol as string) || 'stock future'} · lot ${(workspace?.fno.lot_size as number) || '—'} · expiry ${(workspace?.fno.expiry as string) || '—'}`
                : 'Not in current F&O universe or master file missing.'}
            </div>
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
