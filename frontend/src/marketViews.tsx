import { useMemo, useState } from 'react'
import type { ControlName, DashboardPayload, FnoUnderlying, NewsArticle } from './types'
import { compactDateTime, words } from './format'
import { MetricCard, Panel } from './components'

type Props = {
  dashboard: DashboardPayload
  runControl: (control: ControlName) => Promise<void>
  setSelected?: (symbol: string) => void
  setActive?: (page: string) => void
}

const operationLabel = (kind: string) => words(kind.replace('MARKET_', ''))

function operationTone(status: string): string {
  if (status === 'RUNNING') return 'operation-running'
  if (status === 'SUCCEEDED') return 'operation-succeeded'
  if (status === 'BLOCKED' || status === 'FAILED') return 'operation-failed'
  return 'operation-pending'
}

export function OperationsRibbon({ dashboard }: { dashboard: DashboardPayload }) {
  const active = dashboard.operations.active
  const latestFailure = dashboard.operations.recent.find((item) => item.status === 'FAILED' || item.status === 'BLOCKED')
  return (
    <section className="operations-ribbon">
      <div className={dashboard.operations.running ? 'ops-worker online' : 'ops-worker offline'}>
        <i />
        <div>
          <strong>{dashboard.operations.running ? 'MARKET OPERATIONS ONLINE' : 'MARKET OPERATIONS OFFLINE'}</strong>
          <span>PID {dashboard.operations.worker_pid || '—'} · independent from paper autonomy</span>
        </div>
      </div>
      <div className="ops-active-strip">
        {active.length === 0 && <span className="ops-idle">No market operation is running.</span>}
        {active.slice(0, 4).map((item) => (
          <div className={`operation-chip ${operationTone(item.status)}`} key={item.operation_id}>
            <strong>{operationLabel(item.kind)}</strong>
            <span>{item.status} · {words(item.stage)}</span>
            <small>{item.progress_pct == null ? item.message : `${item.progress_pct.toFixed(0)}% · ${item.message}`}</small>
            {item.progress_pct != null && <b style={{ width: `${Math.max(0, Math.min(100, item.progress_pct))}%` }} />}
          </div>
        ))}
      </div>
      {latestFailure && active.length === 0 && (
        <div className="ops-last-failure">
          <strong>{operationLabel(latestFailure.kind)} {latestFailure.status}</strong>
          <span>{latestFailure.error_message || latestFailure.message}</span>
        </div>
      )}
    </section>
  )
}

function NewsCard({ article, openSymbol }: { article: NewsArticle; openSymbol: (symbol: string) => void }) {
  return (
    <article className="news-card">
      <header>
        <div><span>{words(article.category)} · {words(article.event_type)}</span><strong>{article.impact_score}</strong></div>
        <time>{compactDateTime(article.published_at)}</time>
      </header>
      <h3>{article.headline}</h3>
      <p>{article.why_it_matters}</p>
      <div className="news-meta"><span>{article.source}</span><span>{article.official ? 'Official' : `Tier ${article.source_tier}`}</span><span>{article.corroboration_count} source(s)</span></div>
      <div className="news-symbols">
        {article.mentioned_symbols.slice(0, 8).map((symbol) => <button type="button" key={symbol} onClick={() => openSymbol(symbol)}>{symbol}</button>)}
        {article.fno_symbols.length > 0 && <em>F&O linked: {article.fno_symbols.slice(0, 6).join(', ')}</em>}
      </div>
      {article.url && <a href={article.url} target="_blank" rel="noreferrer">Open source ↗</a>}
    </article>
  )
}

export function NewsView({ dashboard, runControl, setSelected, setActive }: Props) {
  const [category, setCategory] = useState('All')
  const [importantOnly, setImportantOnly] = useState(false)
  const articles = useMemo(() => dashboard.news.articles.filter((item) => {
    if (category !== 'All' && item.category !== category.toLowerCase()) return false
    if (importantOnly && item.impact_score < 70) return false
    return true
  }), [category, dashboard.news.articles, importantOnly])
  const categories = ['All', 'Company', 'Economy', 'Regulation', 'Derivatives', 'Global', 'Market']
  const openSymbol = (symbol: string) => {
    setSelected?.(symbol)
    setActive?.('Stock Intelligence')
  }
  const health = dashboard.news.source_health
  const healthy = health.filter((item) => item.status === 'OK').length
  return (
    <section className="workspace-view">
      <div className="inline-actions">
        <button type="button" onClick={() => void runControl('REFRESH_NEWS_NOW')}>Refresh news now</button>
        <button type="button" onClick={() => setImportantOnly((value) => !value)}>{importantOnly ? 'Show all impact levels' : 'Impact 70+ only'}</button>
      </div>
      <div className="view-metrics">
        <MetricCard label="24H ARTICLES" value={String(dashboard.news.stats.total || 0)} detail={`${dashboard.news.stats.important || 0} high impact`} />
        <MetricCard label="HEALTHY SOURCES" value={`${healthy}/${health.length}`} detail="Source-by-source health" tone={healthy ? 'green' : 'amber'} />
        <MetricCard label="F&O LINKED" value={String(dashboard.news.stats.fno_linked || 0)} detail="News mapped to current derivatives" tone="purple" />
        <MetricCard label="REFRESH" value={String(dashboard.news.latest_refresh?.status || 'NOT RUN')} detail={String(dashboard.news.latest_refresh?.message || 'Run a refresh to inspect sources')} />
      </div>
      <div className="mode-tabs">{categories.map((item) => <button type="button" key={item} className={category === item ? 'active' : ''} onClick={() => setCategory(item)}>{item}</button>)}</div>
      <div className="news-layout">
        <Panel title={`CURATED MARKET NEWS · ${articles.length}`} subtitle="News is context, never an order signal">
          <div className="news-feed">{articles.length ? articles.map((article) => <NewsCard key={article.article_id} article={article} openSymbol={openSymbol} />) : <div className="large-empty">No curated article matches this view. Run Refresh news now and inspect source health.</div>}</div>
        </Panel>
        <Panel title="SOURCE HEALTH" subtitle="No silent provider failures">
          <div className="source-health-list">
            {health.length === 0 && <div className="empty-row">No source-health observations yet.</div>}
            {health.map((source) => <div key={source.source_key}><i className={source.status === 'OK' ? 'healthy' : source.status === 'EMPTY' ? 'empty' : 'failed'} /><strong>{source.source_name}</strong><span>{source.status}</span><b>{source.article_count} articles · {source.latency_ms}ms</b><small>{source.error || compactDateTime(source.fetched_at)}</small></div>)}
          </div>
        </Panel>
      </div>
    </section>
  )
}

function FnoTable({ rows, onSelect }: { rows: FnoUnderlying[]; onSelect: (symbol: string) => void }) {
  return (
    <div className="fno-table wide-table">
      <div className="fno-head"><span>UNDERLYING</span><span>COMPANY</span><span>NEAREST FUTURE</span><span>EXPIRY</span><span>LOT</span><span>CONTRACTS</span></div>
      {rows.length === 0 && <div className="empty-row">No mapped F&O underlyings. Refresh after Zerodha login or inspect the instrument cache.</div>}
      {rows.map((row) => <button type="button" className="fno-row" key={row.symbol} onClick={() => onSelect(row.symbol)}><strong>{row.symbol}</strong><span>{row.company_name}</span><span>{row.future_symbol}</span><span>{row.expiry || '—'}</span><span>{row.lot_size}</span><span>{row.contract_count}</span></button>)}
    </div>
  )
}

export function FnoView({ dashboard, runControl, setSelected, setActive }: Props) {
  const [query, setQuery] = useState('')
  const rows = useMemo(() => {
    const clean = query.trim().toUpperCase()
    if (!clean) return dashboard.fno.underlyings
    return dashboard.fno.underlyings.filter((row) => row.symbol.includes(clean) || row.company_name.toUpperCase().includes(clean))
  }, [dashboard.fno.underlyings, query])
  const select = (symbol: string) => {
    setSelected?.(symbol)
    setActive?.('Stock Intelligence')
  }
  return (
    <section className="workspace-view">
      <div className="inline-actions">
        <button type="button" onClick={() => void runControl('REFRESH_FNO_NOW')}>Refresh F&O instruments</button>
        <input className="inline-search" placeholder="Search underlying…" value={query} onChange={(event: { target: { value: string } }) => setQuery(event.target.value)} />
      </div>
      <div className="view-metrics">
        <MetricCard label="MAPPED STOCKS" value={String(dashboard.fno.mapped_underlyings || 0)} detail={`Source ${dashboard.fno.source || 'unavailable'}`} tone={dashboard.fno.available ? 'green' : 'amber'} />
        <MetricCard label="STOCK UNDERLYINGS" value={String(dashboard.fno.unique_stock_underlyings || 0)} detail="Unique current NFO stock names" />
        <MetricCard label="FUTURE CONTRACTS" value={String(dashboard.fno.total_future_contracts || 0)} detail={`${dashboard.fno.index_future_contracts || 0} index contracts excluded`} tone="purple" />
        <MetricCard label="EXCLUSIONS" value={String(dashboard.fno.exclusions.length)} detail="Unmapped names remain explicit" tone="amber" />
      </div>
      <div className="fno-layout">
        <Panel title={`CURRENT F&O UNIVERSE · ${rows.length}`} subtitle="Cash-equity mapping with nearest future metadata"><FnoTable rows={rows} onSelect={select} /></Panel>
        <Panel title="MAPPING EXCLUSIONS" subtitle="Nothing silently disappears"><div className="exclusion-list">{dashboard.fno.exclusions.length === 0 && <div className="empty-row">No exclusions recorded.</div>}{dashboard.fno.exclusions.slice(0, 100).map((item, index) => <div key={`${item.underlying}-${index}`}><strong>{item.underlying}</strong><span>{words(item.stage)}</span><p>{item.reason}</p></div>)}</div></Panel>
      </div>
    </section>
  )
}
