import { useEffect, useMemo, useState } from 'react'
import type {
  ControlName,
  DashboardPayload,
  FnoUnderlying,
  NewsArticle,
  OptionsChainPayload,
  OptionsEodHistoryPayload,
} from './types'
import { compactDateTime, words } from './format'
import { MetricCard, Panel } from './components'
import { fetchMarketOptions, fetchOptionsEodHistory } from './api'

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
          <span>PID {dashboard.operations.worker_pid || '—'} · research jobs run independently from paper execution</span>
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

const canonicalCategory = (article: NewsArticle) => {
  const text = `${article.category} ${article.event_type} ${(article.tags || []).join(' ')}`.toLowerCase()
  if (/(result|order|contract|promoter|insider|fund rais|company|corporate|dividend|merger|acquisition)/.test(text)) return 'Company'
  if (/(economy|macro|inflation|gdp|rate|rbi|currency|bond)/.test(text)) return 'Economy'
  if (/(regulation|sebi|policy|tax|government|court)/.test(text)) return 'Regulation'
  if (/(derivative|future|option|f&o|expiry|margin)/.test(text)) return 'Derivatives'
  if (/(global|us |china|europe|fed|geopolit)/.test(text)) return 'Global'
  return 'Market'
}

function NewsCard({ article, openSymbol }: { article: NewsArticle; openSymbol: (symbol: string) => void }) {
  return (
    <article className="news-card">
      <header>
        <div><span>{canonicalCategory(article)} · {words(article.event_type)}</span><strong>{article.impact_score}</strong></div>
        <time>{compactDateTime(article.published_at || article.fetched_at)}</time>
      </header>
      <h3>{article.headline}</h3>
      <p>{article.why_it_matters || article.summary || 'No verified impact explanation was recorded.'}</p>
      <div className="news-meta"><span>{article.source}</span><span>{article.official ? 'Official source' : `Source tier ${article.source_tier}`}</span><span>{article.corroboration_count} corroborating source(s)</span></div>
      <div className="news-symbols">
        {article.mentioned_symbols.slice(0, 8).map((symbol) => <button type="button" key={symbol} onClick={() => openSymbol(symbol)}>{symbol}</button>)}
        {article.fno_symbols.length > 0 && <em>F&O linked: {article.fno_symbols.slice(0, 6).join(', ')}</em>}
      </div>
      {article.url && <a href={article.url} target="_blank" rel="noreferrer">Open original source ↗</a>}
    </article>
  )
}

export function NewsView({ dashboard, runControl, setSelected, setActive }: Props) {
  const [category, setCategory] = useState('All')
  const [importantOnly, setImportantOnly] = useState(false)
  const categories = useMemo(() => {
    const present = new Set<string>(dashboard.news.articles.map(canonicalCategory))
    const availableCategories: string[] = ['Company', 'Economy', 'Regulation', 'Derivatives', 'Global', 'Market']
    return ['All', ...availableCategories.filter((item) => present.has(item))]
  }, [dashboard.news.articles])
  const articles = useMemo(() => dashboard.news.articles.filter((item) => {
    if (category !== 'All' && canonicalCategory(item) !== category) return false
    if (importantOnly && item.impact_score < 70) return false
    return true
  }), [category, dashboard.news.articles, importantOnly])
  const openSymbol = (symbol: string) => {
    setSelected?.(symbol)
    setActive?.('Stock Intelligence')
  }
  const health = dashboard.news.source_health
  const healthy = health.filter((item) => item.status === 'OK').length
  const failed = health.filter((item) => item.status !== 'OK').length
  return (
    <section className="workspace-view">
      <div className="feature-purpose">
        <strong>What this page is for</strong>
        <p>Use news to understand dated events, source quality and which stocks may need review. Do not use a headline as a buy or sell instruction.</p>
      </div>
      <div className="inline-actions">
        <button type="button" onClick={() => void runControl('REFRESH_NEWS_NOW')}>Refresh news and filings</button>
        <button type="button" onClick={() => setImportantOnly((value) => !value)}>{importantOnly ? 'Show every impact level' : 'Show impact 70+ only'}</button>
      </div>
      <div className="view-metrics">
        <MetricCard label="24H ARTICLES" value={String(dashboard.news.stats.total || 0)} detail={`${dashboard.news.stats.important || 0} high impact`} />
        <MetricCard label="SOURCE HEALTH" value={`${healthy}/${health.length || 0}`} detail={`${failed} source(s) empty or failed`} tone={healthy ? 'green' : 'amber'} />
        <MetricCard label="F&O LINKED" value={String(dashboard.news.stats.fno_linked || 0)} detail="Articles mapped to current derivative underlyings" tone="purple" />
        <MetricCard label="LATEST REFRESH" value={String(dashboard.news.latest_refresh?.status || 'NOT RUN')} detail={String(dashboard.news.latest_refresh?.error_message || dashboard.news.latest_refresh?.message || 'Run refresh to inspect every source')} />
      </div>
      <div className="mode-tabs">{categories.map((item) => <button type="button" key={item} className={category === item ? 'active' : ''} onClick={() => setCategory(item)}>{item}</button>)}</div>
      <div className="news-layout">
        <Panel title={`CURATED MARKET CONTEXT · ${articles.length}`} subtitle="Every article keeps its source, date, impact and entity mapping">
          <div className="news-feed">{articles.length ? articles.map((article) => <NewsCard key={article.article_id} article={article} openSymbol={openSymbol} />) : <div className="large-empty">No article matches this view. Refresh the store, then inspect the source-health panel for the exact provider failure.</div>}</div>
        </Panel>
        <Panel title="SOURCE HEALTH" subtitle="Provider failures remain visible">
          <div className="source-health-list">
            {health.length === 0 && <div className="empty-row">No source-health observations exist yet. Run Refresh news and filings.</div>}
            {health.map((source) => <div key={source.source_key}><i className={source.status === 'OK' ? 'healthy' : source.status === 'EMPTY' ? 'empty' : 'failed'} /><strong>{source.source_name}</strong><span>{source.status}</span><b>{source.article_count} articles · {source.latency_ms}ms</b><small>{source.error || compactDateTime(source.fetched_at)}</small></div>)}
          </div>
        </Panel>
      </div>
    </section>
  )
}

const INDEX_QUICK = ['NIFTY', 'BANKNIFTY', 'FINNIFTY'] as const

function FnoTable({
  rows,
  selected,
  onSelect,
}: {
  rows: FnoUnderlying[]
  selected: string
  onSelect: (symbol: string) => void
}) {
  return (
    <div className="fno-table wide-table">
      <div className="fno-head"><span>UNDERLYING</span><span>COMPANY</span><span>NEAREST FUTURE</span><span>EXPIRY</span><span>LOT</span><span>CONTRACTS</span></div>
      {rows.length === 0 && <div className="empty-row">No mapped stock derivatives. Refresh after Zerodha login or inspect the instrument-cache failure.</div>}
      {rows.map((row) => (
        <button
          type="button"
          className={`fno-row ${selected === row.symbol ? 'selected' : ''}`}
          key={row.symbol}
          onClick={() => onSelect(row.symbol)}
        >
          <strong>{row.symbol}</strong>
          <span>{row.company_name}</span>
          <span>{row.future_symbol}</span>
          <span>{row.expiry || '—'}</span>
          <span>{row.lot_size}</span>
          <span>{row.contract_count}</span>
        </button>
      ))}
    </div>
  )
}

function ChainContextPanel({
  symbol,
  coverage,
  chain,
  history,
  loading,
  onRetry,
  onOpenIntelligence,
}: {
  symbol: string
  coverage: FnoUnderlying | null
  chain: OptionsChainPayload | null
  history: OptionsEodHistoryPayload | null
  loading: boolean
  onRetry: () => void
  onOpenIntelligence: () => void
}) {
  const biasTone = (chain?.bias || '').toLowerCase() === 'bullish'
    ? 'green'
    : (chain?.bias || '').toLowerCase() === 'bearish'
      ? 'amber'
      : 'purple'
  return (
    <div className="fno-chain-stack">
      <Panel
        title={`CHAIN CONTEXT · ${symbol || '—'}`}
        subtitle={
          coverage
            ? `Mapped future ${coverage.future_symbol} · lot ${coverage.lot_size} · expiry ${coverage.expiry || '—'}`
            : INDEX_QUICK.includes(symbol as (typeof INDEX_QUICK)[number])
              ? 'Index option chain (no stock-future mapping required)'
              : 'Select a mapped underlying or an index quick-pick'
        }
      >
        <div className="inline-actions" style={{ marginBottom: 10 }}>
          <button type="button" disabled={!symbol || loading} onClick={onRetry}>
            {loading ? 'Loading chain…' : 'Refresh live chain'}
          </button>
          <button type="button" disabled={!symbol} onClick={onOpenIntelligence}>
            Open Stock Intelligence
          </button>
        </div>
        {loading && <div className="empty-row">Fetching nearest-expiry OI / IV / PCR…</div>}
        {!loading && chain && !chain.available && (
          <div className="empty-row">{chain.message || 'Option chain unavailable right now.'}</div>
        )}
        {!loading && chain?.available && (
          <>
            <div className="fno-chain-metrics">
              <MetricCard label="PCR (OI)" value={String(chain.pcr ?? '—')} detail={chain.note || 'Put/call open interest'} tone={biasTone} />
              <MetricCard label="MAX PAIN" value={chain.max_pain != null ? String(chain.max_pain) : '—'} detail={`Expiry ${chain.expiry || '—'}`} />
              <MetricCard label="ATM IV" value={chain.atm_iv != null ? `${chain.atm_iv}%` : '—'} detail={chain.iv_rank != null ? `IV rank ${chain.iv_rank}` : 'Nearest strike'} tone="purple" />
              <MetricCard label="BIAS READ" value={chain.bias || '—'} detail={`${(chain.total_pe_oi || 0).toLocaleString('en-IN')} put OI · ${(chain.total_ce_oi || 0).toLocaleString('en-IN')} call OI`} tone={biasTone} />
            </div>
            {chain.honesty && <p className="fno-honesty">{chain.honesty}</p>}
            <div className="fno-oi-columns">
              <div>
                <strong>Top call OI</strong>
                {(chain.top_call_oi || []).length === 0 && <span className="empty-row">No call OI rows</span>}
                {(chain.top_call_oi || []).map((row) => (
                  <div key={`ce-${row.strike}`} className="fno-oi-row">
                    <span>{row.strike}</span>
                    <b>{Number(row.ce_oi || 0).toLocaleString('en-IN')}</b>
                  </div>
                ))}
              </div>
              <div>
                <strong>Top put OI</strong>
                {(chain.top_put_oi || []).length === 0 && <span className="empty-row">No put OI rows</span>}
                {(chain.top_put_oi || []).map((row) => (
                  <div key={`pe-${row.strike}`} className="fno-oi-row">
                    <span>{row.strike}</span>
                    <b>{Number(row.pe_oi || 0).toLocaleString('en-IN')}</b>
                  </div>
                ))}
              </div>
            </div>
            {(chain.chain || []).length > 0 && (
              <div className="fno-mini-chain">
                <div className="fno-mini-head">
                  <span>STRIKE</span><span>CE OI</span><span>CE IV</span><span>PE IV</span><span>PE OI</span>
                </div>
                {(chain.chain || []).slice(0, 16).map((row) => (
                  <div className="fno-mini-row" key={`row-${row.strike}`}>
                    <strong>{row.strike}</strong>
                    <span>{Number(row.ce_oi || 0).toLocaleString('en-IN')}</span>
                    <span>{row.ce_iv != null ? `${Number(row.ce_iv).toFixed(1)}` : '—'}</span>
                    <span>{row.pe_iv != null ? `${Number(row.pe_iv).toFixed(1)}` : '—'}</span>
                    <span>{Number(row.pe_oi || 0).toLocaleString('en-IN')}</span>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </Panel>
      <Panel title="EOD HISTORY" subtitle="Saved daily PCR / ATM IV (not a live Greek stream)">
        {!history?.available && (
          <div className="empty-row">{history?.message || 'No EOD options history for this symbol yet.'}</div>
        )}
        {history?.available && (
          <div className="fno-eod-list">
            {(history.rows || []).slice(0, 10).map((row) => (
              <div key={`${row.as_of}-${row.expiry}`}>
                <strong>{row.as_of}</strong>
                <span>PCR {row.pcr ?? '—'}</span>
                <span>ATM IV {row.atm_iv != null ? `${row.atm_iv}%` : '—'}</span>
                <span>Max pain {row.max_pain ?? '—'}</span>
                <small>{row.expiry}</small>
              </div>
            ))}
          </div>
        )}
      </Panel>
    </div>
  )
}

export function FnoView({ dashboard, runControl, setSelected, setActive }: Props) {
  const [query, setQuery] = useState('')
  const [focus, setFocus] = useState('NIFTY')
  const [chain, setChain] = useState<OptionsChainPayload | null>(null)
  const [history, setHistory] = useState<OptionsEodHistoryPayload | null>(null)
  const [loading, setLoading] = useState(false)
  const [reloadToken, setReloadToken] = useState(0)

  const rows = useMemo(() => {
    const clean = query.trim().toUpperCase()
    if (!clean) return dashboard.fno.underlyings
    return dashboard.fno.underlyings.filter((row) => row.symbol.includes(clean) || row.company_name.toUpperCase().includes(clean))
  }, [dashboard.fno.underlyings, query])

  const coverage = useMemo(
    () => dashboard.fno.underlyings.find((row) => row.symbol === focus) || null,
    [dashboard.fno.underlyings, focus],
  )

  useEffect(() => {
    if (!focus) return
    let cancelled = false
    setLoading(true)
    // Always force on desk loads — background warm fail-backoff must not blank the chain.
    Promise.all([
      fetchMarketOptions(focus, true),
      fetchOptionsEodHistory(focus, 14),
    ])
      .then(([live, eod]) => {
        if (cancelled) return
        setChain(live)
        setHistory(eod)
      })
      .catch((reason) => {
        if (cancelled) return
        setChain({
          available: false,
          symbol: focus,
          message: reason instanceof Error ? reason.message : 'Option chain request failed',
        })
        setHistory(null)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [focus, reloadToken])

  const select = (symbol: string) => {
    setFocus(symbol)
    setSelected?.(symbol)
  }
  const generatedAt = dashboard.fno.generated_at ? new Date(Number(dashboard.fno.generated_at) * 1000).toLocaleString('en-IN') : 'unknown'
  return (
    <section className="workspace-view">
      <div className="feature-purpose">
        <strong>What F&O Coverage does—and does not do</strong>
        <p>
          It maps which stocks have current futures contracts (nearest future, expiry, lot size),
          then loads the live nearest-expiry option chain for a selected name: OI, IV, PCR and max pain,
          plus any saved EOD history. It does <em>not</em> compute Black-Scholes Greeks or issue trade
          direction — PCR bias is positioning context only, not a buy/sell desk.
        </p>
      </div>
      <div className="inline-actions">
        <button type="button" onClick={() => void runControl('REFRESH_FNO_NOW')}>Refresh instrument master</button>
        {INDEX_QUICK.map((sym) => (
          <button
            type="button"
            key={sym}
            className={focus === sym ? 'active' : ''}
            onClick={() => select(sym)}
          >
            {sym}
          </button>
        ))}
        <input className="inline-search" placeholder="Search underlying…" value={query} onChange={(event: { target: { value: string } }) => setQuery(event.target.value)} />
      </div>
      <div className="view-metrics">
        <MetricCard label="MAPPED STOCKS" value={String(dashboard.fno.mapped_underlyings || 0)} detail={`Source ${dashboard.fno.source || 'unavailable'} · as of ${generatedAt}`} tone={dashboard.fno.available ? 'green' : 'amber'} />
        <MetricCard label="STOCK UNDERLYINGS" value={String(dashboard.fno.unique_stock_underlyings || 0)} detail="Unique current stock derivative names" />
        <MetricCard label="FUTURE CONTRACTS" value={String(dashboard.fno.total_future_contracts || 0)} detail={`${dashboard.fno.index_future_contracts || 0} index contracts kept separate`} tone="purple" />
        <MetricCard label="MAPPING GAPS" value={String(dashboard.fno.exclusions.length)} detail="Names that could not be safely mapped" tone="amber" />
      </div>
      <div className="fno-desk-layout">
        <Panel title={`CURRENT STOCK-DERIVATIVES COVERAGE · ${rows.length}`} subtitle="Click a stock to load its option-chain context on this page">
          <FnoTable rows={rows} selected={focus} onSelect={select} />
        </Panel>
        <ChainContextPanel
          symbol={focus}
          coverage={coverage}
          chain={chain}
          history={history}
          loading={loading}
          onRetry={() => setReloadToken((n) => n + 1)}
          onOpenIntelligence={() => {
            if (!focus) return
            setSelected?.(focus)
            setActive?.('Stock Intelligence')
          }}
        />
      </div>
      <Panel title="MAPPING GAPS" subtitle="Nothing silently disappears">
        <div className="exclusion-list">
          {dashboard.fno.exclusions.length === 0 && <div className="empty-row">No mapping exclusions recorded.</div>}
          {dashboard.fno.exclusions.slice(0, 100).map((item, index) => (
            <div key={`${item.underlying}-${index}`}>
              <strong>{item.underlying}</strong>
              <span>{words(item.stage)}</span>
              <p>{item.reason}</p>
            </div>
          ))}
        </div>
      </Panel>
    </section>
  )
}
