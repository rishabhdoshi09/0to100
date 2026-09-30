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

const componentLabels: Record<string, string> = {
  breakout: 'Breakout', trend: 'Trend', rvol: 'RVOL', rsi: 'RSI', adx: 'ADX',
  relative_strength: 'Rel. strength', sector_strength: 'Sector strength',
  nifty_alignment: 'NIFTY align', futures_oi: 'Futures OI',
  delta_fit: 'Delta fit', liquidity: 'Liquidity', theta: 'Theta', iv: 'IV',
  expiry_fit: 'Expiry fit', expected_payoff: 'Expected payoff',
}

function componentsRationale(components: Record<string, number> | undefined): string {
  if (!components || Object.keys(components).length === 0) return ''
  return Object.entries(components)
    .sort((a, b) => b[1] - a[1])
    .map(([key, value]) => `${componentLabels[key] || words(key)} ${value}`)
    .join(' · ')
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

function FnoTable({ rows, onSelect }: { rows: FnoUnderlying[]; onSelect: (symbol: string) => void }) {
  return (
    <div className="fno-table wide-table">
      <div className="fno-head"><span>UNDERLYING</span><span>COMPANY</span><span>NEAREST FUTURE</span><span>EXPIRY</span><span>LOT</span><span>CONTRACTS</span></div>
      {rows.length === 0 && <div className="empty-row">No mapped stock derivatives. Refresh after Zerodha login or inspect the instrument-cache failure.</div>}
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
  const generatedAt = dashboard.fno.generated_at ? new Date(Number(dashboard.fno.generated_at) * 1000).toLocaleString('en-IN') : 'unknown'
  const directional = dashboard.fno.directional || {}
  const candidates = directional.candidates || []
  const paper = dashboard.fno.paper || {}
  const openPaper = paper.open_positions || []
  const learning = dashboard.fno.learning_impact || {}
  const ranking = learning.ranking_impact || {}
  const fmt = (value: number | null | undefined, digits = 2) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—'
  return (
    <section className="workspace-view">
      <div className="feature-purpose">
        <strong>NSE F&O directional options lab — PAPER ONLY</strong>
        <p>QuantTerm now combines breakout/breakdown, VWAP, EMA20/50, RSI, ADX, relative/sector strength, NIFTY direction and futures OI, then ranks eligible CE/PE contracts using Greeks, liquidity, spread, IV, DTE and modelled payoff. Setup and option scores are quality scores—not win probabilities. Live execution remains disabled.</p>
      </div>
      <div className="inline-actions">
        <button type="button" onClick={() => void runControl('REFRESH_FNO_NOW')}>Refresh F&O scan</button>
        <input className="inline-search" placeholder="Search underlying…" value={query} onChange={(event: { target: { value: string } }) => setQuery(event.target.value)} />
      </div>
      <div className="view-metrics">
        <MetricCard label="MAPPED STOCKS" value={String(dashboard.fno.mapped_underlyings || 0)} detail={`Source ${dashboard.fno.source || 'unavailable'} · as of ${generatedAt}`} tone={dashboard.fno.available ? 'green' : 'amber'} />
        <MetricCard label="PAPER CANDIDATES" value={String(directional.candidate_count || 0)} detail={directional.status ? `${directional.status} · quality score ≠ probability` : 'Run F&O refresh for directional evidence'} tone={(directional.candidate_count || 0) > 0 ? 'green' : 'amber'} />
        <MetricCard label="OPEN OPTION PAPERS" value={String(openPaper.length)} detail={`Realized ₹${fmt(paper.status?.realized_pnl)} · live money locked`} tone="purple" />
        <MetricCard label="PRODUCTION EVIDENCE" value={paper.production_evidence_enabled ? 'ENABLED' : 'HELD'} detail={paper.production_evidence_enabled ? 'Fully-costed settled evidence exists' : 'Uncosted/gross-only trades cannot promote models'} tone={paper.production_evidence_enabled ? 'green' : 'amber'} />
      </div>

      <div className="fno-layout">
        <Panel title={`TOP DIRECTIONAL PAPER CANDIDATES · ${candidates.length}`} subtitle="Ranked by setup quality and contract quality; scenarios use constant-IV estimates">
          <div className="exclusion-list">
            {candidates.length === 0 && <div className="empty-row">{directional.code ? `${directional.code} · ` : ''}{directional.decision || 'No eligible directional option candidate in the latest scan.'}</div>}
            {candidates.slice(0, 5).map((candidate, index) => {
              const setup = candidate.setup || {}
              const contract = candidate.selected_contract || {}
              const plan = contract.trade_plan || {}
              const underlyingPlan = setup.underlying_trade_plan || {}
              const expected = setup.expected_move || {}
              const evidence = candidate.forward_evidence || {}
              const coverage = evidence.coverage || {}
              const ivHistory = candidate.iv_history || {}
              const evidenceReady = evidence.probability_claim_available === true
              const cmp = candidate.options?.spot
              const matchedPosition = openPaper.find((position) => (
                (contract.symbol && position.option_symbol === contract.symbol)
                || (position.underlying === candidate.symbol && position.option_type === contract.option_type)
              ))
              const rationale = componentsRationale(contract.components)
              const setupRationale = componentsRationale(setup.components)
              return (
                <button type="button" key={`${candidate.symbol}-${contract.symbol || index}`} onClick={() => select(candidate.symbol)}>
                  <strong>{candidate.symbol} · {candidate.direction} · CMP ₹{fmt(cmp)}</strong>
                  <span>{contract.symbol || contract.option_type || 'No contract'} · setup {fmt(setup.score, 1)}/100 · option {fmt(contract.score, 1)}/100</span>
                  <p>Setup type: {words(candidate.direction || 'unknown')} breakout · futures OI {words(setup.futures_oi_state || 'unknown')} · distance {fmt(setup.breakout_distance_pct, 2)}% · ATR {fmt(setup.atr_pct, 2)}%</p>
                  <small>Underlying plan · entry ₹{fmt(underlyingPlan.entry)} · stop ₹{fmt(underlyingPlan.stop)} · target ₹{fmt(underlyingPlan.target)}</small>
                  <p>{contract.option_type || '—'} {fmt(contract.strike, 0)} · expiry {contract.expiry || 'unavailable'} · DTE {contract.dte ?? '—'} · horizon {words(expected.horizon || 'unknown')}</p>
                  <small>Premium ₹{fmt(contract.premium)} · Δ {fmt(contract.delta, 2)} · IV {fmt(contract.iv, 1)}% · OI {contract.oi ?? '—'} · volume {contract.volume ?? '—'} · spread {contract.spread_pct == null ? 'unavailable' : `${fmt(contract.spread_pct, 2)}%`}</small>
                  {ivHistory.available ? (
                    <small>Forward IV percentile {fmt(ivHistory.percentile_pct, 1)}% · {ivHistory.prior_sessions ?? 0} prior closing sessions · no backfill</small>
                  ) : (
                    <small>Forward IV percentile held · {ivHistory.prior_sessions ?? 0}/{ivHistory.minimum_prior_sessions ?? 60} prior closing sessions · no backfill</small>
                  )}
                  <small>Paper entry ₹{fmt(plan.entry)} · stop ₹{fmt(plan.stop)} · target ₹{fmt(plan.target)} · R:R {fmt(plan.risk_reward, 2)}</small>
                  {rationale && <small>Option-selection rationale · {rationale}</small>}
                  {setupRationale && <small>Setup rationale · {setupRationale}</small>}
                  {matchedPosition ? (
                    <small className="fno-paper-state">
                      Paper state: OPEN · bars held {matchedPosition.bars_held ?? 0}/{matchedPosition.max_holding_sessions ?? '—'}
                      {matchedPosition.trailing_stop_price ? ` · trailing stop ₹${fmt(matchedPosition.trailing_stop_price)}` : ''}
                      {matchedPosition.max_mark ? ` · best mark ₹${fmt(matchedPosition.max_mark)}` : ''}
                    </small>
                  ) : (
                    <small className="fno-paper-state">Paper state: not yet executed (awaiting next paper cycle, entry window or risk gate)</small>
                  )}
                  {evidenceReady ? (
                    <small>Forward evidence · n={evidence.n ?? 0} fully costed + path-valid · observed win {fmt(evidence.win_probability_pct, 1)}% · Wilson floor {fmt(evidence.win_probability_wilson_lb_pct, 1)}% · conservative EV {fmt(evidence.conservative_ev_pct, 2)}%</small>
                  ) : (
                    <small>Forward evidence · {words(evidence.status || 'uncalibrated')} · observed {evidence.observed_n ?? 0} · eligible {evidence.n ?? 0}/{evidence.minimum_required_n ?? 30} · probability held</small>
                  )}
                  {((evidence.excluded_path_observation ?? 0) > 0 || (evidence.excluded_unpriced_costs ?? 0) > 0) && (
                    <small>Held out from probability · path uncertainty {evidence.excluded_path_observation ?? 0} · unpriced costs {evidence.excluded_unpriced_costs ?? 0}</small>
                  )}
                  {coverage.valid_context && (
                    <small>Evidence coverage · thesis {coverage.thesis_n ?? 0} · dir+OI {coverage.direction_oi_n ?? 0} · direction {coverage.direction_n ?? 0} · broader counts research only</small>
                  )}
                </button>
              )
            })}
          </div>
        </Panel>

        <Panel title={`F&O PAPER BOOK · ${openPaper.length} OPEN`} subtitle="Durable simulated positions only; broker order mutations are unavailable">
          <div className="exclusion-list">
            {openPaper.length === 0 && <div className="empty-row">No open option paper positions.</div>}
            {openPaper.slice(0, 10).map((position, index) => (
              <div key={position.trade_id || `${position.option_symbol}-${index}`}>
                <strong>{position.underlying || '—'} · {position.option_symbol || '—'}</strong>
                <span>{position.option_type || '—'} {fmt(position.strike, 0)} · expiry {position.expiry || 'unavailable'} · qty {position.quantity ?? '—'} · setup {fmt(position.setup_score, 1)} · option {fmt(position.option_score, 1)}</span>
                <p>Entry ₹{fmt(position.entry_price)} · stop ₹{fmt(position.stop_price)}{position.trailing_stop_price ? ` (trailing ₹${fmt(position.trailing_stop_price)})` : ''} · target ₹{fmt(position.target_price)} · max hold {position.max_holding_sessions ?? '—'} session(s)</p>
                <small>Supervision · bars held {position.bars_held ?? 0}/{position.max_holding_sessions ?? '—'} · best mark ₹{fmt(position.max_mark)} · worst mark ₹{fmt(position.min_mark)} · entry status {words(position.entry_minute_status || 'unknown')}</small>
              </div>
            ))}
          </div>
        </Panel>
      </div>

      <div className="fno-layout">
        <Panel title="F&O LEARNING LOOP · HISTORICAL (COUNTERFACTUAL)" subtitle="Point-in-time walk-forward replay over real historical OHLC — establishes priors only, can never move ranking">
          {(() => {
            const hist = learning.historical || {}
            if (!hist.available) return <div className="empty-row">{hist.error ? `Unavailable: ${hist.error}` : 'No historical walk-forward run has completed yet.'}</div>
            return (
              <div className="exclusion-list">
                <div>
                  <strong>{hist.historical_simulations_completed ?? 0} session(s) simulated · {hist.decisions_graded ?? 0} decisions graded · {hist.settled ?? 0} settled</strong>
                  <span>Cursor {hist.cursor_date || '—'} · {hist.coverage_complete ? 'fully caught up to available history' : 'still catching up on backlog'}</span>
                  <p>Correct rejects {hist.correct_rejects ?? 0} · missed winners {hist.missed_winners ?? 0} · avoided losers {hist.avoided_losers ?? 0} · ran away without entry {hist.ran_away_without_entry ?? 0}</p>
                  <small>{hist.evidence_cells ?? 0} evidence cell(s) · can affect ranking: {hist.can_affect_ranking ? 'yes' : 'no'}</small>
                  <small>{hist.note}</small>
                </div>
              </div>
            )
          })()}
        </Panel>
        <Panel title="F&O LEARNING LOOP · FORWARD (PAPER_FORWARD)" subtitle="Real settled F&O paper trades — the only evidence class that can change ranking">
          {(() => {
            const fwd = learning.forward || {}
            if (!fwd.available) return <div className="empty-row">{fwd.error ? `Unavailable: ${fwd.error}` : 'No forward paper evidence yet.'}</div>
            return (
              <div className="exclusion-list">
                <div>
                  <strong>{fwd.forward_paper_trades ?? 0} forward paper trade(s) · {fwd.wins ?? 0} win / {fwd.losses ?? 0} loss</strong>
                  <span>{fwd.open_positions ?? 0} open · {fwd.production_evidence_trades ?? 0} fully-costed production-evidence trade(s)</span>
                  <p>{fwd.evidence_cells ?? 0} evidence cell(s) · {fwd.matured_cells ?? 0} matured (≥{fwd.minimum_sample_for_ranking ?? 30} samples)</p>
                  <small>Can affect ranking: {fwd.can_affect_ranking ? 'yes' : 'no'}</small>
                  <small>{fwd.note}</small>
                </div>
              </div>
            )
          })()}
        </Panel>
      </div>
      <div className="fno-layout">
        <Panel title="RANKING IMPACT" subtitle="Only claimed when a real candidate's ranking actually moved this cycle">
          <div className="exclusion-list">
            <div>
              <strong>{ranking.plain || 'Learning impact is being measured.'}</strong>
              <span>Status: {words(ranking.status || 'COLLECTING')}</span>
            </div>
            {(ranking.influenced || []).length === 0 && <div className="empty-row">No candidate's ranking has been changed by validated evidence yet.</div>}
            {(ranking.influenced || []).map((row, index) => (
              <div key={`${row.symbol}-${index}`}>
                <strong>{row.symbol} · {row.direction}</strong>
                <span>Setup demoted after negative forward expectancy · base {fmt(row.base_score, 1)} → ranked {fmt(row.ranking_score, 1)} ({fmt(row.adjustment, 1)})</span>
                <p>{words(row.reason || '')} · n={row.count ?? 0} · expectancy {fmt(row.expectancy_R, 2)}R</p>
              </div>
            ))}
          </div>
        </Panel>
      </div>

      <div className="fno-layout">
        <Panel title={`CURRENT STOCK-DERIVATIVES COVERAGE · ${rows.length}`} subtitle="Click a stock to open its combined cash-market intelligence workspace"><FnoTable rows={rows} onSelect={select} /></Panel>
        <Panel title="MAPPING GAPS" subtitle="Nothing silently disappears"><div className="exclusion-list">{dashboard.fno.exclusions.length === 0 && <div className="empty-row">No mapping exclusions recorded.</div>}{dashboard.fno.exclusions.slice(0, 100).map((item, index) => <div key={`${item.underlying}-${index}`}><strong>{item.underlying}</strong><span>{words(item.stage)}</span><p>{item.reason}</p></div>)}</div></Panel>
      </div>
    </section>
  )
}
