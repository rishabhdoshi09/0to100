import { useEffect, useMemo, useRef, useState } from 'react'
import { ChartWorkspace, Panel } from './components'
import { money, pct, relativeAge, words } from './format'
import {
  addWatchlistItem,
  bootstrapProduct,
  fetchCompareWorkspace,
  fetchProductReadiness,
  fetchRadarHome,
  fetchScannerWorkspace,
  fetchTradePlan,
  fetchWatchlist,
  removeWatchlistItem,
  type CompareWorkspace,
  type ProductReadiness,
  type RadarHome,
  type ScannerWorkspaceRow,
  type TradePlan,
  type WatchlistPayload,
} from './productApi'
import { RiskLensCard } from './productViews'
import { LiveScanBanner, type ExperienceViewProps } from './experience'
import { keepRicher, recall, remember } from './sessionMemory'
import {
  bestSetupsFromRadar,
  scannerEmptyHint,
  scannerFallbackRows,
  scannerMetaFromDashboard,
} from './scannerFallback'

type RadarRow = ScannerWorkspaceRow & {
  breakout_state?: string
  momentum_state?: string
  setup_label?: string
  freshness?: string
  change_5d_pct?: number
  relative_strength?: number
  risk_label?: string
  reason?: string
  company?: string
  classification?: string
  combined_score?: number
  why?: string
  sepa_score?: number
  sepa_passed?: number
  sepa_total?: number
  sepa_verdict?: string
  breakout_grade?: string
  breakout_conviction?: number
  breakout_quality?: number
  fundamental_score?: number
  sniper_candidate?: boolean
  volume_ratio?: number
  rsi?: number
  tech_source?: string
  price_tag?: string
}

const breakoutLabel: Record<string, string> = {
  confirmed_breakout: 'Confirmed',
  near_breakout: 'Near breakout',
  breakout_under_observation: 'Under observation',
  breakout_without_volume: 'No volume confirm',
  insufficient_confirmation: 'Needs confirmation',
  extended_after_breakout: 'Extended',
  failed_breakout: 'Failed',
  failed_or_extended: 'Failed / extended',
  insufficient_data: 'Insufficient data',
  not_in_breakout_lane: '—',
}

const momentumLabel: Record<string, string> = {
  strong_actionable: 'Strong · actionable',
  strong_but_extended: 'Strong · extended',
  steady_leadership: 'Steady leadership',
  improving: 'Improving',
  weakening: 'Weakening',
  high_volatility_momentum: 'High-vol momentum',
  insufficient_history: 'Short history',
  watch_momentum: 'Watch',
  not_momentum: '—',
}

function BestSniperPanel({
  best,
  sniperCount,
  onSelect,
}: {
  best: RadarRow | null | undefined
  sniperCount: number
  onSelect: (symbol: string) => void
}) {
  if (best) {
    const volOk = best.volume_ratio == null || Number(best.volume_ratio) >= 1
    return (
      <div className="radar-best-breakout">
        <Panel
          title={`BEST TECHNICAL BREAKOUT · ${best.symbol}`}
          subtitle={
            [
              sniperCount > 0 ? `${sniperCount} sniper candidate${sniperCount === 1 ? '' : 's'}` : null,
              best.breakout_grade ? `Grade ${best.breakout_grade}` : null,
              best.rsi != null
                ? `RSI ${Math.round(Number(best.rsi))}${best.tech_source === 'live' || best.price_tag === 'LIVE' ? ' LIVE' : ' EOD'}`
                : null,
              best.volume_ratio != null
                ? `Vol ${Number(best.volume_ratio).toFixed(1)}×${volOk ? '' : ' THIN'}`
                : null,
            ].filter(Boolean).join(' · ') || 'Volume ≥1× · not chasing · RSI ≤82 — fundamentals not required'
          }
        >
          <button
            type="button"
            className="radar-best-pick-btn"
            onClick={() => onSelect(String(best.symbol || ''))}
          >
            Score {best.score ?? '—'}
            {best.breakout_quality != null
              ? ` · Quality ${Number(best.breakout_quality).toFixed(0)}`
              : ''}
            {' · '}
            {breakoutLabel[String(best.breakout_state || '')]
              || words(String(best.breakout_state || best.status || ''))}
          </button>
        </Panel>
      </div>
    )
  }
  return (
    <div className="radar-best-breakout radar-best-empty">
      <Panel
        title="BEST TECHNICAL BREAKOUT"
        subtitle="Volume ≥1.0× · not extended · RSI ≤82 — tape only, no fund gate"
      >
        <p className="radar-empty-li">
          {sniperCount === 0
            ? 'No sniper breakouts yet — thin volume / extended names stay out.'
            : 'Sniper pool has names but none ranked as technical best.'}
        </p>
      </Panel>
    </div>
  )
}

function BestAmongFundamentalsPanel({
  best,
  onSelect,
}: {
  best: RadarRow | null | undefined
  onSelect: (symbol: string) => void
}) {
  if (best) {
    const gates = (best as RadarRow & { quality_gates?: Record<string, string> }).quality_gates || {}
    return (
      <div className="radar-best-fundamentals">
        <Panel
          title={`BEST AMONG BREAKOUTS · ${best.symbol}`}
          subtitle={
            [
              'Fundamentals filter',
              best.classification ? String(best.classification).replace(/_/g, ' ') : null,
              best.fundamental_score != null ? `Fund score ${Math.round(Number(best.fundamental_score))}` : null,
              gates.fundamentals ? `Fund ${gates.fundamentals}` : null,
              best.rsi != null ? `RSI ${Math.round(Number(best.rsi))}` : null,
              best.volume_ratio != null ? `Vol ${Number(best.volume_ratio).toFixed(1)}×` : null,
            ].filter(Boolean).join(' · ')
          }
        >
          <button
            type="button"
            className="radar-best-pick-btn"
            onClick={() => onSelect(String(best.symbol || ''))}
          >
            Score {best.score ?? '—'}
            {best.breakout_quality != null
              ? ` · Quality ${Number(best.breakout_quality).toFixed(0)}`
              : ''}
            {' · '}
            {breakoutLabel[String(best.breakout_state || '')]
              || words(String(best.breakout_state || best.status || ''))}
          </button>
        </Panel>
      </div>
    )
  }
  return (
    <div className="radar-best-fundamentals radar-best-empty">
      <Panel
        title="BEST AMONG BREAKOUTS"
        subtitle="Only uses fundamentals among already-valid breakout candidates"
      >
        <p className="radar-empty-li">
          No breakout candidate has usable fundamental coverage yet — run long-term scan, or wait for fund data. Technical sniper lane above is independent.
        </p>
      </Panel>
    </div>
  )
}

function thinVolume(row: RadarRow): boolean {
  const vol = Number(row.volume_ratio)
  return Number.isFinite(vol) && vol > 0 && vol < 1
}

function DenseTable({
  rows,
  selected,
  onSelect,
  depth,
  mode,
  emptyHint,
}: {
  rows: RadarRow[]
  selected: string
  onSelect: (symbol: string) => void
  depth: ExperienceViewProps['depth']
  mode: string
  emptyHint: string
}) {
  const [sortKey, setSortKey] = useState('score')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')

  const sorted = useMemo(() => {
    const copy = [...rows]
    copy.sort((a, b) => {
      const av = (a as Record<string, unknown>)[sortKey]
      const bv = (b as Record<string, unknown>)[sortKey]
      const an = typeof av === 'number' ? av : String(av ?? '')
      const bn = typeof bv === 'number' ? bv : String(bv ?? '')
      if (typeof an === 'number' && typeof bn === 'number') {
        return sortDir === 'asc' ? an - bn : bn - an
      }
      return sortDir === 'asc'
        ? String(an).localeCompare(String(bn))
        : String(bn).localeCompare(String(an))
    })
    return copy
  }, [rows, sortKey, sortDir])

  const toggleSort = (key: string) => {
    if (sortKey === key) setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    else { setSortKey(key); setSortDir('desc') }
  }

  const cols = mode === 'Long-Term'
    ? ['symbol', 'classification', 'combined_score', 'sector', 'coverage_pct', 'risk_label']
    : depth === 'professional'
      ? ['symbol', 'price', 'change_5d_pct', 'sector', 'setup_label', 'breakout_state', 'momentum_state', 'relative_strength', 'risk_label']
      : ['symbol', 'price', 'change_5d_pct', 'sector', 'setup_label', 'risk_label']

  return (
    <div className="radar-table-wrap">
      <table className="radar-table">
        <thead>
          <tr>
            {cols.map((col) => (
              <th key={col} onClick={() => toggleSort(col)}>{words(col.replace(/_/g, ' '))}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.length === 0 && (
            <tr><td colSpan={cols.length} className="radar-empty">{emptyHint}</td></tr>
          )}
          {sorted.map((row) => (
            <tr key={row.symbol} className={selected === row.symbol ? 'selected' : ''} onClick={() => onSelect(row.symbol)}>
              {cols.map((col) => {
                const raw = (row as Record<string, unknown>)[col]
                let cell: string
                if (col === 'breakout_state') cell = breakoutLabel[String(raw)] || words(String(raw))
                else if (col === 'momentum_state') cell = momentumLabel[String(raw)] || words(String(raw))
                else if (col === 'price') cell = money(raw as number)
                else if (col === 'change_5d_pct') cell = pct(raw as number)
                else if (col === 'combined_score' || col === 'relative_strength') cell = raw != null ? String(raw) : '—'
                else cell = String(raw ?? '—')
                return <td key={col}>{cell}</td>
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function RadarHomeView(props: ExperienceViewProps & {
  onCompare: (symbol: string) => void
  onWatchlist: (symbol: string) => void
}) {
  const { dashboard, selected, setSelected, bars, setActive, depth, marketScan, longTermScan, onCompare, onWatchlist } = props
  const [radar, setRadar] = useState<RadarHome | null>(() => recall<RadarHome>('radar-home') ?? null)
  const [plan, setPlan] = useState<TradePlan | null>(null)
  const [readiness, setReadiness] = useState<ProductReadiness | null>(() => recall<ProductReadiness>('product-readiness') ?? null)
  const [bootstrapBusy, setBootstrapBusy] = useState(false)
  const [deskNote, setDeskNote] = useState('')
  const autoBootRef = useRef(false)

  useEffect(() => {
    let alive = true
    const load = () => {
      fetchRadarHome()
        .then((payload) => {
          const kept = keepRicher('radar-home', payload, (row) => {
            const counts = (row.counts?.breakouts || 0) + (row.counts?.momentum || 0) + (row.counts?.long_term_picks || 0)
            return counts === 0 && !(row.best_setups || []).length && !row.best_breakout
          })
          if (alive) setRadar(kept)
        })
        .catch(() => { if (alive && !recall('radar-home')) setRadar(null) })
      fetchProductReadiness()
        .then((payload) => {
          remember('product-readiness', payload)
          if (alive) setReadiness(payload)
        })
        .catch(() => undefined)
    }
    load()
    const watching = marketScan.isActive || Boolean(dashboard.scan_progress?.active)
    const timer = window.setInterval(load, watching ? 4000 : 20_000)
    return () => { alive = false; window.clearInterval(timer) }
  }, [dashboard.scan.scanned_at, dashboard.long_term.scanned_at, dashboard.generated_at, dashboard.scan_progress?.updated_at, marketScan.isActive])

  useEffect(() => {
    if (!selected) { setPlan(null); return }
    let alive = true
    fetchTradePlan(selected)
      .then((payload) => { if (alive) setPlan(payload) })
      .catch(() => { if (alive) setPlan(null) })
    return () => { alive = false }
  }, [selected, dashboard.scan.scanned_at, dashboard.generated_at])

  const scanAt = radar?.scan_scanned_at || dashboard.scan.scanned_at || ''
  const kiteOk = dashboard.autonomy.state !== 'AUTH_REQUIRED'
    && !(dashboard.autonomy.active_failures || []).some((f) => String(f).includes('auth'))
  const emptyDesk = !scanAt
    || (radar != null && ((radar.counts.breakouts || 0) + (radar.counts.momentum || 0) + (radar.counts.long_term_picks || 0) === 0))
  const readinessScore = readiness?.score ?? 0
  const needsBootstrap = emptyDesk || readinessScore < 70 || !dashboard.data.ready

  const runBootstrap = async () => {
    setBootstrapBusy(true)
    setDeskNote('Preparing official history, news and scan…')
    try {
      const result = await bootstrapProduct()
      setReadiness(result.readiness)
      setDeskNote(result.message || 'Data lanes queued')
      if (!marketScan.isBusy) void marketScan.start()
    } catch (reason) {
      setDeskNote(reason instanceof Error ? reason.message : 'Could not start data lanes')
    } finally {
      setBootstrapBusy(false)
      window.setTimeout(() => setDeskNote(''), 4000)
    }
  }

  useEffect(() => {
    if (autoBootRef.current) return
    const cached = recall<RadarHome>('radar-home')
    const cachedCount = (cached?.counts.breakouts || 0) + (cached?.counts.momentum || 0) + (cached?.counts.long_term_picks || 0)
    if (cachedCount > 0 || dashboard.scan.scanned_at) {
      autoBootRef.current = true
      return
    }
    const deskNeedsWork = emptyDesk || !dashboard.data.ready
    if (!deskNeedsWork) return
    autoBootRef.current = true
    void runBootstrap()
  }, [emptyDesk, dashboard.data.ready, dashboard.scan.scanned_at])

  const laneCard = (title: string, rows: RadarRow[], count: number, qualityHint?: number) => (
    <section className="radar-lane-card">
      <header>
        <span>{title}</span>
        <strong>
          {count}
          {qualityHint != null ? ` · ${qualityHint} sniper` : ''}
        </strong>
      </header>
      <ul>
        {rows.slice(0, 6).map((item) => {
          const thin = thinVolume(item)
          return (
          <li key={item.symbol}>
            <button
              type="button"
              className={[selected === item.symbol ? 'active' : '', thin ? 'thin-volume' : ''].filter(Boolean).join(' ')}
              onClick={() => setSelected(item.symbol)}
            >
              <b>
                {item.symbol}
                {item.sniper_candidate ? <em className="sniper-tag"> SNIPER</em> : null}
                {thin ? <em className="thin-tag"> THIN VOL</em> : null}
              </b>
              <span>{thin ? 'No volume confirm' : (item.setup_label || item.status)}</span>
              <small>
                {item.sector}
                {item.volume_ratio != null ? ` · ${Number(item.volume_ratio).toFixed(1)}×` : ''}
                {item.rsi != null ? ` · RSI ${Math.round(Number(item.rsi))}` : ''}
                {' · '}
                {item.reason?.slice(0, 36) || '—'}
              </small>
            </button>
          </li>
          )
        })}
        {rows.length === 0 && (
          <li className="radar-empty-li">
            Preparing official history and scan…
          </li>
        )}
      </ul>
    </section>
  )

  const row = radar?.lanes.breakouts.find((r) => r.symbol === selected)
    || radar?.lanes.momentum.find((r) => r.symbol === selected)
    || radar?.lanes.long_term_picks.find((r) => r.symbol === selected)

  return (
    <section className="radar-home">
      <header className="radar-hero">
        <div>
          <span>MARKET DESK</span>
          <h2>{radar?.market_health || dashboard.market.health}</h2>
          <p>{dashboard.market.summary}</p>
        </div>
        <div className="radar-hero-actions">
          {needsBootstrap && (
            <button type="button" disabled={bootstrapBusy} onClick={() => void runBootstrap()}>
              {bootstrapBusy ? 'Preparing…' : 'Refresh desk'}
            </button>
          )}
          <button type="button" disabled={marketScan.isBusy} onClick={() => void marketScan.start()}>
            {marketScan.isBusy
              ? `Scanning… ${marketScan.percent != null ? `${marketScan.percent}%` : ''}${marketScan.etaLine ? ` · ETA ${marketScan.etaLine}` : ''}`
              : 'Scan now'}
          </button>
        </div>
      </header>

      <div className={`radar-desk-strip ${kiteOk ? '' : 'desk-warn'}`}>
        <div>
          <span>SCAN</span>
          <strong>{relativeAge(scanAt)}</strong>
          <small>{scanAt ? 'signals from last scan · prices refresh live' : 'scan queued on start'}</small>
        </div>
        <div>
          <span>PRICE DATA</span>
          <strong>{dashboard.data.bhavcopy.latest_date || 'Preparing…'}</strong>
          <small>{dashboard.data.ready ? 'official bhavcopy ready' : 'preparing official bhavcopy'}</small>
        </div>
        <div>
          <span>ZERODHA</span>
          <strong>{kiteOk ? 'SESSION OK' : 'LOGIN NEEDED'}</strong>
          <small>
            {kiteOk
              ? 'live quotes / depth available when market is open'
              : (dashboard.autonomy.plain_state || 'python main.py login')}
          </small>
        </div>
        <div>
          <span>NEXT</span>
          <strong>
            {!dashboard.data.ready || readinessScore < 70
              ? 'Preparing data'
              : !scanAt
                ? 'Scanning'
                : selected
                  ? 'Check ₹ risk'
                  : 'Pick one name'}
          </strong>
          <small>{deskNote || dashboard.market.trade_stance}</small>
        </div>
      </div>

      <div className="radar-market-strip">
        <div><span>NIFTY 1D</span><strong>{pct(radar?.nifty_change_1d ?? dashboard.market.nifty_change_1d)}</strong></div>
        <div><span>BREADTH</span><strong>{radar?.breadth || dashboard.market.breadth}</strong></div>
        <div><span>VIX</span><strong>{radar?.vix ?? dashboard.market.vix ?? '—'}</strong></div>
        <div><span>LEADERS</span><strong>{(radar?.leaders || dashboard.market.leaders).slice(0, 3).join(', ') || '—'}</strong></div>
        <div><span>SCAN AGE</span><strong>{relativeAge(scanAt)}</strong></div>
        <div><span>STANCE</span><strong>{dashboard.market.trade_stance?.split(';')[0] || '—'}</strong></div>
      </div>

      <LiveScanBanner scan={marketScan} depth={depth} label="Market scan" />
      {longTermScan.isActive || longTermScan.notice ? (
        <LiveScanBanner scan={longTermScan} depth={depth} label="Long-term scan" />
      ) : null}

      <BestSniperPanel
        best={radar?.best_breakout as RadarRow | null | undefined}
        sniperCount={radar?.counts.sniper_breakouts || radar?.sniper_candidates?.length || 0}
        onSelect={setSelected}
      />

      <BestAmongFundamentalsPanel
        best={radar?.best_among_fundamentals as RadarRow | null | undefined}
        onSelect={setSelected}
      />

      {(radar?.sniper_candidates?.length || 0) > 0 && (
        <section className="radar-sniper-pool">
          <header>
            <span>SNIPER BREAKOUT CANDIDATES</span>
            <strong>{radar?.sniper_candidates?.length}</strong>
          </header>
          <ul>
            {(radar?.sniper_candidates || []).slice(0, 8).map((item) => (
              <li key={item.symbol}>
                <button
                  type="button"
                  className={selected === item.symbol ? 'active' : ''}
                  onClick={() => setSelected(item.symbol)}
                >
                  <b>{item.symbol}</b>
                  <span>
                    {(item as RadarRow).breakout_grade ? `G${(item as RadarRow).breakout_grade}` : '—'}
                    {item.volume_ratio != null ? ` · ${Number(item.volume_ratio).toFixed(1)}×` : ''}
                    {(item as RadarRow).rsi != null
                      ? ` · RSI ${Math.round(Number((item as RadarRow).rsi))}${(item as RadarRow).tech_source === 'live' || (item as RadarRow).price_tag === 'LIVE' ? ' LIVE' : ''}`
                      : ''}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      <div className="radar-three-lanes">
        {laneCard(
          'Breakouts',
          radar?.lanes.breakouts || [],
          radar?.counts.breakouts || 0,
          radar?.counts.sniper_breakouts,
        )}
        {laneCard('Momentum', radar?.lanes.momentum || [], radar?.counts.momentum || 0)}
        {laneCard('Long-Term Picks', radar?.lanes.long_term_picks || [], radar?.counts.long_term_picks || 0)}
      </div>

      <div className="radar-workspace">
        <Panel title={`CHART · ${selected || 'SELECT STOCK'}`} subtitle={`Official history · ${dashboard.data.bhavcopy.latest_date || '—'}`}>
          <ChartWorkspace symbol={selected} bars={bars} row={row} />
        </Panel>
        <Panel title="DECISION PREVIEW" subtitle="Risk before reward · read-only">
          {selected ? (
            <div className="radar-decision-preview">
              <p><strong>{(row as RadarRow)?.reason || plan?.summary || 'Select a stock from a lane above.'}</strong></p>
              {plan?.entry != null && <div>Entry zone: {money(plan.entry)}</div>}
              {plan?.stop != null && <div>Invalidation: {money(plan.stop)}</div>}
              {plan?.target != null && <div>Target: {money(plan.target)}</div>}
              <RiskLensCard plan={plan} />
              <div className="radar-action-row">
                <button type="button" onClick={() => setActive('Stock Intelligence')}>Full research</button>
                <button type="button" onClick={() => onCompare(selected)}>Compare</button>
                <button type="button" onClick={() => onWatchlist(selected)}>Watchlist</button>
              </div>
            </div>
          ) : (
            <p className="radar-empty-li">Pick one name above — then check ₹ risk here before opening research.</p>
          )}
        </Panel>
      </div>
    </section>
  )
}

export function MarketScannerView(props: ExperienceViewProps & { onCompare: (symbol: string) => void }) {
  const { dashboard, selected, setSelected, bars, setActive, depth, marketScan, longTermScan, onCompare } = props
  const [tab, setTab] = useState<'Best Setups' | 'Breakouts' | 'Momentum' | 'Long-Term'>('Best Setups')
  const [rows, setRows] = useState<RadarRow[]>(() => {
    const cached = recall<RadarRow[]>('scanner:Best Setups')
    if (cached?.length) return cached
    return scannerFallbackRows('Best Setups', dashboard) as RadarRow[]
  })
  const [meta, setMeta] = useState(() => {
    const cached = recall<{ scanned_at: string; universe: number }>('scanner-meta:Best Setups')
    if (cached?.universe || cached?.scanned_at) return cached
    return scannerMetaFromDashboard('Best Setups', dashboard)
  })
  const [search, setSearch] = useState('')
  const [sector, setSector] = useState('All')
  const [excludeChase, setExcludeChase] = useState(true)

  const activeScan = tab === 'Long-Term' ? longTermScan : marketScan
  const hasScan = Boolean(dashboard.scan.scanned_at || dashboard.scan.records.length || meta.universe || meta.scanned_at)

  useEffect(() => {
    const seed = scannerFallbackRows(tab, dashboard) as RadarRow[]
    const seedMeta = scannerMetaFromDashboard(tab, dashboard)
    const cachedRows = recall<RadarRow[]>(`scanner:${tab}`)
    const cachedMeta = recall<{ scanned_at: string; universe: number }>(`scanner-meta:${tab}`)
    const opening = (cachedRows?.length ? cachedRows : seed)
    if (opening.length) setRows(opening)
    const openingMeta = (cachedMeta?.universe || cachedMeta?.scanned_at) ? cachedMeta : seedMeta
    if (openingMeta.universe || openingMeta.scanned_at) setMeta(openingMeta)

    const apply = (next: RadarRow[], nextMeta: { scanned_at: string; universe: number }) => {
      const kept = keepRicher(`scanner:${tab}`, next, (items) => items.length === 0)
      const metaToKeep = nextMeta.scanned_at || nextMeta.universe
        ? nextMeta
        : (recall<{ scanned_at: string; universe: number }>(`scanner-meta:${tab}`) || openingMeta)
      remember(`scanner-meta:${tab}`, metaToKeep)
      setRows(kept)
      setMeta(metaToKeep)
    }

    if (tab === 'Best Setups') {
      fetchRadarHome()
        .then((result) => {
          const next = bestSetupsFromRadar(result, dashboard) as RadarRow[]
          apply(next, {
            scanned_at: result.scan_scanned_at || seedMeta.scanned_at,
            universe: result.universe_size || seedMeta.universe,
          })
        })
        .catch(() => apply(opening, openingMeta))
      return
    }
    fetchScannerWorkspace(tab)
      .then((result) => {
        const next = (result.rows?.length ? result.rows : seed) as RadarRow[]
        apply(next, {
          scanned_at: result.scanned_at || seedMeta.scanned_at,
          universe: result.universe_size || seedMeta.universe,
        })
      })
      .catch(() => apply(opening, openingMeta))
  }, [
    tab,
    dashboard.scan.scanned_at,
    dashboard.long_term.scanned_at,
    dashboard.scan.records.length,
    dashboard.long_term.records.length,
    dashboard.scan.universe_size,
    marketScan.succeeded,
    longTermScan.succeeded,
  ])

  const sectors = useMemo(() => [...new Set(rows.map((r) => r.sector).filter(Boolean))].sort(), [rows])
  const filtered = rows.filter((row) => {
    const q = search.trim().toUpperCase()
    if (q && !row.symbol.includes(q) && !String(row.company || '').toUpperCase().includes(q)) return false
    if (sector !== 'All' && row.sector !== sector) return false
    if (excludeChase && row.chase_risk) return false
    return true
  })

  const selectedRow = filtered.find((r) => r.symbol === selected) || rows.find((r) => r.symbol === selected)

  return (
    <section className="market-scanner">
      <header className="scanner-command-bar">
        <div>
          <span>MARKET SCANNER</span>
          <h2>Breakouts, momentum, SEPA and long-term</h2>
          <p>{filtered.length} matches · universe {meta.universe.toLocaleString('en-IN')} · scan {meta.scanned_at || '—'}</p>
        </div>
        <button type="button" disabled={activeScan.isBusy} onClick={() => void activeScan.start()}>
          {activeScan.isBusy
              ? `Scanning… ${activeScan.percent != null ? `${activeScan.percent}%` : ''}${activeScan.etaLine ? ` · ETA ${activeScan.etaLine}` : ''}`
              : tab === 'Long-Term' ? 'Run long-term scan' : 'Scan now'}
        </button>
      </header>

      <LiveScanBanner scan={activeScan} depth={depth} label={tab === 'Long-Term' ? 'Long-term scan' : 'Market scan'} />

      <div className="radar-tab-row">
        {(['Best Setups', 'Breakouts', 'Momentum', 'Long-Term'] as const).map((item) => (
          <button key={item} type="button" className={tab === item ? 'active' : ''} onClick={() => setTab(item)}>{item}</button>
        ))}
      </div>

      <div className="scanner-filter-row">
        <label>Search<input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Symbol" /></label>
        <label>Sector<select value={sector} onChange={(e) => setSector(e.target.value)}><option>All</option>{sectors.map((s) => <option key={s}>{s}</option>)}</select></label>
        <label className="scanner-check"><input type="checkbox" checked={excludeChase} onChange={(e) => setExcludeChase(e.target.checked)} /> Hide extended</label>
      </div>

      <div className="scanner-workspace-grid">
        <Panel title={`${tab.toUpperCase()} · ${filtered.length}`} subtitle="Sorted from persisted backend scan">
          <DenseTable
            rows={filtered}
            selected={selected}
            onSelect={setSelected}
            depth={depth}
            mode={tab}
            emptyHint={scannerEmptyHint(rows.length, filtered.length, hasScan)}
          />
        </Panel>
        <div className="scanner-detail-column">
          <Panel title={`CHART · ${selected || '—'}`}><ChartWorkspace symbol={selected} bars={bars} row={selectedRow} /></Panel>
          <Panel title="ACTIONS">
            <div className="radar-action-row">
              <button type="button" disabled={!selected} onClick={() => setActive('Stock Intelligence')}>Stock Intelligence</button>
              <button type="button" disabled={!selected} onClick={() => selected && onCompare(selected)}>Compare</button>
            </div>
          </Panel>
        </div>
      </div>
    </section>
  )
}

export function CompareView({ symbols, setSymbols, setActive, setSelected }: {
  symbols: string[]
  setSymbols: (s: string[]) => void
  setActive: (page: string) => void
  setSelected: (s: string) => void
}) {
  const [data, setData] = useState<CompareWorkspace | null>(null)
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (symbols.length === 0) { setData(null); return }
    setLoading(true)
    fetchCompareWorkspace(symbols)
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false))
  }, [symbols.join(',')])

  const addSymbol = () => {
    const sym = input.trim().toUpperCase()
    if (!sym || symbols.includes(sym)) return
    if (symbols.length >= 5) return
    setSymbols([...symbols, sym])
    setInput('')
  }

  return (
    <section className="compare-view">
      <header className="radar-hero">
        <div><span>COMPARE</span><h2>Side-by-side fundamentals and market state</h2><p>{data?.disclaimer || 'Add up to 5 NSE symbols.'}</p></div>
      </header>
      <div className="compare-chips">
        {symbols.map((sym) => (
          <button key={sym} type="button" className="compare-chip" onClick={() => { setSelected(sym); setActive('Stock Intelligence') }}>{sym}</button>
        ))}
        <input value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && addSymbol()} placeholder="Add symbol" />
        <button type="button" onClick={addSymbol}>Add</button>
        <button type="button" onClick={() => setSymbols([])}>Clear</button>
      </div>
      {loading && <p>Loading comparison…</p>}
      {data && (
        <div className="compare-grid">
          {Object.entries(data.section_labels).map(([key, label]) => (
            <Panel key={key} title={label.toUpperCase()}>
              <table className="radar-table compare-table">
                <thead><tr><th>Metric</th>{data.rows.map((r) => <th key={r.symbol}>{r.symbol}</th>)}</tr></thead>
                <tbody>
                  {(data.rows[0]?.sections[key] || []).map((_, idx) => (
                    <tr key={idx}>
                      <td>{data.rows[0]?.sections[key]?.[idx]?.label}</td>
                      {data.rows.map((row) => {
                        const m = row.sections[key]?.[idx]
                        return <td key={row.symbol}>{m?.available ? `${m.value}${m.unit ? ` ${m.unit}` : ''}` : '—'}</td>
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </Panel>
          ))}
        </div>
      )}
    </section>
  )
}

export function WatchlistView({ setActive, setSelected, onCompare }: {
  setActive: (page: string) => void
  setSelected: (s: string) => void
  onCompare: (symbol: string) => void
}) {
  const [payload, setPayload] = useState<WatchlistPayload | null>(() => recall<WatchlistPayload>('watchlist') ?? null)
  const [symbol, setSymbol] = useState('')
  const [notes, setNotes] = useState('')
  const [busy, setBusy] = useState(false)

  const reload = () => fetchWatchlist()
    .then((next) => { remember('watchlist', next); setPayload(next) })
    .catch(() => { if (!recall('watchlist')) setPayload(null) })

  useEffect(() => { void reload() }, [])

  const add = async () => {
    const sym = symbol.trim().toUpperCase()
    if (!sym) return
    setBusy(true)
    try {
      await addWatchlistItem({ symbol: sym, notes })
      setSymbol('')
      setNotes('')
      await reload()
    } finally { setBusy(false) }
  }

  return (
    <section className="watchlist-view">
      <header className="radar-hero">
        <div><span>WATCHLIST</span><h2>Track names you want to investigate</h2><p>Personal list — not a second alerts engine.</p></div>
      </header>
      <div className="watchlist-add">
        <input value={symbol} onChange={(e) => setSymbol(e.target.value)} placeholder="NSE symbol" />
        <input value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="Why watching" />
        <button type="button" disabled={busy} onClick={() => void add()}>Add</button>
      </div>
      <table className="radar-table">
        <thead><tr><th>Symbol</th><th>Added</th><th>Setup</th><th>Notes</th><th>Actions</th></tr></thead>
        <tbody>
          {(payload?.items || []).map((item) => (
            <tr key={item.id}>
              <td><button type="button" onClick={() => { setSelected(item.symbol); setActive('Stock Intelligence') }}>{item.symbol}</button></td>
              <td>{item.added_date}</td>
              <td>{String((item.snapshot as RadarRow)?.setup_label || item.snapshot?.status || '—')}</td>
              <td>{item.notes || '—'}</td>
              <td className="radar-action-row">
                <button type="button" onClick={() => onCompare(item.symbol)}>Compare</button>
                <button type="button" onClick={() => void removeWatchlistItem(item.id).then(reload)}>Remove</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {payload?.count === 0 && <p className="radar-empty-li">No watchlist items yet.</p>}
    </section>
  )
}
