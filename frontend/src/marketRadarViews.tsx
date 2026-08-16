import { useEffect, useMemo, useState } from 'react'
import { ChartWorkspace, Panel } from './components'
import { money, pct, relativeAge, words } from './format'
import {
  addWatchlistItem,
  bootstrapProduct,
  fetchCompareWorkspace,
  fetchPreTrade,
  fetchProductReadiness,
  fetchRadarHome,
  fetchScannerWorkspace,
  fetchWatchlist,
  removeWatchlistItem,
  type CompareWorkspace,
  type PreTrade,
  type ProductReadiness,
  type RadarHome,
  type ScannerWorkspaceRow,
  type WatchlistPayload,
} from './productApi'
import { RiskLensCard } from './productViews'
import { LiveScanBanner, type ExperienceViewProps } from './experience'
import type { DashboardPayload } from './types'

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
  breakout_grade?: string
  breakout_conviction?: number
  breakout_quality?: number
  fundamental_score?: number
  sniper_candidate?: boolean
  volume_ratio?: number
  rsi?: number
  tech_source?: string
  price_tag?: string
  pct_below_20d_high?: number
}

const breakoutLabel: Record<string, string> = {
  confirmed_breakout: 'Confirmed',
  near_breakout: 'Near breakout',
  breakout_under_observation: 'Under observation',
  breakout_without_volume: 'No volume confirm',
  insufficient_confirmation: 'Needs confirmation',
  extended_after_breakout: 'Extended',
  faded_breakout: 'Faded',
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
          title={`Best live breakout · ${best.symbol}`}
          subtitle={
            [
              sniperCount > 0 ? `${sniperCount} confirmed setup${sniperCount === 1 ? '' : 's'}` : null,
              best.breakout_grade ? `Grade ${best.breakout_grade}` : null,
              best.rsi != null
                ? `RSI ${Math.round(Number(best.rsi))}${best.tech_source === 'kite' || best.price_tag === 'KITE' ? ' KITE' : best.tech_source === 'live' || best.price_tag === 'LIVE' ? ' LIVE' : ' EOD'}`
                : null,
              best.volume_ratio != null
                ? `Vol ${Number(best.volume_ratio).toFixed(1)}×${volOk ? '' : ' THIN'}`
                : null,
              best.pct_below_20d_high != null
                ? `${Number(best.pct_below_20d_high).toFixed(1)}% off 20d high`
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
        title="Best live breakout"
        subtitle="Volume ≥1.0× · still near the 20-day high · RSI ≤82 — tape only"
      >
        <p className="radar-empty-li">
          {sniperCount === 0
            ? 'No confirmed breakouts — thin volume or faded names stay out.'
            : 'Confirmed pool has names but none still intact on the live bar.'}
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
    const ctx = (best as RadarRow & {
      breakout_context?: { order_book?: { status?: string; note?: string }; concall?: { status?: string; note?: string } }
    }).breakout_context
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
          {ctx && (
            <p className="radar-empty-li" style={{ paddingTop: 8 }}>
              Order book: {ctx.order_book?.status || 'unavailable'}
              {ctx.order_book?.note ? ` — ${ctx.order_book.note}` : ''}
              {' · '}
              Concall: {ctx.concall?.status || 'unavailable'}
              {ctx.concall?.note && ctx.concall.status === 'present' ? ` — ${ctx.concall.note}` : ''}
            </p>
          )}
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
          No breakout candidate has usable fundamental coverage yet — run long-term scan, or wait for fund data. The confirmed-breakout list above is independent.
        </p>
      </Panel>
    </div>
  )
}

function fallbackRadarFromDashboard(dashboard: DashboardPayload): RadarHome {
  const records = (dashboard.scan.records || []) as RadarRow[]
  const breakouts = records.filter((row) => {
    const sigs = row.signals || []
    return sigs.some((item) => String(item).includes('BREAKOUT') || item === 'GOLDEN_CROSS' || item === 'VOL_SQUEEZE')
      || Boolean(row.breakout_grade)
      || row.status === 'Ready to trade'
  })
  const momentum = records.filter((row) => (row.signals || []).includes('MOMENTUM'))
  const longTerm = (dashboard.long_term.records || []) as RadarRow[]
  return {
    generated_at: dashboard.generated_at,
    market_session: dashboard.market.trade_stance,
    market_health: dashboard.market.health,
    breadth: dashboard.market.breadth,
    nifty_change_1d: Number(dashboard.market.nifty_change_1d || 0),
    vix: Number(dashboard.market.vix || 0),
    leaders: dashboard.market.leaders || [],
    laggards: dashboard.market.laggards || [],
    scan_scanned_at: dashboard.scan.scanned_at || '',
    long_term_scanned_at: dashboard.long_term.scanned_at || '',
    price_session: dashboard.data.bhavcopy.latest_date,
    market_as_of: dashboard.market.as_of,
    universe_size: dashboard.scan.universe_size,
    lanes: {
      breakouts: breakouts.slice(0, 12),
      momentum: momentum.slice(0, 12),
      long_term_picks: longTerm.slice(0, 12),
    },
    counts: {
      breakouts: breakouts.length,
      momentum: momentum.length,
      long_term_picks: longTerm.length,
    },
  }
}

function thinVolume(row: RadarRow): boolean {
  const vol = Number(row.volume_ratio)
  return row.breakout_state === 'breakout_without_volume'
    || (Number.isFinite(vol) && vol > 0 && vol < 1)
}

type DeskLane = 'breakouts' | 'momentum' | 'long_term'

function radarBadge(row: RadarRow): { label: string; cls: string } {
  if (thinVolume(row)) return { label: 'THIN VOL', cls: 'is-closed' }
  const state = String(row.breakout_state || '')
  if (state === 'faded_breakout' || state === 'failed_breakout' || state === 'failed_or_extended') {
    return { label: 'FADED', cls: 'is-closed' }
  }
  if (row.sniper_candidate) return { label: 'CONFIRMED', cls: '' }
  if (row.chase_risk || state === 'extended_after_breakout') return { label: 'EXTENDED', cls: 'is-watch' }
  if (row.classification) return { label: String(row.classification).replace(/_/g, ' '), cls: 'is-watch' }
  return { label: (row.setup_label || row.status || 'WATCH').slice(0, 18), cls: 'is-watch' }
}

function radarRiskClass(row: RadarRow): string {
  const label = String(row.risk_label || '')
  if (/high|chase|extended/i.test(label) || row.chase_risk) return 'high'
  if (/low/i.test(label)) return 'low'
  return 'medium'
}

function RadarPickCard({
  row,
  selected,
  featured,
  onSelect,
}: {
  row: RadarRow
  selected: string
  featured?: boolean
  onSelect: (symbol: string) => void
}) {
  const badge = radarBadge(row)
  const off20 = row.pct_below_20d_high
  const last = Number((row as RadarRow & { price?: number }).price)
  return (
    <button
      type="button"
      className={[
        'reco-pick',
        selected === row.symbol ? 'is-active' : '',
        featured ? 'is-featured' : '',
      ].filter(Boolean).join(' ')}
      onClick={() => onSelect(row.symbol)}
    >
      <div className="reco-pick-row1">
        <span className={`reco-buy ${badge.cls}`}>{badge.label}</span>
        <span className={`reco-risk-chip ${radarRiskClass(row)}`}>
          <span className="reco-risk-meter" aria-hidden="true" />
          {row.risk_label || 'Medium'} Risk
        </span>
      </div>
      <h3 className="reco-pick-name">{row.company || row.symbol}</h3>
      <div className="reco-pick-sub">
        <span>{row.symbol}</span>
        {row.sector ? <span className="reco-tag">{row.sector}</span> : null}
        {row.price_tag ? <span>{row.price_tag}</span> : null}
        {row.sniper_candidate ? <span className="reco-tag">Confirmed</span> : null}
      </div>
      <div className="reco-pick-kpis">
        <div>
          <span>Last</span>
          <strong>{Number.isFinite(last) && last > 0 ? money(last, 2) : '—'}</strong>
        </div>
        <div>
          <span>Volume</span>
          <strong>{row.volume_ratio != null ? `${Number(row.volume_ratio).toFixed(1)}×` : '—'}</strong>
        </div>
        <div>
          <span>RSI</span>
          <strong>{row.rsi != null ? Math.round(Number(row.rsi)) : '—'}</strong>
        </div>
        <div className="reco-gain">
          <strong className={off20 != null && Number(off20) > 5 ? 'neg' : ''}>
            {off20 != null ? `${Number(off20).toFixed(1)}%` : '—'}
          </strong>
          <small>off 20-day high</small>
        </div>
      </div>
      {row.reason ? <p className="reco-pick-note">{row.reason}</p> : null}
      <div className="reco-pick-tags" aria-label="Setup tags">
        {row.breakout_grade ? <span className="reco-evidence-tag">grade {row.breakout_grade}</span> : null}
        {row.setup_label ? <span className="reco-evidence-tag">{row.setup_label}</span> : null}
        {row.tech_source ? <span className="reco-evidence-tag">{row.tech_source}</span> : null}
      </div>
    </button>
  )
}

function DenseTable({
  rows,
  selected,
  onSelect,
  depth,
  mode,
}: {
  rows: RadarRow[]
  selected: string
  onSelect: (symbol: string) => void
  depth: ExperienceViewProps['depth']
  mode: string
}) {
  // Breakouts: keep server sniper-first ranking until the user clicks a column.
  const [sortKey, setSortKey] = useState(mode === 'Breakouts' ? '' : 'score')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')

  useEffect(() => {
    setSortKey(mode === 'Breakouts' ? '' : 'score')
    setSortDir('desc')
  }, [mode])

  const sorted = useMemo(() => {
    if (!sortKey) return rows
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
    : mode === 'Breakouts'
      ? depth === 'professional'
        ? ['symbol', 'sniper', 'price', 'volume_ratio', 'rsi', 'breakout_grade', 'breakout_quality', 'breakout_state', 'sector', 'risk_label']
        : ['symbol', 'sniper', 'price', 'volume_ratio', 'rsi', 'breakout_quality', 'setup_label', 'risk_label']
      : depth === 'professional'
        ? ['symbol', 'price', 'change_5d_pct', 'sector', 'setup_label', 'breakout_state', 'momentum_state', 'relative_strength', 'risk_label']
        : ['symbol', 'price', 'change_5d_pct', 'sector', 'setup_label', 'risk_label']

  return (
    <div className="radar-table-wrap">
      <table className="radar-table">
        <thead>
          <tr>
            {cols.map((col) => (
              <th key={col} onClick={() => toggleSort(col)}>
                {col === 'sniper' ? 'Confirmed' : words(col.replace(/_/g, ' '))}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.length === 0 && (
            <tr><td colSpan={cols.length} className="radar-empty">No matches yet — run Scan now (or Make ready if data is incomplete).</td></tr>
          )}
          {sorted.map((row) => {
            const thin = thinVolume(row)
            return (
            <tr
              key={row.symbol}
              className={[
                selected === row.symbol ? 'selected' : '',
                row.sniper_candidate ? 'sniper-row' : '',
                thin ? 'thin-volume-row' : '',
              ].filter(Boolean).join(' ')}
              onClick={() => onSelect(row.symbol)}
            >
              {cols.map((col) => {
                const raw = (row as Record<string, unknown>)[col]
                let cell: string
                let tone = ''
                if (col === 'sniper') cell = row.sniper_candidate ? 'YES' : '—'
                else if (col === 'breakout_state') cell = breakoutLabel[String(raw)] || words(String(raw))
                else if (col === 'momentum_state') cell = momentumLabel[String(raw)] || words(String(raw))
                else if (col === 'price') cell = money(raw as number)
                else if (col === 'change_5d_pct') cell = pct(raw as number)
                else if (col === 'volume_ratio') {
                  if (raw == null) cell = '—'
                  else if (thin) {
                    cell = `${Number(raw).toFixed(1)}× NO VOL`
                    tone = 'cell-warn'
                  } else cell = `${Number(raw).toFixed(1)}×`
                }
                else if (col === 'rsi' || col === 'breakout_quality' || col === 'combined_score' || col === 'relative_strength') {
                  cell = raw != null ? String(Math.round(Number(raw))) : '—'
                }
                else if (col === 'setup_label' && thin) {
                  cell = 'No volume confirm'
                  tone = 'cell-warn'
                }
                else cell = String(raw ?? '—')
                return <td key={col} className={tone || undefined}>{cell}</td>
              })}
            </tr>
            )
          })}
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
  const [radar, setRadar] = useState<RadarHome | null>(null)
  const [preTrade, setPreTrade] = useState<PreTrade | null>(null)
  const [readiness, setReadiness] = useState<ProductReadiness | null>(null)
  const [bootstrapBusy, setBootstrapBusy] = useState(false)
  const [deskNote, setDeskNote] = useState('')
  const [lane, setLane] = useState<DeskLane>('breakouts')
  const [query, setQuery] = useState('')

  // Daily fields (RSI/price/vol) must not freeze on last scan — poll the
  // live-refreshed radar payload on a short timer.
  useEffect(() => {
    let alive = true
    const load = () => {
      fetchRadarHome()
        .then((payload) => { if (alive) setRadar(payload) })
        .catch(() => undefined)
      fetchProductReadiness()
        .then((payload) => { if (alive) setReadiness(payload) })
        .catch(() => undefined)
    }
    load()
    const timer = window.setInterval(load, 20_000)
    return () => { alive = false; window.clearInterval(timer) }
  }, [dashboard.scan.scanned_at, dashboard.long_term.scanned_at, dashboard.generated_at])

  useEffect(() => {
    if (selected) return
    const best = String(radar?.best_breakout?.symbol || '').toUpperCase()
    if (best) setSelected(best)
  }, [radar?.best_breakout, selected, setSelected])

  useEffect(() => {
    if (!selected) { setPreTrade(null); return }
    let alive = true
    const load = () => {
      fetchPreTrade(selected)
        .then((payload) => { if (alive) setPreTrade(payload) })
        .catch(() => { if (alive) setPreTrade(null) })
    }
    load()
    const timer = window.setInterval(load, 20_000)
    return () => { alive = false; window.clearInterval(timer) }
  }, [selected, dashboard.scan.scanned_at, dashboard.generated_at])

  const scanCount = dashboard.scan.records?.length || 0
  const radarCount = (radar?.counts.breakouts || 0) + (radar?.counts.momentum || 0) + (radar?.counts.long_term_picks || 0)
  const desk = radar && radarCount > 0 ? radar : (scanCount > 0 ? fallbackRadarFromDashboard(dashboard) : radar)
  const scanAt = desk?.scan_scanned_at || dashboard.scan.scanned_at || ''
  const priceSession = desk?.price_session || desk?.market_as_of || dashboard.data.bhavcopy.latest_date || ''
  const emptyDesk = scanCount === 0 && !scanAt
  const readinessScore = readiness?.score ?? 0
  const needsBootstrap = emptyDesk || readinessScore < 70 || !dashboard.data.ready

  const runBootstrap = async () => {
    setBootstrapBusy(true)
    setDeskNote('Preparing data lanes…')
    try {
      const result = await bootstrapProduct()
      setReadiness(result.readiness)
      setDeskNote(result.message || 'Bootstrap queued')
      if (!marketScan.isBusy) void marketScan.start()
    } catch (reason) {
      setDeskNote(reason instanceof Error ? reason.message : 'Bootstrap failed')
    } finally {
      setBootstrapBusy(false)
      window.setTimeout(() => setDeskNote(''), 4000)
    }
  }

  const row = desk?.lanes.breakouts.find((r) => r.symbol === selected)
    || desk?.lanes.momentum.find((r) => r.symbol === selected)
    || desk?.lanes.long_term_picks.find((r) => r.symbol === selected)

  const health = desk?.market_health || dashboard.market.health || 'Market'
  const laneRows: Record<DeskLane, RadarRow[]> = {
    breakouts: (desk?.lanes.breakouts || []) as RadarRow[],
    momentum: (desk?.lanes.momentum || []) as RadarRow[],
    long_term: (desk?.lanes.long_term_picks || []) as RadarRow[],
  }
  const q = query.trim().toUpperCase()
  const visible = laneRows[lane].filter((item) => {
    if (!q) return true
    return item.symbol.includes(q) || String(item.company || '').toUpperCase().includes(q)
  })
  const best = desk?.best_breakout as RadarRow | null | undefined
  const fundBest = desk?.best_among_fundamentals as RadarRow | null | undefined
  const kiteLive = Boolean(dashboard.data.kite?.ok)
  const stale = Boolean(dashboard.data.bhavcopy.is_stale)
  const nifty = desk?.nifty_change_1d ?? dashboard.market.nifty_change_1d
  const bannerNote = kiteLive
    ? (stale
      ? `Kite is live, but official history is stale — need ${dashboard.data.bhavcopy.required_session || 'latest session'}.`
      : dashboard.market.quote_source === 'kite'
        ? `Kite is the primary last print. Official session ${priceSession || dashboard.data.bhavcopy.latest_date || '—'}.`
        : `Official bhavcopy ready · ${priceSession || dashboard.data.bhavcopy.latest_date || '—'}.`)
    : (dashboard.data.kite?.note || 'Kite token rejected — run python main.py login')
  const laneLabel = lane === 'breakouts' ? 'Breakouts' : lane === 'momentum' ? 'Momentum' : 'Long-term picks'
  const heroIcon = health.slice(0, 1).toUpperCase() || 'M'

  return (
    <div className="reco-light">
      <LiveScanBanner scan={marketScan} depth={depth} label="Market scan" />

      <nav className="reco-crumb" aria-label="Breadcrumb">
        <button type="button" onClick={() => setActive('Home')}>Home</button>
        <span>›</span>
        <strong>Market desk</strong>
        <span>›</span>
        <strong>{laneLabel}</strong>
      </nav>

      <header className="reco-hero">
        <div className="reco-hero-icon" aria-hidden="true">{heroIcon}</div>
        <div>
          <h2>{health}</h2>
          <p>{dashboard.market.summary}</p>
        </div>
        <div className="reco-hero-actions">
          {needsBootstrap && (
            <button type="button" className="reco-ghost" disabled={bootstrapBusy} onClick={() => void runBootstrap()}>
              {bootstrapBusy ? 'Preparing…' : readinessScore >= 90 ? 'Refresh desk' : 'Make ready'}
            </button>
          )}
          <button type="button" className="reco-primary" disabled={marketScan.isBusy} onClick={() => void marketScan.start()}>
            {marketScan.isBusy ? 'Scanning…' : 'Scan now'}
          </button>
        </div>
      </header>

      <div className="reco-status-row" aria-label="Market status">
        <div className={`reco-status ${Number(nifty) >= 0 ? 'is-good' : 'is-mid'}`}>
          <small>Nifty 1D</small>
          <strong>{pct(nifty)}</strong>
        </div>
        <div className={`reco-status ${/narrow/i.test(String(desk?.breadth || dashboard.market.breadth)) ? 'is-bad' : 'is-mid'}`}>
          <small>Breadth</small>
          <strong>{desk?.breadth || dashboard.market.breadth || '—'}</strong>
        </div>
        <div className="reco-status is-mid">
          <small>VIX</small>
          <strong>{desk?.vix ?? dashboard.market.vix ?? '—'}</strong>
        </div>
        <div className={`reco-status ${kiteLive ? 'is-good' : 'is-warn'}`}>
          <small>Zerodha</small>
          <strong>{kiteLive ? 'Kite live' : 'Login needed'}</strong>
        </div>
        <div className={`reco-status ${stale ? 'is-warn' : 'is-good'}`}>
          <small>Price data</small>
          <strong>{dashboard.data.bhavcopy.latest_date || 'Missing'}</strong>
        </div>
      </div>

      <div className="reco-cmp-banner" role="status">
        <span className="ico" aria-hidden="true">!</span>
        <div>
          <div>{bannerNote}</div>
          <em>
            {scanAt
              ? `Last scan ${relativeAge(scanAt)} · bars as of ${priceSession || 'last session'} EOD`
              : 'Run Scan now to fill the desk'}
            {deskNote ? ` · ${deskNote}` : ''}
          </em>
        </div>
      </div>

      <div className="reco-cat-rail" role="tablist" aria-label="Desk lanes">
        <button type="button" role="tab" aria-selected={lane === 'breakouts'} className={lane === 'breakouts' ? 'active' : ''} onClick={() => setLane('breakouts')}>
          Breakouts · {desk?.counts.breakouts || 0}
          {desk?.counts.sniper_breakouts ? ` · ${desk.counts.sniper_breakouts} confirmed` : ''}
        </button>
        <button type="button" role="tab" aria-selected={lane === 'momentum'} className={lane === 'momentum' ? 'active' : ''} onClick={() => setLane('momentum')}>
          Momentum · {desk?.counts.momentum || 0}
        </button>
        <button type="button" role="tab" aria-selected={lane === 'long_term'} className={lane === 'long_term' ? 'active' : ''} onClick={() => setLane('long_term')}>
          Long-term · {desk?.counts.long_term_picks || 0}
        </button>
      </div>

      <div className="reco-controls">
        <div className="reco-search-wrap">
          <input
            type="search"
            placeholder="Search stocks"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            aria-label="Search stocks"
          />
        </div>
      </div>

      {lane === 'breakouts' && best ? (
        <>
          <p className="reco-featured-label">Best live breakout</p>
          <div className="reco-card-stack">
            <RadarPickCard row={best} selected={selected} featured onSelect={setSelected} />
          </div>
        </>
      ) : null}

      {lane === 'breakouts' && !best && visible.length > 0 ? (
        <div className="reco-empty">
          <strong>No confirmed live breakout</strong>
          <p>Names below are from the last scan. Confirmed only if volume ≥1.0× and still near the 20-day high.</p>
        </div>
      ) : null}

      {lane === 'breakouts' && fundBest ? (
        <>
          <p className="reco-featured-label">Best among breakouts with fundamentals</p>
          <div className="reco-card-stack">
            <RadarPickCard row={fundBest} selected={selected} onSelect={setSelected} />
          </div>
        </>
      ) : null}

      {visible.length === 0 ? (
        <div className="reco-empty">
          <strong>No {laneLabel.toLowerCase()} yet</strong>
          <p>Use Make ready / Scan now above, or clear the search.</p>
        </div>
      ) : (
        <div className="reco-card-stack">
          {visible.slice(0, 8).filter((item) => item.symbol !== best?.symbol || lane !== 'breakouts').map((item) => (
            <RadarPickCard key={item.symbol} row={item} selected={selected} onSelect={setSelected} />
          ))}
        </div>
      )}

      {selected ? (
        <section className="reco-sheet">
          <header className="reco-sheet-hero">
            <p>{(row as RadarRow)?.sector || 'Selected name'}</p>
            <h2>{(row as RadarRow)?.company || selected}</h2>
            <p>{(row as RadarRow)?.reason || preTrade?.plan_summary || preTrade?.meaning || 'Check rupee risk before opening full research.'}</p>
          </header>
          <div className="reco-sheet-kpis">
            <div>
              <span>Entry</span>
              <strong>{money(preTrade?.plan?.entry ?? preTrade?.scan?.entry, 2)}</strong>
            </div>
            <div>
              <span>Stop</span>
              <strong>{money(preTrade?.plan?.stop ?? preTrade?.scan?.stop, 2)}</strong>
            </div>
            <div>
              <span>Target</span>
              <strong>{money(preTrade?.plan?.target ?? preTrade?.scan?.target, 2)}</strong>
            </div>
            <div>
              <span>Pre-trade</span>
              <strong>{preTrade?.verdict || '—'}</strong>
            </div>
          </div>
          <div className="reco-next">{deskNote || dashboard.market.trade_stance}</div>
          {(preTrade?.blockers || []).length > 0 ? (
            <div className="reco-sheet-cols">
              <div>
                <h3>What blocks a trade</h3>
                <ul>{preTrade!.blockers!.slice(0, 4).map((item) => <li key={item}>{item}</li>)}</ul>
              </div>
              <div>
                <h3>Leaders / laggards</h3>
                <p>
                  Leading {(radar?.leaders || dashboard.market.leaders).slice(0, 3).join(', ') || '—'}.
                  Lagging {(radar?.laggards || dashboard.market.laggards).slice(0, 3).join(', ') || '—'}.
                </p>
              </div>
            </div>
          ) : null}
          <RiskLensCard plan={preTrade?.plan || null} />
          <div className="reco-sheet-actions">
            <button type="button" className="reco-primary" onClick={() => setActive('Stock Intelligence')}>Full research</button>
            <button type="button" className="reco-ghost" onClick={() => onCompare(selected)}>Compare</button>
            <button type="button" className="reco-ghost" onClick={() => onWatchlist(selected)}>Watchlist</button>
          </div>
          <div className="reco-chart-card">
            <ChartWorkspace symbol={selected} bars={bars} row={row} />
          </div>
        </section>
      ) : (
        <p className="reco-foot">Pick one name — then check rupee risk here before opening research.</p>
      )}

      <p className="reco-foot">
        Last prints prefer Kite. History stays on official NSE bhavcopy.
        {priceSession ? ` Session ${priceSession}.` : ''}
      </p>
    </div>
  )
}

export function MarketScannerView(props: ExperienceViewProps & { onCompare: (symbol: string) => void }) {
  const { dashboard, selected, setSelected, bars, setActive, depth, marketScan, longTermScan, onCompare } = props
  const scannerTabs = depth === 'professional'
    ? ['Breakouts', 'Momentum', 'Conviction', 'Pre-Breakout', 'Long-Term', 'F&O', 'Avoid']
    : ['Breakouts', 'Momentum', 'Long-Term']
  const [tab, setTab] = useState('Breakouts')
  const [rows, setRows] = useState<RadarRow[]>([])
  const [bestBreakout, setBestBreakout] = useState<RadarRow | null>(null)
  const [bestAmongFund, setBestAmongFund] = useState<RadarRow | null>(null)
  const [sniperCount, setSniperCount] = useState(0)
  const [meta, setMeta] = useState({ scanned_at: '', universe: 0 })
  const [search, setSearch] = useState('')
  const [sector, setSector] = useState('All')
  const [excludeChase, setExcludeChase] = useState(true)
  const [sniperOnly, setSniperOnly] = useState(false)

  const activeScan = tab === 'Long-Term' ? longTermScan : marketScan

  useEffect(() => {
    if (!scannerTabs.includes(tab)) setTab(scannerTabs[0])
  }, [depth])

  useEffect(() => {
    let alive = true
    const load = () => {
      fetchScannerWorkspace(tab)
        .then((result) => {
          if (!alive) return
          setRows(result.rows as RadarRow[])
          setMeta({ scanned_at: result.scanned_at, universe: result.universe_size })
          setBestBreakout((result.best_breakout as RadarRow | null | undefined) || null)
          setBestAmongFund((result.best_among_fundamentals as RadarRow | null | undefined) || null)
          setSniperCount(result.sniper_count ?? (result.sniper_rows?.length || 0))
        })
        .catch(() => {
          if (!alive) return
          setRows([])
          setBestBreakout(null)
          setBestAmongFund(null)
          setSniperCount(0)
        })
    }
    load()
    // Breakouts/Pre-Breakout change every minute — don't wait for a new scan.
    const pollMs = (tab === 'Breakouts' || tab === 'Pre-Breakout') ? 20_000 : 60_000
    const timer = window.setInterval(load, pollMs)
    return () => { alive = false; window.clearInterval(timer) }
  }, [tab, dashboard.scan.scanned_at, dashboard.long_term.scanned_at, dashboard.generated_at])

  const sectors = useMemo(() => [...new Set(rows.map((r) => r.sector).filter(Boolean))].sort(), [rows])
  const filtered = rows.filter((row) => {
    const q = search.trim().toUpperCase()
    if (q && !row.symbol.includes(q) && !String(row.company || '').toUpperCase().includes(q)) return false
    if (sector !== 'All' && row.sector !== sector) return false
    if (excludeChase && row.chase_risk) return false
    if (tab === 'Breakouts' && sniperOnly && !row.sniper_candidate) return false
    return true
  })

  const selectedRow = filtered.find((r) => r.symbol === selected) || rows.find((r) => r.symbol === selected)

  return (
    <section className="reco-light market-scanner">
      <nav className="reco-crumb" aria-label="Breadcrumb">
        <button type="button" onClick={() => setActive('Home')}>Home</button>
        <span>›</span>
        <strong>Market Scanner</strong>
        <span>›</span>
        <strong>{tab}</strong>
      </nav>
      <header className="reco-hero">
        <div className="reco-hero-icon" aria-hidden="true">S</div>
        <div>
          <h2>{tab}</h2>
          <p>
            {filtered.length} matches · universe {meta.universe.toLocaleString('en-IN')}
            {' · scan '}{relativeAge(meta.scanned_at)}
            {' · last prints prefer Kite'}
          </p>
        </div>
        <div className="reco-hero-actions">
          <button type="button" className="reco-primary" disabled={activeScan.isBusy} onClick={() => void activeScan.start()}>
            {activeScan.isBusy ? 'Scanning…' : tab === 'Long-Term' ? 'Run long-term scan' : 'Scan now'}
          </button>
        </div>
      </header>

      <LiveScanBanner scan={activeScan} depth={depth} label={tab === 'Long-Term' ? 'Long-term scan' : 'Market scan'} />

      <div className="reco-cat-rail" role="tablist" aria-label="Scanner lanes">
        {scannerTabs.map((item) => (
          <button key={item} type="button" role="tab" aria-selected={tab === item} className={tab === item ? 'active' : ''} onClick={() => setTab(item)}>{item}</button>
        ))}
      </div>

      {tab === 'Breakouts' && (
        <>
          <BestSniperPanel
            best={bestBreakout}
            sniperCount={sniperCount}
            onSelect={setSelected}
          />
          <BestAmongFundamentalsPanel
            best={bestAmongFund}
            onSelect={setSelected}
          />
        </>
      )}

      <div className="scanner-filter-row">
        <label>Search<input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Symbol" /></label>
        <label>Sector<select value={sector} onChange={(e) => setSector(e.target.value)}><option>All</option>{sectors.map((s) => <option key={s}>{s}</option>)}</select></label>
        <label className="scanner-check"><input type="checkbox" checked={excludeChase} onChange={(e) => setExcludeChase(e.target.checked)} /> Hide extended</label>
        {tab === 'Breakouts' && (
          <label className="scanner-check">
            <input type="checkbox" checked={sniperOnly} onChange={(e) => setSniperOnly(e.target.checked)} />
            Confirmed setups only
          </label>
        )}
      </div>

      <div className="scanner-workspace-grid">
        <Panel
          title={`${tab.toUpperCase()} · ${filtered.length}`}
          subtitle={tab === 'Breakouts'
            ? `Confirmed-first rank · ${sniperCount} intact setup${sniperCount === 1 ? '' : 's'} · thin volume marked red`
            : 'Sorted from persisted backend scan · tape fields refresh live'}
        >
          <DenseTable rows={filtered} selected={selected} onSelect={setSelected} depth={depth} mode={tab} />
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
  const [payload, setPayload] = useState<WatchlistPayload | null>(null)
  const [symbol, setSymbol] = useState('')
  const [notes, setNotes] = useState('')
  const [busy, setBusy] = useState(false)

  const reload = () => fetchWatchlist().then(setPayload).catch(() => setPayload(null))

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
