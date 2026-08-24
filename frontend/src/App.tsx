import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { fetchChart, fetchDashboard, sendControl } from './api'
import {
  CompareView,
  MarketScannerView,
  RadarHomeView,
  WatchlistView,
} from './marketRadarViews'
import { MarketReportsView, RecommendationsView } from './recommendationsViews'
import { DisplayDepthToggle } from './displayDepth'
import {
  EnhancedLongTermView,
  ExperienceHelpDrawer,
} from './experience'
import { MarketSidebar } from './MarketSidebar'
import { EducationView } from './educationViews'
import { NewsView, OperationsRibbon, FnoView } from './marketViews'
import { ProductStockIntelligenceView } from './productViews'
import { ResearchDataView } from './researchData'
import {
  AutomationView,
  MarketInternalsView,
  PortfolioView,
  RecoBacktestView,
} from './views'
import type { DisplayDepth } from './productLanguage'
import { addWatchlistItem, bootstrapProduct } from './productApi'
import { useScanRunner } from './scanRunner'
import {
  readDeskNav,
  readSessionJson,
  writeDeskNav,
  writeSessionJson,
} from './sessionMemory'
import type { ChartBar, ControlName, DashboardPayload, OperationRecord } from './types'
import type { ReactNode } from 'react'

function activeSeed(dashboard: DashboardPayload, kind: string): OperationRecord | null {
  const active = dashboard.operations.active.find((item) => item.kind === kind)
  return active ?? null
}

const emptyDashboard: DashboardPayload = {
  generated_at: '',
  market: {
    available: false,
    health: 'Unavailable',
    summary: 'Preparing official market history…',
    trade_stance: 'QuantTerm is completing data lanes.',
    breadth: '—',
    leaders: [],
    laggards: [],
    nifty_change_1d: null,
    nifty_change_5d: null,
    vix: null,
    technical_details: {},
  },
  scan: { available: false, universe_size: 0, summary: {}, records: [] },
  long_term: { available: false, summary: {}, records: [], job: {} },
  paper: {
    available: false,
    enabled: false,
    supervisor_running: false,
    capital: 0,
    equity: 0,
    equity_curve: [],
    open_risk: 0,
    risk_per_trade_pct: 0.01,
    max_positions: 0,
    open_positions: [],
    closed_trades: [],
    refusals: [],
    last_cycle: {},
  },
  autonomy: {
    available: false,
    running: false,
    process_running: false,
    state: 'UNKNOWN',
    plain_state: 'Checking autonomy status…',
    explanation: '',
    heartbeat_ist: '',
    scheduler_owner_pid: null,
    active_job: {},
    new_paper_entries: false,
    existing_exits: false,
    research_enabled: false,
    capability_notes: [],
    active_failures: [],
    recent_dialogue: [],
    recent_transitions: [],
    jobs: {},
    jobs_recent: [],
    owner_state: {},
    live_feed: {},
    last_cycle: {},
  },
  operations: {
    available: false,
    running: false,
    worker_pid: null,
    heartbeat: '',
    active_lanes: {},
    counts: {},
    active: [],
    recent: [],
    latest: {},
  },
  news: {
    available: false,
    stats: { total: 0, important: 0, fno_linked: 0, macro: 0, sources: 0 },
    articles: [],
    source_health: [],
    latest_refresh: {},
  },
  fno: {
    available: false,
    source: 'unavailable',
    mapped_underlyings: 0,
    underlyings: [],
    exclusions: [],
  },
  data: {
    ready: false,
    snapshot: { ready: false, snapshot_id: '', latest_date: '', source: '' },
    bhavcopy: {
      ready: false,
      symbols: 0,
      sessions: 0,
      latest_date: '',
      csv_files: 0,
      cache_exists: false,
    },
    scan_saved: false,
    scan_records: 0,
    long_term_saved: false,
    long_term_records: 0,
    blockers: [],
  },
  conviction: [],
}

function dashboardHasWork(payload: DashboardPayload): boolean {
  return Boolean(
    payload.scan.scanned_at
    || payload.scan.records.length > 0
    || payload.long_term.records.length > 0
    || payload.data.ready,
  )
}

function slimDashboard(payload: DashboardPayload): DashboardPayload {
  return {
    ...payload,
    scan: { ...payload.scan, records: payload.scan.records.slice(0, 48) },
    long_term: { ...payload.long_term, records: payload.long_term.records.slice(0, 48) },
    news: { ...payload.news, articles: payload.news.articles.slice(0, 24) },
    conviction: payload.conviction.slice(0, 24),
    paper: {
      ...payload.paper,
      equity_curve: (payload.paper.equity_curve || []).slice(-40),
      closed_trades: (payload.paper.closed_trades || []).slice(0, 12),
    },
  }
}

function KeepPage({
  ids,
  active,
  seen,
  children,
}: {
  ids: string[]
  active: string
  seen: string[]
  children: ReactNode
}) {
  if (!ids.some((id) => seen.includes(id))) return null
  return <div hidden={!ids.includes(active)}>{children}</div>
}

function useIstClock() {
  const [now, setNow] = useState('')
  useEffect(() => {
    const tick = () => {
      setNow(new Date().toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour12: false }))
    }
    tick()
    const id = window.setInterval(tick, 1000)
    return () => window.clearInterval(id)
  }, [])
  return now
}

const pageTitles: Record<string, string> = {
  Home: 'Home',
  'Market Scanner': 'Market Scanner',
  Recommendations: 'Recommendations',
  'Market Reports': 'Market Reports',
  'Stock Intelligence': 'Stock Intelligence',
  'Long-Term Picks': 'Long-Term Picks',
  Compare: 'Compare',
  Watchlist: 'Watchlist',
  'Market Overview': 'Market Overview',
  'News & Events': 'News & Events',
  Education: 'Education',
  'Research Data': 'Research Data',
  Backtest: 'Backtest',
  'F&O Desk': 'F&O Desk',
  'Paper Portfolio': 'My Holdings',
  'System Health': 'System Health',
  'Command Center': 'Home',
  Scanner: 'Market Scanner',
  'Long-Term': 'Long-Term Picks',
  Portfolio: 'My Holdings',
  'Market Internals': 'Market Overview',
  Automation: 'System Health',
}

const pageSubtitles: Record<string, string> = {
  Home: 'Daily command centre — Breakouts, Momentum and Long-Term Picks from the saved market scan.',
  'Market Scanner': 'Professional scanner tables for breakouts, momentum, SEPA Best Setups and long-term quality.',
  Recommendations: 'Simple decisions on the outside — buy zone, target, stop, why now — with QuantTerm evidence underneath.',
  'Market Reports': 'Daily Market Pulse archive — trends, sector movers and breakout context from live system state.',
  'Stock Intelligence': 'Company workspace — chart, financials, ratios and pre-trade GO/CAUTION/NO_GO cockpit.',
  'Long-Term Picks': 'Business quality, valuation and timing without fabricated model performance.',
  Compare: 'Side-by-side comparison across market, growth, quality and technical dimensions.',
  Watchlist: 'Names you are tracking with latest scan context.',
  'Market Overview': 'Regime, breadth, volatility and sector leadership.',
  'News & Events': 'Dated market context with source health.',
  Education: 'Crunched news + macro/micro teach-ins for the share market — never invented blogs, never a signal.',
  'Research Data': 'Verified snapshots, data platform jobs, and evidence uploads.',
  Backtest: 'Inspect a paper-loss style on past data. This does not change today’s BUY list.',
  'F&O Desk': 'Mapped futures plus live OI / IV / PCR / max-pain context for a selected underlying.',
  'Paper Portfolio': 'Demat holdings + paper book — sync Zerodha or paste your shares.',
  Portfolio: 'Demat holdings + paper book — sync Zerodha or paste your shares.',
  'System Health': 'Operations, autonomy and infrastructure detail.',
}

function App() {
  const nav = readDeskNav()
  const [dashboard, setDashboard] = useState<DashboardPayload>(() => (
    readSessionJson<DashboardPayload>('quantterm-dashboard') || emptyDashboard
  ))
  const [loading, setLoading] = useState(() => !readSessionJson('quantterm-dashboard'))
  const [error, setError] = useState('')
  const [active, setActive] = useState(nav.active || 'Home')
  const [seen, setSeen] = useState<string[]>([nav.active || 'Home'])
  const [compareSymbols, setCompareSymbols] = useState<string[]>(nav.compare || [])
  const [selected, setSelected] = useState(nav.selected || '')
  const [bars, setBars] = useState<ChartBar[]>([])
  const [controlState, setControlState] = useState('')
  const [query, setQuery] = useState('')
  const [helpOpen, setHelpOpen] = useState(false)
  const autoPrepareRef = useRef(false)
  const istClock = useIstClock()
  const [depth, setDepth] = useState<DisplayDepth>(() => {
    const saved = window.localStorage.getItem('quantterm-display-depth')
    return saved === 'professional' ? 'professional' : 'simple'
  })

  const refresh = useCallback(async () => {
    try {
      const payload = await fetchDashboard()
      setDashboard((prev) => {
        const next = dashboardHasWork(prev) && !dashboardHasWork(payload)
          ? {
              ...payload,
              scan: prev.scan.records.length ? prev.scan : payload.scan,
              long_term: prev.long_term.records.length ? prev.long_term : payload.long_term,
              conviction: prev.conviction.length ? prev.conviction : payload.conviction,
              data: {
                ...payload.data,
                scan_saved: payload.data.scan_saved || prev.data.scan_saved,
                scan_records: Math.max(payload.data.scan_records || 0, prev.data.scan_records || 0),
              },
            }
          : payload
        writeSessionJson('quantterm-dashboard', slimDashboard(next))
        return next
      })
      setError('')
      const allSymbols = [
        ...payload.scan.records.map((row) => row.symbol),
        ...payload.long_term.records.map((row) => row.symbol),
        ...payload.fno.underlyings.map((row) => row.symbol),
      ]
      const first = allSymbols[0] || ''
      setSelected((current) => current || first)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Waiting for the market API')
    } finally {
      setLoading(false)
    }
  }, [])

  const marketScan = useScanRunner('MARKET_SCAN', {
    onComplete: () => void refresh(),
    seedOperation: activeSeed(dashboard, 'MARKET_SCAN'),
  })

  const longTermScan = useScanRunner('LONG_TERM_SCAN', {
    onComplete: () => void refresh(),
    seedOperation: activeSeed(dashboard, 'LONG_TERM_SCAN'),
  })

  const scanPollingActive = marketScan.isActive || longTermScan.isActive

  useEffect(() => {
    void refresh()
    const interval = scanPollingActive ? 8_000 : 12_000
    const timer = window.setInterval(() => void refresh(), interval)
    return () => window.clearInterval(timer)
  }, [refresh, scanPollingActive])

  useEffect(() => {
    if (autoPrepareRef.current || loading || error) return
    if (readSessionJson('quantterm-auto-prepare-done')) {
      autoPrepareRef.current = true
      return
    }
    const scanReady = Boolean(dashboard.scan.available && dashboard.scan.records.length > 0)
      || Boolean(dashboard.scan.scanned_at)
    if (dashboard.data.ready && scanReady) {
      autoPrepareRef.current = true
      writeSessionJson('quantterm-auto-prepare-done', true)
      return
    }
    autoPrepareRef.current = true
    void bootstrapProduct()
      .then(() => {
        writeSessionJson('quantterm-auto-prepare-done', true)
        void refresh()
      })
      .catch(() => { autoPrepareRef.current = false })
  }, [loading, error, dashboard.data.ready, dashboard.scan.available, dashboard.scan.records.length, dashboard.scan.scanned_at, refresh])

  useEffect(() => {
    window.localStorage.setItem('quantterm-display-depth', depth)
  }, [depth])

  useEffect(() => {
    if (!selected) {
      setBars([])
      return
    }
    fetchChart(selected)
      .then((result) => setBars(result.bars))
      .catch(() => setBars([]))
  }, [selected, dashboard.data.bhavcopy.ready, dashboard.data.bhavcopy.latest_date])

  const openPage = useCallback((page: string) => {
    setActive(page)
    setSeen((prev) => (prev.includes(page) ? prev : [...prev, page]))
  }, [])

  useEffect(() => {
    writeDeskNav({ active, selected, compare: compareSymbols })
  }, [active, selected, compareSymbols])

  useEffect(() => {
    window.dispatchEvent(new Event('resize'))
  }, [active])

  const symbols = useMemo(() => {
    const values = [
      ...dashboard.scan.records.map((row) => row.symbol),
      ...dashboard.long_term.records.map((row) => row.symbol),
      ...dashboard.fno.underlyings.map((row) => row.symbol),
    ]
    return [...new Set(values)].sort()
  }, [dashboard.fno.underlyings, dashboard.long_term.records, dashboard.scan.records])

  const openSearch = () => {
    const clean = query.trim().toUpperCase()
    if (!clean) return
    if (!/^[A-Z0-9&.-]{1,32}$/.test(clean)) {
      setControlState('Use a valid NSE symbol such as RELIANCE, TCS or HDFCBANK')
      return
    }
    const match = symbols.find((symbol) => symbol === clean)
      || symbols.find((symbol) => symbol.startsWith(clean))
      || clean
    setSelected(match)
    setQuery(match)
    openPage('Stock Intelligence')
    setControlState(`Opening verified workspace for ${match}`)
    window.setTimeout(() => setControlState(''), 2500)
  }

  const runControl = async (control: ControlName) => {
    setControlState('Starting…')
    try {
      const result = await sendControl(control)
      if (!result.accepted) {
        setControlState('Request was not accepted')
      } else if (result.operation_id) {
        setControlState('Operation queued — watch progress below')
      } else {
        setControlState('Queued successfully')
      }
      window.setTimeout(() => setControlState(''), 4000)
      window.setTimeout(() => void refresh(), 400)
    } catch (reason) {
      setControlState(reason instanceof Error ? reason.message : 'Control failed')
    }
  }

  const reportBase = `${window.location.protocol}//${window.location.hostname}:8766`
  const openEquityReport = () => {
    if (!selected) {
      setControlState('Select a stock before generating an equity evidence PDF')
      return
    }
    window.open(`${reportBase}/reports/equity/${encodeURIComponent(selected)}`, '_blank', 'noopener,noreferrer')
  }
  const openBasketReport = () => {
    window.open(`${reportBase}/reports/basket/long-term?limit=3`, '_blank', 'noopener,noreferrer')
  }

  const addToCompare = (symbol: string) => {
    setCompareSymbols((prev) => [...new Set([...prev, symbol.toUpperCase()])].slice(0, 5))
    openPage('Compare')
  }

  const addToWatchlist = async (symbol: string) => {
    try {
      await addWatchlistItem({ symbol, notes: 'From radar' })
      setControlState(`${symbol} added to watchlist`)
      window.setTimeout(() => setControlState(''), 2500)
    } catch {
      setControlState('Could not add to watchlist')
    }
  }

  const viewProps = {
    dashboard,
    selected,
    setSelected,
    bars,
    setActive: openPage,
    runControl,
    depth,
    marketScan,
    longTermScan,
  }

  const primaryPages = [
    'Home', 'Market Scanner', 'Stock Intelligence', 'Long-Term Picks',
    'Compare', 'Watchlist', 'Command Center', 'Scanner', 'Recommendations', 'Market Reports',
  ]
  const showOpsRibbon = !primaryPages.includes(active)
  const showReportActions = [
    'Stock Intelligence',
    'Research Data',
    'Long-Term Picks',
    'Long-Term',
    'Market Overview',
    'Market Internals',
  ].includes(active)
  const kiteOk = dashboard.autonomy.state !== 'AUTH_REQUIRED'
    && !(dashboard.autonomy.active_failures || []).some((f) => String(f).includes('auth'))

  const keep = (ids: string[], node: ReactNode) => (
    <KeepPage ids={ids} active={active} seen={seen}>{node}</KeepPage>
  )

  const pages = (
    <>
      {keep(['Home', 'Command Center'], <RadarHomeView {...viewProps} onCompare={addToCompare} onWatchlist={addToWatchlist} />)}
      {keep(['Market Scanner', 'Scanner'], <MarketScannerView {...viewProps} onCompare={addToCompare} />)}
      {keep(['Recommendations'], <RecommendationsView {...viewProps} />)}
      {keep(['Market Reports'], <MarketReportsView {...viewProps} />)}
      {keep(['Stock Intelligence'], (
        <ProductStockIntelligenceView
          {...viewProps}
          depth={depth}
          onCompare={addToCompare}
          onWatchlist={addToWatchlist}
        />
      ))}
      {keep(['Long-Term Picks', 'Long-Term'], <EnhancedLongTermView {...viewProps} />)}
      {keep(['Compare'], (
        <CompareView
          symbols={compareSymbols}
          setSymbols={setCompareSymbols}
          setActive={openPage}
          setSelected={setSelected}
          seedSymbols={[
            selected,
            ...dashboard.scan.records.map((row) => row.symbol),
            ...dashboard.long_term.records.map((row) => row.symbol),
          ]}
        />
      ))}
      {keep(['Watchlist'], (
        <WatchlistView
          setActive={openPage}
          setSelected={setSelected}
          onCompare={addToCompare}
          selected={selected}
        />
      ))}
      {keep(['Market Overview', 'Market Internals'], <MarketInternalsView {...viewProps} />)}
      {keep(['News & Events'], <NewsView {...viewProps} />)}
      {keep(['Education'], (
        <EducationView
          runControl={viewProps.runControl}
          setSelected={setSelected}
          setActive={openPage}
          newsRevision={dashboard.news.articles.length}
        />
      ))}
      {keep(['Research Data'], <ResearchDataView symbol={selected} />)}
      {keep(['Backtest'], <RecoBacktestView {...viewProps} />)}
      {keep(['F&O Desk'], <FnoView {...viewProps} />)}
      {keep(['Paper Portfolio', 'Portfolio'], <PortfolioView {...viewProps} />)}
      {keep(['System Health', 'Automation'], <AutomationView {...viewProps} />)}
    </>
  )

  return (
    <div className="terminal-root reco-desk">
      <MarketSidebar active={active} setActive={openPage} dashboard={dashboard} />
      <main className="workspace">
        <header className="topbar">
          <div className="search-box">
            ⌕
            <input
              aria-label="Search NSE symbol"
              placeholder="Search any NSE share…"
              value={query}
              onChange={(event: { target: { value: string } }) => setQuery(event.target.value)}
              onKeyDown={(event: { key: string }) => { if (event.key === 'Enter') openSearch() }}
              list="quantterm-symbols"
            />
            <datalist id="quantterm-symbols">{symbols.slice(0, 800).map((symbol) => <option value={symbol} key={symbol} />)}</datalist>
            <button type="button" onClick={openSearch}>Open stock</button>
          </div>
          <div className="top-status">
            <span className="reco-clock" title="Asia/Kolkata">{istClock || '—:—:—'} IST</span>
            <DisplayDepthToggle depth={depth} onChange={setDepth} />
            <button type="button" className="experience-help-trigger" onClick={() => setHelpOpen(true)}>What is this?</button>
            <span className={dashboard.data.ready ? 'live-pill' : 'work-pill'}><i /> {dashboard.data.ready ? 'DATA READY' : 'PREPARING DATA'}</span>
            <span className={kiteOk ? 'live-pill' : 'offline-pill'} title={dashboard.autonomy.plain_state || ''}>
              <i /> {kiteOk ? 'ZERODHA OK' : 'ZERODHA LOGIN'}
            </span>
            <button type="button" onClick={() => void refresh()} aria-label="Refresh dashboard">↻</button>
          </div>
        </header>

        <section className="page-title">
          <div><h1>{pageTitles[active] || active}</h1><p>{pageSubtitles[active]}</p></div>
          <div className="page-actions">
            {showReportActions && (
              <>
                <button type="button" disabled={!selected} onClick={openEquityReport}>Equity evidence PDF</button>
                <button type="button" onClick={openBasketReport}>Top-3 basket PDF</button>
              </>
            )}
            <span>{controlState || (loading ? 'Loading real state…' : `Updated ${dashboard.generated_at ? new Date(dashboard.generated_at).toLocaleTimeString('en-IN') : '—'}`)}</span>
          </div>
        </section>

        {error && (
          <div className="api-degraded-banner" role="alert">
            <strong>Connecting to the market API…</strong>
            <p>QuantTerm is starting the data lanes. Retry if this stays for more than a minute.</p>
            <details>
              <summary>Technical details</summary>
              <pre>{error}</pre>
            </details>
            <button type="button" onClick={() => void refresh()}>Retry connection</button>
          </div>
        )}
        {showOpsRibbon && <OperationsRibbon dashboard={dashboard} />}
        {pages}
      </main>
      <ExperienceHelpDrawer page={active} open={helpOpen} onClose={() => setHelpOpen(false)} />
    </div>
  )
}

export default App
