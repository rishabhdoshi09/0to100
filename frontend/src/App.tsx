import { useCallback, useEffect, useMemo, useState } from 'react'
import { fetchChart, fetchDashboard, fetchOperationsPayload, sendControl } from './api'
import { LivePriceStrip } from './LivePriceStrip'
import { useQuoteHeartbeat } from './useQuoteHeartbeat'
import {
  CompareView,
  MarketScannerView,
  RadarHomeView,
  WatchlistView,
} from './marketRadarViews'
import { DisplayDepthToggle } from './displayDepth'
import {
  EnhancedLongTermView,
  ExperienceHelpDrawer,
} from './experience'
import { MarketSidebar } from './MarketSidebar'
import { EducationView } from './educationViews'
import { StreetPulseView } from './streetPulseView'
import { BuyBookView } from './buyBookView'
import { NewsView, OperationsRibbon, FnoView } from './marketViews'
import { ProductStockIntelligenceView } from './productViews'
import { UsMarketHome, UsScannerView, UsStockView } from './usMarketViews'
import { ConfirmedBreakoutsView } from './sniperBoardView'
import { ResearchDataView } from './researchData'
import {
  AutomationView,
  MarketInternalsView,
  PortfolioView,
} from './views'
import type { DisplayDepth } from './productLanguage'
import { addWatchlistItem, fetchBuyBook, fetchHoldings, fetchSymbolDirectory } from './productApi'
import { useScanRunner } from './scanRunner'
import { ReportPdfViewer } from './ReportPdfViewer'
import type { ChartBar, ControlName, DashboardPayload, OperationRecord } from './types'

/** In dev, Vite proxies /reports and /evidence to the report API on :8766. */
const reportApiBase = import.meta.env.DEV
  ? ''
  : `${window.location.protocol}//${window.location.hostname}:8766`

function activeSeed(dashboard: DashboardPayload, kind: string): OperationRecord | null {
  const active = dashboard.operations.active.find((item) => item.kind === kind)
  return active ?? null
}

const emptyDashboard: DashboardPayload = {
  generated_at: '',
  market: {
    available: false,
    health: 'Unavailable',
    summary: 'Market state is not available yet.',
    trade_stance: 'Start the QuantTerm API and market-operations worker.',
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
    plain_state: 'Autonomy status unavailable.',
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
  institutional: {
    available: false,
    cash: { available: false, history: [], totals: {} },
    bulk_buy_symbols: [],
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

const pageTitles: Record<string, string> = {
  Home: 'Home',
  'Market Scanner': 'Market Scanner',
  'Confirmed Breakouts': 'Confirmed Breakouts',
  'Stock Intelligence': 'Stock Intelligence',
  'Long-Term Picks': 'Long-Term Picks',
  Compare: 'Compare',
  Watchlist: 'Watchlist',
  'Market Overview': 'Market Overview',
  'News & Events': 'News & Events',
  Education: 'Education',
  'Daily Pulse': 'Daily Pulse',
  'Active Buys': 'Active Buys',
  'US Market': 'US Market',
  'US Scanner': 'US Scanner',
  'US Stock': 'US Stock',
  'Research Data': 'Research Data',
  'F&O Desk': 'F&O Desk',
  'Paper Portfolio': 'My Holdings',
  'System Health': 'System Health',
  // legacy route keys
  'Command Center': 'Home',
  Scanner: 'Market Scanner',
  'Long-Term': 'Long-Term Picks',
  Portfolio: 'My Holdings',
  'Market Internals': 'Market Overview',
  Automation: 'System Health',
}

const pageSubtitles: Record<string, string> = {
  Home: 'Daily command centre — Breakouts, Momentum and Long-Term Picks from the saved market scan.',
  'Market Scanner': 'Professional scanner tables for breakouts, momentum and long-term quality.',
  'Confirmed Breakouts': 'Live sniper confirms collected into one board, then ranked for tomorrow-watch / research.',
  'Stock Intelligence': 'Company workspace — chart, financials, ratios and pre-trade GO/CAUTION/NO_GO cockpit.',
  'Long-Term Picks': 'Business quality, valuation and timing without fabricated model performance.',
  Compare: 'Side-by-side comparison across market, growth, quality and technical dimensions.',
  Watchlist: 'Names you are tracking with latest scan context.',
  'Market Overview': 'Regime, breadth, volatility and sector leadership.',
  'News & Events': 'Dated market context with source health.',
  Education: 'Crunched news + macro/micro teach-ins for the share market — never invented blogs, never a signal.',
  'Daily Pulse': 'Simple daily market digest — what moved, what to watch, options mood. Send to Telegram anytime.',
  'Active Buys': 'Stock results for names you are buying — entry vs now, 1D/5D, plus warnings if averages or support break.',
  'US Market': 'US retail plane — NASDAQ listings, Yahoo EOD, S&P-scoped scan, paper autopilot only.',
  'US Scanner': 'US setups from Yahoo daily bars · liquid quality floor · no options overlay.',
  'US Stock': 'US ticker workspace — daily chart + last scan setup. Fundamentals/options marked unavailable.',
  'Research Data': 'Verified snapshots, data platform jobs, and evidence uploads.',
  'F&O Desk': 'Mapped futures plus live OI / IV / PCR / max-pain context for a selected underlying.',
  'Paper Portfolio': 'Demat holdings + paper book — sync Zerodha or paste your shares.',
  Portfolio: 'Demat holdings + paper book — sync Zerodha or paste your shares.',
  'System Health': 'Operations, autonomy and infrastructure detail.',
}

function App() {
  const [dashboard, setDashboard] = useState<DashboardPayload>(emptyDashboard)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [active, setActive] = useState('Home')
  const [compareSymbols, setCompareSymbols] = useState<string[]>([])
  const [selected, setSelected] = useState('')
  const [bars, setBars] = useState<ChartBar[]>([])
  const [controlState, setControlState] = useState('')
  const [query, setQuery] = useState('')
  const [universeSymbols, setUniverseSymbols] = useState<string[]>([])
  const [remoteSuggestions, setRemoteSuggestions] = useState<string[]>([])
  const [helpOpen, setHelpOpen] = useState(false)
  const [pdfViewer, setPdfViewer] = useState<{ title: string; url: string } | null>(null)
  const [buyBookSymbols, setBuyBookSymbols] = useState<string[]>([])
  const [depth, setDepth] = useState<DisplayDepth>(() => {
    const saved = window.localStorage.getItem('quantterm-display-depth')
    return saved === 'professional' ? 'professional' : 'simple'
  })

  const refresh = useCallback(async (opts?: { soft?: boolean }) => {
    const soft = Boolean(opts?.soft)
    try {
      // Soft polls during an active scan use a shorter timeout and never wipe
      // the last good dashboard if the API is temporarily busy.
      const payload = await fetchDashboard({ timeoutMs: soft ? 12_000 : 25_000 })
      setDashboard(payload)
      setError('')
      const allSymbols = [
        ...payload.scan.records.map((row) => row.symbol),
        ...payload.long_term.records.map((row) => row.symbol),
        ...payload.fno.underlyings.map((row) => row.symbol),
      ]
      const first = allSymbols[0] || ''
      setSelected((current) => current || first)
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : 'Dashboard API unavailable'
      if (soft) {
        // Keep UI usable while market-ops is scanning — merge ops if possible.
        try {
          const ops = await fetchOperationsPayload()
          setDashboard((prev) => ({ ...prev, operations: ops }))
          setError('')
          setControlState('Dashboard busy during scan — showing last good state + live ops')
        } catch {
          setControlState(message)
        }
      } else {
        setError(message)
      }
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

  const sniperBoardEval = useScanRunner('SNIPER_BOARD_EVAL', {
    onComplete: () => void refresh(),
    seedOperation: activeSeed(dashboard, 'SNIPER_BOARD_EVAL'),
  })

  const scanPollingActive = marketScan.isActive || longTermScan.isActive || sniperBoardEval.isActive

  useEffect(() => {
    let cancelled = false
    fetchBuyBook()
      .then((payload) => {
        if (cancelled) return
        setBuyBookSymbols((payload.items || []).map((row) => String(row.symbol || '').toUpperCase()).filter(Boolean))
      })
      .catch(() => {
        if (!cancelled) setBuyBookSymbols([])
      })
    return () => {
      cancelled = true
    }
  }, [active])

  const heartbeatSymbols = useMemo(() => {
    const out = new Set<string>(['NIFTY', 'BANKNIFTY'])
    if (selected) out.add(selected.toUpperCase())
    for (const sym of buyBookSymbols) out.add(sym)
    for (const row of dashboard.paper.open_positions || []) {
      if (row.symbol) out.add(String(row.symbol).toUpperCase())
    }
    for (const row of (dashboard.scan.records || []).slice(0, 8)) {
      if (row.symbol) out.add(String(row.symbol).toUpperCase())
    }
    return [...out].slice(0, 30)
  }, [selected, buyBookSymbols, dashboard.paper.open_positions, dashboard.scan.records])

  // Skip live heartbeat on US pages — Yahoo EOD plane, not Kite.
  const liveEnabled = !String(active).startsWith('US')
  const liveQuotes = useQuoteHeartbeat(heartbeatSymbols, { enabled: liveEnabled })

  useEffect(() => {
    void refresh({ soft: false })
    // During scans: poll ops-friendly soft dashboard less often so the API
    // stays free for market-ops on older Macs.
    const lowPower = import.meta.env.VITE_QT_LOW_POWER === '1'
    const interval = scanPollingActive
      ? (lowPower ? 45_000 : 25_000)
      : (lowPower ? 45_000 : 15_000)
    const timer = window.setInterval(
      () => void refresh({ soft: scanPollingActive }),
      interval,
    )
    return () => window.clearInterval(timer)
  }, [refresh, scanPollingActive])

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

  const symbols = useMemo(() => {
    const values = [
      ...universeSymbols,
      ...dashboard.scan.records.map((row) => row.symbol),
      ...dashboard.long_term.records.map((row) => row.symbol),
      ...dashboard.fno.underlyings.map((row) => row.symbol),
    ]
    return [...new Set(values)].sort()
  }, [universeSymbols, dashboard.fno.underlyings, dashboard.long_term.records, dashboard.scan.records])

  const symbolSuggestions = useMemo(() => {
    const q = query.trim().toUpperCase()
    if (!q) return symbols.slice(0, 80)
    const local = symbols.filter((symbol) => symbol.startsWith(q))
    return [...new Set([...remoteSuggestions, ...local])].sort().slice(0, 80)
  }, [query, symbols, remoteSuggestions])

  useEffect(() => {
    let alive = true
    // limit=0 → complete A→Z directory (do not stop around letter M)
    Promise.all([
      fetchSymbolDirectory({ limit: 0 }).catch(() => null),
      fetchHoldings().catch(() => null),
    ]).then(([directory, book]) => {
      if (!alive) return
      const fromDir = (directory?.symbols || []).map((row) => row.symbol).filter(Boolean)
      const fromHoldings = (book?.holdings || []).flatMap((row) => [
        row.tradingsymbol,
        row.research_symbol || '',
      ]).filter(Boolean)
      setUniverseSymbols([...new Set([...fromDir, ...fromHoldings])])
      if (directory?.truncated) {
        setControlState(`Symbol directory truncated (${directory.count}/${directory.universe_size}) — retry refresh`)
      }
    })
    return () => { alive = false }
  }, [])

  // Server-side typeahead so N…Z prefixes work even if the local cache is thin.
  useEffect(() => {
    const q = query.trim().toUpperCase()
    if (q.length < 1) {
      setRemoteSuggestions([])
      return
    }
    let alive = true
    const timer = window.setTimeout(() => {
      fetchSymbolDirectory({ q, limit: 80 })
        .then((payload) => {
          if (!alive) return
          setRemoteSuggestions((payload.symbols || []).map((row) => row.symbol).filter(Boolean))
        })
        .catch(() => {
          if (alive) setRemoteSuggestions([])
        })
    }, 120)
    return () => {
      alive = false
      window.clearTimeout(timer)
    }
  }, [query])

  const openSearch = () => {
    const clean = query.trim().toUpperCase()
    if (!clean) return
    if (!/^[A-Z0-9&.-]{1,32}$/.test(clean)) {
      setControlState('Use a valid NSE symbol such as RELIANCE, TCS or HDFCBANK')
      return
    }
    const match = symbols.find((symbol) => symbol === clean)
      || remoteSuggestions.find((symbol) => symbol === clean)
      || symbols.find((symbol) => symbol.startsWith(clean))
      || remoteSuggestions.find((symbol) => symbol.startsWith(clean))
      || clean
    setSelected(match)
    setQuery(match)
    setActive('Stock Intelligence')
    setControlState(
      symbols.includes(match) || remoteSuggestions.includes(match)
        ? `Opening verified workspace for ${match}`
        : `Opening ${match} — not in local universe cache; workspace still loads if bhav history exists`,
    )
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

  const reportBase = reportApiBase
  const openReportViewer = (title: string, path: string) => {
    setPdfViewer({ title, url: `${reportBase}${path}` })
  }
  const openEquityReport = () => {
    if (!selected) {
      setControlState('Select a stock before generating an equity evidence PDF')
      return
    }
    openReportViewer(`${selected} · equity evidence`, `/reports/equity/${encodeURIComponent(selected)}`)
  }
  const openBasketReport = () => {
    openReportViewer('Top-3 long-term basket', '/reports/basket/long-term?limit=3')
  }
  const openInstitutionalReport = () => {
    openReportViewer('FII/DII market brief', '/reports/market/institutional?days=30&symbol_limit=4')
  }
  const openStreetPulseReport = () => {
    openReportViewer('Daily Street Pulse', '/reports/market/street-pulse?force=true')
  }

  const addToCompare = (symbol: string) => {
    setCompareSymbols((prev) => [...new Set([...prev, symbol.toUpperCase()])].slice(0, 5))
    setActive('Compare')
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
    setActive,
    runControl,
    depth,
    marketScan,
    longTermScan,
    sniperBoardEval,
  }

  const primaryPages = ['Home', 'Daily Pulse', 'Active Buys', 'Market Scanner', 'Confirmed Breakouts', 'Stock Intelligence', 'Long-Term Picks', 'Compare', 'Watchlist', 'Command Center', 'Scanner']
  const showOpsRibbon = !primaryPages.includes(active)

  const renderView = () => {
    if (active === 'Compare') {
      return (
        <CompareView
          symbols={compareSymbols}
          setSymbols={setCompareSymbols}
          setActive={setActive}
          setSelected={setSelected}
        />
      )
    }
    if (active === 'Watchlist') {
      return <WatchlistView setActive={setActive} setSelected={setSelected} onCompare={addToCompare} />
    }
    if (active === 'Market Scanner' || active === 'Scanner') {
      return <MarketScannerView {...viewProps} onCompare={addToCompare} />
    }
    if (active === 'Confirmed Breakouts') {
      return (
        <ConfirmedBreakoutsView
          selected={selected}
          setSelected={setSelected}
          setActive={setActive}
          evalScan={sniperBoardEval}
        />
      )
    }
    if (active === 'Home' || active === 'Command Center') {
      return <RadarHomeView {...viewProps} onCompare={addToCompare} onWatchlist={addToWatchlist} />
    }
    if (active === 'Stock Intelligence') {
      return (
        <ProductStockIntelligenceView
          {...viewProps}
          depth={depth}
          onCompare={addToCompare}
          onWatchlist={addToWatchlist}
          liveTick={selected ? liveQuotes.get(selected) : undefined}
          liveSessionOpen={liveQuotes.sessionOpen}
        />
      )
    }
    if (active === 'Research Data') return <ResearchDataView symbol={selected} />
    if (active === 'Paper Portfolio' || active === 'Portfolio') return <PortfolioView {...viewProps} />
    if (active === 'Market Overview' || active === 'Market Internals') return <MarketInternalsView {...viewProps} />
    if (active === 'Long-Term Picks' || active === 'Long-Term') return <EnhancedLongTermView {...viewProps} />
    if (active === 'News & Events') return <NewsView {...viewProps} />
    if (active === 'Education') {
      return (
        <EducationView
          runControl={viewProps.runControl}
          setSelected={setSelected}
          setActive={setActive}
        />
      )
    }
    if (active === 'Daily Pulse') {
      return (
        <StreetPulseView
          setSelected={setSelected}
          setActive={setActive}
          onOpenPdf={openStreetPulseReport}
        />
      )
    }
    if (active === 'Active Buys') {
      return (
        <BuyBookView
          selected={selected}
          setSelected={setSelected}
          setActive={setActive}
        />
      )
    }
    if (active === 'US Market') {
      return <UsMarketHome setActive={setActive} setSelected={setSelected} />
    }
    if (active === 'US Scanner') {
      return <UsScannerView setActive={setActive} setSelected={setSelected} />
    }
    if (active === 'US Stock') {
      return <UsStockView symbol={selected} setSymbol={setSelected} />
    }
    if (active === 'F&O Desk') return <FnoView {...viewProps} />
    if (active === 'System Health' || active === 'Automation') return <AutomationView {...viewProps} />
    return <RadarHomeView {...viewProps} onCompare={addToCompare} onWatchlist={addToWatchlist} />
  }

  return (
    <div className="terminal-root">
      <MarketSidebar active={active} setActive={setActive} dashboard={dashboard} />
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
            <datalist id="quantterm-symbols">{symbolSuggestions.map((symbol) => <option value={symbol} key={symbol} />)}</datalist>
            <button type="button" onClick={openSearch}>Open stock</button>
          </div>
          <div className="top-status">
            <DisplayDepthToggle depth={depth} onChange={setDepth} />
            <button type="button" className="experience-help-trigger" onClick={() => setHelpOpen(true)}>What is this?</button>
            <span className={dashboard.data.ready ? 'live-pill' : 'offline-pill'}><i /> {dashboard.data.ready ? 'CORE DATA READY' : 'DATA INCOMPLETE'}</span>
            <span className={dashboard.operations.running ? 'live-pill' : 'offline-pill'}><i /> {dashboard.operations.running ? 'MARKET OPS ONLINE' : 'MARKET OPS OFFLINE'}</span>
            <button type="button" onClick={() => void refresh()} aria-label="Refresh dashboard">↻</button>
          </div>
        </header>
        {liveEnabled && <LivePriceStrip payload={liveQuotes.payload} focusSymbol={selected} />}

        <section className="page-title">
          <div><h1>{pageTitles[active] || active}</h1><p>{pageSubtitles[active]}</p></div>
          <div className="page-actions">
            <button type="button" disabled={!selected} onClick={openEquityReport}>View equity evidence</button>
            <button type="button" onClick={openBasketReport}>View top-3 basket</button>
            <button type="button" onClick={openInstitutionalReport}>View FII/DII brief</button>
            <span>{controlState || (loading ? 'Loading real state…' : `Updated ${dashboard.generated_at ? new Date(dashboard.generated_at).toLocaleTimeString('en-IN') : '—'}`)}</span>
          </div>
        </section>

        {error && (
          <div className="api-degraded-banner" role="alert">
            <strong>
              {scanPollingActive
                ? 'Dashboard is busy while a scan is running.'
                : 'QuantTerm backend is unavailable.'}
            </strong>
            <p>
              {scanPollingActive
                ? 'Market-ops may still be scanning — wait for the scan banner to finish, then Retry. Prefer a full restart only if the worker stays OFFLINE.'
                : 'Dashboard did not load. Market scan needs the Terminal API (:8765) and a live market-ops worker. Prefer a full restart over repeated Retry if this persists.'}
            </p>
            <pre className="api-error-detail">{error || 'No error detail returned by the browser fetch.'}</pre>
            <div className="inline-actions">
              <button type="button" onClick={() => void refresh({ soft: false })}>Retry connection</button>
              <button
                type="button"
                disabled={marketScan.isBusy}
                onClick={() => void marketScan.start()}
              >
                Try market scan anyway
              </button>
            </div>
            <small>
              Restart: bash scripts/stop_quantterm.sh && bash scripts/run_quantterm_low_power.sh
            </small>
          </div>
        )}
        {loading && !error && (
          <div className="api-degraded-banner" role="status">
            <strong>Loading QuantTerm…</strong>
            <p>Fetching dashboard. Sidebar should already be visible.</p>
          </div>
        )}
        {showOpsRibbon && <OperationsRibbon dashboard={dashboard} />}
        {renderView()}
      </main>
      <ExperienceHelpDrawer page={active} open={helpOpen} onClose={() => setHelpOpen(false)} />
      <ReportPdfViewer
        open={pdfViewer != null}
        title={pdfViewer?.title ?? 'Research report'}
        viewUrl={pdfViewer?.url ?? ''}
        onClose={() => setPdfViewer(null)}
      />
    </div>
  )
}

export default App
