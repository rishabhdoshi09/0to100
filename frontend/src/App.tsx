import { useCallback, useEffect, useMemo, useState } from 'react'
import { fetchChart, fetchDashboard, sendControl } from './api'
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
import { NewsView, OperationsRibbon, FnoView } from './marketViews'
import { ProductStockIntelligenceView } from './productViews'
import { ResearchDataView } from './researchData'
import {
  AutomationView,
  MarketInternalsView,
  PortfolioView,
} from './views'
import type { DisplayDepth } from './productLanguage'
import { addWatchlistItem } from './productApi'
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
  'Stock Intelligence': 'Stock Intelligence',
  'Long-Term Picks': 'Long-Term Picks',
  Compare: 'Compare',
  Watchlist: 'Watchlist',
  'Market Overview': 'Market Overview',
  'News & Events': 'News & Events',
  'Research Data': 'Research Data',
  'F&O Desk': 'F&O Desk',
  'Paper Portfolio': 'Paper Portfolio',
  'System Health': 'System Health',
  // legacy route keys
  'Command Center': 'Home',
  Scanner: 'Market Scanner',
  'Long-Term': 'Long-Term Picks',
  Portfolio: 'Paper Portfolio',
  'Market Internals': 'Market Overview',
  Automation: 'System Health',
}

const pageSubtitles: Record<string, string> = {
  Home: 'Daily command centre — Breakouts, Momentum and Long-Term Picks from the saved market scan.',
  'Market Scanner': 'Professional scanner tables for breakouts, momentum and long-term quality.',
  'Stock Intelligence': 'Company workspace — chart, financials, ratios and read-only trade plan.',
  'Long-Term Picks': 'Business quality, valuation and timing without fabricated model performance.',
  Compare: 'Side-by-side comparison across market, growth, quality and technical dimensions.',
  Watchlist: 'Names you are tracking with latest scan context.',
  'Market Overview': 'Regime, breadth, volatility and sector leadership.',
  'News & Events': 'Dated market context with source health.',
  'Research Data': 'Verified snapshots, data platform jobs, and evidence uploads.',
  'F&O Desk': 'Stock derivative coverage — mapped underlyings, expiries and lot sizes.',
  'Paper Portfolio': 'Recorded paper positions and outcomes — secondary evidence.',
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
  const [helpOpen, setHelpOpen] = useState(false)
  const [pdfViewer, setPdfViewer] = useState<{ title: string; url: string } | null>(null)
  const [depth, setDepth] = useState<DisplayDepth>(() => {
    const saved = window.localStorage.getItem('quantterm-display-depth')
    return saved === 'professional' ? 'professional' : 'simple'
  })

  const refresh = useCallback(async () => {
    try {
      const payload = await fetchDashboard()
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
      setError(reason instanceof Error ? reason.message : 'Dashboard API unavailable')
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
    setActive('Stock Intelligence')
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
  }

  const primaryPages = ['Home', 'Market Scanner', 'Stock Intelligence', 'Long-Term Picks', 'Compare', 'Watchlist', 'Command Center', 'Scanner']
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
        />
      )
    }
    if (active === 'Research Data') return <ResearchDataView symbol={selected} />
    if (active === 'Paper Portfolio' || active === 'Portfolio') return <PortfolioView {...viewProps} />
    if (active === 'Market Overview' || active === 'Market Internals') return <MarketInternalsView {...viewProps} />
    if (active === 'Long-Term Picks' || active === 'Long-Term') return <EnhancedLongTermView {...viewProps} />
    if (active === 'News & Events') return <NewsView {...viewProps} />
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
              placeholder="Open any NSE symbol…"
              value={query}
              onChange={(event: { target: { value: string } }) => setQuery(event.target.value)}
              onKeyDown={(event: { key: string }) => { if (event.key === 'Enter') openSearch() }}
              list="quantterm-symbols"
            />
            <datalist id="quantterm-symbols">{symbols.slice(0, 800).map((symbol) => <option value={symbol} key={symbol} />)}</datalist>
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

        <section className="page-title">
          <div><h1>{pageTitles[active] || active}</h1><p>{pageSubtitles[active]}</p></div>
          <div className="page-actions">
            <button type="button" disabled={!selected} onClick={openEquityReport}>View equity evidence PDF</button>
            <button type="button" onClick={openBasketReport}>View top-3 basket PDF</button>
            <button type="button" onClick={openInstitutionalReport}>View FII/DII market brief</button>
            <span>{controlState || (loading ? 'Loading real state…' : `Updated ${dashboard.generated_at ? new Date(dashboard.generated_at).toLocaleTimeString('en-IN') : '—'}`)}</span>
          </div>
        </section>

        {error && (
          <div className="api-degraded-banner" role="alert">
            <strong>QuantTerm backend is unavailable.</strong>
            <p>Existing information may be incomplete. Reconnect or start the backend, then retry.</p>
            <details>
              <summary>Technical details</summary>
              <pre>{error}</pre>
            </details>
            <button type="button" onClick={() => void refresh()}>Retry connection</button>
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
