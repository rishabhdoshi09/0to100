import { useCallback, useEffect, useMemo, useState } from 'react'
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
import { hubOf, wrapInHub } from './hubs'
import { EducationView } from './educationViews'
import { NewsView, OperationsRibbon, FnoView } from './marketViews'
import { ProductStockIntelligenceView } from './productViews'
import { ResearchDataView } from './researchData'
import {
  AutomationView,
  MarketInternalsView,
  PortfolioView,
} from './views'
import type { DisplayDepth } from './productLanguage'
import { addWatchlistItem, fetchHoldings, fetchSymbolDirectory } from './productApi'
import { useScanRunner } from './scanRunner'
import { ReportPdfViewer } from './ReportPdfViewer'
import type { ChartBar, ControlName, DashboardPayload, OperationRecord } from './types'

/** In dev, Vite proxies /reports to :8766 (PDFs) and /evidence to :8765 (one origin). */
const reportApiBase = import.meta.env.DEV
  ? ''
  : `${window.location.protocol}//${window.location.hostname}:8766`

function activeSeed(dashboard: DashboardPayload, kind: string): OperationRecord | null {
  const active = dashboard.operations.active.find((item) => item.kind === kind)
  return active ?? null
}

const emptyDashboard: DashboardPayload = {
  generated_at: '',
  session: { available: false, banner: '', last_session: '' },
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
  'Market Scanner': 'Ideas',
  Recommendations: 'Ideas',
  'Market Reports': 'Context',
  'Stock Intelligence': 'Stock',
  'Long-Term Picks': 'Ideas',
  Compare: 'Compare',
  Watchlist: 'Watchlist',
  'Market Overview': 'Home',
  'News & Events': 'Context',
  Education: 'Context',
  'Research Data': 'System',
  'F&O Desk': 'Ideas',
  'Paper Portfolio': 'Holdings',
  'System Health': 'System',
  'Command Center': 'Home',
  Scanner: 'Ideas',
  'Long-Term': 'Ideas',
  Portfolio: 'Holdings',
  'Market Internals': 'Home',
  Automation: 'System',
}

const pageSubtitles: Record<string, string> = {
  Home: 'Desk of names plus the market weather they sit in.',
  'Market Scanner': 'The same scan as a dense table.',
  Recommendations: 'Research shortlist — categories, table, long-term and F&O.',
  'Market Reports': 'Daily digest, source list, and the same flow taught.',
  'Stock Intelligence': 'One company — chart, financials, options and evidence.',
  'Long-Term Picks': 'Quality, valuation and timing without fabricated performance.',
  Compare: 'Side-by-side across market, growth, quality and technicals.',
  Watchlist: 'Names you are tracking with latest scan context.',
  'Market Overview': 'Regime, breadth, volatility and FII/DII — the weather for the desk.',
  'News & Events': 'Dated market context with source health.',
  Education: 'The news flow, taught — never invented blogs, never a signal.',
  'Research Data': 'Which files behind a stock are fresh, stale or missing.',
  'F&O Desk': 'Derivatives floor of a name — mapped futures, then the live chain.',
  'Paper Portfolio': 'Demat holdings + paper book — sync Zerodha or paste your shares.',
  Portfolio: 'Demat holdings + paper book — sync Zerodha or paste your shares.',
  'System Health': 'Workers alive, then the files behind a stock.',
}

function App() {
  const [dashboard, setDashboard] = useState<DashboardPayload>(emptyDashboard)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [active, setActive] = useState('Home')
  const [compareSymbols, setCompareSymbols] = useState<string[]>([])
  const [selected, setSelected] = useState('')
  const [intelTab, setIntelTab] = useState<string | undefined>()
  const [bars, setBars] = useState<ChartBar[]>([])
  const [controlState, setControlState] = useState('')
  const [query, setQuery] = useState('')
  const [universeSymbols, setUniverseSymbols] = useState<string[]>([])
  const [remoteSuggestions, setRemoteSuggestions] = useState<string[]>([])
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
      // Do not auto-open scan.records[0]. That list is raw scan rank — a faded
      // name (YATHARTH) can sit on top and then get chart/pre-trade polled forever.
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
    let alive = true
    const load = () => {
      fetchChart(selected)
        .then((result) => { if (alive) setBars(result.bars) })
        .catch(() => { if (alive) setBars([]) })
    }
    load()
    // Intraday bar changes — don't wait for a new bhavcopy date.
    const timer = window.setInterval(load, 20_000)
    return () => { alive = false; window.clearInterval(timer) }
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
    setIntelTab(undefined)
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

  useEffect(() => {
    if (active !== 'Stock Intelligence') setIntelTab(undefined)
  }, [active])

  const hub = hubOf(active)
  const hidePageTitle = hub === 'Home' || hub === 'Ideas' || hub === 'Context' || hub === 'System'
  const showOpsRibbon = hub === 'System'
  const showReportActions = [
    'Stock Intelligence',
    'Research Data',
    'Long-Term Picks',
    'Long-Term',
    'Market Overview',
    'Market Internals',
  ].includes(active)
  const kiteOk = Boolean(dashboard.data.kite?.ok)

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
    if (active === 'Recommendations') {
      return <RecommendationsView {...viewProps} />
    }
    if (active === 'Market Reports') {
      return <MarketReportsView {...viewProps} />
    }
    if (active === 'Home' || active === 'Command Center') {
      return (
        <RadarHomeView
          {...viewProps}
          onCompare={addToCompare}
          onWatchlist={addToWatchlist}
          onOpenFloor={(page) => setActive(page)}
        />
      )
    }
    if (active === 'Stock Intelligence') {
      return (
        <ProductStockIntelligenceView
          {...viewProps}
          depth={depth}
          initialTab={intelTab}
          onCompare={addToCompare}
          onWatchlist={addToWatchlist}
        />
      )
    }
    if (active === 'Research Data') {
      return (
        <ResearchDataView
          symbol={selected}
          onOpenStock={() => setActive(selected ? 'Stock Intelligence' : 'Recommendations')}
        />
      )
    }
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
    if (active === 'F&O Desk') {
      return (
        <FnoView
          {...viewProps}
          onOpenStock={(symbol) => {
            setSelected(symbol)
            setIntelTab('Options')
            setActive('Stock Intelligence')
          }}
        />
      )
    }
    if (active === 'System Health' || active === 'Automation') return <AutomationView {...viewProps} />
    return (
      <RadarHomeView
        {...viewProps}
        onCompare={addToCompare}
        onWatchlist={addToWatchlist}
        onOpenFloor={(page) => setActive(page)}
      />
    )
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
            <span className={dashboard.data.ready ? 'live-pill' : 'offline-pill'}><i /> {dashboard.data.ready ? 'DATA READY' : 'DATA INCOMPLETE'}</span>
            <span className={kiteOk ? 'live-pill' : 'offline-pill'} title={dashboard.data.kite?.note || dashboard.autonomy.plain_state || ''}>
              <i /> {kiteOk ? 'KITE LIVE' : 'KITE LOGIN'}
            </span>
            <button type="button" onClick={() => void refresh()} aria-label="Refresh dashboard">↻</button>
          </div>
        </header>

        <section className={`page-title ${hidePageTitle ? 'is-hidden' : ''}`}>
          <div><h1>{pageTitles[active] || active}</h1><p>{pageSubtitles[active]}</p></div>
          <div className="page-actions">
            {showReportActions && (
              <>
                <button type="button" disabled={!selected} onClick={openEquityReport}>Equity evidence PDF</button>
                <button type="button" onClick={openBasketReport}>Top-3 basket PDF</button>
                <button type="button" onClick={openInstitutionalReport}>FII/DII brief PDF</button>
              </>
            )}
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
        {wrapInHub(active, setActive, renderView())}
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
