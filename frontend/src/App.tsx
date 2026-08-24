import { useCallback, useEffect, useMemo, useState } from 'react'
import { fetchChart, fetchDashboard, sendControl } from './api'
import {
  CompareView,
  MarketScannerView,
  RadarHomeView,
  WatchlistView,
} from './marketRadarViews'
import { DisplayDepthToggle } from './displayDepth'
import { ExperienceHelpDrawer } from './experience'
import { MarketSidebar } from './MarketSidebar'
import { NewsView, OperationsRibbon } from './marketViews'
import { ProductStockIntelligenceView } from './productViews'
import { ResearchDataView } from './researchData'
import { DeskHub } from './deskHub'
import {
  MarketInternalsView,
  PortfolioView,
  RecoBacktestView,
} from './views'
import type { DisplayDepth } from './productLanguage'
import { addWatchlistItem } from './productApi'
import { useScanRunner } from './scanRunner'
import type { ChartBar, ControlName, DashboardPayload, OperationRecord } from './types'

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

function istSessionOpen() {
  const parts = new Intl.DateTimeFormat('en-GB', {
    timeZone: 'Asia/Kolkata', weekday: 'short', hour: '2-digit', minute: '2-digit', hour12: false,
  }).formatToParts(new Date())
  const week = parts.find((p) => p.type === 'weekday')?.value || ''
  const hour = Number(parts.find((p) => p.type === 'hour')?.value || 0)
  const minute = Number(parts.find((p) => p.type === 'minute')?.value || 0)
  if (['Sat', 'Sun'].includes(week)) return false
  const mins = hour * 60 + minute
  return mins >= 9 * 60 + 15 && mins <= 15 * 60 + 30
}

const pageTitles: Record<string, string> = {
  Today: 'Top Stocks',
  Setups: 'Recommendations',
  'Paper Desk': 'Momentum',
  Backtest: 'Backtest',
  Portfolio: 'Wealth Builders',
  Desk: 'Market Reports',
  Home: 'Today',
  'Market Scanner': 'Setups',
  'Stock Intelligence': 'Stock Intelligence',
  'Long-Term Picks': 'Setups',
  Compare: 'Compare',
  Watchlist: 'Watchlist',
  'Market Overview': 'Market',
  'News & Events': 'News',
  'Research Data': 'Data',
  'Paper Portfolio': 'Paper Desk',
  'System Health': 'Desk',
  'Command Center': 'Today',
  Scanner: 'Setups',
  'Long-Term': 'Setups',
  'Market Internals': 'Market',
  Automation: 'Desk',
}

const pageSubtitles: Record<string, string> = {
  Today: 'SEPA-qualified names first, then the scanner watchlist. CMP can be delayed.',
  Setups: 'Best Setups, Momentum and Long-term. Do not mix them.',
  'Paper Desk': 'Simulated book. The bot learns from closed trades. Live orders stay locked.',
  Backtest: 'Inspect a paper-loss style on past data. This does not change today’s BUY list.',
  Portfolio: 'Paper positions, equity and what the bot learned.',
  Desk: 'Market, news, data and system health.',
  Home: 'SEPA-style Best Setups first, then the scanner watchlist.',
  'Market Scanner': 'Breakouts, Momentum, Long-term.',
  'Paper Portfolio': 'Simulated book. Live orders stay locked.',
}

function App() {
  const [dashboard, setDashboard] = useState<DashboardPayload>(emptyDashboard)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [active, setActive] = useState('Today')
  const [compareSymbols, setCompareSymbols] = useState<string[]>([])
  const [selected, setSelected] = useState('')
  const [bars, setBars] = useState<ChartBar[]>([])
  const [controlState, setControlState] = useState('')
  const [query, setQuery] = useState('')
  const [helpOpen, setHelpOpen] = useState(false)
  const istClock = useIstClock()
  const sessionOpen = istSessionOpen()
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

  const primaryPages = ['Today', 'Setups', 'Home', 'Market Scanner', 'Command Center', 'Scanner', 'Paper Desk', 'Backtest', 'Portfolio', 'Paper Portfolio']
  const showOpsRibbon = !primaryPages.includes(active)
  const recoDesk = ['Today', 'Setups', 'Paper Desk', 'Backtest', 'Portfolio', 'Home', 'Command Center', 'Paper Portfolio'].includes(active)

  const renderView = () => {
    if (active === 'Today' || active === 'Home' || active === 'Command Center') {
      return <RadarHomeView {...viewProps} onCompare={addToCompare} onWatchlist={addToWatchlist} />
    }
    if (active === 'Setups' || active === 'Market Scanner' || active === 'Scanner' || active === 'Long-Term Picks' || active === 'Long-Term') {
      return <MarketScannerView {...viewProps} onCompare={addToCompare} />
    }
    if (active === 'Paper Desk' || active === 'Paper Portfolio' || active === 'Portfolio') {
      return <PortfolioView {...viewProps} />
    }
    if (active === 'Backtest') return <RecoBacktestView {...viewProps} />
    if (active === 'Desk' || active === 'System Health' || active === 'Automation') {
      return (
        <DeskHub
          {...viewProps}
          compareSymbols={compareSymbols}
          setCompareSymbols={setCompareSymbols}
          onCompare={addToCompare}
          onWatchlist={addToWatchlist}
          depth={depth}
        />
      )
    }
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
    if (active === 'Market Overview' || active === 'Market Internals') return <MarketInternalsView {...viewProps} />
    if (active === 'News & Events') return <NewsView {...viewProps} />
    return <RadarHomeView {...viewProps} onCompare={addToCompare} onWatchlist={addToWatchlist} />
  }

  return (
    <div className="rw-app">
      <header className="rw-topbar">
        <div className="rw-wordmark">Reco Wealth</div>
        <div className="search-box">
          ⌕
          <input
            aria-label="Search NSE symbol"
            placeholder="Search stocks"
            value={query}
            onChange={(event: { target: { value: string } }) => setQuery(event.target.value)}
            onKeyDown={(event: { key: string }) => { if (event.key === 'Enter') openSearch() }}
            list="quantterm-symbols"
          />
          <datalist id="quantterm-symbols">{symbols.slice(0, 800).map((symbol) => <option value={symbol} key={symbol} />)}</datalist>
          <button type="button" onClick={openSearch}>Search</button>
        </div>
        <div className="rw-top-meta">
          <span className="rw-clock">{istClock || '—:—:—'} IST</span>
          <span className={sessionOpen ? 'rw-session open' : 'rw-session closed'}>
            ● {sessionOpen ? 'MARKET OPEN' : 'MARKET CLOSED'}
          </span>
          {!recoDesk && <DisplayDepthToggle depth={depth} onChange={setDepth} />}
          <button type="button" className="experience-help-trigger" onClick={() => setHelpOpen(true)}>What is this?</button>
          <button type="button" onClick={() => void refresh()} aria-label="Refresh dashboard">↻</button>
          <div className="rw-avatar" aria-hidden="true">R</div>
        </div>
      </header>
      <div className="rw-shell">
      <MarketSidebar active={active} setActive={setActive} dashboard={dashboard} />
      <main className="workspace">

        <section className="page-title">
          <div><h1>{pageTitles[active] || active}</h1><p>{pageSubtitles[active]}</p></div>
          <div className="page-actions">
            {!recoDesk && (
              <>
                <button type="button" disabled={!selected} onClick={openEquityReport}>Equity Evidence PDF</button>
                <button type="button" onClick={openBasketReport}>Top-3 Basket PDF</button>
              </>
            )}
            <span>{controlState || (loading ? 'Loading real state…' : `Updated ${dashboard.generated_at ? new Date(dashboard.generated_at).toLocaleTimeString('en-IN') : '—'}`)}</span>
          </div>
        </section>

        {error && (
          <div className="api-degraded-banner" role="alert">
            <strong>RecoWealth desk is waiting for the market API.</strong>
            <p>Start the QuantTerm stack, then retry. Cards stay empty until the last scan is readable.</p>
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
      </div>
    </div>
  )
}

export default App
