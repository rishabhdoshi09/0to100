import { useEffect, useMemo, useState } from 'react'
import { fetchChart, fetchDashboard, sendControl } from './api'
import { MarketSidebar } from './MarketSidebar'
import { FnoView, NewsView, OperationsRibbon } from './marketViews'
import { ResearchDataView } from './researchData'
import {
  AutomationView,
  CommandCenterView,
  LongTermView,
  MarketInternalsView,
  PortfolioView,
  ScannerView,
  StockIntelligenceView,
} from './views'
import type { ChartBar, ControlName, DashboardPayload } from './types'

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

const pageSubtitles: Record<string, string> = {
  'Command Center': 'Market posture, opportunities, portfolio and system health in one decision surface.',
  Scanner: 'Immediate whole-market momentum, conviction, breakout and pre-breakout research.',
  'Stock Intelligence': 'Price structure, evidence, quality, risk and invalidation for the selected stock.',
  'Research Data': 'Strict as-of dates, source links, templates and evidence uploads for the selected stock.',
  Portfolio: 'Paper capital, open risk, recorded equity, positions and closed-trade attribution.',
  'Market Internals': 'Regime, breadth, volatility, sector leadership and scanner coverage.',
  'Long-Term': 'Business quality, valuation and technical timing on a dedicated research lane.',
  'News & Events': 'Curated market context with source health, entity mapping and F&O linkage.',
  'F&O Desk': 'Current stock-derivatives universe, contract mapping and explicit exclusions.',
  Automation: 'Paper supervisor heartbeat, durable jobs, failures, controls and operational dialogue.',
}

function App() {
  const [dashboard, setDashboard] = useState<DashboardPayload>(emptyDashboard)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [active, setActive] = useState('Command Center')
  const [selected, setSelected] = useState('')
  const [bars, setBars] = useState<ChartBar[]>([])
  const [controlState, setControlState] = useState('')
  const [query, setQuery] = useState('')

  const refresh = async () => {
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
      setSelected((current) => (current && allSymbols.includes(current) ? current : first))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Dashboard API unavailable')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void refresh()
    const timer = window.setInterval(() => void refresh(), 2_000)
    return () => window.clearInterval(timer)
  }, [])

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
    const match = symbols.find((symbol) => symbol === clean)
      || symbols.find((symbol) => symbol.startsWith(clean))
      || symbols.find((symbol) => symbol.includes(clean))
    if (match) {
      setSelected(match)
      setQuery(match)
      setActive('Stock Intelligence')
      setControlState(`Opened ${match}`)
      window.setTimeout(() => setControlState(''), 1800)
    } else {
      setControlState(`No saved QuantTerm record found for ${clean}`)
    }
  }

  const runControl = async (control: ControlName) => {
    setControlState('Starting operation…')
    try {
      const result = await sendControl(control)
      if (!result.accepted) {
        setControlState('Control was not accepted')
      } else if (result.operation_id) {
        setControlState(`${control.replaceAll('_', ' ')} · ${result.operation_status || 'PENDING'} · ${result.operation_id.slice(0, 8)}`)
      } else {
        setControlState(`${control.replaceAll('_', ' ')} queued`)
      }
      window.setTimeout(() => setControlState(''), 5000)
      window.setTimeout(() => void refresh(), 250)
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

  const viewProps = {
    dashboard,
    selected,
    setSelected,
    bars,
    setActive,
    runControl,
  }

  const renderView = () => {
    if (active === 'Scanner') return <ScannerView {...viewProps} />
    if (active === 'Stock Intelligence') return <StockIntelligenceView {...viewProps} />
    if (active === 'Research Data') return <ResearchDataView symbol={selected} />
    if (active === 'Portfolio') return <PortfolioView {...viewProps} />
    if (active === 'Market Internals') return <MarketInternalsView {...viewProps} />
    if (active === 'Long-Term') return <LongTermView {...viewProps} />
    if (active === 'News & Events') return <NewsView {...viewProps} />
    if (active === 'F&O Desk') return <FnoView {...viewProps} />
    if (active === 'Automation') return <AutomationView {...viewProps} />
    return <CommandCenterView {...viewProps} />
  }

  return (
    <div className="terminal-root">
      <MarketSidebar active={active} setActive={setActive} dashboard={dashboard} />
      <main className="workspace">
        <header className="topbar">
          <div className="search-box">
            ⌕
            <input
              aria-label="Search saved QuantTerm symbols"
              placeholder="Search saved stocks…"
              value={query}
              onChange={(event: { target: { value: string } }) => setQuery(event.target.value)}
              onKeyDown={(event: { key: string }) => { if (event.key === 'Enter') openSearch() }}
              list="quantterm-symbols"
            />
            <datalist id="quantterm-symbols">{symbols.slice(0, 800).map((symbol) => <option value={symbol} key={symbol} />)}</datalist>
            <button type="button" onClick={openSearch}>Open</button>
          </div>
          <div className="top-status">
            <span className={dashboard.operations.running ? 'live-pill' : 'offline-pill'}>
              <i /> {dashboard.operations.running ? 'MARKET OPS ONLINE' : 'MARKET OPS OFFLINE'}
            </span>
            <span className={dashboard.autonomy.running ? 'live-pill' : 'offline-pill'}>
              <i /> {dashboard.autonomy.running ? 'AUTONOMY ONLINE' : 'AUTONOMY OFFLINE'}
            </span>
            <button type="button" onClick={() => void refresh()} aria-label="Refresh dashboard">↻</button>
          </div>
        </header>

        <section className="page-title">
          <div><h1>{active}</h1><p>{pageSubtitles[active]}</p></div>
          <div className="page-actions">
            <button type="button" disabled={!selected} onClick={openEquityReport}>Equity Evidence PDF</button>
            <button type="button" onClick={openBasketReport}>Top-3 Basket PDF</button>
            <span>{controlState || (loading ? 'Loading real state…' : `Updated ${dashboard.generated_at ? new Date(dashboard.generated_at).toLocaleTimeString('en-IN') : '—'}`)}</span>
          </div>
        </section>

        {error && <div className="api-warning">API unavailable: {error}. No numbers below should be treated as current until the local API reconnects.</div>}
        <OperationsRibbon dashboard={dashboard} />
        {renderView()}
      </main>
    </div>
  )
}

export default App
