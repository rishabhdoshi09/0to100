import { useEffect, useMemo, useState } from 'react'
import { fetchChart, fetchDashboard, sendControl } from './api'
import { Sidebar } from './components'
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
    trade_stance: 'Start the QuantTerm API and autonomy supervisor.',
    breadth: '—',
    leaders: [],
    laggards: [],
    nifty_change_1d: null,
    nifty_change_5d: null,
    vix: null,
    technical_details: {},
  },
  scan: { available: false, universe_size: 0, summary: {}, records: [] },
  long_term: { available: false, summary: {}, records: [] },
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
  conviction: [],
}

const pageSubtitles: Record<string, string> = {
  'Command Center': 'Market posture, opportunities, paper portfolio and system health in one decision surface.',
  Scanner: 'One ranked workspace for momentum, conviction, breakouts, pre-breakouts and avoid lists.',
  'Stock Intelligence': 'Price structure, evidence, quality, risk and invalidation for the selected stock.',
  Portfolio: 'Paper capital, open risk, recorded equity, positions and closed-trade attribution.',
  'Market Internals': 'Regime, breadth, volatility, sector leadership and scanner coverage.',
  'Long-Term': 'Current business-quality, valuation and technical-timing research with explicit data coverage.',
  Automation: 'Live supervisor heartbeat, durable jobs, failures, controls and operational dialogue.',
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
      const first = payload.scan.records[0]?.symbol || payload.long_term.records[0]?.symbol || ''
      setSelected((current) => current || first)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Dashboard API unavailable')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void refresh()
    const timer = window.setInterval(() => void refresh(), 5_000)
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
  }, [selected])

  const symbols = useMemo(() => {
    const values = [
      ...dashboard.scan.records.map((row) => row.symbol),
      ...dashboard.long_term.records.map((row) => row.symbol),
    ]
    return [...new Set(values)].sort()
  }, [dashboard.long_term.records, dashboard.scan.records])

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
    setControlState('Queuing owner control…')
    try {
      const result = await sendControl(control)
      setControlState(result.accepted ? `${control.replaceAll('_', ' ')} queued` : 'Control was not accepted')
      window.setTimeout(() => setControlState(''), 3000)
      window.setTimeout(() => void refresh(), 750)
    } catch (reason) {
      setControlState(reason instanceof Error ? reason.message : 'Control failed')
    }
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
    if (active === 'Portfolio') return <PortfolioView {...viewProps} />
    if (active === 'Market Internals') return <MarketInternalsView {...viewProps} />
    if (active === 'Long-Term') return <LongTermView {...viewProps} />
    if (active === 'Automation') return <AutomationView {...viewProps} />
    return <CommandCenterView {...viewProps} />
  }

  return (
    <div className="terminal-root">
      <Sidebar active={active} setActive={setActive} dashboard={dashboard} />
      <main className="workspace">
        <header className="topbar">
          <div className="search-box">
            ⌕
            <input
              aria-label="Search saved QuantTerm symbols"
              placeholder="Search saved stocks…"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={(event) => { if (event.key === 'Enter') openSearch() }}
              list="quantterm-symbols"
            />
            <datalist id="quantterm-symbols">{symbols.slice(0, 500).map((symbol) => <option value={symbol} key={symbol} />)}</datalist>
            <button type="button" onClick={openSearch}>Open</button>
          </div>
          <div className="top-status">
            <span className={dashboard.autonomy.running ? 'live-pill' : 'offline-pill'}>
              <i /> {dashboard.autonomy.running ? 'AUTONOMY ONLINE' : 'AUTONOMY OFFLINE'}
            </span>
            <span>Heartbeat {dashboard.autonomy.heartbeat_ist || '—'}</span>
            <button type="button" onClick={() => void refresh()} aria-label="Refresh dashboard">↻</button>
          </div>
        </header>

        <section className="page-title">
          <div><h1>{active}</h1><p>{pageSubtitles[active]}</p></div>
          <div className="page-actions">
            <span>{controlState || (loading ? 'Loading real state…' : `Updated ${dashboard.generated_at ? new Date(dashboard.generated_at).toLocaleTimeString('en-IN') : '—'}`)}</span>
          </div>
        </section>

        {error && <div className="api-warning">API unavailable: {error}. No numbers below should be treated as current until the local API reconnects.</div>}
        {renderView()}
      </main>
    </div>
  )
}

export default App
