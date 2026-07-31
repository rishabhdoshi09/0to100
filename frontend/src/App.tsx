import { useEffect, useMemo, useState } from 'react'
import { fetchChart, fetchDashboard, sendControl } from './api'
import { PriceChart } from './PriceChart'
import type { ChartBar, DashboardPayload, LongTermRecord, ScanRecord } from './types'

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
  },
  scan: { available: false, universe_size: 0, summary: {}, records: [] },
  long_term: { available: false, summary: {}, records: [] },
  paper: {
    enabled: false,
    supervisor_running: false,
    capital: 0,
    equity: 0,
    open_risk: 0,
    risk_per_trade_pct: 0.01,
    max_positions: 0,
    open_positions: [],
    closed_trades: [],
  },
  autonomy: {
    running: false,
    state: 'UNKNOWN',
    plain_state: 'Autonomy status unavailable.',
    explanation: '',
    heartbeat_ist: '',
    new_paper_entries: false,
    recent_dialogue: [],
    jobs: {},
  },
  conviction: [],
}

const money = (value?: number) =>
  Number.isFinite(value) ? `₹${Number(value).toLocaleString('en-IN', { maximumFractionDigits: 0 })}` : '—'

const pct = (value?: number | null) =>
  Number.isFinite(value) ? `${Number(value) >= 0 ? '+' : ''}${Number(value).toFixed(2)}%` : '—'

const score = (value?: number) => Number.isFinite(value) ? Math.round(Number(value)) : 0

function Logo() {
  return (
    <div className="brand-mark" aria-hidden="true">
      <span />
      <span />
      <span />
    </div>
  )
}

function Sidebar({ active, setActive }: { active: string; setActive: (value: string) => void }) {
  const items = [
    ['⌘', 'Command Center'],
    ['◉', 'Scanner'],
    ['◎', 'Stock Intelligence'],
    ['▣', 'Portfolio'],
    ['↗', 'Market Internals'],
    ['◇', 'Long-Term'],
    ['◫', 'Research'],
    ['◌', 'Automation'],
    ['⚙', 'Settings'],
  ]
  return (
    <aside className="sidebar">
      <div className="brand"><Logo /><div><strong>QUANTTERM</strong><small>PROFESSIONAL</small></div></div>
      <nav>
        {items.map(([icon, label]) => (
          <button
            key={label}
            className={active === label ? 'nav-item active' : 'nav-item'}
            type="button"
            onClick={() => setActive(label)}
          >
            <span>{icon}</span>{label}
          </button>
        ))}
      </nav>
      <div className="sidebar-spacer" />
      <div className="broker-card">
        <div className="broker-row"><strong>ZERODHA</strong><span className="status-dot" /> </div>
        <small>Research and paper mode</small>
        <div className="broker-stats">
          <div><span>Paper Equity</span><strong id="side-equity">—</strong></div>
          <div><span>Mode</span><strong>PAPER</strong></div>
        </div>
      </div>
      <div className="system-mini">
        <span>System health</span>
        <strong>Supervisor-owned</strong>
        <div className="health-track"><i /></div>
      </div>
    </aside>
  )
}

function MetricCard({ label, value, detail, tone = 'cyan' }: {
  label: string; value: string; detail?: string; tone?: 'cyan' | 'green' | 'purple' | 'amber'
}) {
  return (
    <article className={`metric metric-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail || 'No additional reading'}</small>
    </article>
  )
}

function MomentumTable({ rows, selected, onSelect }: {
  rows: ScanRecord[]; selected?: string; onSelect: (symbol: string) => void
}) {
  return (
    <div className="table-shell">
      <div className="table-head"><span>#</span><span>STOCK</span><span>SCORE</span><span>PRICE</span><span>CHANGE</span><span>SETUP</span></div>
      {rows.length === 0 && <div className="empty-row">No saved scan candidates yet.</div>}
      {rows.map((row, index) => (
        <button
          key={row.symbol}
          type="button"
          className={selected === row.symbol ? 'table-row selected' : 'table-row'}
          onClick={() => onSelect(row.symbol)}
        >
          <span>{index + 1}</span>
          <strong>{row.symbol}</strong>
          <span className="score-cell">{score(row.score)}</span>
          <span>{money(row.price)}</span>
          <span className={(row.momentum_5d || 0) >= 0 ? 'positive' : 'negative'}>{pct(row.momentum_5d)}</span>
          <span>{row.signals?.[0]?.replaceAll('_', ' ') || row.status || 'Watch'}</span>
        </button>
      ))}
    </div>
  )
}

function LongTermTable({ rows }: { rows: LongTermRecord[] }) {
  return (
    <div className="compact-list">
      {rows.length === 0 && <div className="empty-row">Long-term shortlist has not been generated.</div>}
      {rows.slice(0, 5).map((row, index) => (
        <div className="compact-row" key={row.symbol}>
          <span>{index + 1}</span>
          <strong>{row.symbol}</strong>
          <span>{score(row.fundamental_score)}</span>
          <span>{score(row.technical_score)}</span>
          <b>{score(row.combined_score)}</b>
        </div>
      ))}
    </div>
  )
}

function App() {
  const [dashboard, setDashboard] = useState<DashboardPayload>(emptyDashboard)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [active, setActive] = useState('Command Center')
  const [selected, setSelected] = useState('')
  const [bars, setBars] = useState<ChartBar[]>([])
  const [controlState, setControlState] = useState('')

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
    const timer = window.setInterval(() => void refresh(), 30_000)
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

  useEffect(() => {
    const node = document.getElementById('side-equity')
    if (node) node.textContent = money(dashboard.paper.equity)
  }, [dashboard.paper.equity])

  const momentum = useMemo(
    () => dashboard.scan.records
      .filter((row) => row.signals?.includes('MOMENTUM') || row.verdict === 'BUY')
      .sort((a, b) => (b.score || 0) - (a.score || 0))
      .slice(0, 6),
    [dashboard.scan.records],
  )

  const longTerm = useMemo(
    () => dashboard.long_term.records
      .filter((row) => ['QUALITY_COMPOUNDER', 'GARP_CANDIDATE', 'QUALITY_BUT_EXPENSIVE'].includes(row.classification || ''))
      .sort((a, b) => (b.combined_score || 0) - (a.combined_score || 0)),
    [dashboard.long_term.records],
  )

  const selectedRow = dashboard.scan.records.find((row) => row.symbol === selected)
    || dashboard.long_term.records.find((row) => row.symbol === selected)

  const runControl = async (control: 'RUN_SCAN_NOW' | 'RUN_LONG_TERM_SCAN_NOW' | 'RUN_CYCLE_NOW') => {
    setControlState('Queuing…')
    try {
      const result = await sendControl(control)
      setControlState(result.accepted ? 'Queued with autonomy' : 'Not accepted')
      window.setTimeout(() => setControlState(''), 2500)
    } catch (reason) {
      setControlState(reason instanceof Error ? reason.message : 'Control failed')
    }
  }

  const marketTone = dashboard.market.health.toLowerCase() === 'healthy' ? 'green' : 'amber'
  const scanSummary = dashboard.scan.summary
  const ltSummary = dashboard.long_term.summary
  const paperReturn = dashboard.paper.capital > 0
    ? ((dashboard.paper.equity / dashboard.paper.capital) - 1) * 100
    : null

  return (
    <div className="terminal-root">
      <Sidebar active={active} setActive={setActive} />
      <main className="workspace">
        <header className="topbar">
          <div className="search-box">⌕ <input aria-label="Search" placeholder="Search stocks, sectors, themes…" /></div>
          <div className="top-status">
            <span className={dashboard.autonomy.running ? 'live-pill' : 'offline-pill'}>
              <i /> {dashboard.autonomy.running ? 'AUTONOMY ONLINE' : 'AUTONOMY OFFLINE'}
            </span>
            <span>Heartbeat {dashboard.autonomy.heartbeat_ist || '—'}</span>
            <button type="button" onClick={() => void refresh()}>↻</button>
          </div>
        </header>

        <section className="page-title">
          <div><h1>{active}</h1><p>Institutional discipline, translated for a serious retail trader.</p></div>
          <div className="page-actions">
            <span>{controlState}</span>
            <button type="button" onClick={() => void runControl('RUN_SCAN_NOW')}>Run Fresh Scan</button>
          </div>
        </section>

        {error && <div className="api-warning">API unavailable: {error}. The interface is live, but real values require the QuantTerm terminal API.</div>}

        <section className="metric-grid">
          <MetricCard label="MARKET HEALTH" value={dashboard.market.health.toUpperCase()} detail={dashboard.market.breadth} tone={marketTone} />
          <MetricCard label="NIFTY TODAY" value={pct(dashboard.market.nifty_change_1d)} detail={`5D ${pct(dashboard.market.nifty_change_5d)}`} tone="green" />
          <MetricCard label="ENTRY READY" value={String(scanSummary.ready_to_trade ?? 0)} detail={`${scanSummary.near_breakout ?? 0} near breakout`} />
          <MetricCard label="VOLATILITY" value={dashboard.market.vix ? dashboard.market.vix.toFixed(2) : '—'} detail="India VIX" tone="purple" />
          <MetricCard label="LONG-HORIZON" value={String((ltSummary.quality_compounder ?? 0) + (ltSummary.garp_candidate ?? 0))} detail={`${ltSummary.coverage_pct ?? 0}% fundamental coverage`} tone="purple" />
          <MetricCard label="SYSTEM EDGE" value={dashboard.autonomy.state || 'UNKNOWN'} detail={dashboard.autonomy.plain_state} tone="cyan" />
        </section>

        <section className="dashboard-grid">
          <article className="panel momentum-panel">
            <div className="panel-title"><div><strong>TOP MOMENTUM SETUPS</strong><small>{dashboard.scan.universe_size.toLocaleString('en-IN')} stocks evaluated</small></div><button type="button" onClick={() => setActive('Scanner')}>View All</button></div>
            <MomentumTable rows={momentum} selected={selected} onSelect={setSelected} />
            <footer><span>{dashboard.scan.scanned_at ? `Updated ${dashboard.scan.scanned_at.slice(0, 19)}` : 'No saved scan'}</span><button type="button" onClick={() => void runControl('RUN_SCAN_NOW')}>Refresh</button></footer>
          </article>

          <article className="panel chart-panel">
            <div className="panel-title">
              <div><strong>CHART · {selected || 'SELECT STOCK'}</strong><small>{selectedRow && 'price' in selectedRow ? `${money(selectedRow.price)} · ${selectedRow.sector || 'Sector unavailable'}` : 'No selected security'}</small></div>
              <div className="timeframes"><span>1m</span><span>5m</span><span>15m</span><span>1h</span><b>1D</b></div>
            </div>
            {bars.length > 0 ? <PriceChart symbol={selected} bars={bars} /> : <div className="chart-empty"><div className="chart-grid" /><strong>No chart data</strong><span>Start the API or select a symbol with saved bhavcopy history.</span></div>}
            <div className="ohlc-strip">
              <div><span>ENTRY</span><strong>{money((selectedRow as ScanRecord | undefined)?.entry)}</strong></div>
              <div><span>STOP</span><strong className="negative">{money((selectedRow as ScanRecord | undefined)?.stop)}</strong></div>
              <div><span>TARGET</span><strong className="positive">{money((selectedRow as ScanRecord | undefined)?.target)}</strong></div>
              <div><span>RSI</span><strong>{Number.isFinite((selectedRow as ScanRecord | undefined)?.rsi) ? Number((selectedRow as ScanRecord).rsi).toFixed(0) : '—'}</strong></div>
              <div><span>VOLUME</span><strong>{Number.isFinite((selectedRow as ScanRecord | undefined)?.volume_ratio) ? `${Number((selectedRow as ScanRecord).volume_ratio).toFixed(1)}×` : '—'}</strong></div>
            </div>
          </article>

          <aside className="right-stack">
            <article className="panel portfolio-panel">
              <div className="panel-title"><strong>PORTFOLIO OVERVIEW</strong><button type="button">View Details</button></div>
              <span>Total Paper Equity</span>
              <h2>{money(dashboard.paper.equity)} <small className={(paperReturn || 0) >= 0 ? 'positive' : 'negative'}>{pct(paperReturn)}</small></h2>
              <div className="sparkline"><svg viewBox="0 0 220 64" preserveAspectRatio="none"><polyline points="0,52 22,45 44,49 66,34 88,39 110,24 132,31 154,18 176,22 198,10 220,4" /></svg></div>
              <div className="portfolio-stats"><div><span>Open Positions</span><strong>{dashboard.paper.open_positions.length}</strong></div><div><span>Open Risk</span><strong>{money(dashboard.paper.open_risk)}</strong></div><div><span>Max Positions</span><strong>{dashboard.paper.max_positions}</strong></div></div>
            </article>

            <article className="panel alerts-panel">
              <div className="panel-title"><strong>RECENT SYSTEM INSIGHTS</strong><button type="button">View All</button></div>
              <div className="insight"><i className="cyan" /> <div><strong>{dashboard.market.health} market regime</strong><span>{dashboard.market.trade_stance}</span></div></div>
              <div className="insight"><i className="green" /> <div><strong>{scanSummary.with_any_setup ?? 0} current setups</strong><span>{scanSummary.momentum ?? 0} momentum candidates</span></div></div>
              <div className="insight"><i className="purple" /> <div><strong>{ltSummary.quality_compounder ?? 0} quality compounders</strong><span>{ltSummary.garp_candidate ?? 0} GARP candidates</span></div></div>
              <div className="insight"><i className="amber" /> <div><strong>{dashboard.autonomy.state}</strong><span>{dashboard.autonomy.explanation || dashboard.autonomy.plain_state}</span></div></div>
            </article>
          </aside>

          <article className="panel longterm-panel">
            <div className="panel-title"><div><strong>LONG-TERM SCREENER</strong><small>Quality, growth, value and technical timing</small></div><button type="button" onClick={() => void runControl('RUN_LONG_TERM_SCAN_NOW')}>Run LT Scan</button></div>
            <div className="compact-head"><span>#</span><span>STOCK</span><span>QUALITY</span><span>TECH</span><span>SCORE</span></div>
            <LongTermTable rows={longTerm} />
          </article>

          <article className="panel sector-panel">
            <div className="panel-title"><strong>SECTOR LEADERSHIP</strong><small>Current regime engine</small></div>
            <div className="sector-bars">
              {[...dashboard.market.leaders.slice(0, 4), ...dashboard.market.laggards.slice(0, 2)].map((sector, index) => {
                const leader = dashboard.market.leaders.includes(sector)
                const width = Math.max(24, 88 - index * 9)
                return <div key={`${sector}-${index}`}><span>{sector}</span><i className={leader ? 'bar-positive' : 'bar-negative'} style={{ width: `${width}%` }} /><b className={leader ? 'positive' : 'negative'}>{leader ? 'LEADING' : 'LAGGING'}</b></div>
              })}
              {dashboard.market.leaders.length + dashboard.market.laggards.length === 0 && <div className="empty-row">Sector leadership unavailable.</div>}
            </div>
          </article>

          <article className="panel automation-panel">
            <div className="panel-title"><strong>AUTOMATION STATUS</strong><small>Single mutation owner</small></div>
            <div className="automation-state"><span className={dashboard.autonomy.running ? 'pulse green' : 'pulse red'} /><div><small>Autonomy</small><strong>{dashboard.autonomy.running ? 'ACTIVE' : 'OFFLINE'}</strong></div></div>
            <div className="automation-grid"><div><span>Live Data</span><strong>{dashboard.scan.available ? 'Saved + Ready' : 'Unavailable'}</strong></div><div><span>Paper Entries</span><strong>{dashboard.autonomy.new_paper_entries ? 'Allowed' : 'Blocked'}</strong></div><div><span>Heartbeat</span><strong>{dashboard.autonomy.heartbeat_ist || '—'}</strong></div><div><span>Paper Mode</span><strong>{dashboard.paper.enabled ? 'Enabled' : 'Disabled'}</strong></div></div>
            <div className="automation-actions"><button type="button" onClick={() => void runControl('RUN_CYCLE_NOW')}>Run Paper Cycle</button><button type="button" onClick={() => setActive('Automation')}>Open Controls</button></div>
          </article>

          <article className="panel positions-panel">
            <div className="panel-title"><strong>ACTIVE PAPER POSITIONS</strong><small>Simulation only · LIVE orders locked</small></div>
            <div className="positions-head"><span>STOCK</span><span>ENTRY</span><span>CURRENT</span><span>P&amp;L</span><span>STOP</span><span>TARGET</span><span>STRATEGY</span></div>
            {dashboard.paper.open_positions.length === 0 && <div className="empty-row">No active paper positions.</div>}
            {dashboard.paper.open_positions.slice(0, 5).map((position, index) => (
              <div className="positions-row" key={`${String(position.symbol)}-${index}`}>
                <strong>{String(position.symbol || '—')}</strong><span>{money(Number(position.entry_price || 0))}</span><span>{money(Number(position.current_price || 0))}</span><span className={Number(position.pnl || 0) >= 0 ? 'positive' : 'negative'}>{money(Number(position.pnl || 0))}</span><span>{money(Number(position.stop || 0))}</span><span>{money(Number(position.target || 0))}</span><span>{String(position.strategy || 'Research')}</span>
              </div>
            ))}
          </article>
        </section>

        {loading && <div className="loading-overlay">Loading QuantTerm intelligence…</div>}
      </main>
    </div>
  )
}

export default App
