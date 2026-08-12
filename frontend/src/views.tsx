import { useEffect, useMemo, useState } from 'react'
import {
  ChartWorkspace,
  EvidenceList,
  JobLedger,
  LongTermTable,
  MetricCard,
  Panel,
  PositionsTable,
  SecurityTable,
} from './components'
import { boolLabel, money, pct, score, words } from './format'
import {
  exportCorporateActionGaps,
  fetchCorporateActionsStatus,
  fetchHoldings,
  fetchInstitutionalStack,
  fetchSignalBacktestStatus,
  fetchTargetPortfolio,
  importHoldings,
  refreshFiiDiiStore,
  syncHoldings,
  verifyCorporateActions,
  type CorporateActionsStatus,
  type HoldingsBook,
  type InstitutionalDomain,
  type InstitutionalStack,
  type SignalBacktestStatus,
} from './productApi'
import { longTermPicks } from './longTermPicks'
import type {
  ChartBar,
  ControlName,
  ConvictionRecord,
  DashboardPayload,
  LongTermRecord,
  ScanRecord,
} from './types'

type ViewProps = {
  dashboard: DashboardPayload
  selected: string
  setSelected: (symbol: string) => void
  bars: ChartBar[]
  setActive: (page: string) => void
  runControl: (control: ControlName) => Promise<void>
}

const findRow = (dashboard: DashboardPayload, symbol: string) =>
  dashboard.conviction.find((row) => row.symbol === symbol)
  || dashboard.scan.records.find((row) => row.symbol === symbol)
  || dashboard.long_term.records.find((row) => row.symbol === symbol)

const momentumRows = (dashboard: DashboardPayload) => dashboard.scan.records
  .filter((row) => row.signals?.includes('MOMENTUM') || row.verdict === 'BUY')
  .sort((a, b) => (b.score || 0) - (a.score || 0))

const qualityRows = (dashboard: DashboardPayload) => longTermPicks(dashboard.long_term.records)
  .sort((a, b) => (b.combined_score || 0) - (a.combined_score || 0))

function EquityCurve({ values }: { values?: number[] }) {
  if (!values || values.length < 2) {
    return <div className="curve-unavailable">Equity history unavailable; QuantTerm will not draw a synthetic curve.</div>
  }
  const width = 420
  const height = 110
  const min = Math.min(...values)
  const max = Math.max(...values)
  const span = Math.max(1, max - min)
  const points = values.map((value, index) => {
    const x = (index / Math.max(1, values.length - 1)) * width
    const y = height - ((value - min) / span) * (height - 12) - 6
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
  return (
    <svg className="real-equity-curve" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Recorded paper equity curve">
      <polyline points={points} />
    </svg>
  )
}

function DataReadinessPanel({ dashboard }: { dashboard: DashboardPayload }) {
  const data = dashboard.data
  const history = data.bhavcopy
  const snapshot = data.snapshot
  return (
    <Panel title="DATA PIPELINE" subtitle={data.ready ? 'Canonical market data is available' : 'One or more authoritative inputs are not ready'}>
      <div className="fact-grid">
        <div><span>Market operations</span><strong className={dashboard.operations.running ? 'positive' : 'negative'}>{dashboard.operations.running ? 'ONLINE' : 'OFFLINE'}</strong></div>
        <div><span>Official history</span><strong className={history.ready ? 'positive' : 'negative'}>{history.ready ? `${history.sessions} sessions` : 'NOT READY'}</strong></div>
        <div><span>History symbols</span><strong>{history.symbols.toLocaleString('en-IN')}</strong></div>
        <div><span>Saved scanner rows</span><strong>{data.scan_records.toLocaleString('en-IN')}</strong></div>
        <div><span>Long-term rows</span><strong>{data.long_term_records.toLocaleString('en-IN')}</strong></div>
        <div><span>Verified paper snapshot</span><strong className={snapshot.ready ? 'positive' : 'negative'}>{snapshot.ready ? snapshot.latest_date || 'READY' : 'MISSING'}</strong></div>
      </div>
      <EvidenceList title="Current blockers" items={data.blockers} tone={data.blockers.length ? 'red' : 'green'} />
    </Panel>
  )
}

export function CommandCenterView(props: ViewProps) {
  const { dashboard, selected, setSelected, bars, setActive, runControl } = props
  const momentum = useMemo(() => momentumRows(dashboard), [dashboard])
  const longTerm = useMemo(() => qualityRows(dashboard), [dashboard])
  const selectedRow = findRow(dashboard, selected)
  const summary = dashboard.scan.summary
  const lt = dashboard.long_term.summary
  const latestScan = dashboard.operations.latest.MARKET_SCAN
  const latestLongTerm = dashboard.operations.latest.LONG_TERM_SCAN
  const insights = [
    dashboard.market.trade_stance,
    dashboard.operations.running
      ? `${dashboard.operations.active.length} market operation(s) active across independent lanes.`
      : 'Market operations worker is offline; direct scans are unavailable.',
    `${summary.with_any_setup ?? 0} saved setup(s); ${summary.ready_to_trade ?? 0} entry-ready.`,
    `${(lt.quality_compounder ?? 0) + (lt.garp_candidate ?? 0)} quality/GARP long-horizon candidate(s).`,
  ]

  return (
    <>
      {!dashboard.data.ready && <div className="api-warning">Market-data pipeline is incomplete. QuantTerm is showing only persisted facts; missing values are not simulated.</div>}
      <section className="metric-grid">
        <MetricCard label="MARKET HEALTH" value={dashboard.market.health.toUpperCase()} detail={dashboard.market.breadth} tone={dashboard.market.health.toLowerCase() === 'healthy' ? 'green' : 'amber'} />
        <MetricCard label="MARKET OPS" value={dashboard.operations.running ? 'ONLINE' : 'OFFLINE'} detail={`${dashboard.operations.active.length} active · PID ${dashboard.operations.worker_pid || '—'}`} tone={dashboard.operations.running ? 'green' : 'amber'} />
        <MetricCard label="ENTRY READY" value={String(summary.ready_to_trade ?? 0)} detail={`${summary.near_breakout ?? 0} near breakout`} />
        <MetricCard label="LONG-HORIZON" value={String((lt.quality_compounder ?? 0) + (lt.garp_candidate ?? 0))} detail={`${lt.coverage_pct ?? 0}% fundamental coverage`} tone="purple" />
        <MetricCard label="NEWS 24H" value={String(dashboard.news.stats.total || 0)} detail={`${dashboard.news.stats.important || 0} high-impact · ${dashboard.news.stats.sources || 0} sources`} tone="cyan" />
        <MetricCard label="F&O UNIVERSE" value={String(dashboard.fno.mapped_underlyings || 0)} detail={`Source ${dashboard.fno.source || 'unavailable'}`} tone={dashboard.fno.available ? 'green' : 'amber'} />
      </section>

      <section className="dashboard-grid">
        <Panel title="TOP MOMENTUM SETUPS" subtitle={`${dashboard.scan.universe_size.toLocaleString('en-IN')} stocks evaluated`} className="momentum-panel" action={<button type="button" onClick={() => setActive('Scanner')}>View all</button>}>
          <SecurityTable rows={momentum} selected={selected} onSelect={setSelected} limit={7} />
          <footer><span>{dashboard.scan.scanned_at ? `Updated ${dashboard.scan.scanned_at.slice(0, 19)}` : 'No saved scan'}</span><button type="button" onClick={() => void runControl('RUN_SCAN_NOW')}>Run scan now</button></footer>
        </Panel>

        <Panel title={`CHART · ${selected || 'SELECT STOCK'}`} subtitle="Daily history · official saved bhavcopy source" className="chart-panel">
          <ChartWorkspace symbol={selected} bars={bars} row={selectedRow} />
        </Panel>

        <aside className="right-stack">
          <Panel title="MARKET OPERATIONS" action={<button type="button" onClick={() => setActive('Automation')}>Inspect</button>}>
            <div className="key-value-list">
              <div><span>Worker</span><strong className={dashboard.operations.running ? 'positive' : 'negative'}>{dashboard.operations.running ? `ONLINE · ${dashboard.operations.worker_pid || '—'}` : 'OFFLINE'}</strong></div>
              <div><span>Momentum scan</span><strong>{latestScan ? `${latestScan.status} · ${words(latestScan.stage)}` : 'NOT RUN'}</strong></div>
              <div><span>Long-term scan</span><strong>{latestLongTerm ? `${latestLongTerm.status} · ${words(latestLongTerm.stage)}` : 'NOT RUN'}</strong>{latestLongTerm?.message ? <small style={{ display: 'block', opacity: 0.75 }}>{latestLongTerm.message}</small> : null}</div>
              <div><span>History</span><strong>{dashboard.data.bhavcopy.ready ? `${dashboard.data.bhavcopy.sessions} sessions` : 'PREPARING'}</strong></div>
            </div>
          </Panel>
          <Panel title="MARKET KNOWLEDGE" action={<button type="button" onClick={() => setActive('News & Events')}>Open news</button>}>
            <div className="fact-grid">
              <div><span>High-impact news</span><strong>{dashboard.news.stats.important || 0}</strong></div>
              <div><span>F&O-linked news</span><strong>{dashboard.news.stats.fno_linked || 0}</strong></div>
              <div><span>F&O underlyings</span><strong>{dashboard.fno.mapped_underlyings || 0}</strong></div>
              <div><span>Data blockers</span><strong className={dashboard.data.blockers.length ? 'negative' : 'positive'}>{dashboard.data.blockers.length}</strong></div>
            </div>
          </Panel>
          <Panel title="SYSTEM INSIGHTS">
            <div className="insight-list">{insights.map((item, index) => <div className="insight" key={`${item}-${index}`}><i className={index === 1 && dashboard.operations.running ? 'green' : 'cyan'} /><span>{item}</span></div>)}</div>
          </Panel>
        </aside>

        <Panel title="LONG-TERM INTELLIGENCE" subtitle="Current quality + valuation + timing" className="longterm-panel" action={<button type="button" onClick={() => setActive('Long-Term')}>Open</button>}>
          <LongTermTable rows={longTerm} selected={selected} onSelect={setSelected} limit={6} />
        </Panel>

        <Panel title="MARKET LEADERSHIP" subtitle={dashboard.market.summary} className="sector-panel" action={<button type="button" onClick={() => setActive('Market Internals')}>Details</button>}>
          <div className="sector-columns"><div><strong className="positive">LEADING</strong>{dashboard.market.leaders.length ? dashboard.market.leaders.map((item) => <span key={item}>{item}</span>) : <span>No clear leader</span>}</div><div><strong className="negative">LAGGING</strong>{dashboard.market.laggards.length ? dashboard.market.laggards.map((item) => <span key={item}>{item}</span>) : <span>No clear laggard</span>}</div></div>
        </Panel>

        <Panel title="PAPER PORTFOLIO · SECONDARY EXECUTION LAYER" subtitle={`${dashboard.paper.open_positions.length} open · equity ${money(dashboard.paper.equity)}`} className="positions-panel" action={<button type="button" onClick={() => setActive('Portfolio')}>Open portfolio</button>}>
          <PositionsTable rows={dashboard.paper.open_positions.slice(0, 8)} />
        </Panel>
      </section>
      {!dashboard.data.ready && <section className="workspace-view"><DataReadinessPanel dashboard={dashboard} /></section>}
    </>
  )
}

export function ScannerView(props: ViewProps) {
  const { dashboard, selected, setSelected, bars, runControl } = props
  const [mode, setMode] = useState('Momentum')
  const rows = useMemo<Array<ScanRecord | ConvictionRecord>>(() => {
    if (mode === 'Conviction') return [...dashboard.conviction].sort((a, b) => (b.conviction_score || 0) - (a.conviction_score || 0))
    if (mode === 'Breakouts') return dashboard.scan.records.filter((row) => row.signals?.some((item) => item.includes('BREAKOUT')) || row.status === 'Ready to trade')
    if (mode === 'Pre-Breakout') return dashboard.scan.records.filter((row) => row.signals?.includes('PRE_BREAKOUT') || row.status === 'Watch for breakout')
    if (mode === 'Avoid') return dashboard.scan.records.filter((row) => row.chase_risk || row.status === 'Wait for pullback')
    return momentumRows(dashboard)
  }, [dashboard, mode])
  const selectedRow = findRow(dashboard, selected) as ScanRecord | ConvictionRecord | undefined

  return (
    <section className="workspace-view">
      {!dashboard.data.bhavcopy.ready && <DataReadinessPanel dashboard={dashboard} />}
      <div className="mode-tabs">
        {['Momentum', 'Conviction', 'Breakouts', 'Pre-Breakout', 'Avoid'].map((item) => <button type="button" key={item} className={mode === item ? 'active' : ''} onClick={() => setMode(item)}>{item}</button>)}
        <button className="mode-action" type="button" onClick={() => void runControl('RUN_SCAN_NOW')}>Scan whole market now</button>
      </div>
      <div className="split-workspace">
        <Panel title={`${mode.toUpperCase()} · ${rows.length} MATCHES`} subtitle={`Saved scan ${dashboard.scan.scanned_at || 'not available'}`}>
          <SecurityTable rows={rows} selected={selected} onSelect={setSelected} />
        </Panel>
        <div className="detail-stack">
          <Panel title={`PRICE STRUCTURE · ${selected || 'SELECT STOCK'}`} subtitle="Daily official history; no synthetic candles">
            <ChartWorkspace symbol={selected} bars={bars} row={selectedRow} />
          </Panel>
          <Panel title="DECISION EVIDENCE">
            <div className="evidence-grid"><EvidenceList title="Why it qualified" items={selectedRow?.reasons} tone="green" /><EvidenceList title="What can invalidate it" items={(selectedRow as ConvictionRecord | undefined)?.risks || (selectedRow?.chase_risk ? ['Price is extended; do not chase.'] : [])} tone="red" /></div>
          </Panel>
        </div>
      </div>
    </section>
  )
}

export function StockIntelligenceView(props: ViewProps) {
  const { dashboard, selected, bars } = props
  const scan = dashboard.scan.records.find((row) => row.symbol === selected)
  const conviction = dashboard.conviction.find((row) => row.symbol === selected)
  const longTerm = dashboard.long_term.records.find((row) => row.symbol === selected)
  const row = conviction || scan || longTerm

  if (!selected || !row) return <section className="workspace-view"><div className="large-empty">Select a stock from Scanner, Long-Term or Command Center to open its intelligence workspace.</div></section>
  return (
    <section className="workspace-view">
      <div className="stock-hero"><div><span>{row.sector || 'Sector unavailable'}</span><h2>{selected}</h2><p>{scan?.company || selected}</p></div><div className="stock-score"><span>Primary score</span><strong>{score(conviction?.conviction_score ?? longTerm?.combined_score ?? scan?.score)}</strong></div></div>
      <div className="intelligence-grid">
        <Panel title="PRICE & STRUCTURE" className="intelligence-chart"><ChartWorkspace symbol={selected} bars={bars} row={row} /></Panel>
        <Panel title="TRADE PLAN"><div className="fact-grid"><div><span>Entry</span><strong>{money(scan?.entry)}</strong></div><div><span>Stop</span><strong className="negative">{money(scan?.stop)}</strong></div><div><span>Target</span><strong className="positive">{money(scan?.target)}</strong></div><div><span>Status</span><strong>{words(conviction?.classification || scan?.status || longTerm?.classification)}</strong></div></div></Panel>
        <Panel title="TECHNICAL EVIDENCE"><EvidenceList title="Recorded confirmations" items={scan?.reasons || longTerm?.quality_factors} tone="green" /></Panel>
        <Panel title="RISK & INVALIDATION"><EvidenceList title="Recorded risks" items={conviction?.risks || longTerm?.risk_flags || (scan?.chase_risk ? ['Chase risk is active.'] : [])} tone="red" /></Panel>
        <Panel title="LONG-HORIZON QUALITY"><div className="fact-grid"><div><span>Fundamental</span><strong>{score(longTerm?.fundamental_score)}</strong></div><div><span>Technical</span><strong>{score(longTerm?.technical_score)}</strong></div><div><span>Coverage</span><strong>{Number.isFinite(longTerm?.fundamental_coverage) ? `${Number(longTerm?.fundamental_coverage) * 100}%` : '—'}</strong></div><div><span>Timing</span><strong>{words(longTerm?.timing)}</strong></div></div></Panel>
        <Panel title="MARKET CONTEXT"><p className="panel-copy">{dashboard.market.summary}</p><p className="panel-copy">{dashboard.market.trade_stance}</p></Panel>
      </div>
    </section>
  )
}

export function PortfolioView({ dashboard, runControl, setSelected, setActive }: ViewProps) {
  const [target, setTarget] = useState<Awaited<ReturnType<typeof fetchTargetPortfolio>> | null>(null)
  const [holdings, setHoldings] = useState<HoldingsBook | null>(null)
  const [holdingsBusy, setHoldingsBusy] = useState('')
  const [importText, setImportText] = useState('')
  const [holdingsError, setHoldingsError] = useState('')
  const paperReturn = dashboard.paper.capital > 0 ? ((dashboard.paper.equity / dashboard.paper.capital) - 1) * 100 : null

  const reloadHoldings = async () => {
    try {
      setHoldings(await fetchHoldings())
      setHoldingsError('')
    } catch (reason) {
      setHoldings(null)
      setHoldingsError(reason instanceof Error ? reason.message : 'Holdings unavailable')
    }
  }

  useEffect(() => {
    let alive = true
    fetchTargetPortfolio()
      .then((payload) => { if (alive) setTarget(payload) })
      .catch(() => { if (alive) setTarget(null) })
    return () => { alive = false }
  }, [dashboard.autonomy.heartbeat_ist, dashboard.paper.equity])

  useEffect(() => {
    void reloadHoldings()
  }, [])

  const summary = target?.summary
  const hSummary = holdings?.summary

  const onSync = async () => {
    setHoldingsBusy('sync')
    setHoldingsError('')
    try {
      const book = await syncHoldings()
      setHoldings(book)
      if (!book.available) setHoldingsError(book.message || 'Sync returned no holdings')
    } catch (reason) {
      setHoldingsError(reason instanceof Error ? reason.message : 'Zerodha sync failed')
    } finally {
      setHoldingsBusy('')
    }
  }

  const onImport = async () => {
    const lines = importText.split(/\n+/).map((line) => line.trim()).filter(Boolean)
    const rows: Array<Record<string, unknown>> = []
    for (const line of lines) {
      // SYMBOL qty avg [ltp]  — e.g. RELIANCE 10 2500 2550
      const parts = line.split(/[\s,]+/).filter(Boolean)
      if (parts.length < 3) continue
      const symbol = parts[0].toUpperCase()
      const quantity = Number(parts[1])
      const average_price = Number(parts[2])
      const last_price = parts[3] != null ? Number(parts[3]) : average_price
      if (!symbol || !(quantity > 0) || !(average_price >= 0)) continue
      rows.push({ tradingsymbol: symbol, quantity, average_price, last_price })
    }
    if (!rows.length) {
      setHoldingsError('Paste your own lines like: RELIANCE 10 2500 2550')
      return
    }
    setHoldingsBusy('import')
    setHoldingsError('')
    try {
      const book = await importHoldings(rows, 'paste')
      setHoldings(book)
      setImportText('')
    } catch (reason) {
      setHoldingsError(reason instanceof Error ? reason.message : 'Import failed')
    } finally {
      setHoldingsBusy('')
    }
  }

  const openHolding = (symbol: string) => {
    const clean = symbol.trim().toUpperCase()
    if (!clean) return
    // Stock Intelligence uses EQ research ticker; strip series suffix for workspace.
    const research = clean.replace(/-(BE|BZ|BL|SM)$/i, '')
    setSelected(research || clean)
    setActive('Stock Intelligence')
  }

  return (
    <section className="workspace-view">
      <div className="inline-actions">
        <button type="button" disabled={!!holdingsBusy} onClick={() => void onSync()}>
          {holdingsBusy === 'sync' ? 'Syncing Zerodha…' : 'Sync Zerodha holdings'}
        </button>
        <button type="button" onClick={() => void runControl('RUN_CYCLE_NOW')}>Request paper cycle</button>
        <button type="button" onClick={() => void runControl(dashboard.autonomy.new_paper_entries ? 'PAUSE_NEW_PAPER_ENTRIES' : 'RESUME_NEW_PAPER_ENTRIES')}>
          {dashboard.autonomy.new_paper_entries ? 'Pause new entries' : 'Resume new entries'}
        </button>
      </div>
      {holdingsError && <div className="api-warning">{holdingsError}</div>}
      <div className="view-metrics">
        <MetricCard label="DEMAT INVESTED" value={money(hSummary?.invested || 0)} detail={holdings?.available ? `${hSummary?.count || 0} holdings · ${holdings.source || 'book'}` : 'Not synced'} />
        <MetricCard label="DEMAT VALUE" value={money(hSummary?.current_value || 0)} detail={hSummary ? pct(hSummary.pnl_pct) : '—'} tone="green" />
        <MetricCard label="DEMAT P&L" value={money(hSummary?.pnl || 0)} detail={hSummary?.day_pnl != null ? `Day ${money(hSummary.day_pnl)}` : '—'} tone={(hSummary?.pnl || 0) >= 0 ? 'green' : 'amber'} />
        <MetricCard label="PAPER EQUITY" value={money(dashboard.paper.equity)} detail={pct(paperReturn)} tone="purple" />
        {summary && (
          <MetricCard label="TARGET PORTFOLIO" value={target?.available ? `${summary.target_positions} targets` : 'EMPTY'} detail={target?.available ? `Exec ${summary.executable_changes} · blocked ${summary.blocked_changes}` : target?.message || '—'} tone={target?.available ? 'cyan' : 'amber'} />
        )}
      </div>
      <div className="portfolio-workspace">
        <Panel title="MY HOLDINGS · DEMAT" subtitle="Zerodha CNC book · includes -BE series · not paper">
          {!holdings && <p className="panel-copy">Loading demat holdings…</p>}
          {holdings && !holdings.available && (
            <div>
              <p className="panel-copy">{holdings.message || 'No holdings saved yet.'}</p>
              <p className="panel-copy">Sync from Zerodha, or paste lines: SYMBOL QTY AVG [LTP]</p>
            </div>
          )}
          {holdings?.available && (
            <table className="radar-table">
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Qty</th>
                  <th>Avg</th>
                  <th>LTP</th>
                  <th>Invested</th>
                  <th>P&L</th>
                </tr>
              </thead>
              <tbody>
                {holdings.holdings.map((row) => (
                  <tr key={row.tradingsymbol}>
                    <td>
                      <button type="button" className="linkish" onClick={() => openHolding(row.tradingsymbol)}>
                        {row.tradingsymbol}
                      </button>
                    </td>
                    <td>{row.quantity}</td>
                    <td>{money(row.average_price)}</td>
                    <td>{money(row.last_price)}</td>
                    <td>{money(row.invested)}</td>
                    <td className={row.pnl >= 0 ? 'positive' : 'negative'}>
                      {money(row.pnl)} ({pct(row.pnl_pct)})
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <div className="holdings-import" style={{ marginTop: 12 }}>
            <textarea
              aria-label="Paste holdings"
              placeholder={'RELIANCE 10 2500 2550\nTCS 5 3800 3900\nINFY 20 1500'}
              value={importText}
              onChange={(event) => setImportText(event.target.value)}
              rows={4}
              style={{ width: '100%', fontFamily: 'inherit' }}
            />
            <button type="button" disabled={!!holdingsBusy || !importText.trim()} onClick={() => void onImport()}>
              {holdingsBusy === 'import' ? 'Importing…' : 'Import pasted holdings'}
            </button>
            <p className="panel-copy" style={{ marginTop: 8 }}>
              Empty until you Sync your Zerodha account or paste your own demat rows. QuantTerm never invents holdings.
            </p>
          </div>
        </Panel>
        <Panel title="CANONICAL TARGET PORTFOLIO" subtitle="Read-only intelligence target book · not demat">
          {!target && <p className="panel-copy">Loading target portfolio projection…</p>}
          {target && !target.available && <p className="panel-copy">{target.message || 'No persisted target portfolio yet.'}</p>}
          {target?.available && summary && (
            <div className="key-value-list">
              <div><span>Current positions</span><strong>{summary.current_positions}</strong></div>
              <div><span>Target positions</span><strong>{summary.target_positions}</strong></div>
              <div><span>Open risk (current)</span><strong>{summary.current_open_risk_pct}%</strong></div>
              <div><span>Open risk (target)</span><strong>{summary.target_open_risk_pct}%</strong></div>
              <div><span>Available cash</span><strong>{money(summary.available_cash)}</strong></div>
            </div>
          )}
          {target?.positions && target.positions.length > 0 && (
            <table className="radar-table" style={{ marginTop: '12px' }}>
              <thead><tr><th>Symbol</th><th>Qty</th><th>Status</th><th>Risk %</th></tr></thead>
              <tbody>
                {target.positions.slice(0, 20).map((row, idx) => (
                  <tr key={String(row.symbol ?? idx)}>
                    <td>{String(row.symbol ?? '—')}</td>
                    <td>{String(row.desired_quantity ?? row.required_quantity ?? '—')}</td>
                    <td>{String(row.status ?? '—')}</td>
                    <td>{row.target_risk_pct != null ? String(row.target_risk_pct) : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Panel>
        <Panel title="RECORDED EQUITY CURVE" subtitle="Paper evidence · no synthetic history"><EquityCurve values={dashboard.paper.equity_curve} /></Panel>
        <Panel title="OPEN PAPER POSITIONS" subtitle="Secondary paper layer"><PositionsTable rows={dashboard.paper.open_positions} /></Panel>
        <Panel title="RECENT CLOSED TRADES"><PositionsTable rows={[...dashboard.paper.closed_trades].reverse().slice(0, 50)} closed /></Panel>
      </div>
    </section>
  )
}

export function MarketInternalsView({ dashboard, runControl }: ViewProps) {
  const details = dashboard.market.technical_details || {}
  const inst = dashboard.institutional
  const cash = inst?.cash
  const history = cash?.history || []
  const niftyOpts = inst?.nifty_options
  const [instBusy, setInstBusy] = useState(false)

  const refreshInstitutional = async () => {
    setInstBusy(true)
    try {
      await refreshFiiDiiStore()
      await runControl('REFRESH_DATA_NOW')
    } finally {
      setInstBusy(false)
    }
  }

  return (
    <section className="workspace-view">
      {!dashboard.data.ready && <DataReadinessPanel dashboard={dashboard} />}
      <div className="inline-actions">
        <button type="button" disabled={instBusy} onClick={() => void refreshInstitutional()}>
          {instBusy ? 'Syncing FII/DII…' : 'Refresh FII/DII store'}
        </button>
      </div>
      <div className="view-metrics">
        <MetricCard label="REGIME" value={dashboard.market.health} detail={String(details.market_regime || dashboard.market.breadth)} tone={dashboard.market.health.toLowerCase() === 'healthy' ? 'green' : 'amber'} />
        <MetricCard label="NIFTY 1D" value={pct(dashboard.market.nifty_change_1d)} detail={`5D ${pct(dashboard.market.nifty_change_5d)}`} />
        <MetricCard label="INDIA VIX" value={Number.isFinite(dashboard.market.vix) ? Number(dashboard.market.vix).toFixed(2) : '—'} tone="purple" />
        <MetricCard label="FII NET (30D)" value={cash?.totals?.fii_net_cr != null ? `₹${Number(cash.totals.fii_net_cr).toLocaleString('en-IN')} Cr` : '—'} detail={inst?.insight || cash?.note || 'Syncs from NSE when you open this page'} tone={Number(cash?.totals?.fii_net_cr || 0) >= 0 ? 'green' : 'amber'} />
        <MetricCard label="DII NET (30D)" value={cash?.totals?.dii_net_cr != null ? `₹${Number(cash.totals.dii_net_cr).toLocaleString('en-IN')} Cr` : '—'} detail={`Bias ${cash?.bias || '—'}`} />
        <MetricCard label="BULK BUYS" value={String(inst?.bulk_buy_symbols?.length || 0)} detail="Net bulk-buy symbols today" tone="cyan" />
      </div>
      <div className="market-grid">
        <Panel title="MARKET NARRATIVE"><p className="lead-copy">{dashboard.market.summary}</p><p className="panel-copy">{dashboard.market.trade_stance}</p>{inst?.insight && <p className="panel-copy"><strong>Institutional:</strong> {inst.insight}</p>}</Panel>
        <Panel title="FII / DII CASH FLOWS" subtitle="NSE official · ₹ Crore · persisted store">
          {!inst?.available && <p className="panel-copy">FII/DII data loads automatically from NSE on first visit (one quick sync). If NSE is unreachable, numbers stay empty — nothing is fabricated.</p>}
          {history.length > 0 && (
            <div className="fno-table wide-table">
              <div className="fno-head"><span>DATE</span><span>FII NET</span><span>DII NET</span></div>
              {history.slice(0, 12).map((row) => (
                <div className="fno-row" key={row.date} style={{ display: 'grid', cursor: 'default' }}>
                  <strong>{row.date}</strong>
                  <span>{Number(row.fii_net).toLocaleString('en-IN')} Cr</span>
                  <span>{Number(row.dii_net).toLocaleString('en-IN')} Cr</span>
                </div>
              ))}
            </div>
          )}
        </Panel>
        <Panel title="SECTOR LEADERS"><div className="tag-cloud">{dashboard.market.leaders.length ? dashboard.market.leaders.map((item) => <span className="positive-tag" key={item}>{item}</span>) : <span>No clear leaders recorded.</span>}</div></Panel>
        <Panel title="SECTOR LAGGARDS"><div className="tag-cloud">{dashboard.market.laggards.length ? dashboard.market.laggards.map((item) => <span className="negative-tag" key={item}>{item}</span>) : <span>No clear laggards recorded.</span>}</div></Panel>
        <Panel title="BULK DEAL BUYS" subtitle="Symbol-level institutional footprint">
          <div className="tag-cloud">{inst?.bulk_buy_symbols?.length ? inst.bulk_buy_symbols.slice(0, 24).map((sym) => <span className="positive-tag" key={sym}>{sym}</span>) : <span>No net bulk buys in current cache.</span>}</div>
        </Panel>
        <Panel title="BULK DEAL DETAIL" subtitle="NSE largedeal snapshot · client, qty, price">
          {(inst?.bulk_deals?.length ?? 0) > 0 ? (
            <div className="fno-table wide-table">
              <div className="fno-head"><span>SYMBOL</span><span>SIDE</span><span>QTY</span><span>PRICE</span><span>CLIENT</span></div>
              {inst!.bulk_deals!.slice(0, 24).map((deal, idx) => (
                <div className="fno-row" key={`${deal.symbol}-${idx}`} style={{ display: 'grid', cursor: 'default' }}>
                  <strong>{String(deal.symbol ?? '—')}</strong>
                  <span className={String(deal.side) === 'BUY' ? 'positive-tag' : 'negative-tag'}>{String(deal.side ?? '—')}</span>
                  <span>{Number(deal.qty ?? 0).toLocaleString('en-IN')}</span>
                  <span>{money(Number(deal.price ?? 0))}</span>
                  <span>{String(deal.client ?? '').slice(0, 40) || '—'}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="panel-copy">No bulk deals in cache. Data syncs from NSE when institutional flows refresh (same path as Brain bulk tags).</p>
          )}
        </Panel>
        <Panel title="NIFTY OPTIONS" subtitle="Nearest expiry · NSE chain when available">
          {niftyOpts?.available
            ? <div className="key-value-list"><div><span>PCR</span><strong>{String(niftyOpts.pcr)}</strong></div><div><span>Max pain</span><strong>{String(niftyOpts.max_pain)}</strong></div><div><span>Bias</span><strong>{String(niftyOpts.bias)}</strong></div><div><span>Note</span><strong>{String(niftyOpts.note || '')}</strong></div></div>
            : <p className="panel-copy">Index option chain unavailable (NSE may block off-hours). Stock options load on Stock Intelligence → Options tab.</p>}
        </Panel>
        <Panel title="REGIME ENGINE DETAILS"><div className="key-value-list">{Object.entries(details).map(([key, value]) => <div key={key}><span>{words(key)}</span><strong>{String(value ?? '—')}</strong></div>)}</div></Panel>
      </div>
    </section>
  )
}

export function LongTermView(props: ViewProps) {
  const { dashboard, selected, setSelected, bars, runControl } = props
  const [classification, setClassification] = useState('All')
  const rows = useMemo(() => {
    const all = [...dashboard.long_term.records].sort((a, b) => (b.combined_score || 0) - (a.combined_score || 0))
    if (classification === 'Quality') {
      return longTermPicks(all).filter(
        (row) => row.classification === 'QUALITY_COMPOUNDER' || row.classification === 'GARP_CANDIDATE',
      )
    }
    if (classification === 'Expensive') return all.filter((row) => row.classification === 'QUALITY_BUT_EXPENSIVE')
    if (classification === 'Needs Data') return all.filter((row) => row.classification === 'NEEDS_FUNDAMENTALS')
    if (classification === 'Avoid') return all.filter((row) => row.classification === 'AVOID_REVIEW')
    return all
  }, [classification, dashboard.long_term.records])

  useEffect(() => {
    if (rows.length && !rows.some((item) => item.symbol === selected)) setSelected(rows[0].symbol)
  }, [rows, selected, setSelected])

  const row = rows.find((item) => item.symbol === selected)
  const viewSymbol = row?.symbol || ''
  const viewBars = viewSymbol === selected ? bars : []
  const latestOperation = dashboard.operations.latest.LONG_TERM_REFRESH || dashboard.operations.latest.LONG_TERM_SCAN
  const runLabel = dashboard.data.bhavcopy.ready ? 'Run long-term scan now' : 'Prepare history & run scan'
  return (
    <section className="workspace-view">
      {(!dashboard.data.bhavcopy.ready || !dashboard.long_term.available) && <DataReadinessPanel dashboard={dashboard} />}
      <div className="mode-tabs">{['All', 'Quality', 'Expensive', 'Needs Data', 'Avoid'].map((item) => <button type="button" key={item} className={classification === item ? 'active' : ''} onClick={() => setClassification(item)}>{item}</button>)}<button className="mode-action" type="button" onClick={() => void runControl('RUN_LONG_TERM_SCAN_NOW')}>{runLabel}</button><button className="mode-action" type="button" disabled={!dashboard.long_term.records.length} onClick={() => void runControl('REFRESH_LONG_TERM_NOW')}>Fill missing fundamentals</button></div>
      {!dashboard.long_term.available && <div className="api-warning">Latest long-term operation: {String(latestOperation?.status || 'not run')} · {String(latestOperation?.error_message || latestOperation?.message || 'waiting for the dedicated long-term lane')}</div>}
      <div className="split-workspace"><Panel title={`${classification.toUpperCase()} · ${rows.length} RECORDS`} subtitle={`Coverage ${dashboard.long_term.summary.coverage_pct ?? 0}% · ${dashboard.long_term.fundamentals_source || 'current snapshot'}`}><LongTermTable rows={rows} selected={viewSymbol} onSelect={setSelected} /></Panel><div className="detail-stack"><Panel title={`LONG-TERM CHART · ${viewSymbol || 'SELECT STOCK'}`}><ChartWorkspace symbol={viewSymbol} bars={viewBars} row={row} /></Panel><Panel title="QUALITY & RISKS"><div className="evidence-grid"><EvidenceList title="Quality factors" items={row?.quality_factors} tone="green" /><EvidenceList title="Risk flags" items={row?.risk_flags} tone="red" /></div></Panel></div></div>
    </section>
  )
}

function InstitutionalStackPanel() {
  const [stack, setStack] = useState<InstitutionalStack | null>(null)

  useEffect(() => {
    let alive = true
    fetchInstitutionalStack()
      .then((payload) => { if (alive) setStack(payload) })
      .catch(() => { if (alive) setStack(null) })
    return () => { alive = false }
  }, [])

  const readiness = stack?.readiness
  const readinessOk = readiness && 'domains' in readiness ? readiness : null
  const domains = readinessOk?.domains || []
  const systemState = readinessOk?.system_state || 'UNAVAILABLE'

  const serviceCards = stack ? [
    { label: 'OMS', data: stack.oms, detail: String(stack.oms.summary?.orders ?? stack.oms.summary?.open_orders ?? '—') },
    { label: 'RISK GOVERNOR', data: stack.risk_governor, detail: String(stack.risk_governor.mode ?? stack.risk_governor.summary?.decisions ?? '—') },
    { label: 'RECONCILIATION', data: stack.reconciliation, detail: String(stack.reconciliation.summary?.latest_status ?? '—') },
    { label: 'PROTECTION', data: stack.protection, detail: String(stack.protection.summary?.fully_protected ?? '—') },
    { label: 'TCA', data: stack.tca, detail: String(stack.tca.summary?.assessments ?? '—') },
    {
      label: 'BROKER OBSERVER',
      data: stack.broker_observer,
      detail: stack.broker_observer.running ? 'RUNNING' : 'OFF',
      available: Boolean(stack.broker_observer.running) || Boolean(stack.broker_observer.snapshots?.available),
    },
  ] : []

  return (
    <Panel title="INSTITUTIONAL EXECUTION STACK" subtitle="Read-only projections · no live orders from this panel">
      {!stack && <p className="panel-copy">Loading institutional APIs…</p>}
      {stack && (
        <>
          <div className="view-metrics">
            <MetricCard label="SYSTEM STATE" value={systemState} detail={String(readinessOk?.summary || '')} tone={systemState.includes('ELIGIBLE') ? 'green' : 'amber'} />
            <MetricCard label="HARD BLOCKERS" value={String(readinessOk?.hard_blockers?.length ?? 0)} detail={(readinessOk?.hard_blockers || []).slice(0, 3).join(', ') || 'None listed'} tone="purple" />
          </div>
          <div className="runtime-grid">
            {domains.map((domain: InstitutionalDomain) => (
              <article key={domain.key}>
                <span>{domain.label}</span>
                <strong className={`evidence-status ${domain.status === 'READY' ? 'fresh' : domain.status === 'PARTIAL' ? 'stale' : 'missing'}`}>{domain.status}</strong>
                <small>{domain.summary}</small>
                {domain.blockers.length > 0 && <small>Blockers: {domain.blockers.join('; ')}</small>}
              </article>
            ))}
          </div>
          <div className="view-metrics" style={{ marginTop: '12px' }}>
            {serviceCards.map((card) => (
              <MetricCard
                key={card.label}
                label={card.label}
                value={(card.available ?? card.data.available) ? 'AVAILABLE' : 'EMPTY'}
                detail={card.detail}
                tone={(card.available ?? card.data.available) ? 'cyan' : 'amber'}
              />
            ))}
          </div>
          {stack.broker_observer.message && <p className="panel-copy">{stack.broker_observer.message}</p>}
        </>
      )}
    </Panel>
  )
}

export function AutomationView({ dashboard, runControl }: ViewProps) {
  const a = dashboard.autonomy
  const activeJob = a.active_job || {}
  const [bt, setBt] = useState<SignalBacktestStatus | null>(null)
  const [ca, setCa] = useState<CorporateActionsStatus | null>(null)
  const [caBusy, setCaBusy] = useState('')

  useEffect(() => {
    let alive = true
    const load = () => {
      fetchSignalBacktestStatus()
        .then((payload) => { if (alive) setBt(payload) })
        .catch(() => { if (alive) setBt(null) })
      fetchCorporateActionsStatus()
        .then((payload) => { if (alive) setCa(payload) })
        .catch(() => { if (alive) setCa(null) })
    }
    load()
    const timer = window.setInterval(load, 5000)
    return () => { alive = false; window.clearInterval(timer) }
  }, [dashboard.autonomy.heartbeat_ist])

  const uni = bt?.universe || {}
  const btDetail = bt?.running
    ? `${bt.progress || 0}/${bt.total || 0} symbols`
    : bt?.has_report
      ? `${uni.run || bt.symbols_run || 0} stocks · ${bt.generated_at || '—'}`
      : 'No full-universe report yet'

  const caTone = ca?.adjustment_verified ? 'green' : (ca?.events ? 'amber' : 'amber')
  const caValue = ca?.adjustment_verified
    ? 'VERIFIED'
    : (ca?.events ? 'PARTIAL' : 'MISSING')

  const runCaGaps = async () => {
    setCaBusy('Exporting gap TODO…')
    try {
      await exportCorporateActionGaps(400)
      setCa(await fetchCorporateActionsStatus())
    } catch {
      /* surface stays honest empty */
    } finally {
      setCaBusy('')
    }
  }
  const runCaVerify = async () => {
    setCaBusy('Verifying adjustment…')
    try {
      setCa(await verifyCorporateActions(80))
    } catch {
      /* keep prior */
    } finally {
      setCaBusy('')
    }
  }

  return (
    <section className="workspace-view">
      <div className="inline-actions"><button type="button" onClick={() => void runControl('RUN_SCAN_NOW')}>Start market scan</button><button type="button" onClick={() => void runControl('RUN_CYCLE_NOW')}>Request paper cycle</button><button type="button" onClick={() => void runControl('REFRESH_DATA_NOW')}>Prepare market data</button><button type="button" onClick={() => void runControl('RUN_FULL_UNIVERSE_BACKTEST_NOW')}>Backtest all stocks</button><button type="button" onClick={() => void runControl(a.new_paper_entries ? 'PAUSE_NEW_PAPER_ENTRIES' : 'RESUME_NEW_PAPER_ENTRIES')}>{a.new_paper_entries ? 'Pause entries' : 'Resume entries'}</button></div>
      <div className="view-metrics">
        <MetricCard label="PAPER SUPERVISOR" value={a.running ? 'ONLINE' : 'OFFLINE'} detail={`PID ${a.scheduler_owner_pid || '—'}`} tone={a.running ? 'green' : 'amber'} />
        <MetricCard label="STATE" value={a.state} detail={a.plain_state} />
        <MetricCard label="ACTIVE PAPER JOB" value={String(activeJob.job_type || 'IDLE').toUpperCase()} detail={activeJob.elapsed_s ? `${activeJob.elapsed_s}s elapsed` : 'No paper worker job reported'} tone="cyan" />
        <MetricCard
          label="SIGNAL BACKTEST"
          value={bt?.running ? 'RUNNING' : (bt?.has_report ? (uni.truncated ? 'PARTIAL' : 'READY') : 'MISSING')}
          detail={btDetail}
          tone={bt?.running ? 'cyan' : (bt?.has_report && !uni.truncated ? 'green' : 'amber')}
        />
        <MetricCard
          label="CORPORATE ACTIONS"
          value={caValue}
          detail={ca ? `${ca.events} events · ${ca.symbols} symbols` : 'Ledger unread'}
          tone={caTone}
        />
        <MetricCard label="FAILURES" value={String(a.active_failures?.length || 0)} detail={(a.active_failures || []).join(', ') || 'None active'} tone="purple" />
      </div>
      <div className="automation-grid">
        <Panel title="GETTING SMARTER" subtitle="Full-universe backtest feeds scanner ranking + pre-trade gates">
          <div className="key-value-list">
            <div><span>Last report</span><strong>{bt?.generated_at || 'Never'}</strong></div>
            <div><span>Symbols measured</span><strong>{uni.run || bt?.symbols_run || 0}</strong></div>
            <div><span>Truncated</span><strong>{uni.truncated ? 'Yes — re-run full backtest' : 'No'}</strong></div>
            <div><span>Live locked</span><strong>{boolLabel(bt?.live_locked ?? true)}</strong></div>
          </div>
          <p className="panel-copy">After each full backtest, the next market scan demotes proven-loser combos and ranks by measured edge. Pre-trade GO/CAUTION/NO_GO reads that edge.</p>
        </Panel>
        <Panel title="CORPORATE ACTIONS" subtitle="Detect phantom gaps · never invent factors · adjust-on-read">
          <div className="key-value-list">
            <div><span>Events</span><strong>{ca?.events ?? 0}</strong></div>
            <div><span>Verified</span><strong>{boolLabel(Boolean(ca?.adjustment_verified))}</strong></div>
            <div><span>Gap rate</span><strong>{ca?.gap_rate == null ? '—' : `${(Number(ca.gap_rate) * 100).toFixed(1)}%`}</strong></div>
            <div><span>Todo gaps</span><strong>{ca?.todo_gaps ?? 0}</strong></div>
          </div>
          <div className="inline-actions" style={{ marginTop: '10px' }}>
            <button type="button" disabled={Boolean(caBusy)} onClick={() => void runCaGaps()}>Export gap TODO</button>
            <button type="button" disabled={Boolean(caBusy)} onClick={() => void runCaVerify()}>Verify adjustment</button>
          </div>
          <p className="panel-copy">
            {caBusy || ca?.next_action || ca?.honesty || 'Fill factor/type from NSE filings into the TODO CSV, then ingest. QuantTerm will not invent corporate actions.'}
          </p>
        </Panel>
        <Panel title="PAPER-AUTONOMY JOB LEDGER" subtitle="Execution and learning only · market scans use separate lanes" className="job-panel"><JobLedger jobs={a.jobs_recent || []} /></Panel>
        <Panel title="OPERATING STATE"><div className="key-value-list"><div><span>Heartbeat</span><strong>{a.heartbeat_ist || '—'}</strong></div><div><span>Live feed</span><strong>{String(a.live_feed?.connected ?? 'Unavailable')}</strong></div><div><span>Subscriptions</span><strong>{String(a.live_feed?.subscriptions ?? '—')}</strong></div><div><span>Existing exits</span><strong>{boolLabel(a.existing_exits)}</strong></div><div><span>Research</span><strong>{boolLabel(a.research_enabled)}</strong></div></div><p className="panel-copy">{a.explanation || a.plain_state}</p></Panel>
        <Panel title="RECENT SUPERVISOR DIALOGUE"><div className="dialogue-list">{a.recent_dialogue.length === 0 && <div className="empty-row">No dialogue records.</div>}{[...a.recent_dialogue].reverse().slice(0, 20).map((record, index) => <div key={index}><strong>{words(String(record.record_type || record.decision || 'event'))}</strong><span>{String(record.claim || record.explanation || record.summary || JSON.stringify(record))}</span></div>)}</div></Panel>
        <Panel title="CAPABILITY NOTES"><EvidenceList title="Current constraints" items={[...(a.capability_notes || []), ...(a.active_failures || [])]} tone={a.active_failures?.length ? 'red' : 'cyan'} /></Panel>
        <InstitutionalStackPanel />
        <DataReadinessPanel dashboard={dashboard} />
      </div>
    </section>
  )
}
