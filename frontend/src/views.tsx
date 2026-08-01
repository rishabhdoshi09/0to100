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

const qualityRows = (dashboard: DashboardPayload) => dashboard.long_term.records
  .filter((row) => ['QUALITY_COMPOUNDER', 'GARP_CANDIDATE', 'QUALITY_BUT_EXPENSIVE'].includes(row.classification || ''))
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
              <div><span>Long-term scan</span><strong>{latestLongTerm ? `${latestLongTerm.status} · ${words(latestLongTerm.stage)}` : 'NOT RUN'}</strong></div>
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

export function PortfolioView({ dashboard, runControl }: ViewProps) {
  const paperReturn = dashboard.paper.capital > 0 ? ((dashboard.paper.equity / dashboard.paper.capital) - 1) * 100 : null
  return (
    <section className="workspace-view">
      <div className="inline-actions"><button type="button" onClick={() => void runControl('RUN_CYCLE_NOW')}>Request paper cycle</button><button type="button" onClick={() => void runControl(dashboard.autonomy.new_paper_entries ? 'PAUSE_NEW_PAPER_ENTRIES' : 'RESUME_NEW_PAPER_ENTRIES')}>{dashboard.autonomy.new_paper_entries ? 'Pause new entries' : 'Resume new entries'}</button></div>
      <div className="view-metrics"><MetricCard label="PAPER CAPITAL" value={money(dashboard.paper.capital)} /><MetricCard label="PAPER EQUITY" value={money(dashboard.paper.equity)} detail={pct(paperReturn)} tone="green" /><MetricCard label="OPEN RISK" value={money(dashboard.paper.open_risk)} detail={`${(dashboard.paper.risk_per_trade_pct * 100).toFixed(1)}% risk/trade`} tone="amber" /><MetricCard label="POSITIONS" value={String(dashboard.paper.open_positions.length)} detail={`Max ${dashboard.paper.max_positions}`} tone="purple" /></div>
      <div className="portfolio-workspace">
        <Panel title="RECORDED EQUITY CURVE" subtitle="No synthetic history"><EquityCurve values={dashboard.paper.equity_curve} /></Panel>
        <Panel title="OPEN PAPER POSITIONS"><PositionsTable rows={dashboard.paper.open_positions} /></Panel>
        <Panel title="RECENT CLOSED TRADES"><PositionsTable rows={[...dashboard.paper.closed_trades].reverse().slice(0, 50)} closed /></Panel>
      </div>
    </section>
  )
}

export function MarketInternalsView({ dashboard }: ViewProps) {
  const details = dashboard.market.technical_details || {}
  return (
    <section className="workspace-view">
      {!dashboard.data.ready && <DataReadinessPanel dashboard={dashboard} />}
      <div className="view-metrics"><MetricCard label="REGIME" value={dashboard.market.health} detail={String(details.market_regime || dashboard.market.breadth)} tone={dashboard.market.health.toLowerCase() === 'healthy' ? 'green' : 'amber'} /><MetricCard label="NIFTY 1D" value={pct(dashboard.market.nifty_change_1d)} detail={`5D ${pct(dashboard.market.nifty_change_5d)}`} /><MetricCard label="INDIA VIX" value={Number.isFinite(dashboard.market.vix) ? Number(dashboard.market.vix).toFixed(2) : '—'} tone="purple" /><MetricCard label="SCAN COVERAGE" value={dashboard.scan.universe_size.toLocaleString('en-IN')} detail={`${dashboard.scan.summary.with_any_setup ?? 0} with setups`} /></div>
      <div className="market-grid">
        <Panel title="MARKET NARRATIVE"><p className="lead-copy">{dashboard.market.summary}</p><p className="panel-copy">{dashboard.market.trade_stance}</p></Panel>
        <Panel title="SECTOR LEADERS"><div className="tag-cloud">{dashboard.market.leaders.length ? dashboard.market.leaders.map((item) => <span className="positive-tag" key={item}>{item}</span>) : <span>No clear leaders recorded.</span>}</div></Panel>
        <Panel title="SECTOR LAGGARDS"><div className="tag-cloud">{dashboard.market.laggards.length ? dashboard.market.laggards.map((item) => <span className="negative-tag" key={item}>{item}</span>) : <span>No clear laggards recorded.</span>}</div></Panel>
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
    if (classification === 'Quality') return all.filter((row) => ['QUALITY_COMPOUNDER', 'GARP_CANDIDATE'].includes(row.classification || ''))
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
      <div className="mode-tabs">{['All', 'Quality', 'Expensive', 'Needs Data', 'Avoid'].map((item) => <button type="button" key={item} className={classification === item ? 'active' : ''} onClick={() => setClassification(item)}>{item}</button>)}<button className="mode-action" type="button" onClick={() => void runControl('RUN_LONG_TERM_SCAN_NOW')}>{runLabel}</button><button className="mode-action" type="button" disabled={!dashboard.long_term.records.length} onClick={() => void runControl('REFRESH_LONG_TERM_NOW')}>Refresh fundamentals</button></div>
      {!dashboard.long_term.available && <div className="api-warning">Latest long-term operation: {String(latestOperation?.status || 'not run')} · {String(latestOperation?.error_message || latestOperation?.message || 'waiting for the dedicated long-term lane')}</div>}
      <div className="split-workspace"><Panel title={`${classification.toUpperCase()} · ${rows.length} RECORDS`} subtitle={`Coverage ${dashboard.long_term.summary.coverage_pct ?? 0}% · ${dashboard.long_term.fundamentals_source || 'current snapshot'}`}><LongTermTable rows={rows} selected={viewSymbol} onSelect={setSelected} /></Panel><div className="detail-stack"><Panel title={`LONG-TERM CHART · ${viewSymbol || 'SELECT STOCK'}`}><ChartWorkspace symbol={viewSymbol} bars={viewBars} row={row} /></Panel><Panel title="QUALITY & RISKS"><div className="evidence-grid"><EvidenceList title="Quality factors" items={row?.quality_factors} tone="green" /><EvidenceList title="Risk flags" items={row?.risk_flags} tone="red" /></div></Panel></div></div>
    </section>
  )
}

export function AutomationView({ dashboard, runControl }: ViewProps) {
  const a = dashboard.autonomy
  const activeJob = a.active_job || {}
  return (
    <section className="workspace-view">
      <div className="inline-actions"><button type="button" onClick={() => void runControl('RUN_SCAN_NOW')}>Start market scan</button><button type="button" onClick={() => void runControl('RUN_CYCLE_NOW')}>Request paper cycle</button><button type="button" onClick={() => void runControl('REFRESH_DATA_NOW')}>Prepare market data</button><button type="button" onClick={() => void runControl(a.new_paper_entries ? 'PAUSE_NEW_PAPER_ENTRIES' : 'RESUME_NEW_PAPER_ENTRIES')}>{a.new_paper_entries ? 'Pause entries' : 'Resume entries'}</button></div>
      <div className="view-metrics"><MetricCard label="PAPER SUPERVISOR" value={a.running ? 'ONLINE' : 'OFFLINE'} detail={`PID ${a.scheduler_owner_pid || '—'}`} tone={a.running ? 'green' : 'amber'} /><MetricCard label="STATE" value={a.state} detail={a.plain_state} /><MetricCard label="ACTIVE PAPER JOB" value={String(activeJob.job_type || 'IDLE').toUpperCase()} detail={activeJob.elapsed_s ? `${activeJob.elapsed_s}s elapsed` : 'No paper worker job reported'} tone="cyan" /><MetricCard label="FAILURES" value={String(a.active_failures?.length || 0)} detail={(a.active_failures || []).join(', ') || 'None active'} tone="purple" /></div>
      <div className="automation-grid">
        <Panel title="PAPER-AUTONOMY JOB LEDGER" subtitle="Execution and learning only · market scans use separate lanes" className="job-panel"><JobLedger jobs={a.jobs_recent || []} /></Panel>
        <Panel title="OPERATING STATE"><div className="key-value-list"><div><span>Heartbeat</span><strong>{a.heartbeat_ist || '—'}</strong></div><div><span>Live feed</span><strong>{String(a.live_feed?.connected ?? 'Unavailable')}</strong></div><div><span>Subscriptions</span><strong>{String(a.live_feed?.subscriptions ?? '—')}</strong></div><div><span>Existing exits</span><strong>{boolLabel(a.existing_exits)}</strong></div><div><span>Research</span><strong>{boolLabel(a.research_enabled)}</strong></div></div><p className="panel-copy">{a.explanation || a.plain_state}</p></Panel>
        <Panel title="RECENT SUPERVISOR DIALOGUE"><div className="dialogue-list">{a.recent_dialogue.length === 0 && <div className="empty-row">No dialogue records.</div>}{[...a.recent_dialogue].reverse().slice(0, 20).map((record, index) => <div key={index}><strong>{words(String(record.record_type || record.decision || 'event'))}</strong><span>{String(record.claim || record.explanation || record.summary || JSON.stringify(record))}</span></div>)}</div></Panel>
        <Panel title="CAPABILITY NOTES"><EvidenceList title="Current constraints" items={[...(a.capability_notes || []), ...(a.active_failures || [])]} tone={a.active_failures?.length ? 'red' : 'cyan'} /></Panel>
        <DataReadinessPanel dashboard={dashboard} />
      </div>
    </section>
  )
}
