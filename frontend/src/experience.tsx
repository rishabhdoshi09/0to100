import { useEffect, useMemo, useState } from 'react'
import {
  ChartWorkspace,
  EvidenceList,
  LongTermTable,
  MetricCard,
  Panel,
  SecurityTable,
} from './components'
import { money, pct, score, words } from './format'
import {
  fetchCommandCenterWorkspace,
  fetchScannerWorkspace,
  type CommandCenterWorkspace,
  type ScannerWorkspaceRow,
} from './productApi'
import type {
  ChartBar,
  ControlName,
  ConvictionRecord,
  DashboardPayload,
  LongTermRecord,
  ScanRecord,
} from './types'
import { depthLabel, GLOSSARY, PAGE_GUIDE, type DisplayDepth } from './productLanguage'

export type ExperienceViewProps = {
  dashboard: DashboardPayload
  selected: string
  setSelected: (symbol: string) => void
  bars: ChartBar[]
  setActive: (page: string) => void
  runControl: (control: ControlName) => Promise<void>
  depth: DisplayDepth
}

const scoreOf = (row: ScannerWorkspaceRow) => Number(
  row.conviction_score ?? row.combined_score ?? row.score ?? 0,
)

const selectedRecord = (dashboard: DashboardPayload, symbol: string) =>
  dashboard.conviction.find((row) => row.symbol === symbol)
  || dashboard.scan.records.find((row) => row.symbol === symbol)
  || dashboard.long_term.records.find((row) => row.symbol === symbol)

const fallbackRows = (mode: string, dashboard: DashboardPayload): ScannerWorkspaceRow[] => {
  if (mode === 'Conviction') return [...dashboard.conviction]
  if (mode === 'Long-Term') return [...dashboard.long_term.records] as ScannerWorkspaceRow[]
  if (mode === 'Breakouts') {
    return dashboard.scan.records.filter((row) => row.signals?.some((signal) => signal.includes('BREAKOUT')) || row.status === 'Ready to trade')
  }
  if (mode === 'Pre-Breakout') {
    return dashboard.scan.records.filter((row) => row.signals?.includes('PRE_BREAKOUT') || row.status === 'Watch for breakout')
  }
  if (mode === 'Avoid') {
    return dashboard.scan.records.filter((row) => row.chase_risk || row.status === 'Wait for pullback')
  }
  return dashboard.scan.records.filter((row) => row.signals?.includes('MOMENTUM') || row.verdict === 'BUY')
}

const operationFor = (dashboard: DashboardPayload, kind: string) =>
  dashboard.operations.active.find((item) => item.kind === kind)
  || dashboard.operations.latest[kind]

function relativeDate(value?: string) {
  if (!value) return 'not available'
  const stamp = new Date(value)
  if (Number.isNaN(stamp.getTime())) return value
  const hours = Math.max(0, (Date.now() - stamp.getTime()) / 3_600_000)
  if (hours < 1) return `${Math.round(hours * 60)} min old`
  if (hours < 48) return `${Math.round(hours)} hr old`
  return `${Math.round(hours / 24)} day old`
}

function closedTradeEvidence(dashboard: DashboardPayload) {
  const trades = dashboard.paper.closed_trades
    .map((row) => Number(row.pnl ?? 0))
    .filter((value) => Number.isFinite(value))
  const wins = trades.filter((value) => value > 0)
  const losses = trades.filter((value) => value < 0)
  const grossProfit = wins.reduce((sum, value) => sum + value, 0)
  const grossLoss = Math.abs(losses.reduce((sum, value) => sum + value, 0))
  const cumulative: number[] = []
  trades.reduce((sum, value) => {
    const next = sum + value
    cumulative.push(next)
    return next
  }, 0)
  let peak = 0
  let maxDrawdown = 0
  cumulative.forEach((value) => {
    peak = Math.max(peak, value)
    maxDrawdown = Math.min(maxDrawdown, value - peak)
  })
  return {
    count: trades.length,
    net: trades.reduce((sum, value) => sum + value, 0),
    winRate: trades.length ? wins.length / trades.length * 100 : null,
    profitFactor: grossLoss > 0 ? grossProfit / grossLoss : null,
    maxDrawdown,
    mature: trades.length >= 20,
  }
}

export function DisplayDepthToggle({ depth, onChange }: {
  depth: DisplayDepth
  onChange: (value: DisplayDepth) => void
}) {
  return (
    <div className="display-depth-toggle" aria-label="Interface depth">
      {(['simple', 'professional'] as DisplayDepth[]).map((item) => (
        <button
          type="button"
          key={item}
          className={depth === item ? 'active' : ''}
          onClick={() => onChange(item)}
        >
          {depthLabel(item)}
        </button>
      ))}
    </div>
  )
}

export function ExperienceHelpDrawer({ page, open, onClose }: {
  page: string
  open: boolean
  onClose: () => void
}) {
  if (!open) return null
  const guide = PAGE_GUIDE[page] || PAGE_GUIDE['Command Center']
  return (
    <div className="experience-help-backdrop" role="presentation" onClick={onClose}>
      <aside className="experience-help" role="dialog" aria-modal="true" aria-label={`${guide.title} help`} onClick={(event: { stopPropagation: () => void }) => event.stopPropagation()}>
        <header><div><span>WHAT IS THIS?</span><h2>{guide.title}</h2></div><button type="button" onClick={onClose}>×</button></header>
        <p className="experience-help-purpose">{guide.purpose}</p>
        <section><strong>Questions this page should answer</strong>{guide.questions.map((item) => <div key={item}>✓ {item}</div>)}</section>
        <section><strong>Best next action</strong><p>{guide.action}</p></section>
        <section className="help-warning"><strong>Do not misunderstand it</strong><p>{guide.doesNot}</p></section>
        <section><strong>Useful terms</strong><div className="help-glossary">{Object.entries(GLOSSARY).slice(0, 6).map(([term, meaning]) => <div key={term}><b>{term}</b><span>{meaning}</span></div>)}</div></section>
      </aside>
    </div>
  )
}

function TodayStrip({ dashboard }: { dashboard: DashboardPayload }) {
  const scanOperation = operationFor(dashboard, 'MARKET_SCAN')
  return (
    <div className="today-strip">
      <div><span>MARKET</span><strong>{dashboard.market.health.toUpperCase()}</strong><small>{dashboard.market.trade_stance}</small></div>
      <div><span>NIFTY 1D</span><strong className={(dashboard.market.nifty_change_1d || 0) >= 0 ? 'positive' : 'negative'}>{pct(dashboard.market.nifty_change_1d)}</strong><small>{dashboard.market.breadth}</small></div>
      <div><span>VOLATILITY</span><strong>{dashboard.market.vix == null ? '—' : Number(dashboard.market.vix).toFixed(2)}</strong><small>VIX / regime input</small></div>
      <div><span>PRICE DATA</span><strong>{dashboard.data.bhavcopy.latest_date || 'MISSING'}</strong><small>{dashboard.data.bhavcopy.sessions} sessions</small></div>
      <div><span>MARKET SCAN</span><strong>{scanOperation?.status || 'NOT RUN'}</strong><small>{scanOperation?.progress_pct == null ? relativeDate(dashboard.scan.scanned_at) : `${scanOperation.progress_pct.toFixed(0)}% · ${words(scanOperation.stage)}`}</small></div>
    </div>
  )
}

function OperationMiniBoard({ dashboard }: { dashboard: DashboardPayload }) {
  const kinds = ['DATA_PREPARE', 'MARKET_SCAN', 'LONG_TERM_SCAN', 'NEWS_REFRESH']
  return (
    <div className="operation-mini-board">
      {kinds.map((kind) => {
        const item = operationFor(dashboard, kind)
        return <div key={kind}><span>{words(kind)}</span><strong>{item?.status || 'NOT RUN'}</strong><small>{item?.message || item?.stage || 'No durable operation yet'}</small></div>
      })}
    </div>
  )
}

export function EnhancedCommandCenterView(props: ExperienceViewProps) {
  const { dashboard, selected, setSelected, bars, setActive, runControl, depth } = props
  const [workspace, setWorkspace] = useState<CommandCenterWorkspace | null>(null)
  const evidence = useMemo(() => closedTradeEvidence(dashboard), [dashboard.paper.closed_trades])

  useEffect(() => {
    let alive = true
    const load = () => fetchCommandCenterWorkspace().then((value) => { if (alive) setWorkspace(value) }).catch(() => undefined)
    void load()
    const timer = window.setInterval(load, 10_000)
    return () => { alive = false; window.clearInterval(timer) }
  }, [])

  const momentum = workspace?.top_setups?.length
    ? workspace.top_setups
    : fallbackRows('Momentum', dashboard).slice(0, 8) as Array<ScanRecord | ConvictionRecord>
  const longTerm = workspace?.top_long_term?.length
    ? workspace.top_long_term
    : dashboard.long_term.records.slice(0, 8)
  const row = selectedRecord(dashboard, selected)
  const insights = workspace?.insights?.length ? workspace.insights : [dashboard.market.summary, dashboard.market.trade_stance]

  return (
    <section className={`experience-command ${depth}`}>
      <header className="experience-hero">
        <div><span>QUANTTERM DAILY BOARD</span><h2>{dashboard.market.health || 'Market state unavailable'}</h2><p>{dashboard.market.summary}</p></div>
        <div className="experience-hero-actions">
          <button type="button" onClick={() => void runControl('RUN_SCAN_NOW')}>Run momentum scan</button>
          <button type="button" onClick={() => void runControl('RUN_LONG_TERM_SCAN_NOW')}>Run long-term scan</button>
        </div>
      </header>
      <TodayStrip dashboard={dashboard} />

      {depth === 'simple' && (
        <div className="simple-decision-board">
          <article><span>1</span><div><strong>Check readiness</strong><p>{dashboard.data.ready ? 'Core price data is ready.' : `Resolve: ${dashboard.data.blockers[0] || 'market data is incomplete'}`}</p></div></article>
          <article><span>2</span><div><strong>Run one research job</strong><p>Momentum finds current technical strength. Long-Term adds current business quality.</p></div></article>
          <article><span>3</span><div><strong>Review one stock</strong><p>Open Stock Intelligence before considering any paper decision.</p></div></article>
        </div>
      )}

      <div className="terminal-triad">
        <Panel title="MOMENTUM DISCOVERY" subtitle={`${dashboard.scan.universe_size.toLocaleString('en-IN')} stocks evaluated · ${dashboard.scan.scanned_at || 'no completed scan'}`} action={<button type="button" onClick={() => setActive('Scanner')}>Open Discover</button>}>
          <SecurityTable rows={momentum as Array<ScanRecord | ConvictionRecord>} selected={selected} onSelect={setSelected} limit={depth === 'simple' ? 5 : 8} />
        </Panel>
        <Panel title={`PRICE STRUCTURE · ${selected || 'SELECT STOCK'}`} subtitle={`Official daily history · ${dashboard.data.bhavcopy.latest_date || 'date unavailable'}`}>
          <ChartWorkspace symbol={selected} bars={bars} row={row} />
          <footer className="experience-chart-footer"><button type="button" disabled={!selected} onClick={() => setActive('Stock Intelligence')}>Explain this stock</button></footer>
        </Panel>
        <Panel title="LONG-TERM QUALITY" subtitle={`${dashboard.long_term.records.length} saved candidates · as of ${dashboard.long_term.scanned_at || 'not run'}`} action={<button type="button" onClick={() => setActive('Long-Term')}>Open research</button>}>
          <LongTermTable rows={longTerm} selected={selected} onSelect={setSelected} limit={depth === 'simple' ? 5 : 8} />
        </Panel>
      </div>

      <div className="experience-lower-grid">
        <Panel title="WHAT THE SYSTEM KNOWS" subtitle="Plain-language synthesis from persisted state">
          <div className="system-insight-board">{insights.map((item, index) => <div key={`${item}-${index}`}><i /><span>{item}</span></div>)}</div>
        </Panel>
        <Panel title="MEASURED PAPER EVIDENCE" subtitle="Only recorded closed trades; no synthetic portfolio curve">
          {evidence.mature ? (
            <div className="evidence-stat-grid">
              <div><span>Closed trades</span><strong>{evidence.count}</strong></div>
              <div><span>Win rate</span><strong>{evidence.winRate?.toFixed(1)}%</strong></div>
              <div><span>Profit factor</span><strong>{evidence.profitFactor?.toFixed(2) || '—'}</strong></div>
              <div><span>Net P&L</span><strong className={evidence.net >= 0 ? 'positive' : 'negative'}>{money(evidence.net)}</strong></div>
              <div><span>Trade P&L drawdown</span><strong className="negative">{money(evidence.maxDrawdown)}</strong></div>
            </div>
          ) : (
            <div className="evidence-not-mature"><strong>Performance evidence is not mature.</strong><p>{evidence.count} closed trade(s) recorded. QuantTerm waits for at least 20 before showing stable win-rate and profit-factor cards.</p><small>Current net recorded P&L: {money(evidence.net)}</small></div>
          )}
        </Panel>
        <Panel title="BACKEND OPERATIONS" subtitle="Real lanes, durable stages and exact results"><OperationMiniBoard dashboard={dashboard} /></Panel>
      </div>
    </section>
  )
}

export function EnhancedScannerView(props: ExperienceViewProps) {
  const { dashboard, selected, setSelected, bars, runControl, depth } = props
  const modes = depth === 'simple'
    ? ['Momentum', 'Breakouts', 'Long-Term', 'Avoid']
    : ['Momentum', 'Conviction', 'Breakouts', 'Pre-Breakout', 'Long-Term', 'Avoid']
  const [mode, setMode] = useState('Momentum')
  const [rows, setRows] = useState<ScannerWorkspaceRow[]>([])
  const [query, setQuery] = useState('')
  const [sector, setSector] = useState('All')
  const [minScore, setMinScore] = useState(depth === 'simple' ? 60 : 0)
  const [excludeChase, setExcludeChase] = useState(true)
  const [sourceMessage, setSourceMessage] = useState('')

  useEffect(() => {
    if (!modes.includes(mode)) setMode(modes[0])
  }, [depth])

  useEffect(() => {
    let alive = true
    fetchScannerWorkspace(mode)
      .then((result) => {
        if (!alive) return
        setRows(result.rows)
        setSourceMessage(`${result.source} · ${result.scanned_at || 'date unavailable'}`)
      })
      .catch(() => {
        if (!alive) return
        setRows(fallbackRows(mode, dashboard))
        setSourceMessage('Local dashboard fallback')
      })
    return () => { alive = false }
  }, [mode, dashboard.scan.scanned_at, dashboard.long_term.scanned_at])

  const sectors = useMemo(() => [...new Set(rows.map((row) => row.sector).filter(Boolean) as string[])].sort(), [rows])
  const filtered = useMemo(() => rows.filter((row) => {
    const clean = query.trim().toUpperCase()
    if (clean && !row.symbol.includes(clean) && !String(row.company || '').toUpperCase().includes(clean)) return false
    if (sector !== 'All' && row.sector !== sector) return false
    if (scoreOf(row) < minScore) return false
    if (excludeChase && row.chase_risk) return false
    return true
  }).sort((a, b) => scoreOf(b) - scoreOf(a)), [rows, query, sector, minScore, excludeChase])

  const selectedRow = filtered.find((row) => row.symbol === selected)
    || rows.find((row) => row.symbol === selected)
    || selectedRecord(dashboard, selected) as ScannerWorkspaceRow | undefined
  const scanOperation = operationFor(dashboard, mode === 'Long-Term' ? 'LONG_TERM_SCAN' : 'MARKET_SCAN')
  const reasons = selectedRow?.reasons || selectedRow?.quality_factors || []
  const risks = selectedRow?.risks || selectedRow?.risk_flags || (selectedRow?.chase_risk ? ['Price is extended; do not chase.'] : [])

  return (
    <section className={`enhanced-scanner ${depth}`}>
      <header className="scanner-command-bar">
        <div><span>DISCOVER</span><h2>Momentum and long-horizon research, one stock at a time</h2><p>{sourceMessage}</p></div>
        <button type="button" onClick={() => void runControl(mode === 'Long-Term' ? 'RUN_LONG_TERM_SCAN_NOW' : 'RUN_SCAN_NOW')}>{mode === 'Long-Term' ? 'Run long-term scan now' : 'Scan whole market now'}</button>
      </header>

      {scanOperation && (
        <div className={`scan-progress ${String(scanOperation.status).toLowerCase()}`}>
          <div><strong>{words(scanOperation.kind)}</strong><span>{scanOperation.status} · {words(scanOperation.stage)}</span></div>
          <p>{scanOperation.message}</p>
          <div className="scan-progress-track"><b style={{ width: `${Math.max(0, Math.min(100, scanOperation.progress_pct || 0))}%` }} /></div>
          <small>{scanOperation.progress_current || 0} / {scanOperation.progress_total || 0} · attempt {scanOperation.attempt}</small>
        </div>
      )}

      <div className="scanner-lane-cards">
        <article><span>MOMENTUM LANE</span><strong>{fallbackRows('Momentum', dashboard).length}</strong><small>Current technical-strength matches</small></article>
        <article><span>LONG-TERM LANE</span><strong>{dashboard.long_term.records.length}</strong><small>Quality, valuation and timing candidates</small></article>
        <article><span>UNIVERSE</span><strong>{dashboard.scan.universe_size.toLocaleString('en-IN')}</strong><small>Stocks in the saved market scan</small></article>
        <article><span>PRICE DATA</span><strong>{dashboard.data.bhavcopy.latest_date || 'MISSING'}</strong><small>{dashboard.data.bhavcopy.sessions} official sessions</small></article>
      </div>

      <div className="scanner-mode-row">{modes.map((item) => <button type="button" key={item} className={mode === item ? 'active' : ''} onClick={() => setMode(item)}>{item}</button>)}</div>
      <div className="scanner-filter-row">
        <label>Search<input value={query} onChange={(event: { target: { value: string } }) => setQuery(event.target.value)} placeholder="Symbol or company" /></label>
        <label>Sector<select value={sector} onChange={(event: { target: { value: string } }) => setSector(event.target.value)}><option>All</option>{sectors.map((item) => <option key={item}>{item}</option>)}</select></label>
        <label>Minimum score<input type="number" min="0" max="100" value={minScore} onChange={(event: { target: { value: string } }) => setMinScore(Number(event.target.value) || 0)} /></label>
        <label className="scanner-check"><input type="checkbox" checked={excludeChase} onChange={(event: { target: { checked: boolean } }) => setExcludeChase(event.target.checked)} /> Hide chase-risk names</label>
      </div>

      <div className="scanner-workspace-grid">
        <Panel title={`${mode.toUpperCase()} · ${filtered.length} MATCHES`} subtitle="Every row comes from persisted backend research">
          {mode === 'Long-Term'
            ? <LongTermTable rows={filtered as LongTermRecord[]} selected={selected} onSelect={setSelected} />
            : <SecurityTable rows={filtered as Array<ScanRecord | ConvictionRecord>} selected={selected} onSelect={setSelected} />}
        </Panel>
        <div className="scanner-detail-column">
          <Panel title={`PRICE STRUCTURE · ${selected || 'SELECT STOCK'}`} subtitle={`Official daily history · ${dashboard.data.bhavcopy.latest_date || 'date unavailable'}`}><ChartWorkspace symbol={selected} bars={bars} row={selectedRow} /></Panel>
          <Panel title="WHY IT IS HERE"><div className="evidence-grid"><EvidenceList title="Qualification evidence" items={reasons} tone="green" /><EvidenceList title="Invalidation and risk" items={risks} tone="red" /></div></Panel>
        </div>
      </div>
    </section>
  )
}

function median(values: number[]) {
  if (!values.length) return null
  const ordered = [...values].sort((a, b) => a - b)
  const middle = Math.floor(ordered.length / 2)
  return ordered.length % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2
}

export function EnhancedLongTermView(props: ExperienceViewProps) {
  const { dashboard, selected, setSelected, bars, runControl, depth } = props
  const [classification, setClassification] = useState('All')
  const [sector, setSector] = useState('All')
  const [minCoverage, setMinCoverage] = useState(depth === 'simple' ? 50 : 0)
  const rows = dashboard.long_term.records
  const classes = [...new Set(rows.map((row) => row.classification).filter(Boolean) as string[])].sort()
  const sectors = [...new Set(rows.map((row) => row.sector).filter(Boolean) as string[])].sort()
  const filtered = rows.filter((row) => {
    if (classification !== 'All' && row.classification !== classification) return false
    if (sector !== 'All' && row.sector !== sector) return false
    const coverage = Number(row.fundamental_coverage || 0) * 100
    return coverage >= minCoverage
  }).sort((a, b) => Number(b.combined_score || 0) - Number(a.combined_score || 0))
  const current = filtered.find((row) => row.symbol === selected) || rows.find((row) => row.symbol === selected)
  const qualityCount = rows.filter((row) => row.classification === 'QUALITY_COMPOUNDER').length
  const garpCount = rows.filter((row) => row.classification === 'GARP_CANDIDATE').length
  const needsData = rows.filter((row) => row.classification === 'NEEDS_FUNDAMENTALS').length
  const medianFundamental = median(rows.map((row) => Number(row.fundamental_score)).filter(Number.isFinite))
  const medianTechnical = median(rows.map((row) => Number(row.technical_score)).filter(Number.isFinite))
  const operation = operationFor(dashboard, 'LONG_TERM_SCAN')

  return (
    <section className={`enhanced-long-term ${depth}`}>
      <header className="long-term-hero">
        <div><span>LONG-TERM RESEARCH</span><h2>Quality, valuation and timing without fake model performance</h2><p>Current fundamentals are combined with official daily technical history. Missing historical evidence remains visible.</p></div>
        <button type="button" onClick={() => void runControl('RUN_LONG_TERM_SCAN_NOW')}>Run long-term scan now</button>
      </header>

      <div className="long-term-metric-strip">
        <MetricCard label="QUALITY COMPOUNDERS" value={String(qualityCount)} detail="Passed current quality classification" tone="green" />
        <MetricCard label="GARP CANDIDATES" value={String(garpCount)} detail="Growth and valuation balance" />
        <MetricCard label="NEEDS FUNDAMENTALS" value={String(needsData)} detail="Coverage too weak for a quality conclusion" tone="amber" />
        <MetricCard label="MEDIAN FUNDAMENTAL" value={medianFundamental == null ? '—' : medianFundamental.toFixed(1)} detail="Across the saved long-term set" tone="purple" />
        <MetricCard label="MEDIAN TECHNICAL" value={medianTechnical == null ? '—' : medianTechnical.toFixed(1)} detail="Official-history timing score" />
      </div>

      {operation && <div className="long-term-operation"><strong>{operation.status} · {words(operation.stage)}</strong><span>{operation.message}</span><small>{operation.progress_pct == null ? 'Progress not reported' : `${operation.progress_pct.toFixed(0)}% complete`}</small></div>}

      <div className="long-term-filter-row">
        <label>Classification<select value={classification} onChange={(event: { target: { value: string } }) => setClassification(event.target.value)}><option>All</option>{classes.map((item) => <option key={item}>{item}</option>)}</select></label>
        <label>Sector<select value={sector} onChange={(event: { target: { value: string } }) => setSector(event.target.value)}><option>All</option>{sectors.map((item) => <option key={item}>{item}</option>)}</select></label>
        <label>Minimum coverage<input type="number" min="0" max="100" value={minCoverage} onChange={(event: { target: { value: string } }) => setMinCoverage(Number(event.target.value) || 0)} /><span>%</span></label>
      </div>

      <div className="long-term-workspace-grid">
        <Panel title={`FILTERED RESEARCH · ${filtered.length}`} subtitle={`Saved run ${dashboard.long_term.scanned_at || 'not available'}`}><LongTermTable rows={filtered} selected={selected} onSelect={setSelected} /></Panel>
        <div className="long-term-detail-column">
          <Panel title={`PRICE & TIMING · ${selected || 'SELECT STOCK'}`}><ChartWorkspace symbol={selected} bars={bars} row={current} /></Panel>
          <Panel title="QUALITY OVERLAYS"><div className="quality-overlay-grid"><div><span>Classification</span><strong>{words(current?.classification || 'Unavailable')}</strong></div><div><span>Fundamental score</span><strong>{score(current?.fundamental_score)}</strong></div><div><span>Technical score</span><strong>{score(current?.technical_score)}</strong></div><div><span>Combined score</span><strong>{score(current?.combined_score)}</strong></div><div><span>Coverage</span><strong>{current?.fundamental_coverage == null ? '—' : `${(Number(current.fundamental_coverage) * 100).toFixed(0)}%`}</strong></div><div><span>Timing</span><strong>{words(current?.timing || 'Unavailable')}</strong></div></div></Panel>
          <Panel title="EVIDENCE AND RISKS"><div className="evidence-grid"><EvidenceList title="Quality factors" items={current?.quality_factors} tone="green" /><EvidenceList title="Risk flags" items={current?.risk_flags} tone="red" /></div></Panel>
          <div className="no-fake-performance"><strong>Why there is no glossy model-performance chart</strong><p>This shortlist does not yet carry a matched strategy-level out-of-sample performance series. QuantTerm will not display a decorative Sharpe, CAGR or projection without that evidence.</p></div>
        </div>
      </div>
    </section>
  )
}
