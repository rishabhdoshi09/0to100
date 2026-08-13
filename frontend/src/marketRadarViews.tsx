import { useEffect, useMemo, useState } from 'react'
import { ChartWorkspace, Panel } from './components'
import { money, pct, words } from './format'
import {
  addWatchlistItem,
  fetchCompareWorkspace,
  fetchPreTrade,
  fetchRadarHome,
  fetchScannerWorkspace,
  fetchWatchlist,
  removeWatchlistItem,
  type CompareWorkspace,
  type PreTrade,
  type RadarHome,
  type ScannerWorkspaceRow,
  type WatchlistPayload,
} from './productApi'
import { LiveScanBanner, type ExperienceViewProps } from './experience'

type RadarRow = ScannerWorkspaceRow & {
  breakout_state?: string
  momentum_state?: string
  setup_label?: string
  freshness?: string
  change_5d_pct?: number
  relative_strength?: number
  risk_label?: string
  reason?: string
  company?: string
  classification?: string
  combined_score?: number
  breakout_grade?: string
  breakout_conviction?: number
  breakout_quality?: number
  fundamental_score?: number
  sniper_candidate?: boolean
  volume_ratio?: number
  rsi?: number
  tech_source?: string
  price_tag?: string
  rsi_scan?: number
}

const breakoutLabel: Record<string, string> = {
  confirmed_breakout: 'Confirmed',
  near_breakout: 'Near breakout',
  breakout_under_observation: 'Under observation',
  breakout_without_volume: 'No volume confirm',
  insufficient_confirmation: 'Needs confirmation',
  extended_after_breakout: 'Extended',
  failed_breakout: 'Failed',
  failed_or_extended: 'Failed / extended',
  insufficient_data: 'Insufficient data',
  not_in_breakout_lane: '—',
}

const momentumLabel: Record<string, string> = {
  strong_actionable: 'Strong · actionable',
  strong_but_extended: 'Strong · extended',
  steady_leadership: 'Steady leadership',
  improving: 'Improving',
  weakening: 'Weakening',
  high_volatility_momentum: 'High-vol momentum',
  insufficient_history: 'Short history',
  watch_momentum: 'Watch',
  not_momentum: '—',
}

function BestSniperPanel({
  best,
  sniperCount,
  onSelect,
}: {
  best: RadarRow | null | undefined
  sniperCount: number
  onSelect: (symbol: string) => void
}) {
  if (best) {
    const gates = (best as RadarRow & { quality_gates?: Record<string, string> }).quality_gates || {}
    const ctx = (best as RadarRow & {
      breakout_context?: { order_book?: { status?: string; note?: string }; concall?: { status?: string; note?: string } }
    }).breakout_context
    return (
      <div className="radar-best-breakout">
        <Panel
          title={`BEST SNIPER CANDIDATE · ${best.symbol}`}
          subtitle={
            [
              sniperCount > 0 ? `${sniperCount} gated candidate${sniperCount === 1 ? '' : 's'}` : null,
              best.breakout_grade ? `Grade ${best.breakout_grade}` : null,
              best.rsi != null
                ? `RSI ${Math.round(Number(best.rsi))}${best.tech_source === 'live' || best.price_tag === 'LIVE' ? ' LIVE' : ''}`
                : null,
              best.volume_ratio != null
                ? `Vol ${Number(best.volume_ratio).toFixed(1)}×`
                : null,
              best.classification
                ? String(best.classification).replace(/_/g, ' ')
                : null,
              gates.fundamentals ? `Fund ${gates.fundamentals}` : null,
              gates.trend ? `Trend ${gates.trend}` : null,
            ].filter(Boolean).join(' · ') || 'Vol≥1× · RSI≤70 · fund/tech gated'
          }
        >
          <button
            type="button"
            className="radar-best-pick-btn"
            onClick={() => onSelect(String(best.symbol || ''))}
          >
            Score {best.score ?? '—'}
            {best.breakout_quality != null
              ? ` · Quality ${Number(best.breakout_quality).toFixed(0)}`
              : ''}
            {' · '}
            {breakoutLabel[String(best.breakout_state || '')]
              || words(String(best.breakout_state || best.status || ''))}
          </button>
          {ctx && (
            <p className="radar-empty-li" style={{ paddingTop: 8 }}>
              Order book: {ctx.order_book?.status || 'unavailable'}
              {ctx.order_book?.note ? ` — ${ctx.order_book.note}` : ''}
              {' · '}
              Concall: {ctx.concall?.status || 'unavailable'}
              {ctx.concall?.note && ctx.concall.status === 'present' ? ` — ${ctx.concall.note}` : ''}
            </p>
          )}
        </Panel>
      </div>
    )
  }
  return (
    <div className="radar-best-breakout radar-best-empty">
      <Panel
        title="BEST SNIPER CANDIDATE"
        subtitle="Need vol≥1.0×, RSI≤70, no chase, no AVOID_REVIEW — thin tape never qualifies"
      >
        <p className="radar-empty-li">
          {sniperCount === 0
            ? 'No gated sniper candidates — thin-volume / blow-off names are excluded.'
            : 'Sniper pool has names but none cleared the best-pick gate.'}
        </p>
      </Panel>
    </div>
  )
}

function DenseTable({
  rows,
  selected,
  onSelect,
  depth,
  mode,
}: {
  rows: RadarRow[]
  selected: string
  onSelect: (symbol: string) => void
  depth: ExperienceViewProps['depth']
  mode: string
}) {
  // Breakouts: keep server sniper-first ranking until the user clicks a column.
  const [sortKey, setSortKey] = useState(mode === 'Breakouts' ? '' : 'score')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')

  useEffect(() => {
    setSortKey(mode === 'Breakouts' ? '' : 'score')
    setSortDir('desc')
  }, [mode])

  const sorted = useMemo(() => {
    if (!sortKey) return rows
    const copy = [...rows]
    copy.sort((a, b) => {
      const av = (a as Record<string, unknown>)[sortKey]
      const bv = (b as Record<string, unknown>)[sortKey]
      const an = typeof av === 'number' ? av : String(av ?? '')
      const bn = typeof bv === 'number' ? bv : String(bv ?? '')
      if (typeof an === 'number' && typeof bn === 'number') {
        return sortDir === 'asc' ? an - bn : bn - an
      }
      return sortDir === 'asc'
        ? String(an).localeCompare(String(bn))
        : String(bn).localeCompare(String(an))
    })
    return copy
  }, [rows, sortKey, sortDir])

  const toggleSort = (key: string) => {
    if (sortKey === key) setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    else { setSortKey(key); setSortDir('desc') }
  }

  const cols = mode === 'Long-Term'
    ? ['symbol', 'classification', 'combined_score', 'sector', 'coverage_pct', 'risk_label']
    : mode === 'Breakouts'
      ? depth === 'professional'
        ? ['symbol', 'sniper', 'price', 'volume_ratio', 'rsi', 'breakout_grade', 'breakout_quality', 'breakout_state', 'sector', 'risk_label']
        : ['symbol', 'sniper', 'price', 'volume_ratio', 'rsi', 'breakout_quality', 'setup_label', 'risk_label']
      : depth === 'professional'
        ? ['symbol', 'price', 'change_5d_pct', 'sector', 'setup_label', 'breakout_state', 'momentum_state', 'relative_strength', 'risk_label']
        : ['symbol', 'price', 'change_5d_pct', 'sector', 'setup_label', 'risk_label']

  return (
    <div className="radar-table-wrap">
      <table className="radar-table">
        <thead>
          <tr>
            {cols.map((col) => (
              <th key={col} onClick={() => toggleSort(col)}>
                {col === 'sniper' ? 'Sniper' : words(col.replace(/_/g, ' '))}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.length === 0 && (
            <tr><td colSpan={cols.length} className="radar-empty">No matches in saved scan data. Run Scan Now.</td></tr>
          )}
          {sorted.map((row) => (
            <tr
              key={row.symbol}
              className={[
                selected === row.symbol ? 'selected' : '',
                row.sniper_candidate ? 'sniper-row' : '',
              ].filter(Boolean).join(' ')}
              onClick={() => onSelect(row.symbol)}
            >
              {cols.map((col) => {
                const raw = (row as Record<string, unknown>)[col]
                let cell: string
                if (col === 'sniper') cell = row.sniper_candidate ? 'YES' : '—'
                else if (col === 'breakout_state') cell = breakoutLabel[String(raw)] || words(String(raw))
                else if (col === 'momentum_state') cell = momentumLabel[String(raw)] || words(String(raw))
                else if (col === 'price') cell = money(raw as number)
                else if (col === 'change_5d_pct') cell = pct(raw as number)
                else if (col === 'volume_ratio') cell = raw != null ? `${Number(raw).toFixed(1)}×` : '—'
                else if (col === 'rsi' || col === 'breakout_quality' || col === 'combined_score' || col === 'relative_strength') {
                  cell = raw != null ? String(Math.round(Number(raw))) : '—'
                }
                else cell = String(raw ?? '—')
                return <td key={col}>{cell}</td>
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function RadarHomeView(props: ExperienceViewProps & {
  onCompare: (symbol: string) => void
  onWatchlist: (symbol: string) => void
}) {
  const { dashboard, selected, setSelected, bars, setActive, depth, marketScan, longTermScan, onCompare, onWatchlist } = props
  const [radar, setRadar] = useState<RadarHome | null>(null)
  const [preTrade, setPreTrade] = useState<PreTrade | null>(null)

  useEffect(() => {
    fetchRadarHome().then(setRadar).catch(() => setRadar(null))
  }, [dashboard.scan.scanned_at, dashboard.long_term.scanned_at])

  useEffect(() => {
    if (!selected) { setPreTrade(null); return }
    fetchPreTrade(selected).then(setPreTrade).catch(() => setPreTrade(null))
  }, [selected, dashboard.scan.scanned_at])

  const laneCard = (title: string, rows: RadarRow[], count: number, sniperHint?: number) => (
    <section className="radar-lane-card">
      <header>
        <span>{title}</span>
        <strong>
          {count}
          {sniperHint != null ? ` · ${sniperHint} sniper` : ''}
        </strong>
      </header>
      <ul>
        {rows.slice(0, 6).map((row) => (
          <li key={row.symbol}>
            <button type="button" className={selected === row.symbol ? 'active' : ''} onClick={() => setSelected(row.symbol)}>
              <b>
                {row.symbol}
                {row.sniper_candidate ? <em className="sniper-tag"> SNIPER</em> : null}
              </b>
              <span>{row.setup_label || row.status}</span>
              <small>{row.sector} · {row.reason?.slice(0, 48) || '—'}</small>
            </button>
          </li>
        ))}
        {rows.length === 0 && <li className="radar-empty-li">No saved matches</li>}
      </ul>
    </section>
  )

  const row = radar?.lanes.breakouts.find((r) => r.symbol === selected)
    || radar?.lanes.momentum.find((r) => r.symbol === selected)
    || radar?.lanes.long_term_picks.find((r) => r.symbol === selected)

  return (
    <section className="radar-home">
      <header className="radar-hero">
        <div>
          <span>MARKET COMMAND</span>
          <h2>{radar?.market_health || dashboard.market.health}</h2>
          <p>{dashboard.market.summary}</p>
        </div>
        <div className="radar-hero-actions">
          <button type="button" disabled={marketScan.isBusy} onClick={() => void marketScan.start()}>
            {marketScan.isBusy ? 'Scanning…' : 'Scan now'}
          </button>
        </div>
      </header>

      <div className="radar-market-strip">
        <div><span>NIFTY 1D</span><strong>{pct(radar?.nifty_change_1d ?? dashboard.market.nifty_change_1d)}</strong></div>
        <div><span>BREADTH</span><strong>{radar?.breadth || dashboard.market.breadth}</strong></div>
        <div><span>VIX</span><strong>{radar?.vix ?? dashboard.market.vix ?? '—'}</strong></div>
        <div><span>LEADERS</span><strong>{(radar?.leaders || dashboard.market.leaders).slice(0, 3).join(', ') || '—'}</strong></div>
        <div><span>LAST SCAN</span><strong>{radar?.scan_scanned_at || dashboard.scan.scanned_at || 'Not run'}</strong></div>
        <div><span>DATA</span><strong>{dashboard.data.bhavcopy.latest_date || '—'}</strong></div>
      </div>

      <LiveScanBanner scan={marketScan} depth={depth} label="Market scan" />

      <BestSniperPanel
        best={radar?.best_breakout as RadarRow | null | undefined}
        sniperCount={radar?.counts.sniper_breakouts || radar?.sniper_candidates?.length || 0}
        onSelect={setSelected}
      />

      {(radar?.sniper_candidates?.length || 0) > 0 && (
        <section className="radar-sniper-pool">
          <header>
            <span>SNIPER BREAKOUT CANDIDATES</span>
            <strong>{radar?.sniper_candidates?.length}</strong>
          </header>
          <ul>
            {(radar?.sniper_candidates || []).slice(0, 8).map((row) => (
              <li key={row.symbol}>
                <button
                  type="button"
                  className={selected === row.symbol ? 'active' : ''}
                  onClick={() => setSelected(row.symbol)}
                >
                  <b>{row.symbol}</b>
                  <span>
                    {row.breakout_grade ? `G${row.breakout_grade}` : '—'}
                    {row.volume_ratio != null ? ` · ${Number(row.volume_ratio).toFixed(1)}×` : ''}
                    {row.rsi != null
                      ? ` · RSI ${Math.round(Number(row.rsi))}${(row as RadarRow).tech_source === 'live' || (row as RadarRow).price_tag === 'LIVE' ? '·L' : ''}`
                      : ''}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      <div className="radar-three-lanes">
        {laneCard(
          'Breakouts',
          radar?.lanes.breakouts || [],
          radar?.counts.breakouts || 0,
          radar?.counts.sniper_breakouts,
        )}
        {laneCard('Momentum', radar?.lanes.momentum || [], radar?.counts.momentum || 0)}
        {laneCard('Long-Term Picks', radar?.lanes.long_term_picks || [], radar?.counts.long_term_picks || 0)}
      </div>

      <div className="radar-workspace">
        <Panel title={`CHART · ${selected || 'SELECT STOCK'}`} subtitle={`Official history · ${dashboard.data.bhavcopy.latest_date || '—'}`}>
          <ChartWorkspace symbol={selected} bars={bars} row={row} />
        </Panel>
        <Panel title="DECISION PREVIEW">
          {selected ? (
            <div className="radar-decision-preview">
              {preTrade?.verdict && (
                <div className={`radar-pretrade-badge radar-pretrade-${String(preTrade.verdict).toLowerCase().replace('_', '-')}`}>
                  Pre-trade · {preTrade.verdict}
                </div>
              )}
              <p><strong>{(row as RadarRow)?.reason || preTrade?.plan_summary || preTrade?.meaning || 'Select a stock from a lane above.'}</strong></p>
              {(preTrade?.plan?.entry ?? preTrade?.scan?.entry) != null && (
                <div>Entry zone: {money(preTrade?.plan?.entry ?? preTrade?.scan?.entry)}</div>
              )}
              {(preTrade?.plan?.stop ?? preTrade?.scan?.stop) != null && (
                <div>Invalidation: {money(preTrade?.plan?.stop ?? preTrade?.scan?.stop)}</div>
              )}
              {(preTrade?.plan?.target ?? preTrade?.scan?.target) != null && (
                <div>Target: {money(preTrade?.plan?.target ?? preTrade?.scan?.target)}</div>
              )}
              <div className="radar-action-row">
                <button type="button" onClick={() => setActive('Stock Intelligence')}>Stock Intelligence</button>
                <button type="button" onClick={() => onCompare(selected)}>Compare</button>
                <button type="button" onClick={() => onWatchlist(selected)}>Watchlist</button>
              </div>
            </div>
          ) : (
            <p className="radar-empty-li">Select a stock to preview the trade plan and next steps.</p>
          )}
        </Panel>
      </div>
    </section>
  )
}

export function MarketScannerView(props: ExperienceViewProps & { onCompare: (symbol: string) => void }) {
  const { dashboard, selected, setSelected, bars, setActive, depth, marketScan, longTermScan, onCompare } = props
  const scannerTabs = depth === 'professional'
    ? ['Breakouts', 'Momentum', 'Conviction', 'Pre-Breakout', 'Long-Term', 'F&O', 'Avoid']
    : ['Breakouts', 'Momentum', 'Long-Term']
  const [tab, setTab] = useState('Breakouts')
  const [rows, setRows] = useState<RadarRow[]>([])
  const [bestBreakout, setBestBreakout] = useState<RadarRow | null>(null)
  const [sniperCount, setSniperCount] = useState(0)
  const [meta, setMeta] = useState({ scanned_at: '', universe: 0 })
  const [search, setSearch] = useState('')
  const [sector, setSector] = useState('All')
  const [excludeChase, setExcludeChase] = useState(true)
  const [sniperOnly, setSniperOnly] = useState(false)

  const activeScan = tab === 'Long-Term' ? longTermScan : marketScan

  useEffect(() => {
    if (!scannerTabs.includes(tab)) setTab(scannerTabs[0])
  }, [depth])

  useEffect(() => {
    fetchScannerWorkspace(tab)
      .then((result) => {
        setRows(result.rows as RadarRow[])
        setMeta({ scanned_at: result.scanned_at, universe: result.universe_size })
        setBestBreakout((result.best_breakout as RadarRow | null | undefined) || null)
        setSniperCount(result.sniper_count ?? (result.sniper_rows?.length || 0))
      })
      .catch(() => {
        setRows([])
        setBestBreakout(null)
        setSniperCount(0)
      })
  }, [tab, dashboard.scan.scanned_at, dashboard.long_term.scanned_at])

  const sectors = useMemo(() => [...new Set(rows.map((r) => r.sector).filter(Boolean))].sort(), [rows])
  const filtered = rows.filter((row) => {
    const q = search.trim().toUpperCase()
    if (q && !row.symbol.includes(q) && !String(row.company || '').toUpperCase().includes(q)) return false
    if (sector !== 'All' && row.sector !== sector) return false
    if (excludeChase && row.chase_risk) return false
    if (tab === 'Breakouts' && sniperOnly && !row.sniper_candidate) return false
    return true
  })

  const selectedRow = filtered.find((r) => r.symbol === selected) || rows.find((r) => r.symbol === selected)

  return (
    <section className="market-scanner">
      <header className="scanner-command-bar">
        <div>
          <span>MARKET SCANNER</span>
          <h2>Breakouts · Momentum · Conviction · F&O</h2>
          <p>{filtered.length} matches · universe {meta.universe.toLocaleString('en-IN')} · scan {meta.scanned_at || '—'}</p>
        </div>
        <button type="button" disabled={activeScan.isBusy} onClick={() => void activeScan.start()}>
          {activeScan.isBusy ? 'Scanning…' : tab === 'Long-Term' ? 'Run long-term scan' : 'Scan now'}
        </button>
      </header>

      <LiveScanBanner scan={activeScan} depth={depth} label={tab === 'Long-Term' ? 'Long-term scan' : 'Market scan'} />

      <div className="radar-tab-row">
        {scannerTabs.map((item) => (
          <button key={item} type="button" className={tab === item ? 'active' : ''} onClick={() => setTab(item)}>{item}</button>
        ))}
      </div>

      {tab === 'Breakouts' && (
        <BestSniperPanel
          best={bestBreakout}
          sniperCount={sniperCount}
          onSelect={setSelected}
        />
      )}

      <div className="scanner-filter-row">
        <label>Search<input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Symbol" /></label>
        <label>Sector<select value={sector} onChange={(e) => setSector(e.target.value)}><option>All</option>{sectors.map((s) => <option key={s}>{s}</option>)}</select></label>
        <label className="scanner-check"><input type="checkbox" checked={excludeChase} onChange={(e) => setExcludeChase(e.target.checked)} /> Hide extended</label>
        {tab === 'Breakouts' && (
          <label className="scanner-check">
            <input type="checkbox" checked={sniperOnly} onChange={(e) => setSniperOnly(e.target.checked)} />
            Sniper candidates only
          </label>
        )}
      </div>

      <div className="scanner-workspace-grid">
        <Panel
          title={`${tab.toUpperCase()} · ${filtered.length}`}
          subtitle={tab === 'Breakouts'
            ? `Server sniper-first rank · ${sniperCount} sniper candidate${sniperCount === 1 ? '' : 's'}`
            : 'Sorted from persisted backend scan'}
        >
          <DenseTable rows={filtered} selected={selected} onSelect={setSelected} depth={depth} mode={tab} />
        </Panel>
        <div className="scanner-detail-column">
          <Panel title={`CHART · ${selected || '—'}`}><ChartWorkspace symbol={selected} bars={bars} row={selectedRow} /></Panel>
          <Panel title="ACTIONS">
            <div className="radar-action-row">
              <button type="button" disabled={!selected} onClick={() => setActive('Stock Intelligence')}>Stock Intelligence</button>
              <button type="button" disabled={!selected} onClick={() => selected && onCompare(selected)}>Compare</button>
            </div>
          </Panel>
        </div>
      </div>
    </section>
  )
}

export function CompareView({ symbols, setSymbols, setActive, setSelected }: {
  symbols: string[]
  setSymbols: (s: string[]) => void
  setActive: (page: string) => void
  setSelected: (s: string) => void
}) {
  const [data, setData] = useState<CompareWorkspace | null>(null)
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (symbols.length === 0) { setData(null); return }
    setLoading(true)
    fetchCompareWorkspace(symbols)
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false))
  }, [symbols.join(',')])

  const addSymbol = () => {
    const sym = input.trim().toUpperCase()
    if (!sym || symbols.includes(sym)) return
    if (symbols.length >= 5) return
    setSymbols([...symbols, sym])
    setInput('')
  }

  return (
    <section className="compare-view">
      <header className="radar-hero">
        <div><span>COMPARE</span><h2>Side-by-side fundamentals and market state</h2><p>{data?.disclaimer || 'Add up to 5 NSE symbols.'}</p></div>
      </header>
      <div className="compare-chips">
        {symbols.map((sym) => (
          <button key={sym} type="button" className="compare-chip" onClick={() => { setSelected(sym); setActive('Stock Intelligence') }}>{sym}</button>
        ))}
        <input value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && addSymbol()} placeholder="Add symbol" />
        <button type="button" onClick={addSymbol}>Add</button>
        <button type="button" onClick={() => setSymbols([])}>Clear</button>
      </div>
      {loading && <p>Loading comparison…</p>}
      {data && (
        <div className="compare-grid">
          {Object.entries(data.section_labels).map(([key, label]) => (
            <Panel key={key} title={label.toUpperCase()}>
              <table className="radar-table compare-table">
                <thead><tr><th>Metric</th>{data.rows.map((r) => <th key={r.symbol}>{r.symbol}</th>)}</tr></thead>
                <tbody>
                  {(data.rows[0]?.sections[key] || []).map((_, idx) => (
                    <tr key={idx}>
                      <td>{data.rows[0]?.sections[key]?.[idx]?.label}</td>
                      {data.rows.map((row) => {
                        const m = row.sections[key]?.[idx]
                        return <td key={row.symbol}>{m?.available ? `${m.value}${m.unit ? ` ${m.unit}` : ''}` : '—'}</td>
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </Panel>
          ))}
        </div>
      )}
    </section>
  )
}

export function WatchlistView({ setActive, setSelected, onCompare }: {
  setActive: (page: string) => void
  setSelected: (s: string) => void
  onCompare: (symbol: string) => void
}) {
  const [payload, setPayload] = useState<WatchlistPayload | null>(null)
  const [symbol, setSymbol] = useState('')
  const [notes, setNotes] = useState('')
  const [busy, setBusy] = useState(false)

  const reload = () => fetchWatchlist().then(setPayload).catch(() => setPayload(null))

  useEffect(() => { void reload() }, [])

  const add = async () => {
    const sym = symbol.trim().toUpperCase()
    if (!sym) return
    setBusy(true)
    try {
      await addWatchlistItem({ symbol: sym, notes })
      setSymbol('')
      setNotes('')
      await reload()
    } finally { setBusy(false) }
  }

  return (
    <section className="watchlist-view">
      <header className="radar-hero">
        <div><span>WATCHLIST</span><h2>Track names you want to investigate</h2><p>Personal list — not a second alerts engine.</p></div>
      </header>
      <div className="watchlist-add">
        <input value={symbol} onChange={(e) => setSymbol(e.target.value)} placeholder="NSE symbol" />
        <input value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="Why watching" />
        <button type="button" disabled={busy} onClick={() => void add()}>Add</button>
      </div>
      <table className="radar-table">
        <thead><tr><th>Symbol</th><th>Added</th><th>Setup</th><th>Notes</th><th>Actions</th></tr></thead>
        <tbody>
          {(payload?.items || []).map((item) => (
            <tr key={item.id}>
              <td><button type="button" onClick={() => { setSelected(item.symbol); setActive('Stock Intelligence') }}>{item.symbol}</button></td>
              <td>{item.added_date}</td>
              <td>{String((item.snapshot as RadarRow)?.setup_label || item.snapshot?.status || '—')}</td>
              <td>{item.notes || '—'}</td>
              <td className="radar-action-row">
                <button type="button" onClick={() => onCompare(item.symbol)}>Compare</button>
                <button type="button" onClick={() => void removeWatchlistItem(item.id).then(reload)}>Remove</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {payload?.count === 0 && <p className="radar-empty-li">No watchlist items yet.</p>}
    </section>
  )
}
