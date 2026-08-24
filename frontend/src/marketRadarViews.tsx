import { useEffect, useMemo, useState } from 'react'
import { ChartWorkspace, Panel } from './components'
import { BotLearningPanel } from './views'
import { money, pct, words } from './format'
import {
  addWatchlistItem,
  fetchCompareWorkspace,
  fetchRadarHome,
  fetchScannerWorkspace,
  fetchWatchlist,
  removeWatchlistItem,
  type CompareWorkspace,
  type RadarHome,
  type ScannerWorkspaceRow,
  type WatchlistPayload,
} from './productApi'
import { LiveScanBanner, type ExperienceViewProps } from './experience'
import { fetchTradePlan, type TradePlan } from './productApi'

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
  why?: string
  sepa_score?: number
  sepa_passed?: number
  sepa_total?: number
  sepa_verdict?: string
}

function recoBadge(row: RadarRow): [string, string] {
  const verdict = String(row.sepa_verdict || '').toUpperCase()
  if (verdict === 'STRONG') return ['SEPA qualified', 'buy']
  if (verdict === 'CONSTRUCTIVE') return ['Setup forming', 'watch']
  if (row.chase_risk) return ['Avoid', 'avoid']
  if (row.status === 'Ready to trade' || String(row.verdict || '').toUpperCase() === 'BUY') return ['Buy Setup', 'buy']
  if (row.status === 'Wait for pullback') return ['Wait', 'wait']
  return ['Watch', 'watch']
}

function RecoCard({
  row,
  selected,
  onSelect,
}: {
  row: RadarRow
  selected: boolean
  onSelect: (symbol: string) => void
}) {
  const [label, kind] = recoBadge(row)
  const buy = kind === 'buy' ? 'Buy' : kind === 'avoid' ? 'Avoid' : kind === 'wait' ? 'Wait' : 'Watch'
  const entry = Number(row.entry || 0)
  const target = Number(row.target || 0)
  const price = Number(row.price || 0)
  const upside = entry > 0 && target > entry ? ((target - entry) / entry) * 100 : null
  const fromNow = price > 0 && target > 0 ? ((target - price) / price) * 100 : null
  const risk = row.chase_risk ? 'High Risk' : kind === 'buy' ? 'Setup' : 'Medium Risk'
  const inr = (value: number) => `₹${value.toLocaleString('en-IN', { maximumFractionDigits: 2 })}`
  return (
    <button
      type="button"
      className={selected ? 'reco-card rw-stock-card active' : 'reco-card rw-stock-card'}
      onClick={() => onSelect(row.symbol)}
    >
      <span className={`rw-buy ${kind}`}>{buy}</span>
      <div className="rw-stock-id">
        <div className="rw-logo">{row.symbol.slice(0, 1)}</div>
        <div>
          <strong>{row.company || row.symbol}</strong>
          <small>{row.symbol}</small>
        </div>
      </div>
      <div className="rw-tags">
        <span>{label}</span>
        <span className="risk">{risk}</span>
        {row.sepa_score != null && <span>SEPA {row.sepa_score}/100</span>}
      </div>
      <div className="rw-money">
        <div>
          <span>Target</span>
          <b>{target ? inr(target) : '—'}</b>
          <small>Entry Price: {entry ? inr(entry) : 'n/a'}</small>
        </div>
        <div className="upside">
          <b>{upside != null ? `↗ ${upside.toFixed(1)}%` : '—'}</b>
          <small>Upside from entry</small>
          {fromNow != null && <small>{fromNow.toFixed(1)}% from current</small>}
        </div>
      </div>
      <div className="rw-cmp">
        <span>Current Price</span>
        <strong>{price ? inr(price) : 'n/a'}</strong>
      </div>
    </button>
  )
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
  const [sortKey, setSortKey] = useState('score')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')

  const sorted = useMemo(() => {
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
    : depth === 'professional'
      ? ['symbol', 'price', 'change_5d_pct', 'sector', 'setup_label', 'breakout_state', 'momentum_state', 'relative_strength', 'risk_label']
      : ['symbol', 'price', 'change_5d_pct', 'sector', 'setup_label', 'risk_label']

  return (
    <div className="radar-table-wrap">
      <table className="radar-table">
        <thead>
          <tr>
            {cols.map((col) => (
              <th key={col} onClick={() => toggleSort(col)}>{words(col.replace(/_/g, ' '))}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.length === 0 && (
            <tr><td colSpan={cols.length} className="radar-empty">No matches in saved scan data. Run Scan Now.</td></tr>
          )}
          {sorted.map((row) => (
            <tr key={row.symbol} className={selected === row.symbol ? 'selected' : ''} onClick={() => onSelect(row.symbol)}>
              {cols.map((col) => {
                const raw = (row as Record<string, unknown>)[col]
                let cell: string
                if (col === 'breakout_state') cell = breakoutLabel[String(raw)] || words(String(raw))
                else if (col === 'momentum_state') cell = momentumLabel[String(raw)] || words(String(raw))
                else if (col === 'price') cell = money(raw as number)
                else if (col === 'change_5d_pct') cell = pct(raw as number)
                else if (col === 'combined_score' || col === 'relative_strength') cell = raw != null ? String(raw) : '—'
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
  const [plan, setPlan] = useState<TradePlan | null>(null)

  useEffect(() => {
    fetchRadarHome().then(setRadar).catch(() => setRadar(null))
  }, [dashboard.scan.scanned_at, dashboard.long_term.scanned_at])

  useEffect(() => {
    if (!selected) { setPlan(null); return }
    fetchTradePlan(selected).then(setPlan).catch(() => setPlan(null))
  }, [selected, dashboard.scan.scanned_at])

  const row = (radar?.best_setups || []).find((r) => r.symbol === selected)
    || radar?.lanes.breakouts.find((r) => r.symbol === selected)
    || radar?.lanes.momentum.find((r) => r.symbol === selected)
    || radar?.lanes.long_term_picks.find((r) => r.symbol === selected)

  const sepaCards = (radar?.best_setups || []) as RadarRow[]
  const watchlist = ((radar?.lanes.breakouts?.length ? radar.lanes.breakouts : radar?.lanes.momentum) || []).slice(0, 6) as RadarRow[]

  return (
    <section className="radar-home reco-desk">
      <div className="rw-crumb">Home &gt; Recommendations &gt; Top Stocks</div>
      <header className="radar-hero">
        <div>
          <span>TODAY · RECO WEALTH</span>
          <h2>Top Stocks</h2>
          <p>{radar?.market_health || dashboard.market.health} · {dashboard.market.summary}</p>
        </div>
        <div className="radar-hero-actions">
          <button type="button" disabled={marketScan.isBusy} onClick={() => void marketScan.start()}>
            {marketScan.isBusy ? 'Scanning…' : 'Scan now'}
          </button>
        </div>
      </header>
      <div className="rw-delay">⚠ CMP is delayed by up to 15 minutes.</div>
      <div className="rw-quote">
        <div className="mark">“</div>
        <p>The stock market is a device for transferring money from the impatient to the patient.</p>
        <cite>— Warren Buffett</cite>
      </div>

      <div className="radar-market-strip">
        <div><span>NIFTY 1D</span><strong>{pct(radar?.nifty_change_1d ?? dashboard.market.nifty_change_1d)}</strong></div>
        <div><span>BREADTH</span><strong>{radar?.breadth || dashboard.market.breadth}</strong></div>
        <div><span>VIX</span><strong>{radar?.vix ?? dashboard.market.vix ?? '—'}</strong></div>
        <div><span>LEADERS</span><strong>{(radar?.leaders || dashboard.market.leaders).slice(0, 3).join(', ') || '—'}</strong></div>
        <div><span>LAST SCAN</span><strong>{radar?.scan_scanned_at || dashboard.scan.scanned_at || 'Not run'}</strong></div>
        <div><span>DATA</span><strong>{dashboard.data.bhavcopy.latest_date || '—'}</strong></div>
      </div>

      <LiveScanBanner scan={marketScan} depth={depth} label="Market scan" />

      <div className="reco-how">
        <div className="qt-eyebrow">How to use this desk</div>
        <ol>
          <li><span className="k">Today</span> — SEPA-qualified Top Stocks first, then the scanner watchlist. If SEPA is empty, the scan names did not clear the Stage-2 floor.</li>
          <li><span className="k">Setups</span> — Best Setups (SEPA), Momentum, Conviction, Long-term. Do not mix them.</li>
          <li><span className="k">Paper Desk</span> — simulated trades. The bot learns daily. No broker orders here.</li>
          <li><span className="k">Backtest</span> — inspect a paper loss. It does not change today’s BUY list.</li>
        </ol>
      </div>

      <div className="reco-section">
        <div className="qt-eyebrow">Top stocks</div>
        <h3>Best Setups · SEPA qualified <span className="rw-live">● LIVE</span></h3>
        <p className="reco-note">{radar?.best_setups_note || 'Minervini 7-rule Stage-2 template on official NSE history. A qualify is research — not a buy order.'}</p>
      </div>
      <div className="reco-card-grid">
        {sepaCards.map((item) => (
          <RecoCard key={item.symbol} row={item} selected={selected === item.symbol} onSelect={setSelected} />
        ))}
      </div>
      {!sepaCards.length && (
        <p className="empty-row">No SEPA-qualified names in the last scan yet. Keep autonomy running, or open Setups and queue a scan.</p>
      )}

      <div className="reco-section">
        <div className="qt-eyebrow">Scanner watchlist</div>
        <h3>What the momentum scan is watching</h3>
        <p className="reco-note">Saved whole-market scan. Not the same as SEPA-qualified Best Setups above.</p>
      </div>
      <div className="reco-card-grid">
        {watchlist.map((item) => (
          <RecoCard key={`watch-${item.symbol}`} row={item} selected={selected === item.symbol} onSelect={setSelected} />
        ))}
      </div>
      {!watchlist.length && (
        <p className="empty-row">No saved scan yet. An empty list is not the same as “no trade today”.</p>
      )}

      <BotLearningPanel dashboard={dashboard} />

      {selected && (
        <div className="radar-workspace">
          <Panel title={`CHART · ${selected}`} subtitle={`Official history · ${dashboard.data.bhavcopy.latest_date || '—'}`}>
            <ChartWorkspace symbol={selected} bars={bars} row={row} />
          </Panel>
          <Panel title="DECISION PREVIEW">
            <div className="radar-decision-preview">
              <p><strong>{(row as RadarRow)?.reason || plan?.summary || 'Select a stock from a card above.'}</strong></p>
              {plan?.entry != null && <div>Entry zone: {money(plan.entry)}</div>}
              {plan?.stop != null && <div>Invalidation: {money(plan.stop)}</div>}
              {plan?.target != null && <div>Target: {money(plan.target)}</div>}
              <div className="radar-action-row">
                <button type="button" onClick={() => setActive('Desk')}>Open on Desk</button>
                <button type="button" onClick={() => onCompare(selected)}>Compare</button>
                <button type="button" onClick={() => onWatchlist(selected)}>Watchlist</button>
              </div>
            </div>
          </Panel>
        </div>
      )}
    </section>
  )
}

export function MarketScannerView(props: ExperienceViewProps & { onCompare: (symbol: string) => void }) {
  const { dashboard, selected, setSelected, bars, setActive, depth, marketScan, longTermScan, onCompare } = props
  const [tab, setTab] = useState<'Best Setups' | 'Breakouts' | 'Momentum' | 'Long-Term'>('Best Setups')
  const [rows, setRows] = useState<RadarRow[]>([])
  const [meta, setMeta] = useState({ scanned_at: '', universe: 0 })
  const [search, setSearch] = useState('')
  const [sector, setSector] = useState('All')
  const [excludeChase, setExcludeChase] = useState(true)

  const activeScan = tab === 'Long-Term' ? longTermScan : marketScan

  useEffect(() => {
    if (tab === 'Best Setups') {
      fetchRadarHome()
        .then((result) => {
          setRows((result.best_setups || []) as RadarRow[])
          setMeta({ scanned_at: result.scan_scanned_at, universe: result.universe_size })
        })
        .catch(() => setRows([]))
      return
    }
    fetchScannerWorkspace(tab)
      .then((result) => {
        setRows(result.rows as RadarRow[])
        setMeta({ scanned_at: result.scanned_at, universe: result.universe_size })
      })
      .catch(() => setRows([]))
  }, [tab, dashboard.scan.scanned_at, dashboard.long_term.scanned_at])

  const sectors = useMemo(() => [...new Set(rows.map((r) => r.sector).filter(Boolean))].sort(), [rows])
  const filtered = rows.filter((row) => {
    const q = search.trim().toUpperCase()
    if (q && !row.symbol.includes(q) && !String(row.company || '').toUpperCase().includes(q)) return false
    if (sector !== 'All' && row.sector !== sector) return false
    if (excludeChase && row.chase_risk) return false
    return true
  })

  const selectedRow = filtered.find((r) => r.symbol === selected) || rows.find((r) => r.symbol === selected)

  return (
    <section className="market-scanner">
      <header className="scanner-command-bar">
        <div>
          <span>SETUPS · RECO WEALTH</span>
          <h2>Recommendations</h2>
          <p>{filtered.length} matches · universe {meta.universe.toLocaleString('en-IN')} · scan {meta.scanned_at || '—'}</p>
        </div>
        <button type="button" disabled={activeScan.isBusy} onClick={() => void activeScan.start()}>
          {activeScan.isBusy ? 'Scanning…' : tab === 'Long-Term' ? 'Run long-term scan' : 'Scan now'}
        </button>
      </header>

      <LiveScanBanner scan={activeScan} depth={depth} label={tab === 'Long-Term' ? 'Long-term scan' : 'Market scan'} />

      <div className="radar-tab-row">
        {(['Best Setups', 'Breakouts', 'Momentum', 'Long-Term'] as const).map((item) => (
          <button key={item} type="button" className={tab === item ? 'active' : ''} onClick={() => setTab(item)}>{item}</button>
        ))}
      </div>

      <div className="scanner-filter-row">
        <label>Search<input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Symbol" /></label>
        <label>Sector<select value={sector} onChange={(e) => setSector(e.target.value)}><option>All</option>{sectors.map((s) => <option key={s}>{s}</option>)}</select></label>
        <label className="scanner-check"><input type="checkbox" checked={excludeChase} onChange={(e) => setExcludeChase(e.target.checked)} /> Hide extended</label>
      </div>

      <div className="scanner-workspace-grid">
        <Panel title={`${tab.toUpperCase()} · ${filtered.length}`} subtitle="Sorted from persisted backend scan">
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
