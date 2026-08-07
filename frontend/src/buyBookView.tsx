import { useCallback, useEffect, useState } from 'react'
import { EmptyState, StatusBadge } from './designSystem'
import { EvidenceList, MetricCard, Panel } from './components'
import { money, pct, words } from './format'
import {
  addBuyBookItem,
  fetchBuyBook,
  refreshBuyBookResearch,
  removeBuyBookItem,
  syncBuyBookFromHoldings,
  type BuyBookItem,
  type BuyBookPayload,
} from './productApi'

type Props = {
  selected?: string
  setSelected?: (symbol: string) => void
  setActive?: (page: string) => void
}

function severityTone(sev?: string): 'green' | 'amber' | 'purple' | 'cyan' {
  const s = String(sev || '').toLowerCase()
  if (s === 'good') return 'green'
  if (s === 'critical' || s === 'warn') return 'amber'
  if (s === 'info') return 'purple'
  return 'cyan'
}

function resultClass(label?: string | null, vs?: number | null): string {
  const tag = String(label || '').toUpperCase()
  if (tag === 'UP' || (vs != null && vs > 0)) return 'positive'
  if (tag === 'DOWN' || (vs != null && vs < 0)) return 'negative'
  return ''
}

function signedPct(value?: number | null): string {
  return pct(value)
}

function ratioText(value?: number | null, suffix = ''): string {
  if (value == null || Number.isNaN(value)) return '—'
  return `${value}${suffix}`
}

export function BuyBookView({ selected, setSelected, setActive }: Props) {
  const [book, setBook] = useState<BuyBookPayload | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [symbol, setSymbol] = useState(selected || '')
  const [entry, setEntry] = useState('')
  const [stop, setStop] = useState('')
  const [qty, setQty] = useState('')
  const [notes, setNotes] = useState('')
  const [busy, setBusy] = useState(false)
  const [note, setNote] = useState('')
  const [fetchResearch, setFetchResearch] = useState(true)
  const [forceFund, setForceFund] = useState(false)

  const refresh = useCallback(async (fresh = false) => {
    setLoading(true)
    setError('')
    try {
      const payload = await fetchBuyBook({ fresh })
      setBook(payload)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Active Buys unavailable')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh(false)
  }, [refresh])

  useEffect(() => {
    if (selected) setSymbol(selected)
  }, [selected])

  const onAdd = async () => {
    const clean = symbol.trim().toUpperCase()
    if (!clean) return
    setBusy(true)
    setNote('')
    try {
      await addBuyBookItem({
        symbol: clean,
        entry_price: entry ? Number(entry) : undefined,
        stop_price: stop ? Number(stop) : undefined,
        quantity: qty ? Number(qty) : undefined,
        notes: notes.trim() || undefined,
      })
      setEntry('')
      setStop('')
      setQty('')
      setNotes('')
      setNote(`${clean} added — results refresh with live/EOD price`)
      await refresh(true)
    } catch (err) {
      setNote(err instanceof Error ? err.message : 'Could not add')
    } finally {
      setBusy(false)
    }
  }

  const onRemove = async (item: BuyBookItem) => {
    setBusy(true)
    try {
      await removeBuyBookItem(item.id)
      await refresh(true)
    } catch (err) {
      setNote(err instanceof Error ? err.message : 'Could not remove')
    } finally {
      setBusy(false)
    }
  }

  const onSyncHoldings = async () => {
    setBusy(true)
    setNote(
      fetchResearch
        ? 'Syncing Zerodha holdings + fetching fundamentals & technicals…'
        : 'Syncing Zerodha holdings into Active Buys…',
    )
    try {
      const result = await syncBuyBookFromHoldings({
        refresh_kite: true,
        notify: false,
        fetch_research: fetchResearch,
        force_fundamentals: forceFund,
      })
      if (result.book) setBook(result.book)
      const closed = result.closed_stale_zerodha?.length
        ? ` · closed ${result.closed_stale_zerodha.length} stale Zerodha row(s)`
        : ''
      const researchMsg = result.research?.message ? ` · ${result.research.message}` : ''
      setNote(
        result.holdings_available === false && !result.upserted
          ? result.holdings_message || 'Zerodha not connected — connect Kite or import holdings first.'
          : `Tracking ${result.upserted} holding(s) from ${result.synced_from || 'Zerodha'}${closed}${researchMsg}`,
      )
      await refresh(true)
    } catch (err) {
      setNote(err instanceof Error ? err.message : 'Holdings sync failed')
    } finally {
      setBusy(false)
    }
  }

  const onFetchResearch = async () => {
    setBusy(true)
    setNote('Fetching Screener fundamentals + warming official technicals for Active Buys…')
    try {
      const result = await refreshBuyBookResearch({ force_fundamentals: forceFund })
      if (result.book) setBook(result.book as BuyBookPayload)
      setNote(result.message || 'Research refresh finished')
      await refresh(true)
    } catch (err) {
      setNote(err instanceof Error ? err.message : 'Fund + tech fetch failed')
    } finally {
      setBusy(false)
    }
  }

  const openStock = (sym: string) => {
    setSelected?.(sym)
    setActive?.('Stock Intelligence')
  }

  const openHoldings = () => setActive?.('Paper Portfolio')

  const summary = book?.summary
  const results = book?.results
  const items = book?.items || []
  const sortedByResult = [...items].sort((a, b) => {
    const av = a.vs_entry_pct
    const bv = b.vs_entry_pct
    if (av == null && bv == null) return a.symbol.localeCompare(b.symbol)
    if (av == null) return 1
    if (bv == null) return -1
    return bv - av
  })
  const fundMissing = items.filter((item) => !item.fundamentals?.available).length
  const zerodhaCount = items.filter((item) => item.source === 'zerodha' || item.source === 'holdings').length

  return (
    <section className="workspace-view">
      <header className="stock-workspace-hero" style={{ marginBottom: 16 }}>
        <div>
          <span>Your buys · Zerodha track · fundamentals + technicals</span>
          <h2>Active Buys</h2>
          <p>
            Sync demat holdings from Zerodha, then watch each name on two equal pillars:
            technical structure (EMA / support / volume) and fundamentals (P/E, ROE, growth from Screener cache).
            Research only — never places orders.
          </p>
        </div>
      </header>

      <p className="panel-copy">{book?.honesty || 'Not a buy/sell ticket.'}</p>
      {results?.honesty ? <p className="panel-copy">{results.honesty}</p> : null}

      <Panel title="ZERODHA → ACTIVE BUYS" subtitle="Track CNC holdings · optional fundamentals + technicals fetch">
        <label className="panel-copy" style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
          <input
            type="checkbox"
            checked={fetchResearch}
            onChange={(event) => setFetchResearch(event.target.checked)}
            disabled={busy}
          />
          Also fetch fundamentals + technicals after sync (Screener + official bhav)
        </label>
        <label className="panel-copy" style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
          <input
            type="checkbox"
            checked={forceFund}
            onChange={(event) => setForceFund(event.target.checked)}
            disabled={busy}
          />
          Force re-fetch fundamentals (ignore Screener cache)
        </label>
        <div className="inline-actions" style={{ flexWrap: 'wrap', gap: 8 }}>
          <button type="button" disabled={busy} onClick={() => void onSyncHoldings()}>
            {busy
              ? fetchResearch
                ? 'Syncing + fetching research…'
                : 'Syncing…'
              : fetchResearch
                ? 'Track Zerodha + fetch fund/tech'
                : 'Track Zerodha holdings'}
          </button>
          <button type="button" disabled={busy || !items.length} onClick={() => void onFetchResearch()}>
            {busy ? 'Fetching…' : 'Fetch fund + tech now'}
          </button>
          <button type="button" onClick={openHoldings}>
            Open My Holdings
          </button>
          <button type="button" disabled={loading || busy} onClick={() => void refresh(true)}>
            Re-score from cache
          </button>
        </div>
        <p className="panel-copy" style={{ marginTop: 8 }}>
          Fundamentals = Screener.in (cache-first). Technicals = official NSE bhav history already on disk.
          Low-power mode caps the batch so your Air stays usable. Missing stays missing — never invents PE/prices.
        </p>
        <div className="metric-grid" style={{ marginTop: 12 }}>
          <MetricCard label="TRACKED" value={String(summary?.total ?? 0)} detail="Active buy rows" />
          <MetricCard label="FROM ZERODHA" value={String(zerodhaCount)} detail="Synced demat names" tone="cyan" />
          <MetricCard
            label="FUND MISSING"
            value={String(fundMissing)}
            detail={fundMissing ? 'Tap Fetch fund + tech now' : 'Screener cache present'}
            tone={fundMissing ? 'amber' : 'green'}
          />
          <MetricCard label="AT RISK" value={String(summary?.critical ?? 0)} detail="Tech/fund damage" tone="amber" />
        </div>
        {note ? <p className="panel-copy">{note}</p> : null}
      </Panel>

      <Panel title="STOCK RESULTS" subtitle="Entry/avg vs live LTP or EOD · missing entry stays missing">
        <div className="metric-grid">
          <MetricCard label="UP" value={String(results?.up ?? 0)} detail="Above your entry" tone="green" />
          <MetricCard label="DOWN" value={String(results?.down ?? 0)} detail="Below your entry" tone="amber" />
          <MetricCard
            label="AVG RESULT"
            value={results?.avg_vs_entry_pct == null ? '—' : signedPct(results.avg_vs_entry_pct)}
            detail={results?.with_entry ? `${results.with_entry} with entry` : 'Add entry / sync holdings'}
            tone={
              results?.avg_vs_entry_pct == null
                ? 'cyan'
                : results.avg_vs_entry_pct >= 0
                  ? 'green'
                  : 'amber'
            }
          />
          <MetricCard
            label="EST. ₹ P&L"
            value={results?.est_pnl_total == null ? '—' : money(results.est_pnl_total, 0)}
            detail="Qty × (now − entry) · demat P&L shown per row when synced"
            tone={
              results?.est_pnl_total == null
                ? 'cyan'
                : results.est_pnl_total >= 0
                  ? 'green'
                  : 'amber'
            }
          />
        </div>
      </Panel>

      <Panel title="ADD A BUY" subtitle="Manual add · or sync Zerodha above">
        <div className="inline-actions" style={{ flexWrap: 'wrap', gap: 8 }}>
          <input
            aria-label="Symbol"
            placeholder="SYMBOL"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            style={{ minWidth: 120 }}
          />
          <input
            aria-label="Entry price"
            placeholder="Entry ₹"
            value={entry}
            onChange={(e) => setEntry(e.target.value)}
            style={{ minWidth: 100 }}
          />
          <input
            aria-label="Quantity"
            placeholder="Qty"
            value={qty}
            onChange={(e) => setQty(e.target.value)}
            style={{ minWidth: 80 }}
          />
          <input
            aria-label="Stop price"
            placeholder="Stop ₹"
            value={stop}
            onChange={(e) => setStop(e.target.value)}
            style={{ minWidth: 100 }}
          />
          <input
            aria-label="Notes"
            placeholder="Notes"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            style={{ minWidth: 160, flex: 1 }}
          />
          <button type="button" disabled={busy || !symbol.trim()} onClick={() => void onAdd()}>
            {busy ? 'Saving…' : 'Add to Active Buys'}
          </button>
        </div>
      </Panel>

      {loading && <div className="large-empty">Loading stock results…</div>}
      {error && <EmptyState title="Active Buys unavailable" detail={error} />}

      {!loading && items.length === 0 && (
        <EmptyState
          title="No stock results yet"
          detail="Tap Track Zerodha holdings (when Kite is connected), or add a symbol with entry. Each name is scored on technicals and fundamentals."
        />
      )}

      {!loading && sortedByResult.length > 0 && (
        <Panel title="RESULTS BOARD" subtitle="Sorted by vs-entry · Tech + Fund labels on every row">
          <div className="radar-table-wrap">
            <table className="radar-table">
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Source</th>
                  <th>Result</th>
                  <th>vs entry</th>
                  <th>Now</th>
                  <th>1D / 5D</th>
                  <th>Technicals</th>
                  <th>Fundamentals</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {sortedByResult.map((item) => {
                  const label = item.result_label || (item.vs_entry_pct == null ? 'NO ENTRY' : '—')
                  return (
                    <tr key={item.id} onClick={() => openStock(item.symbol)}>
                      <td>
                        <strong>{item.symbol}</strong>
                      </td>
                      <td>{item.source === 'zerodha' || item.source === 'holdings' ? 'Zerodha' : 'Manual'}</td>
                      <td className={resultClass(label, item.vs_entry_pct)}>
                        <strong>{label}</strong>
                      </td>
                      <td className={resultClass(label, item.vs_entry_pct)}>
                        {item.vs_entry_pct == null ? '—' : signedPct(item.vs_entry_pct)}
                      </td>
                      <td>{item.price != null ? money(item.price, 1) : '—'}</td>
                      <td>
                        <span className={resultClass(undefined, item.chg_1d_pct)}>{signedPct(item.chg_1d_pct)}</span>
                        {' / '}
                        <span className={resultClass(undefined, item.chg_5d_pct)}>{signedPct(item.chg_5d_pct)}</span>
                      </td>
                      <td>
                        <StatusBadge status={String(item.tech_label || item.status_label || 'UNKNOWN')} />
                      </td>
                      <td>
                        <StatusBadge status={String(item.fund_label || 'MISSING')} />
                      </td>
                      <td onClick={(e) => e.stopPropagation()}>
                        <div className="inline-actions">
                          <button type="button" onClick={() => openStock(item.symbol)}>
                            Open
                          </button>
                          <button type="button" disabled={busy} onClick={() => void onRemove(item)}>
                            Remove
                          </button>
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </Panel>
      )}

      <div className="stock-context-grid">
        {sortedByResult.map((item) => {
          const health = item.health
          const tech = item.technicals || health?.technicals
          const fund = item.fundamentals || health?.fundamentals
          const avgs = tech?.averages || health?.averages || {}
          const supports = tech?.supports || health?.supports || {}
          const techWarnings = (tech?.warnings || health?.warnings || []).map((w) => w.text)
          const fundFlags = (fund?.flags || []).map((w) => w.text)
          const ratios = fund?.ratios || {}
          return (
            <Panel
              key={`detail-${item.id}`}
              title={`${item.symbol} · ${words(item.status_label || item.severity || 'Unknown')}`}
              subtitle={
                health?.available
                  ? `Now ${money(item.price, 1)} · ${health.price_source || '—'} · ${item.source === 'zerodha' ? 'Zerodha' : 'Manual'}`
                  : 'History incomplete'
              }
              action={
                <div className="inline-actions">
                  <StatusBadge status={String(item.status_label || item.severity || 'UNKNOWN')} />
                  <button type="button" onClick={() => openStock(item.symbol)}>
                    Open desk
                  </button>
                  <button type="button" disabled={busy} onClick={() => void onRemove(item)}>
                    Remove
                  </button>
                </div>
              }
            >
              <div className="fact-grid">
                <div>
                  <span>Result</span>
                  <strong className={resultClass(item.result_label, item.vs_entry_pct)}>
                    {item.result_label || '—'} · {item.vs_entry_pct == null ? 'add entry' : signedPct(item.vs_entry_pct)}
                  </strong>
                </div>
                <div>
                  <span>Entry → Now</span>
                  <strong>
                    {item.entry_price != null ? money(item.entry_price, 1) : '—'} →{' '}
                    {item.price != null ? money(item.price, 1) : '—'}
                  </strong>
                </div>
                <div>
                  <span>Qty / Demat P&L</span>
                  <strong>
                    {item.quantity != null ? String(item.quantity) : '—'}
                    {' · '}
                    {item.demat_pnl == null ? '—' : money(item.demat_pnl, 0)}
                    {item.demat_pnl_pct != null ? ` (${signedPct(item.demat_pnl_pct)})` : ''}
                  </strong>
                </div>
                <div>
                  <span>Stop</span>
                  <strong>{item.stop_price != null ? money(item.stop_price, 1) : '—'}</strong>
                </div>
              </div>

              <Panel title="TECHNICALS" subtitle={tech?.note || 'EMA stack · swing support · volume'}>
                <div className="fact-grid">
                  <div>
                    <span>Tech status</span>
                    <strong>{words(tech?.status_label || item.tech_label || '—')}</strong>
                  </div>
                  <div>
                    <span>EMA 20 / 50 / 200</span>
                    <strong>
                      {money(avgs.ema20, 1)} / {money(avgs.ema50, 1)} / {money(avgs.ema200, 1)}
                    </strong>
                  </div>
                  <div>
                    <span>Support 20d / 60d</span>
                    <strong>
                      {money(supports.swing_20d, 1)} / {money(supports.swing_60d, 1)}
                    </strong>
                  </div>
                  <div>
                    <span>1D / 5D / Vol</span>
                    <strong>
                      {signedPct(item.chg_1d_pct)} / {signedPct(item.chg_5d_pct)} /{' '}
                      {ratioText(Number(tech?.structure?.volume_ratio ?? health?.structure?.volume_ratio), '×')}
                    </strong>
                  </div>
                </div>
                <EvidenceList
                  title="Technical warnings"
                  items={techWarnings}
                  tone={item.severity === 'good' ? 'green' : item.severity === 'critical' ? 'red' : 'cyan'}
                />
              </Panel>

              <Panel
                title="FUNDAMENTALS"
                subtitle={fund?.available ? `Cache ${fund.freshness || fund.status || ''} · ${fund.fetched_at || ''}` : 'Screener cache missing'}
              >
                {fund?.about ? <p className="panel-copy">{fund.about}</p> : null}
                <div className="fact-grid">
                  <div>
                    <span>P/E</span>
                    <strong>{ratioText(ratios.pe, 'x')}</strong>
                  </div>
                  <div>
                    <span>ROE</span>
                    <strong>{ratioText(ratios.roe, '%')}</strong>
                  </div>
                  <div>
                    <span>ROCE</span>
                    <strong>{ratioText(ratios.roce, '%')}</strong>
                  </div>
                  <div>
                    <span>Debt/Equity</span>
                    <strong>{ratioText(ratios.debt_to_equity)}</strong>
                  </div>
                  <div>
                    <span>Sales growth</span>
                    <strong>{ratios.sales_growth_pct == null ? '—' : signedPct(ratios.sales_growth_pct)}</strong>
                  </div>
                  <div>
                    <span>Profit growth</span>
                    <strong>{ratios.profit_growth_pct == null ? '—' : signedPct(ratios.profit_growth_pct)}</strong>
                  </div>
                </div>
                <EvidenceList
                  title="Fundamental flags"
                  items={
                    fundFlags.length
                      ? fundFlags
                      : [fund?.note || 'Open Stock Intelligence → Retry fundamentals to fill this cache.']
                  }
                  tone={!fund?.available ? 'amber' : fund.severity === 'warn' || fund.severity === 'critical' ? 'red' : 'green'}
                />
                {!fund?.available ? (
                  <button type="button" onClick={() => openStock(item.symbol)}>
                    Fetch fundamentals on desk
                  </button>
                ) : null}
              </Panel>

              {item.notes ? <p className="panel-copy">{item.notes}</p> : null}
              <MetricCard
                label="BLEND RISK"
                value={String(health?.risk_score ?? '—')}
                detail={`${words(item.severity || 'unknown')} · tech + fund`}
                tone={severityTone(item.severity)}
              />
            </Panel>
          )
        })}
      </div>
    </section>
  )
}
