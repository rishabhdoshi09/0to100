import { useCallback, useEffect, useState } from 'react'
import { EmptyState, StatusBadge } from './designSystem'
import { EvidenceList, MetricCard, Panel } from './components'
import { money, pct, words } from './format'
import {
  addBuyBookItem,
  fetchBuyBook,
  removeBuyBookItem,
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

  const refresh = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const payload = await fetchBuyBook()
      setBook(payload)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Active Buys unavailable')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
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
      await refresh()
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
      await refresh()
    } catch (err) {
      setNote(err instanceof Error ? err.message : 'Could not remove')
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

  return (
    <section className="workspace-view">
      <header className="stock-workspace-hero" style={{ marginBottom: 16 }}>
        <div>
          <span>Your buys · stock results + health guard</span>
          <h2>Active Buys</h2>
          <p>
            Add stocks you are buying. See entry → now result %, 1D/5D move, and warnings if price
            breaks major averages or swing support. Research only — not sell orders.
          </p>
        </div>
      </header>

      <p className="panel-copy">{book?.honesty || 'Not a buy/sell ticket.'}</p>
      {results?.honesty ? <p className="panel-copy">{results.honesty}</p> : null}

      <Panel title="STOCK RESULTS" subtitle="Entry vs live LTP or EOD · missing entry stays missing">
        <div className="metric-grid">
          <MetricCard label="UP" value={String(results?.up ?? 0)} detail="Above your entry" tone="green" />
          <MetricCard label="DOWN" value={String(results?.down ?? 0)} detail="Below your entry" tone="amber" />
          <MetricCard
            label="AVG RESULT"
            value={results?.avg_vs_entry_pct == null ? '—' : signedPct(results.avg_vs_entry_pct)}
            detail={results?.with_entry ? `${results.with_entry} with entry` : 'Add entry prices'}
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
            detail="Only when you typed qty · not demat"
            tone={
              results?.est_pnl_total == null
                ? 'cyan'
                : results.est_pnl_total >= 0
                  ? 'green'
                  : 'amber'
            }
          />
        </div>
        <div className="metric-grid" style={{ marginTop: 12 }}>
          <MetricCard label="ACTIVE" value={String(summary?.total ?? 0)} detail="Symbols in buy book" />
          <MetricCard label="AT RISK" value={String(summary?.critical ?? 0)} detail="Critical technical damage" tone="amber" />
          <MetricCard label="WEAKENING" value={String(summary?.warn ?? 0)} detail="Below key averages / soft support" tone="amber" />
          <MetricCard label="NO ENTRY" value={String(results?.missing_entry ?? 0)} detail="Result % needs your entry" />
        </div>
        <div className="inline-actions" style={{ marginTop: 12 }}>
          <button type="button" onClick={openHoldings}>
            Open My Holdings for demat ₹ P&L
          </button>
        </div>
      </Panel>

      <Panel title="ADD A BUY" subtitle="Symbol + entry recommended · qty optional for ₹ estimate">
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
          <button type="button" disabled={loading} onClick={() => void refresh()}>
            Refresh results
          </button>
        </div>
        {note ? <p className="panel-copy">{note}</p> : null}
      </Panel>

      {loading && <div className="large-empty">Loading stock results…</div>}
      {error && <EmptyState title="Active Buys unavailable" detail={error} />}

      {!loading && items.length === 0 && (
        <EmptyState
          title="No stock results yet"
          detail="Add a symbol with your entry price to see result %. The system also watches 20/50/200-day averages and swing support."
        />
      )}

      {!loading && sortedByResult.length > 0 && (
        <Panel title="RESULTS BOARD" subtitle="Sorted by vs-entry · health warnings under each row">
          <div className="radar-table-wrap">
            <table className="radar-table">
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Result</th>
                  <th>vs entry</th>
                  <th>Entry</th>
                  <th>Now</th>
                  <th>1D</th>
                  <th>5D</th>
                  <th>Est. P&L</th>
                  <th>Health</th>
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
                      <td className={resultClass(label, item.vs_entry_pct)}>
                        <strong>{label}</strong>
                      </td>
                      <td className={resultClass(label, item.vs_entry_pct)}>
                        {item.vs_entry_pct == null ? '—' : signedPct(item.vs_entry_pct)}
                      </td>
                      <td>{item.entry_price != null ? money(item.entry_price, 1) : '—'}</td>
                      <td>{item.price != null ? money(item.price, 1) : '—'}</td>
                      <td className={resultClass(undefined, item.chg_1d_pct)}>
                        {signedPct(item.chg_1d_pct)}
                      </td>
                      <td className={resultClass(undefined, item.chg_5d_pct)}>
                        {signedPct(item.chg_5d_pct)}
                      </td>
                      <td className={resultClass(undefined, item.est_pnl)}>
                        {item.est_pnl == null ? '—' : money(item.est_pnl, 0)}
                      </td>
                      <td>
                        <StatusBadge status={String(item.status_label || item.severity || 'UNKNOWN')} />
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
          const avgs = health?.averages || {}
          const supports = health?.supports || {}
          const warnings = (health?.warnings || []).map((w) => w.text)
          return (
            <Panel
              key={`detail-${item.id}`}
              title={`${item.symbol} · ${words(item.status_label || item.severity || 'Unknown')}`}
              subtitle={
                health?.available
                  ? `Now ${money(item.price, 1)} · ${health.price_source || '—'} · as of ${health.as_of || '—'}`
                  : 'History incomplete'
              }
              action={
                <div className="inline-actions">
                  <StatusBadge status={String(item.status_label || item.severity || 'UNKNOWN')} />
                  <button type="button" onClick={() => openStock(item.symbol)}>
                    Open
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
                  <span>1D / 5D</span>
                  <strong>
                    <span className={resultClass(undefined, item.chg_1d_pct)}>{signedPct(item.chg_1d_pct)}</span>
                    {' / '}
                    <span className={resultClass(undefined, item.chg_5d_pct)}>{signedPct(item.chg_5d_pct)}</span>
                  </strong>
                </div>
                <div>
                  <span>Est. P&L</span>
                  <strong className={resultClass(undefined, item.est_pnl)}>
                    {item.est_pnl == null
                      ? item.quantity
                        ? 'Need entry + price'
                        : 'Add qty for ₹ estimate'
                      : money(item.est_pnl, 0)}
                  </strong>
                </div>
                <div>
                  <span>Stop</span>
                  <strong>{item.stop_price != null ? money(item.stop_price, 1) : '—'}</strong>
                </div>
                <div>
                  <span>Qty</span>
                  <strong>{item.quantity != null ? String(item.quantity) : '—'}</strong>
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
              </div>
              {item.notes ? <p className="panel-copy">{item.notes}</p> : null}
              <EvidenceList
                title="Health warnings"
                items={warnings}
                tone={item.severity === 'good' ? 'green' : item.severity === 'critical' ? 'red' : 'cyan'}
              />
              <MetricCard
                label="RISK SCORE"
                value={String(health?.risk_score ?? '—')}
                detail={words(item.severity || 'unknown')}
                tone={severityTone(item.severity)}
              />
            </Panel>
          )
        })}
      </div>
    </section>
  )
}
