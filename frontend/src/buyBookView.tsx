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

export function BuyBookView({ selected, setSelected, setActive }: Props) {
  const [book, setBook] = useState<BuyBookPayload | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [symbol, setSymbol] = useState(selected || '')
  const [entry, setEntry] = useState('')
  const [stop, setStop] = useState('')
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
        notes: notes.trim() || undefined,
      })
      setEntry('')
      setStop('')
      setNotes('')
      setNote(`${clean} added to Active Buys`)
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

  const summary = book?.summary
  const items = book?.items || []

  return (
    <section className="workspace-view">
      <header className="stock-workspace-hero" style={{ marginBottom: 16 }}>
        <div>
          <span>Your buys · technical health guard</span>
          <h2>Active Buys</h2>
          <p>
            Add stocks you are buying. QuantTerm warns if price breaks major averages or swing support,
            or shows heavy selling. Research warnings only — not sell orders.
          </p>
        </div>
      </header>

      <p className="panel-copy">{book?.honesty || 'Not a buy/sell ticket.'}</p>

      <div className="metric-grid">
        <MetricCard label="ACTIVE" value={String(summary?.total ?? 0)} detail="Symbols in your buy book" />
        <MetricCard label="AT RISK" value={String(summary?.critical ?? 0)} detail="Critical technical damage" tone="amber" />
        <MetricCard label="WEAKENING" value={String(summary?.warn ?? 0)} detail="Below key averages / soft support" tone="amber" />
        <MetricCard label="HEALTHY" value={String(summary?.good ?? 0)} detail="Holding structure" tone="green" />
      </div>

      <Panel title="ADD A BUY" subtitle="Symbol required · entry and stop optional but recommended">
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
            style={{ minWidth: 180, flex: 1 }}
          />
          <button type="button" disabled={busy || !symbol.trim()} onClick={() => void onAdd()}>
            {busy ? 'Saving…' : 'Add to Active Buys'}
          </button>
          <button type="button" disabled={loading} onClick={() => void refresh()}>
            Refresh health
          </button>
        </div>
        {note ? <p className="panel-copy">{note}</p> : null}
      </Panel>

      {loading && <div className="large-empty">Checking technical health…</div>}
      {error && <EmptyState title="Active Buys unavailable" detail={error} />}

      {!loading && items.length === 0 && (
        <EmptyState
          title="No active buys yet"
          detail="Add a symbol you are buying. The system will watch 20/50/200-day averages, swing support, and volume dumps."
        />
      )}

      <div className="stock-context-grid">
        {items.map((item) => {
          const health = item.health
          const avgs = health?.averages || {}
          const supports = health?.supports || {}
          const warnings = (health?.warnings || []).map((w) => w.text)
          return (
            <Panel
              key={item.id}
              title={`${item.symbol} · ${words(item.status_label || item.severity || 'Unknown')}`}
              subtitle={
                health?.available
                  ? `Price ${money(item.price, 1)} · source ${health.price_source || '—'} · as of ${health.as_of || '—'}`
                  : 'History incomplete'
              }
              action={
                <div className="inline-actions">
                  <StatusBadge status={String(item.status_label || item.severity || 'UNKNOWN')} />
                  <button type="button" onClick={() => openStock(item.symbol)}>Open</button>
                  <button type="button" disabled={busy} onClick={() => void onRemove(item)}>Remove</button>
                </div>
              }
            >
              <div className="fact-grid">
                <div><span>Entry</span><strong>{item.entry_price != null ? money(item.entry_price, 1) : '—'}</strong></div>
                <div><span>Stop</span><strong>{item.stop_price != null ? money(item.stop_price, 1) : '—'}</strong></div>
                <div><span>vs entry</span><strong>{pct(item.vs_entry_pct)}</strong></div>
                <div><span>EMA 20</span><strong>{money(avgs.ema20, 1)}</strong></div>
                <div><span>EMA 50</span><strong>{money(avgs.ema50, 1)}</strong></div>
                <div><span>EMA 200</span><strong>{money(avgs.ema200, 1)}</strong></div>
                <div><span>Support 20d</span><strong>{money(supports.swing_20d, 1)}</strong></div>
                <div><span>Support 60d</span><strong>{money(supports.swing_60d, 1)}</strong></div>
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
