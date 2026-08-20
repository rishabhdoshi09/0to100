import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { money, pct } from './format'
import { deskSymbol } from './deskThesis'
import { loadCachedJson, saveCachedJson } from './deskSession'
import {
  fetchStockPeek,
  type IntelligenceMetric,
  type RecommendationCard,
  type StockPeekPayload,
  type StockWorkspace,
  type SymbolRatioRow,
} from './productApi'
import { SepaScoreChip } from './SepaMonitor'
import {
  filledPeekMetrics,
  formatPeekValue,
  mergePeekMetrics,
  orderPeekMetrics,
  PEEK_FETCH_MS,
  PEEK_FUND_KEYS,
  PEEK_TECHNICAL_KEYS,
  peekNumber,
  snapshotFromCard,
  type PeekMetric,
} from './stockPeek'
import './stockPeek.css'

function asMetrics(items?: Array<{ key?: string; label?: string; value?: unknown; unit?: string }> | IntelligenceMetric[] | null): PeekMetric[] {
  return (items || [])
    .filter((item) => item && item.key)
    .map((item) => ({
      key: String(item.key),
      label: String(item.label || item.key),
      value: item.value,
      unit: item.unit,
    }))
}

function technicalFromWorkspace(ws: StockWorkspace | StockPeekPayload | null): PeekMetric[] {
  const t = ws?.technical
  if (!t) return []
  if (t.metrics?.length) return asMetrics(t.metrics)
  return [
    { key: 'close', label: 'Close', value: t.close, unit: '' },
    { key: 'change_pct', label: 'Change', value: t.change_pct, unit: '%' },
    { key: 'rsi14', label: 'RSI 14', value: t.rsi14 },
    { key: 'ema20', label: 'EMA 20', value: t.ema20 },
    { key: 'ema50', label: 'EMA 50', value: t.ema50 },
    { key: 'ema200', label: 'EMA 200', value: t.ema200 },
    { key: 'atr_pct', label: 'ATR %', value: t.atr_pct, unit: '%' },
    { key: 'volume_ratio', label: 'Vol vs 20d', value: t.volume_ratio, unit: 'x' },
    { key: 'high_52w', label: '52w high', value: t.high_52w },
    { key: 'from_high_pct', label: 'From 52w high', value: t.from_high_pct, unit: '%' },
  ]
}

function technicalsFromCard(rec: RecommendationCard): PeekMetric[] {
  const snap = snapshotFromCard(rec as Record<string, unknown>)
  return filledPeekMetrics([
    { key: 'close', label: 'Close', value: snap.cmp },
    { key: 'change_pct', label: 'Change', value: snap.change, unit: '%' },
    { key: 'rsi14', label: 'RSI', value: snap.rsi },
    { key: 'volume_ratio', label: 'Vol vs 20d', value: snap.volumeRatio, unit: 'x' },
  ])
}

function MetricGrid({
  title,
  items,
  empty,
}: {
  title: string
  items: PeekMetric[]
  empty: string
}) {
  const show = items.slice(0, 12)
  return (
    <section className="stock-peek-section">
      <h3>{title}</h3>
      {items.length === 0 ? (
        <p className="stock-peek-empty">{empty}</p>
      ) : (
        <div className="stock-peek-grid">
          {show.map((item) => (
            <article key={item.key}>
              <span>{item.label}</span>
              <strong>{formatPeekValue(item.value, item.unit)}</strong>
            </article>
          ))}
        </div>
      )}
    </section>
  )
}

export function StockPeekPopup({
  symbol,
  card,
  onClose,
  onOpenResearch,
  onCompare,
  onWatchlist,
}: {
  symbol: string
  card?: RecommendationCard | Record<string, unknown> | null
  onClose: () => void
  onOpenResearch: () => void
  onCompare: () => void
  onWatchlist: () => void
}) {
  const clean = deskSymbol(symbol)
  const rec = (card || {}) as RecommendationCard
  const [peek, setPeek] = useState<StockPeekPayload | null>(
    () => loadCachedJson<StockPeekPayload>(`peek:${clean}`),
  )
  const [workspace, setWorkspace] = useState<StockWorkspace | null>(
    () => loadCachedJson<StockWorkspace>(`stock:${clean}`),
  )
  const [ratios, setRatios] = useState<SymbolRatioRow[]>(
    () => loadCachedJson<SymbolRatioRow[]>(`ratios:${clean}`)
      || loadCachedJson<StockPeekPayload>(`peek:${clean}`)?.ratios
      || [],
  )
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(!loadCachedJson(`peek:${clean}`) && !workspace)

  useEffect(() => {
    document.documentElement.classList.add('stock-peek-open')
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => {
      document.documentElement.classList.remove('stock-peek-open')
      window.removeEventListener('keydown', onKey)
    }
  }, [onClose])

  useEffect(() => {
    let alive = true
    const cachedPeek = loadCachedJson<StockPeekPayload>(`peek:${clean}`)
    const cachedWs = loadCachedJson<StockWorkspace>(`stock:${clean}`)
    if (cachedPeek) {
      setPeek(cachedPeek)
      if (cachedPeek.ratios?.length) setRatios(cachedPeek.ratios)
      setLoading(false)
    } else if (cachedWs) {
      setWorkspace(cachedWs)
      setLoading(false)
    } else {
      setLoading(true)
    }
    const cachedRatios = loadCachedJson<SymbolRatioRow[]>(`ratios:${clean}`)
    if (cachedRatios) setRatios(cachedRatios)
    const ac = new AbortController()
    const to = window.setTimeout(() => ac.abort(), PEEK_FETCH_MS)
    fetchStockPeek(clean, ac.signal)
      .then((payload) => {
        if (!alive) return
        setPeek(payload)
        saveCachedJson(`peek:${clean}`, payload)
        if (payload.ratios?.length) {
          setRatios(payload.ratios)
          saveCachedJson(`ratios:${clean}`, payload.ratios)
        }
        setError('')
      })
      .catch((reason: Error) => {
        if (!alive) return
        if (reason?.name === 'AbortError' || /failed to fetch/i.test(reason.message || '')) {
          return
        }
        if (!cachedPeek && !cachedWs) {
          setError(reason.message || 'Snapshot unread')
        }
      })
      .finally(() => {
        window.clearTimeout(to)
        if (alive) setLoading(false)
      })
    return () => {
      alive = false
      ac.abort()
      window.clearTimeout(to)
    }
  }, [clean])

  const fromCard = snapshotFromCard(rec as Record<string, unknown>)
  const cmp = peekNumber(peek?.cmp) ?? fromCard.cmp ?? workspace?.technical.close ?? null
  const change = peekNumber(peek?.change_pct) ?? fromCard.change ?? workspace?.technical.change_pct ?? null
  const buy = peekNumber(peek?.entry) ?? fromCard.buy
  const stop = peekNumber(peek?.stop) ?? fromCard.stop
  const target = peekNumber(peek?.target) ?? fromCard.target
  const upside = peekNumber(peek?.upside_from_buy_pct) ?? fromCard.upside
  const fundFromCard = asMetrics(rec.fundamentals?.metrics)
  const fundFromPeek = asMetrics(peek?.fundamentals?.metrics)
  const fundFromWs = asMetrics(workspace?.fundamentals?.metrics)
  const fundamentals = filledPeekMetrics(orderPeekMetrics(
    mergePeekMetrics(fundFromPeek, mergePeekMetrics(fundFromWs, fundFromCard)),
    PEEK_FUND_KEYS,
  ))
  const technicals = filledPeekMetrics(orderPeekMetrics(
    mergePeekMetrics(
      technicalFromWorkspace(peek),
      mergePeekMetrics(technicalFromWorkspace(workspace), technicalsFromCard(rec)),
    ),
    PEEK_TECHNICAL_KEYS,
  ))
  const sepa = peek?.sepa || workspace?.sepa
  const filledRatios = (peek?.ratios?.length ? peek.ratios : ratios).filter((row) => peekNumber(row.value) != null)
  const showError = error && technicals.length === 0 && fundamentals.length === 0
  const dialog = (
    <div className="stock-peek-backdrop" onClick={onClose} role="presentation">
      <div
        className="stock-peek"
        role="dialog"
        aria-modal="true"
        aria-label={`${clean} snapshot`}
        onClick={(event) => event.stopPropagation()}
      >
        <header className="stock-peek-head">
          <div>
            <p>{peek?.sector || workspace?.sector || rec.sector || 'Sector not classified'}</p>
            <h2>{peek?.company || workspace?.company || rec.company || clean}</h2>
            <em>{clean} · {peek?.price_tag || rec.price_tag || (workspace?.technical.latest_date ? 'EOD' : 'Snapshot')}</em>
          </div>
          <div className="stock-peek-last">
            <strong>{money(cmp, 2)}</strong>
            <span className={(change ?? 0) >= 0 ? 'pos' : 'neg'}>{pct(change)}</span>
          </div>
          <button type="button" className="stock-peek-close" onClick={onClose} aria-label="Close snapshot">✕</button>
        </header>

        <div className="stock-peek-kpis">
          <article><span>Buy</span><strong>{money(buy, 2)}</strong></article>
          <article><span>Stop</span><strong>{money(stop, 2)}</strong></article>
          <article><span>Target</span><strong>{money(target, 2)}</strong></article>
          <article>
            <span>Upside</span>
            <strong className={(upside ?? 0) < 0 ? 'neg' : ''}>
              {upside != null ? pct(upside) : '—'}
            </strong>
          </article>
        </div>

        <div className="stock-peek-chips">
          <SepaScoreChip
            score={rec.sepa_score ?? sepa?.score}
            max={rec.sepa_max ?? sepa?.max_score}
            passed={rec.sepa_passed ?? sepa?.passed}
            total={rec.sepa_total ?? sepa?.total}
            verdict={rec.sepa_verdict ?? sepa?.verdict}
            headline={rec.sepa_headline ?? sepa?.headline}
          />
          {rec.stage_label || sepa?.stage?.label ? <span className="stock-peek-pill">{rec.stage_label || sepa?.stage?.label}</span> : null}
          {rec.rs_label || sepa?.rs?.label ? (
            <span className="stock-peek-pill">
              {rec.rs_label || sepa?.rs?.label}
              {rec.rs_excess_pp != null ? ` ${rec.rs_excess_pp >= 0 ? '+' : ''}${rec.rs_excess_pp} pp` : ''}
            </span>
          ) : null}
        </div>

        {showError ? <p className="stock-peek-empty">{error}</p> : null}
        {error && !showError ? <p className="stock-peek-note">{error}</p> : null}
        {peek?.history_note ? <p className="stock-peek-note">{peek.history_note}</p> : null}
        {loading ? <p className="stock-peek-note">Loading on-file numbers…</p> : null}

        <MetricGrid
          title="Technicals"
          items={technicals}
          empty="Official history is not on file for this name yet. Nothing is simulated."
        />
        <MetricGrid
          title="Fundamentals"
          items={fundamentals}
          empty="No calculated pack on file. This popup does not scrape to invent P/E or ROE."
        />

        <section className="stock-peek-section">
          <h3>Ratios</h3>
          {filledRatios.length === 0 ? (
            <p className="stock-peek-empty">No ratio table on file for {clean}.</p>
          ) : (
            <table className="stock-peek-ratios">
              <thead>
                <tr><th>Ratio</th><th>Value</th><th>Period</th></tr>
              </thead>
              <tbody>
                {filledRatios.slice(0, 16).map((row) => (
                  <tr key={row.key}>
                    <td>{row.label}</td>
                    <td>{formatPeekValue(row.value)}</td>
                    <td>{row.period || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>

        {sepa?.criteria?.length ? (
          <section className="stock-peek-section">
            <h3>SEPA 7 rules</h3>
            <ul className="stock-peek-sepa">
              {sepa.criteria.map((item) => (
                <li key={item.id} className={item.passed === true ? 'is-pass' : item.passed === false ? 'is-fail' : ''}>
                  <b>{item.passed === true ? '✓' : item.passed === false ? '✕' : '–'}</b>
                  <span>{item.title}</span>
                  <em>{item.detail}</em>
                </li>
              ))}
            </ul>
          </section>
        ) : null}

        <footer className="stock-peek-actions">
          <button type="button" className="reco-primary" onClick={onOpenResearch}>Full research</button>
          <button type="button" className="reco-ghost" onClick={onCompare}>Compare</button>
          <button type="button" className="reco-ghost" onClick={onWatchlist}>Watchlist</button>
        </footer>
      </div>
    </div>
  )

  if (typeof document === 'undefined') return null
  return createPortal(dialog, document.body)
}
