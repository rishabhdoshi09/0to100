import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { money, pct } from './format'
import { deskSymbol } from './deskThesis'
import { loadCachedJson, saveCachedJson } from './deskSession'
import {
  fetchStockIntelligence,
  fetchSymbolRatios,
  type IntelligenceMetric,
  type RecommendationCard,
  type StockWorkspace,
  type SymbolRatioRow,
} from './productApi'
import { SepaScoreChip } from './SepaMonitor'
import {
  formatPeekValue,
  orderPeekMetrics,
  PEEK_FUND_KEYS,
  PEEK_TECHNICAL_KEYS,
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

function technicalFromWorkspace(ws: StockWorkspace | null): PeekMetric[] {
  if (!ws?.technical) return []
  if (ws.technical.metrics?.length) return asMetrics(ws.technical.metrics)
  const t = ws.technical
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
  const [workspace, setWorkspace] = useState<StockWorkspace | null>(
    () => loadCachedJson<StockWorkspace>(`stock:${clean}`),
  )
  const [ratios, setRatios] = useState<SymbolRatioRow[]>(
    () => loadCachedJson<SymbolRatioRow[]>(`ratios:${clean}`) || [],
  )
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(!workspace)

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
    const cached = loadCachedJson<StockWorkspace>(`stock:${clean}`)
    if (cached) {
      setWorkspace(cached)
      setLoading(false)
    } else {
      setWorkspace(null)
      setLoading(true)
    }
    const cachedRatios = loadCachedJson<SymbolRatioRow[]>(`ratios:${clean}`)
    if (cachedRatios) setRatios(cachedRatios)
    Promise.all([
      fetchStockIntelligence(clean),
      fetchSymbolRatios(clean).catch(() => ({ symbol: clean, ratios: [] as SymbolRatioRow[] })),
    ])
      .then(([ws, ratioPayload]) => {
        if (!alive) return
        setWorkspace(ws)
        saveCachedJson(`stock:${clean}`, ws)
        const nextRatios = ratioPayload.ratios || []
        setRatios(nextRatios)
        saveCachedJson(`ratios:${clean}`, nextRatios)
        setError('')
      })
      .catch((reason: Error) => {
        if (alive && !cached) setError(reason.message || 'Workspace unread')
      })
      .finally(() => { if (alive) setLoading(false) })
    return () => { alive = false }
  }, [clean])

  const rec = (card || {}) as RecommendationCard
  const cmp = rec.cmp ?? workspace?.technical.close ?? null
  const change = rec.change_pct ?? workspace?.technical.change_pct ?? null
  const buy = rec.entry ?? rec.cmp ?? null
  const fundFromCard = asMetrics(rec.fundamentals?.metrics)
  const fundFromWs = asMetrics(workspace?.fundamentals?.metrics)
  const fundamentals = orderPeekMetrics(
    fundFromWs.length ? fundFromWs : fundFromCard,
    PEEK_FUND_KEYS,
  )
  const technicals = orderPeekMetrics(
    technicalFromWorkspace(workspace).length
      ? technicalFromWorkspace(workspace)
      : [
          { key: 'rsi14', label: 'RSI', value: rec.rsi },
          { key: 'volume_ratio', label: 'Vol vs 20d', value: rec.volume_ratio, unit: 'x' },
          { key: 'change_pct', label: 'Change', value: rec.change_pct, unit: '%' },
        ],
    PEEK_TECHNICAL_KEYS,
  )
  const sepa = workspace?.sepa
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
            <p>{workspace?.sector || rec.sector || 'Sector not classified'}</p>
            <h2>{workspace?.company || rec.company || clean}</h2>
            <em>{clean} · {rec.price_tag || (workspace?.technical.latest_date ? 'EOD' : 'Snapshot')}</em>
          </div>
          <div className="stock-peek-last">
            <strong>{money(cmp, 2)}</strong>
            <span className={(change ?? 0) >= 0 ? 'pos' : 'neg'}>{pct(change)}</span>
          </div>
          <button type="button" className="stock-peek-close" onClick={onClose} aria-label="Close snapshot">✕</button>
        </header>

        <div className="stock-peek-kpis">
          <article><span>Buy</span><strong>{money(buy, 2)}</strong></article>
          <article><span>Stop</span><strong>{money(rec.stop, 2)}</strong></article>
          <article><span>Target</span><strong>{money(rec.target, 2)}</strong></article>
          <article>
            <span>Upside</span>
            <strong className={(rec.upside_from_buy_pct ?? 0) < 0 ? 'neg' : ''}>
              {rec.upside_from_buy_pct != null ? pct(rec.upside_from_buy_pct) : '—'}
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

        {error ? <p className="stock-peek-empty">{error}</p> : null}
        {loading ? <p className="stock-peek-note">Opening on-file numbers…</p> : null}

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
          {ratios.length === 0 ? (
            <p className="stock-peek-empty">No ratio table on file for {clean}.</p>
          ) : (
            <table className="stock-peek-ratios">
              <thead>
                <tr><th>Ratio</th><th>Value</th><th>Period</th></tr>
              </thead>
              <tbody>
                {ratios.slice(0, 16).map((row) => (
                  <tr key={row.key}>
                    <td>{row.label}</td>
                    <td>{row.value != null ? formatPeekValue(row.value) : (row.missing_reason || 'Not on file')}</td>
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
