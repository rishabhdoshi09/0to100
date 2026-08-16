import { useEffect, useRef, useState } from 'react'
import { ChartWorkspace } from './components'
import { money, pct } from './format'
import { isPhoneLayout, thesisSheetClassName } from './phoneLayout'
import {
  fetchBuyThesis,
  fetchStockFundamentals,
  type BuyThesis,
} from './productApi'
import type { ChartBar, ConvictionRecord, LongTermRecord, ScanRecord } from './types'

function metricLine(label: string, value: unknown, unit = '') {
  if (value == null || value === '') return `${label}: not in cache`
  if (typeof value === 'number') {
    const n = Math.abs(value) >= 1000
      ? value.toLocaleString('en-IN', { maximumFractionDigits: 2 })
      : value.toFixed(2).replace(/\.00$/, '')
    return `${label}: ${n}${unit}`
  }
  return `${label}: ${value}${unit}`
}

export function BuyThesisSheet({
  symbol,
  bars,
  row,
  onClose,
  onOpenResearch,
  onCompare,
  onWatchlist,
}: {
  symbol: string
  bars: ChartBar[]
  row?: Record<string, unknown> | null
  onClose?: () => void
  onOpenResearch: () => void
  onCompare: () => void
  onWatchlist: () => void
}) {
  const [thesis, setThesis] = useState<BuyThesis | null>(null)
  const [loading, setLoading] = useState(false)
  const [fetching, setFetching] = useState(false)
  const [error, setError] = useState('')
  const sheetRef = useRef<HTMLElement | null>(null)

  useEffect(() => {
    let alive = true
    setLoading(true)
    setError('')
    fetchBuyThesis(symbol, false)
      .then((payload) => {
        if (!alive) return
        setThesis(payload)
        const thin = !payload.fundamentals.available || Number(payload.fundamentals.coverage_pct || 0) < 40
        if (!thin) return
        setFetching(true)
        return fetchStockFundamentals(symbol, false)
          .then(() => fetchBuyThesis(symbol, false))
          .then((next) => { if (alive) setThesis(next) })
          .catch((reason) => {
            if (alive) setError(reason instanceof Error ? reason.message : 'Could not fetch filings')
          })
          .finally(() => { if (alive) setFetching(false) })
      })
      .catch((reason) => {
        if (alive) setError(reason instanceof Error ? reason.message : 'Could not load thesis')
      })
      .finally(() => { if (alive) setLoading(false) })
    return () => { alive = false }
  }, [symbol])

  useEffect(() => {
    sheetRef.current?.scrollTo(0, 0)
    if (!onClose) return
    const applyLock = () => {
      document.documentElement.classList.toggle('thesis-open', isPhoneLayout())
    }
    applyLock()
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    window.addEventListener('resize', applyLock)
    return () => {
      document.documentElement.classList.remove('thesis-open')
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('resize', applyLock)
    }
  }, [onClose, symbol])

  const plan = thesis?.plan
  const book = thesis?.order_book
  const sales = thesis?.sales
  return (
    <section
      ref={sheetRef}
      className={thesisSheetClassName(Boolean(onClose))}
      role={onClose ? 'dialog' : undefined}
      aria-modal={onClose ? true : undefined}
      aria-label={`Buy thesis ${symbol}`}
    >
      {onClose ? (
        <div className="thesis-toolbar">
          <button type="button" className="thesis-close" onClick={onClose}>
            ← Close
          </button>
          <strong>{symbol}</strong>
        </div>
      ) : null}
      <header className="reco-sheet-hero">
        <p>{thesis?.sector || 'Selected name'}</p>
        <h2>{thesis?.company || symbol}</h2>
        <p>{thesis?.headline || (loading ? 'Loading why this name is on the desk…' : 'Clicked name — evidence below.')}</p>
      </header>
      {error ? <p className="home-path-error">{error}</p> : null}
      {fetching ? <p className="home-path-hint">Fetching company filings from Screener / Yahoo — current cache was thin.</p> : null}

      <div className="reco-sheet-kpis">
        <div>
          <span>Buy</span>
          <strong>{money(plan?.buy, 2)}</strong>
        </div>
        <div>
          <span>Stop</span>
          <strong>{money(plan?.stop, 2)}</strong>
        </div>
        <div>
          <span>Target</span>
          <strong>{money(plan?.target, 2)}</strong>
        </div>
        <div className="reco-gain">
          <strong>{plan?.upside_from_buy_pct != null ? `↗ ${pct(plan.upside_from_buy_pct)}` : '—'}</strong>
          <small>% upside from buy</small>
        </div>
      </div>

      <div className="thesis-grid">
        <article>
          <h3>Why it is on the desk</h3>
          <ul>
            {(thesis?.why || []).map((item) => <li key={item}>{item}</li>)}
          </ul>
        </article>
        <article>
          <h3>Sector wave</h3>
          <p className={`thesis-wave thesis-wave-${String(thesis?.sector_wave?.wave || 'NO_CLAIM').toLowerCase()}`}>
            {thesis?.sector_wave?.headline || (loading ? 'Identifying sector…' : 'Sector not identified yet.')}
          </p>
          <ul>
            {(thesis?.sector_wave?.bullets || []).map((item) => <li key={item}>{item}</li>)}
          </ul>
          {thesis?.sector_wave?.note ? <small>{thesis.sector_wave.note}</small> : null}
        </article>
        <article>
          <h3>FII / DII / named buyers</h3>
          <p>{thesis?.smart_money?.headline || 'Checking shareholding and NSE prints…'}</p>
          <ul>
            {(thesis?.smart_money?.bullets || []).map((item) => <li key={item}>{item}</li>)}
          </ul>
          {thesis?.smart_money?.note ? <small>{thesis.smart_money.note}</small> : null}
        </article>
        <article>
          <h3>Earnings, margins, valuations</h3>
          {thesis?.earnings?.available ? (
            <ul>
              {(thesis.earnings.bullets || []).map((item) => <li key={item}>{item}</li>)}
              {(thesis.fundamentals.metrics || [])
                .filter((m) => ['roe', 'roce', 'debt_to_equity', 'promoter_holding'].includes(String(m.key)))
                .map((m) => (
                  <li key={String(m.key)}>{metricLine(String(m.label), m.value, m.unit === '%' ? '%' : m.unit ? ` ${m.unit}` : '')}</li>
                ))}
            </ul>
          ) : thesis?.fundamentals.available ? (
            <ul>
              {(thesis.fundamentals.metrics || []).slice(0, 10).map((m) => (
                <li key={String(m.key)}>{metricLine(String(m.label), m.value, m.unit === '%' ? '%' : m.unit ? ` ${m.unit}` : '')}</li>
              ))}
            </ul>
          ) : (
            <p>Filings not in cache yet. {fetching ? 'Fetching now…' : 'A fetch was attempted from Screener, then Yahoo.'}</p>
          )}
          {sales?.cagr_3y != null ? <p>3-year sales CAGR {Number(sales.cagr_3y).toFixed(1)}% (annual table)</p> : null}
          {sales?.series && sales.series.length > 0 ? (
            <ul>
              {sales.series.map((row) => (
                <li key={row.period}>{row.period}: ₹{Number(row.sales_cr).toLocaleString('en-IN')} cr</li>
              ))}
            </ul>
          ) : null}
          {thesis?.fundamentals.about ? <p className="thesis-about">{thesis.fundamentals.about}</p> : null}
        </article>
        <article>
          <h3>Order book</h3>
          <p>{book?.note || 'No live depth.'}</p>
          {book?.source ? <small>Source: {book.source}</small> : null}
          {book?.available && (book.bids?.length || book.asks?.length) ? (
            <div className="thesis-book">
              <div>
                <strong>Bids</strong>
                {(book.bids || []).slice(0, 5).map((lv, i) => (
                  <span key={`b${i}`}>{money(lv.price, 2)} × {Number(lv.quantity || 0).toLocaleString('en-IN')}</span>
                ))}
              </div>
              <div>
                <strong>Asks</strong>
                {(book.asks || []).slice(0, 5).map((lv, i) => (
                  <span key={`a${i}`}>{money(lv.price, 2)} × {Number(lv.quantity || 0).toLocaleString('en-IN')}</span>
                ))}
              </div>
            </div>
          ) : null}
        </article>
      </div>

      <div className="reco-sheet-actions">
        <button type="button" className="reco-primary" onClick={onOpenResearch}>Full research</button>
        <button type="button" className="reco-ghost" onClick={onCompare}>Compare</button>
        <button type="button" className="reco-ghost" onClick={onWatchlist}>Watchlist</button>
      </div>
      <details className="thesis-chart-fold" open={!isPhoneLayout()}>
        <summary>Chart</summary>
        <div className="reco-chart-card">
          <ChartWorkspace
            symbol={symbol}
            bars={bars}
            row={(row || undefined) as ScanRecord | ConvictionRecord | LongTermRecord | undefined}
          />
        </div>
      </details>
    </section>
  )
}
