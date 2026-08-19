import './sepaMonitor.css'
import type { RecommendationCard, RecommendationsWorkspace } from './productApi'
import { deskSymbol } from './deskThesis'

function money(value: number | null | undefined): string {
  if (value == null || Number.isNaN(Number(value))) return '—'
  return `₹${Number(value).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
}

function changeLabel(value: number | null | undefined): string {
  if (value == null || Number.isNaN(Number(value))) return '—'
  const n = Number(value)
  return `${n >= 0 ? '+' : ''}${n.toFixed(2)}%`
}

function rsClass(label?: string): string {
  const v = (label || '').toUpperCase()
  if (v === 'LEADER') return 'is-leader'
  if (v === 'LAGGARD') return 'is-laggard'
  if (v === 'IN LINE') return 'is-inline'
  return ''
}

export function TopStocksBoard({
  cards,
  selected,
  onSelect,
  session,
  tape,
  indices,
  indexNote,
}: {
  cards: RecommendationCard[]
  selected?: string
  onSelect: (symbol: string) => void
  session?: RecommendationsWorkspace['session']
  tape?: RecommendationsWorkspace['tape']
  indices?: RecommendationsWorkspace['indices']
  indexNote?: string
}) {
  const openNow = session?.open === true
  const strip = (indices || []).filter((row) => row.available)
  return (
    <section className="top-stocks" aria-label="Top stocks">
      <header className="top-stocks-head">
        <div>
          <p className="sepa-kicker">Market monitor</p>
          <h3>Top Stocks</h3>
          <em>SEPA · stage · RS vs Nifty 50 on official NSE history</em>
        </div>
        <div className="sepa-session">
          <span className={openNow ? 'is-open' : 'is-closed'}>{session?.label || 'SESSION'}</span>
          {session?.clock ? <b>{session.clock}</b> : null}
        </div>
      </header>
      {strip.length > 0 ? (
        <div className="index-strip" aria-label="Official index strip">
          {strip.map((row) => (
            <article key={row.id}>
              <span>{row.label}</span>
              <strong>{row.close != null ? row.close.toLocaleString('en-IN', { maximumFractionDigits: 2 }) : '—'}</strong>
              <em className={(row.change_pct ?? 0) >= 0 ? 'pos' : 'neg'}>{changeLabel(row.change_pct)}</em>
            </article>
          ))}
        </div>
      ) : null}
      <div className="top-stocks-live">
        <strong>TOP STOCKS</strong>
        <em className={cards.some((c) => (c.price_tag || '') === 'LIVE') ? 'is-live' : ''}>
          {cards.some((c) => (c.price_tag || '') === 'LIVE') ? 'LIVE' : 'EOD'}
        </em>
      </div>
      <ol className="top-stocks-list">
        {cards.map((card, idx) => {
          const symbol = deskSymbol(card.symbol)
          const change = card.change_pct
          const fund = card.fundamentals
          const rs = card.rs_label
          return (
            <li key={`${symbol}-${idx}`}>
              <button
                type="button"
                className={deskSymbol(selected) === symbol ? 'is-active' : ''}
                onClick={() => { if (symbol) onSelect(symbol) }}
              >
                <span className="top-rank">{idx + 1}</span>
                <span className="top-sym">
                  <b>{symbol}</b>
                  <small>
                    {card.sepa_score != null ? `${card.sepa_score}/100 · ${card.sepa_passed}/${card.sepa_total}` : null}
                    {card.stage_label ? ` · ${card.stage_label}` : ''}
                  </small>
                  {rs ? (
                    <small className={`top-rs ${rsClass(rs)}`}>
                      {rs}{card.rs_excess_pp != null ? ` ${card.rs_excess_pp >= 0 ? '+' : ''}${card.rs_excess_pp}pp` : ''}
                    </small>
                  ) : null}
                  {fund?.available && (fund.metrics || []).length > 0 ? (
                    <small className="top-fund">
                      {(fund.metrics || []).slice(0, 3).map((m) => (
                        <i key={m.key}>{m.label} {m.value}{m.unit === '%' ? '%' : m.unit === 'x' ? 'x' : ''}</i>
                      ))}
                    </small>
                  ) : (
                    <small className="top-fund is-missing">Fundamentals not on file</small>
                  )}
                </span>
                <span className="top-px">{money(card.cmp)}</span>
                <span className="top-chg">{changeLabel(change)}</span>
              </button>
            </li>
          )
        })}
      </ol>
      {indexNote ? <p className="top-stocks-note">{indexNote}</p> : null}
      {tape?.technical ? <p className="top-stocks-note">{tape.technical}</p> : null}
      {tape?.fundamental ? <p className="top-stocks-note">{tape.fundamental}</p> : null}
    </section>
  )
}
