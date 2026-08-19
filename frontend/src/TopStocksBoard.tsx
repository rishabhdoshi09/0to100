import './sepaMonitor.css'
import type { RecommendationCard, RecommendationsWorkspace } from './productApi'
import { deskSymbol } from './deskThesis'

function money(value: number | null | undefined): string {
  if (value == null || Number.isNaN(Number(value))) return '—'
  return `₹${Number(value).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
}

function changeLabel(value: number | null | undefined): string {
  if (value == null || Number.isNaN(Number(value))) return '—'
  return `${Number(value).toFixed(2)}%`
}

export function TopStocksBoard({
  cards,
  selected,
  onSelect,
  session,
  tape,
}: {
  cards: RecommendationCard[]
  selected?: string
  onSelect: (symbol: string) => void
  session?: RecommendationsWorkspace['session']
  tape?: RecommendationsWorkspace['tape']
}) {
  const openNow = session?.open === true
  return (
    <section className="top-stocks" aria-label="Top stocks">
      <header className="top-stocks-head">
        <div>
          <p className="sepa-kicker">Market monitor</p>
          <h3>Top Stocks</h3>
          <em>SEPA technicals on official NSE history · on-file valuations</em>
        </div>
        <div className="sepa-session">
          <span className={openNow ? 'is-open' : 'is-closed'}>{session?.label || 'SESSION'}</span>
          {session?.clock ? <b>{session.clock}</b> : null}
        </div>
      </header>
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
                  {card.sepa_score != null ? (
                    <small>{card.sepa_score}/100 · {card.sepa_passed}/{card.sepa_total}</small>
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
      {tape?.fundamental ? <p className="top-stocks-note">{tape.fundamental}</p> : null}
    </section>
  )
}
