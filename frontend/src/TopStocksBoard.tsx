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
  breadth,
  breadthNote,
  news,
}: {
  cards: RecommendationCard[]
  selected?: string
  onSelect: (symbol: string) => void
  session?: RecommendationsWorkspace['session']
  tape?: RecommendationsWorkspace['tape']
  indices?: RecommendationsWorkspace['indices']
  indexNote?: string
  breadth?: RecommendationsWorkspace['breadth']
  breadthNote?: string
  news?: RecommendationsWorkspace['news_tape']
}) {
  const openNow = session?.open === true
  const strip = (indices || []).filter((row) => row.available)
  const nifty = strip.find((row) => row.id === '^NSEI')
  const newsItems = (news?.items || []).slice(0, 8)
  const history = (breadth?.history || []).slice(0, 6)
  return (
    <section className="top-stocks" aria-label="Top stocks">
      <header className="top-stocks-head">
        <div>
          <p className="sepa-kicker">Market monitor</p>
          <h3>Top Stocks</h3>
          <em>News · breadth · SEPA on official NSE history</em>
        </div>
        <div className="sepa-session">
          <span className={openNow ? 'is-open' : 'is-closed'}>{session?.label || 'SESSION'}</span>
          {session?.clock ? <b>{session.clock}</b> : null}
        </div>
      </header>
      {newsItems.length > 0 ? (
        <div className="news-tape" aria-label="On-file news">
          {newsItems.map((item, idx) => (
            <article key={`${item.tag}-${idx}`}>
              {item.tag ? <span className={`news-tag is-${(item.tag || '').toLowerCase()}`}>{item.tag}</span> : null}
              <p>{item.headline}</p>
              {item.source ? <em>{item.source}</em> : null}
            </article>
          ))}
        </div>
      ) : null}
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
      {breadth?.available ? (
        <div className="breadth-strip" aria-label="Market breadth">
          <article>
            <span>NIFTY</span>
            <strong>{nifty?.close != null ? nifty.close.toLocaleString('en-IN', { maximumFractionDigits: 1 }) : '—'}</strong>
            <em className={(nifty?.change_pct ?? 0) >= 0 ? 'pos' : 'neg'}>{changeLabel(nifty?.change_pct)}</em>
          </article>
          <article>
            <span>A/D RATIO</span>
            <strong>{breadth.adv_ratio != null ? breadth.adv_ratio.toFixed(2) : '—'}</strong>
            <em>{breadth.advancers}/{breadth.decliners}</em>
          </article>
          <article>
            <span>% ABV 20 DMA</span>
            <strong>{breadth.pct_above_20 != null ? `${breadth.pct_above_20}%` : '—'}</strong>
            <em>{breadth.verdict || ''}</em>
          </article>
          <article>
            <span>% ABV 40 DMA</span>
            <strong>{breadth.pct_above_40 != null ? `${breadth.pct_above_40}%` : '—'}</strong>
            <em>{breadth.up_4pct != null ? `${breadth.up_4pct} up 4%` : ''}</em>
          </article>
        </div>
      ) : null}
      {history.length > 1 ? (
        <table className="breadth-table">
          <thead>
            <tr>
              <th>Day</th>
              <th>Adv</th>
              <th>Dec</th>
              <th>Up 4%</th>
              <th>Dn 4%</th>
              <th>%20</th>
              <th>%40</th>
              <th>Nifty</th>
            </tr>
          </thead>
          <tbody>
            {history.map((row, idx) => (
              <tr key={`${row.date || idx}`}>
                <td>{row.date ? row.date.slice(5) : idx === 0 ? 'Now' : `-${idx}`}</td>
                <td>{row.advancers}</td>
                <td>{row.decliners}</td>
                <td>{row.up_4pct}</td>
                <td>{row.down_4pct}</td>
                <td>{row.pct_above_20 ?? '—'}</td>
                <td>{row.pct_above_40 ?? '—'}</td>
                <td>{row.nifty_close != null ? row.nifty_close.toLocaleString('en-IN') : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
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
      {breadthNote ? <p className="top-stocks-note">{breadthNote}</p> : null}
      {tape?.technical ? <p className="top-stocks-note">{tape.technical}</p> : null}
      {tape?.fundamental ? <p className="top-stocks-note">{tape.fundamental}</p> : null}
    </section>
  )
}
