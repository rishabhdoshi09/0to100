import { money } from './format'
import type { StockAnalyser, StockAnalyserCriterion, StockAnalyserQuote } from './productApi'
import { recentSymbols } from './sessionMemory'

function rupee(value: number | null | undefined): string {
  return money(value, 2)
}

function criterionMark(passed: boolean | null): string {
  if (passed === true) return '✓'
  if (passed === false) return '✕'
  return '·'
}

function QuoteTile({
  label,
  value,
  tone,
}: {
  label: string
  value: string
  tone?: 'up' | 'down' | ''
}) {
  return (
    <div className={`analyser-quote-tile ${tone || ''}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function CriterionRow({ item }: { item: StockAnalyserCriterion }) {
  const state = item.passed === true ? 'pass' : item.passed === false ? 'fail' : 'unknown'
  return (
    <article className={`analyser-criterion is-${state}`}>
      <div className="analyser-criterion-mark" aria-hidden="true">{criterionMark(item.passed)}</div>
      <div>
        <header>
          <h3>{item.title}</h3>
          <small>{item.passed === true ? 'Passed' : item.passed === false ? 'Failed' : 'Unknown'}</small>
        </header>
        <p className="analyser-criterion-detail">{item.detail}</p>
        <p className="analyser-criterion-note">{item.note}</p>
      </div>
    </article>
  )
}

export function StockAnalyserPanel({
  analyser,
  symbol,
  onOpenSymbol,
}: {
  analyser?: StockAnalyser | null
  symbol: string
  onOpenSymbol?: (next: string) => void
}) {
  const recents = recentSymbols().filter((item) => item !== symbol).slice(0, 5)
  if (!analyser) {
    return <div className="large-empty">Analyser is still loading for {symbol}.</div>
  }
  const quote: StockAnalyserQuote = analyser.quote || {}
  const verdict = String(analyser.verdict || 'INCOMPLETE').toLowerCase()
  const template = analyser.trend_template
  return (
    <div className="stock-analyser">
      {recents.length > 0 && onOpenSymbol ? (
        <div className="analyser-recents" aria-label="Recently opened stocks">
          {recents.map((item) => (
            <button type="button" key={item} onClick={() => onOpenSymbol(item)}>{item}</button>
          ))}
        </div>
      ) : null}

      <section className={`analyser-hero is-${verdict}`}>
        <div className="analyser-score">
          <span>SEPA SCORE</span>
          <strong>{analyser.score}</strong>
          <b>/{analyser.max_score || 100}</b>
        </div>
        <div className="analyser-hero-copy">
          <span>STOCK ANALYSER · MINERVINI SEPA · {symbol}</span>
          <h2>{analyser.headline}</h2>
          <p>
            {analyser.passed}/{analyser.total} criteria passed
            {analyser.benchmark ? ` · vs ${analyser.benchmark}` : ''}
          </p>
          <p className="analyser-advice">{analyser.advice}</p>
        </div>
      </section>

      <div className="analyser-quote-row">
        <QuoteTile label="OPEN" value={rupee(quote.open)} />
        <QuoteTile label="HIGH" value={rupee(quote.high)} tone="up" />
        <QuoteTile label="LOW" value={rupee(quote.low)} tone="down" />
        <QuoteTile label="PREV CLOSE" value={rupee(quote.prev_close)} />
        <QuoteTile label="52W HIGH" value={rupee(quote.high_52w)} tone="up" />
      </div>

      <div className="analyser-criteria">
        {(analyser.criteria || []).map((item) => (
          <CriterionRow key={item.id} item={item} />
        ))}
      </div>

      {template?.criteria?.length ? (
        <section className="analyser-template">
          <header>
            <span>MINERVINI DMA TREND TEMPLATE</span>
            <strong>{template.score ?? 0}/100 · {template.passed ?? 0}/{template.total ?? 7}</strong>
          </header>
          <p>{template.headline || 'Stage-2 moving-average stack on official history.'}</p>
          <ul>
            {template.criteria.map((item) => (
              <li key={item.id} className={`is-${item.passed === true ? 'pass' : item.passed === false ? 'fail' : 'unknown'}`}>
                <b>{criterionMark(item.passed)}</b>
                <span>{item.title}</span>
                <small>{item.detail}</small>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      {analyser.disclaimer ? <p className="analyser-disclaimer">{analyser.disclaimer}</p> : null}
    </div>
  )
}
