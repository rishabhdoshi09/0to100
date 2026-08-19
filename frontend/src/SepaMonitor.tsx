import './sepaMonitor.css'

export type SepaCriterion = {
  id: string
  title: string
  rule?: string
  points: number
  awarded: number
  passed: boolean | null
  detail: string
  note: string
}

export type SepaQuote = {
  open?: number | null
  high?: number | null
  low?: number | null
  close?: number | null
  prev_close?: number | null
  change_pct?: number | null
  as_of?: string
}

export type SepaSession = {
  label?: string
  open?: boolean | null
  clock?: string
}

export type SepaPayload = {
  available: boolean
  score: number
  max_score: number
  passed: number
  total: number
  unknown?: number
  verdict: string
  headline: string
  advice: string
  disclaimer?: string
  method?: string
  criteria: SepaCriterion[]
  quote?: SepaQuote | null
  session?: SepaSession | null
}

function money(value: number | null | undefined, digits = 2): string {
  if (value == null || Number.isNaN(Number(value))) return '—'
  return `₹${Number(value).toLocaleString('en-IN', { minimumFractionDigits: digits, maximumFractionDigits: digits })}`
}

function verdictTone(verdict: string): string {
  const v = (verdict || '').toUpperCase()
  if (v === 'STRONG') return 'strong'
  if (v === 'CONSTRUCTIVE') return 'constructive'
  if (v === 'MIXED') return 'mixed'
  if (v === 'INCOMPLETE') return 'incomplete'
  return 'weak'
}

function ScoreRing({ score, max, tone }: { score: number; max: number; tone: string }) {
  const radius = 42
  const circ = 2 * Math.PI * radius
  const pct = max > 0 ? Math.max(0, Math.min(1, score / max)) : 0
  return (
    <svg viewBox="0 0 108 108" className={`sepa-ring is-${tone}`} aria-hidden="true">
      <circle className="sepa-ring-track" cx="54" cy="54" r={radius} />
      <circle
        className="sepa-ring-value"
        cx="54"
        cy="54"
        r={radius}
        strokeDasharray={`${circ * pct} ${circ}`}
        transform="rotate(-90 54 54)"
      />
      <text className="sepa-ring-score" x="54" y="52">{score}</text>
      <text className="sepa-ring-max" x="54" y="70">/{max}</text>
    </svg>
  )
}

function Stat({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div className={`sepa-stat${tone ? ` is-${tone}` : ''}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

export function SepaMonitor({
  sepa,
  symbol,
  company,
  changePct,
}: {
  sepa: SepaPayload | null | undefined
  symbol: string
  company?: string
  changePct?: number | null
}) {
  if (!sepa) return null
  const tone = verdictTone(sepa.verdict)
  const quote = sepa.quote || {}
  const session = sepa.session || {}
  const change = changePct ?? quote.change_pct
  const price = quote.close
  const openNow = session.open === true
  return (
    <section className={`sepa-monitor is-${tone}`} aria-label="SEPA setup monitor">
      <header className="sepa-monitor-top">
        <div>
          <p className="sepa-kicker">Setup monitor</p>
          <h3>Best stock setups</h3>
          <em>Minervini SEPA · 7 published Stage-2 rules · official NSE history</em>
        </div>
        <div className="sepa-session">
          <span className={openNow ? 'is-open' : 'is-closed'}>
            {session.label || 'SESSION'}
          </span>
          {session.clock ? <b>{session.clock}</b> : null}
        </div>
      </header>

      <div className="sepa-identity">
        <div>
          <h2>{symbol}</h2>
          {company && company !== symbol ? <p>{company}</p> : null}
        </div>
        <div className="sepa-last">
          <strong>{money(price)}</strong>
          {change != null ? (
            <span className={change >= 0 ? 'pos' : 'neg'}>
              {change >= 0 ? '+' : ''}{change.toFixed(2)}%
            </span>
          ) : null}
        </div>
      </div>

      <div className="sepa-summary">
        <ScoreRing score={sepa.score} max={sepa.max_score} tone={tone} />
        <div>
          <p className="sepa-headline">{sepa.headline}</p>
          <small>{sepa.passed}/{sepa.total} criteria passed</small>
        </div>
      </div>

      <aside className="sepa-advice">{sepa.advice}</aside>

      <div className="sepa-stats" aria-label="Session statistics">
        <Stat label="Open" value={money(quote.open)} />
        <Stat label="High" value={money(quote.high)} tone="pos" />
        <Stat label="Low" value={money(quote.low)} tone="neg" />
        <Stat label="Prev close" value={money(quote.prev_close)} />
      </div>

      <h4 className="sepa-break-title">SEPA criteria breakdown</h4>
      <div className="sepa-criteria">
        {(sepa.criteria || []).map((item) => {
          const state = item.passed === true ? 'pass' : item.passed === false ? 'fail' : 'unknown'
          return (
            <article key={item.id} className={`sepa-criterion is-${state}`}>
              <header>
                <span className="sepa-mark" aria-hidden="true">
                  {item.passed === true ? '✓' : item.passed === false ? '✕' : '–'}
                </span>
                <h5>{item.title}</h5>
                <b className={item.awarded > 0 ? 'pos' : ''}>
                  {item.awarded > 0 ? `+${item.awarded}` : '+0'}
                </b>
              </header>
              <p>{item.detail}</p>
              <em>{item.note}</em>
            </article>
          )
        })}
      </div>
      {sepa.disclaimer ? <p className="sepa-disclaimer">{sepa.disclaimer}</p> : null}
    </section>
  )
}

export function SepaScoreChip({
  score,
  max,
  passed,
  total,
  verdict,
  headline,
}: {
  score?: number | null
  max?: number | null
  passed?: number | null
  total?: number | null
  verdict?: string
  headline?: string
}) {
  if (score == null) return null
  const tone = verdictTone(verdict || '')
  return (
    <div className={`sepa-chip is-${tone}`}>
      <strong>{score}/{max ?? 100}</strong>
      <span>{headline || verdict || 'SEPA'}</span>
      {passed != null && total != null ? <em>{passed}/{total} rules</em> : null}
    </div>
  )
}
