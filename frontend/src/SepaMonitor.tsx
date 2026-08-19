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

export type SepaStage = {
  id?: string
  label?: string
  note?: string
}

export type SepaRs = {
  available?: boolean
  lookback?: number
  stock_pct?: number | null
  benchmark_pct?: number | null
  excess_pp?: number | null
  label?: string
  note?: string
  benchmark?: string
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
  stage?: SepaStage | null
  rs?: SepaRs | null
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

function contextTone(label: string | undefined): string {
  const v = (label || '').toUpperCase()
  if (v.includes('STAGE 2') && !v.includes('?')) return 'leader'
  if (v === 'LEADER') return 'leader'
  if (v.includes('STAGE 4') || v === 'LAGGARD') return 'laggard'
  if (v.includes('STAGE 3') || v.includes('STAGE 1') || v.includes('?')) return 'mixed'
  return 'inline'
}

function ContextChip({ label, detail, note }: { label?: string; detail?: string; note?: string }) {
  if (!label) return null
  return (
    <div className={`sepa-context-chip is-${contextTone(label)}`} title={note || ''}>
      <strong>{label}</strong>
      {detail ? <span>{detail}</span> : null}
    </div>
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
  fundamentals,
}: {
  sepa: SepaPayload | null | undefined
  symbol: string
  company?: string
  changePct?: number | null
  fundamentals?: {
    available?: boolean
    coverage_pct?: number | null
    classification?: string
    fetched_at?: string
    metrics?: Array<{ key: string; label: string; value: number | string | null; unit?: string }>
  } | null
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
          <em>Minervini SEPA · stage + RS vs Nifty 50 · official NSE history</em>
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

      <div className="sepa-context" aria-label="Stage and relative strength">
        <ContextChip label={sepa.stage?.label} note={sepa.stage?.note} />
        <ContextChip
          label={sepa.rs?.available ? sepa.rs.label : undefined}
          detail={sepa.rs?.excess_pp != null ? `${sepa.rs.excess_pp >= 0 ? '+' : ''}${sepa.rs.excess_pp} pp vs Nifty` : undefined}
          note={sepa.rs?.note}
        />
      </div>

      <aside className="sepa-advice">{sepa.advice}</aside>

      <div className="sepa-stats" aria-label="Session statistics">
        <Stat label="Open" value={money(quote.open)} />
        <Stat label="High" value={money(quote.high)} tone="pos" />
        <Stat label="Low" value={money(quote.low)} tone="neg" />
        <Stat label="Prev close" value={money(quote.prev_close)} />
      </div>

      <div className="sepa-fund">
        <div className="sepa-fund-head">
          <strong>On-file fundamentals</strong>
          <span>{fundamentals?.classification || (fundamentals?.available ? 'On file' : 'Not on file')}</span>
        </div>
        {fundamentals?.metrics && fundamentals.metrics.filter((m) => m.value != null && ['pe', 'roe', 'roce', 'debt_to_equity', 'sales_growth_3y', 'profit_growth_3y'].includes(m.key)).length > 0 ? (
          <div className="sepa-fund-grid">
            {fundamentals.metrics.filter((m) => m.value != null && ['pe', 'roe', 'roce', 'debt_to_equity', 'sales_growth_3y', 'profit_growth_3y'].includes(m.key)).slice(0, 6).map((m) => (
              <div className="sepa-stat" key={m.key}>
                <span>{m.label}</span>
                <strong>{typeof m.value === 'number' ? `${Number(m.value).toFixed(m.unit === 'x' ? 2 : 1)}${m.unit ? ` ${m.unit}` : ''}` : `${m.value}${m.unit ? ` ${m.unit}` : ''}`}</strong>
              </div>
            ))}
          </div>
        ) : (
          <p className="sepa-fund-empty">No calculated pack on file. This monitor does not scrape to invent P/E or ROE.</p>
        )}
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
