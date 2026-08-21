import { useEffect, useState } from 'react'
import { money } from './format'
import { EvChip } from './evChip'
import { deskSymbol } from './deskThesis'
import { StockPeekPopup } from './StockPeekPopup'
import { SepaScoreChip } from './SepaMonitor'
import type { ExperienceViewProps } from './experience'
import {
  armPaperAutopilot,
  disarmAutopilot,
  fetchBacktestLab,
  fetchLiveJourney,
  fetchReadyQueue,
  type BacktestLabPayload,
  type LiveJourneyPayload,
  type ReadyQueuePayload,
  type ReadyTradeCard,
} from './productApi'
import { journeyTone, labKidTone, labStatusTone, readyLaneLabel } from './tradeDesk'
import './tradeDesk.css'

function ReadyCard({
  card,
  onOpen,
}: {
  card: ReadyTradeCard
  onOpen: (symbol: string) => void
}) {
  const symbol = deskSymbol(card.symbol)
  return (
    <article className={`trade-card lane-${card.lane}`} role="button" tabIndex={0} onClick={() => onOpen(symbol)} onKeyDown={(event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault()
        onOpen(symbol)
      }
    }}>
      <div className="trade-card-top">
        <span className={`trade-lane ${card.lane}`}>{readyLaneLabel(card.lane)}</span>
        <span className="trade-sector">{card.sector || '—'}</span>
      </div>
      <h3>{card.company || card.symbol}</h3>
      <p className="trade-sym">{card.symbol}</p>
      <div className="trade-kpis">
        <div><span>Buy</span><strong>{money(card.entry, 2)}</strong></div>
        <div><span>Stop</span><strong>{money(card.stop, 2)}</strong></div>
        <div><span>Target</span><strong>{money(card.target, 2)}</strong></div>
        <div><span>R:R</span><strong>{card.reward_risk != null ? `${card.reward_risk.toFixed(1)}` : '—'}</strong></div>
      </div>
      <div className="trade-meta">
        {card.atq != null ? <em>ATQ {(card.atq * 100).toFixed(0)}</em> : null}
        {card.edge_r != null ? <em>Edge {card.edge_r >= 0 ? '+' : ''}{card.edge_r.toFixed(2)}R</em> : <em>No measured edge_r</em>}
        <EvChip row={card} />
      </div>
      {card.sepa_score != null ? (
        <SepaScoreChip
          score={card.sepa_score}
          passed={card.sepa_passed}
          total={card.sepa_total}
          verdict={card.sepa_verdict || undefined}
          headline={card.sepa_headline || card.stage_label || undefined}
        />
      ) : null}
      {card.why?.length ? <p className="trade-why">{card.why.join(' · ')}</p> : null}
      <p className="trade-honesty">{card.honesty}</p>
    </article>
  )
}

export function ReadyTradesView({ setSelected, setActive }: ExperienceViewProps) {
  const [payload, setPayload] = useState<ReadyQueuePayload | null>(null)
  const [error, setError] = useState('')
  const [peek, setPeek] = useState('')

  useEffect(() => {
    let alive = true
    fetchReadyQueue()
      .then((next) => {
        if (!alive) return
        setPayload(next)
        setError('')
      })
      .catch((reason: Error) => {
        if (alive) setError(reason.message || 'Ready queue unread')
      })
    return () => { alive = false }
  }, [])

  const prime = payload?.prime || []
  const stage2 = payload?.stage2 || []
  const tickets = payload?.actionable || []
  const allCards = stage2.concat(tickets, prime)

  return (
    <section className="reco-light trade-desk">
      <nav className="reco-crumb" aria-label="Breadcrumb">
        <strong>Trade</strong>
        <span>/</span>
        <span>Ready</span>
        <span>/</span>
        <button type="button" onClick={() => setActive('Backtest Lab')}>Lab</button>
        <span>/</span>
        <button type="button" onClick={() => setActive('Live Journey')}>Journey</button>
      </nav>
      <header className="reco-hero">
        <div className="reco-hero-icon">R</div>
        <div>
          <h2>Ready to trade</h2>
          <p>
            Stage-2 SEPA from the last Ideas ranking, plus scanner BUY/Ready tickets with a buy and a stop.
            Ranked by ATQ — structure, not a win-rate claim. Prime is an overlay when those gates actually pass.
            Paper on Journey. Live stays locked.
          </p>
        </div>
      </header>
      {error ? <p className="stock-peek-note">{error}</p> : null}
      {payload?.method ? <p className="trade-lede">{payload.method}</p> : null}
      {payload?.empty ? (
        <div className="trade-empty">
          <strong>No complete ticket today</strong>
          {(payload.empty_why || []).map((line) => <p key={line}>{line}</p>)}
          <p>{payload.next}</p>
          <div className="trade-actions">
            <button type="button" className="reco-primary" onClick={() => setActive('Backtest Lab')}>Open Lab</button>
            <button type="button" className="reco-ghost" onClick={() => setActive('Home')}>Fill the desk</button>
          </div>
        </div>
      ) : null}
      {stage2.length ? (
        <section className="trade-section">
          <h3>Stage 2 — SEPA template</h3>
          <p className="trade-lede">Minervini 7-rule score on official NSE history. Cached from Ideas. Not a guarantee.</p>
          <div className="trade-grid">
            {stage2.map((card) => <ReadyCard key={`s2-${card.symbol}`} card={card} onOpen={setPeek} />)}
          </div>
        </section>
      ) : null}
      {tickets.length ? (
        <section className="trade-section">
          <h3>Complete tickets — Pattern, PreBreakout, Pullback count</h3>
          <p className="trade-lede">Buy + stop exist and the combo is not a proven loser. ATQ ranks them. Missing EV stays missing.</p>
          <div className="trade-grid">
            {tickets.map((card) => <ReadyCard key={`tk-${card.symbol}`} card={card} onOpen={setPeek} />)}
          </div>
        </section>
      ) : null}
      {prime.length ? (
        <section className="trade-section">
          <h3>Prime — every evidence gate passed</h3>
          <p className="trade-lede">Same gates as Telegram. Rare when conservative EV is still unmeasured.</p>
          <div className="trade-grid">
            {prime.map((card) => <ReadyCard key={`pm-${card.symbol}`} card={card} onOpen={setPeek} />)}
          </div>
        </section>
      ) : null}
      {payload?.disclaimer ? <p className="trade-disclaimer">{payload.disclaimer}</p> : null}
      {peek ? (
        <StockPeekPopup
          symbol={peek}
          card={allCards.find((c) => deskSymbol(c.symbol) === peek) as never}
          onClose={() => setPeek('')}
          onOpenResearch={() => {
            setSelected(peek)
            setPeek('')
            setActive('Stock Intelligence')
          }}
          onCompare={() => { setSelected(peek); setPeek(''); setActive('Compare') }}
          onWatchlist={() => setPeek('')}
        />
      ) : null}
    </section>
  )
}

export function BacktestLabView({ runControl, setActive }: ExperienceViewProps) {
  const [lab, setLab] = useState<BacktestLabPayload | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const load = () => {
    fetchBacktestLab()
      .then((next) => { setLab(next); setError('') })
      .catch((reason: Error) => setError(reason.message || 'Lab unread'))
  }

  const running = Boolean(lab?.running || busy)

  useEffect(() => {
    load()
    const t = window.setInterval(load, running ? 2_000 : 12_000)
    return () => window.clearInterval(t)
  }, [running])

  const run = async () => {
    setBusy(true)
    try {
      await runControl('RUN_FULL_UNIVERSE_BACKTEST_NOW')
      load()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not start backtest')
    } finally {
      setBusy(false)
    }
  }

  const lesson = lab?.lesson
  const board = lab?.scoreboard
  const done = Number(lab?.progress || 0)
  const total = Number(lab?.total || 0)
  const pct = total > 0 ? Math.min(100, Math.round((100 * done) / total)) : (running ? 8 : 0)
  const cta = running
    ? `${lesson?.cta_running || 'Practicing…'}${total ? ` ${done}/${total}` : ''}`
    : (lesson?.cta || 'Run the practice test')

  const lanes: Array<{ key: 'keep' | 'skip' | 'quiet'; title: string; rows: NonNullable<BacktestLabPayload['scoreboard']>['keep'] }> = [
    { key: 'keep', title: 'Passed — keep using', rows: board?.keep || [] },
    { key: 'skip', title: 'Failed — skip', rows: board?.skip || [] },
    { key: 'quiet', title: 'Too few tries — stay quiet', rows: board?.quiet || [] },
  ]

  return (
    <section className="reco-light trade-desk">
      <nav className="reco-crumb" aria-label="Breadcrumb">
        <button type="button" onClick={() => setActive('Ready Trades')}>Trade</button>
        <span>/</span>
        <strong>Lab</strong>
        <span>/</span>
        <button type="button" onClick={() => setActive('Live Journey')}>Journey</button>
      </nav>
      <header className="reco-hero">
        <div className="reco-hero-icon">L</div>
        <div>
          <h2>{lesson?.title || 'What is a backtest?'}</h2>
          <p>{lesson?.plain || 'A practice test on official NSE history. It never places an order.'}</p>
        </div>
      </header>
      <p className="trade-lede">{lesson?.now}</p>
      <ol className="lab-steps">
        {(lesson?.steps || []).map((step) => (
          <li key={step.n}>
            <b>{step.n}</b>
            <div>
              <strong>{step.title}</strong>
              <p>{step.body}</p>
            </div>
          </li>
        ))}
      </ol>
      <div className="trade-actions">
        <button type="button" className="reco-primary" disabled={running} onClick={() => void run()}>
          {cta}
        </button>
        <span className="trade-note">{lab?.evidence_note || 'no measured backtest yet'}</span>
        {lab?.live_locked ? <span className="trade-note">Live locked. No orders.</span> : null}
      </div>
      {running ? (
        <div className="lab-progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct}>
          <span style={{ width: `${pct}%` }} />
        </div>
      ) : null}
      {error ? <p className="stock-peek-note">{error}</p> : null}
      <section className="trade-section">
        <h3>Scoreboard</h3>
        <p className="trade-lede">{board?.headline || 'No practice test on file yet.'}</p>
        {lesson?.r_plain ? <p className="trade-lede">{lesson.r_plain}</p> : null}
        <div className="lab-scoreboard">
          {lanes.map((lane) => (
            <article key={lane.key} className={`lab-lane ${labKidTone(lane.key)}`}>
              <h4>{lane.title}</h4>
              {lane.rows.length ? (
                <ul>
                  {lane.rows.map((row) => (
                    <li key={row.signal}>
                      <strong>{row.signal}</strong>
                      <span>
                        {row.closed || row.trades} tries
                        {row.expectancy_r != null ? ` · ${row.expectancy_r >= 0 ? '+' : ''}${row.expectancy_r}R` : ''}
                        {row.win_rate != null ? ` · ${row.win_rate}% wins` : ''}
                      </span>
                    </li>
                  ))}
                </ul>
              ) : <p className="trade-note">None yet.</p>}
            </article>
          ))}
        </div>
      </section>
      <div className="trade-usecases">
        {(lab?.use_cases || []).map((item) => (
          <article key={item.id} className={`trade-usecase ${labStatusTone(item.status)}`}>
            <header>
              <span>{item.status}</span>
              <h3>{item.kid_title || item.title}</h3>
            </header>
            <p>{item.result}</p>
            {item.best?.length ? (
              <ul>{item.best.map((row) => <li key={String(row.signal)}>{String(row.signal)} · {String(row.expectancy_r)}R</li>)}</ul>
            ) : null}
            {item.avoid?.length ? <p className="trade-avoid">Skip: {item.avoid.join(', ')}</p> : null}
            {item.goto ? (
              <button type="button" className="reco-ghost" onClick={() => setActive('Live Journey')}>{item.goto}</button>
            ) : null}
          </article>
        ))}
      </div>
      {lab?.signals?.length ? (
        <section className="trade-section">
          <h3>The full report card</h3>
          <table className="trade-table">
            <thead>
              <tr><th>Signal</th><th>Tries</th><th>Wins</th><th>Avg result</th><th>Report card</th></tr>
            </thead>
            <tbody>
              {lab.signals.map((row) => (
                <tr key={row.signal} className={labKidTone(row.kid_lane || '')}>
                  <td>{row.signal}</td>
                  <td>{row.closed || row.trades}</td>
                  <td>{row.win_rate != null ? `${row.win_rate}%` : '—'}</td>
                  <td>{row.expectancy_r != null ? `${row.expectancy_r >= 0 ? '+' : ''}${row.expectancy_r}R` : '—'}</td>
                  <td>{row.kid_label || row.verdict}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      ) : null}
      {lesson?.rules?.length ? (
        <ul className="lab-rules">
          {lesson.rules.map((rule) => <li key={rule}>{rule}</li>)}
        </ul>
      ) : null}
      {lab?.disclaimer ? <p className="trade-disclaimer">{lab.disclaimer}</p> : null}
    </section>
  )
}

export function LiveJourneyView({ runControl, setActive }: ExperienceViewProps) {
  const [journey, setJourney] = useState<LiveJourneyPayload | null>(null)
  const [alloc, setAlloc] = useState('25000')
  const [note, setNote] = useState('')
  const [busy, setBusy] = useState(false)

  const load = () => {
    fetchLiveJourney()
      .then((next) => {
        setJourney(next)
        if (next.autopilot.allocation) setAlloc(String(Math.round(Number(next.autopilot.allocation))))
      })
      .catch((reason: Error) => setNote(reason.message || 'Journey unread'))
  }

  useEffect(() => {
    load()
    const t = window.setInterval(load, 15_000)
    return () => window.clearInterval(t)
  }, [])

  const arm = async () => {
    setBusy(true)
    try {
      const n = Number(alloc)
      const out = await armPaperAutopilot(Number.isFinite(n) ? n : undefined)
      setNote(out.message)
      load()
    } catch (reason) {
      setNote(reason instanceof Error ? reason.message : 'Paper arm failed')
    } finally {
      setBusy(false)
    }
  }

  const disarm = async () => {
    setBusy(true)
    try {
      await disarmAutopilot()
      setNote('Paper autopilot disarmed')
      load()
    } finally {
      setBusy(false)
    }
  }

  const stats = journey?.report_card.stats || {}
  const armed = Boolean(journey?.autopilot.armed)

  return (
    <section className="reco-light trade-desk">
      <nav className="reco-crumb" aria-label="Breadcrumb">
        <button type="button" onClick={() => setActive('Ready Trades')}>Trade</button>
        <span>/</span>
        <button type="button" onClick={() => setActive('Backtest Lab')}>Lab</button>
        <span>/</span>
        <strong>Journey</strong>
      </nav>
      <header className="reco-hero">
        <div className="reco-hero-icon">J</div>
        <div>
          <h2>Paper → live journey</h2>
          <p>
            {journey?.rung.label || 'Paper is production. Live is earned.'}
            {' '}This page can arm paper. It cannot arm live.
          </p>
        </div>
      </header>
      <div className="trade-journey-hero">
        <article>
          <span>Rung</span>
          <strong>{journey?.rung.id || '—'}</strong>
        </article>
        <article>
          <span>Paper closed</span>
          <strong>{journey?.paper_closed ?? 0}/{journey?.paper_e4_n ?? 300}</strong>
        </article>
        <article>
          <span>Report card</span>
          <strong>{journey?.report_card.verdict || 'COLLECTING_EVIDENCE'}</strong>
        </article>
        <article>
          <span>Live from this desk</span>
          <strong>LOCKED</strong>
        </article>
      </div>
      <ol className="trade-steps">
        {(journey?.steps || []).map((step) => (
          <li key={step.id} className={journeyTone(step.status)}>
            <b>{step.status}</b>
            <div>
              <strong>{step.title}</strong>
              <p>{step.detail}</p>
              {step.next_action ? <em>{step.next_action}</em> : null}
            </div>
          </li>
        ))}
      </ol>
      <section className="trade-section">
        <h3>Arm paper autopilot</h3>
        <p className="trade-lede">
          {journey?.autopilot.headline || 'Default is disarmed. Allocation compounds from closed paper P&L.'}
        </p>
        <div className="trade-arm">
          <label>
            Paper allocation (₹)
            <input
              type="number"
              min={5000}
              step={1000}
              value={alloc}
              onChange={(event) => setAlloc(event.target.value)}
            />
          </label>
          <button type="button" className="reco-primary" disabled={busy} onClick={() => void arm()}>
            {armed ? 'Re-arm paper' : 'Arm paper'}
          </button>
          <button type="button" className="reco-ghost" disabled={busy || !armed} onClick={() => void disarm()}>
            Disarm
          </button>
          <button type="button" className="reco-ghost" onClick={() => void runControl('RUN_CYCLE_NOW')}>
            Request paper cycle
          </button>
        </div>
        {note ? <p className="trade-note">{note}</p> : null}
        <p className="trade-lede">
          Expectancy {stats.expectancy_r != null ? `${Number(stats.expectancy_r) >= 0 ? '+' : ''}${stats.expectancy_r}R` : '—'}
          {' · '}PF {stats.profit_factor ?? '—'}
          {' · '}n {stats.n ?? 0}
        </p>
        {journey?.scaling?.reason ? <p className="trade-note">{journey.scaling.action}: {journey.scaling.reason}</p> : null}
      </section>
      {journey?.disclaimer ? <p className="trade-disclaimer">{journey.disclaimer}</p> : null}
    </section>
  )
}
