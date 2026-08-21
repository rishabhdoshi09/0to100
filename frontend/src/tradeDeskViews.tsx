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
  feedPaperClassroom,
  fetchBacktestLab,
  fetchLiveJourney,
  fetchReadyQueue,
  type BacktestLabPayload,
  type LabClassroom,
  type LiveJourneyPayload,
  type ReadyQueuePayload,
  type ReadyTradeCard,
} from './productApi'
import { journeyTone, labKidTone, labLoopTone, labStatusTone, readyLaneLabel } from './tradeDesk'
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
      {payload?.lab_applied?.plain ? (
        <p className={`trade-learning ${payload.lab_applied.applied ? 'is-on' : ''}`}>
          {payload.lab_applied.applied ? 'Lab learning on this board' : 'Lab learning'}
          {' — '}
          {payload.lab_applied.plain}
          {(payload.lab_applied.skip || []).length ? ` Skip: ${payload.lab_applied.skip?.join(', ')}.` : ''}
        </p>
      ) : null}
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

function ClassroomBook({ classroom, onArm, onDisarm, onFeed, busy, note }: {
  classroom?: LabClassroom
  onArm: () => void
  onDisarm: () => void
  onFeed: () => void
  busy: boolean
  note?: string
}) {
  const armed = Boolean(classroom?.armed)
  return (
    <section className="trade-section lab-classroom">
      <h3>Paper classroom</h3>
      <p className="trade-lede">
        Fake money on Ready tickets. Every right and wrong is recorded. The next scan
        demotes proven losers — it never inflates a win rate. Live stays locked.
      </p>
      <div className="lab-class-kpis">
        <article>
          <span>Status</span>
          <strong>{armed ? (classroom?.in_window ? 'In session' : 'Armed · waiting') : 'Disarmed'}</strong>
        </article>
        <article>
          <span>Open paper</span>
          <strong>{classroom?.open_n ?? 0}</strong>
        </article>
        <article>
          <span>Closed paper</span>
          <strong>{classroom?.closed_n ?? 0}</strong>
        </article>
        <article>
          <span>Today</span>
          <strong>{classroom?.trades_today ?? 0} taken · {classroom?.considered_today ?? 0} seen</strong>
        </article>
      </div>
      <p className="lab-next">{classroom?.next_action || classroom?.headline}</p>
      <div className="trade-actions">
        <button type="button" className="reco-primary" disabled={busy} onClick={onArm}>
          {armed ? 'Re-arm + feed Ready tickets' : 'Start paper classroom'}
        </button>
        <button type="button" className="reco-ghost" disabled={busy || !armed} onClick={onFeed}>
          Feed Ready tickets now
        </button>
        <button type="button" className="reco-ghost" disabled={busy || !armed} onClick={onDisarm}>
          Disarm
        </button>
      </div>
      {note ? <p className="trade-note">{note}</p> : null}
      {(classroom?.blockers || []).length ? (
        <ul className="lab-blockers">
          {(classroom?.blockers || []).map((line) => <li key={line}>{line}</li>)}
        </ul>
      ) : null}
      {(classroom?.funnel || []).length ? (
        <p className="trade-lede">
          Why tickets were skipped today:{' '}
          {(classroom?.funnel || []).map((row) => `${row.reason} ×${row.n}`).join(' · ')}
        </p>
      ) : null}
      {(classroom?.open || []).length ? (
        <ul className="lab-open">
          {(classroom?.open || []).map((row) => (
            <li key={row.symbol}>
              <strong>{row.symbol}</strong>
              <span>
                {row.qty} @ {money(row.entry, 2)}
                {row.live != null ? ` · live ${money(row.live, 2)}` : ''}
                {row.pnl != null ? ` · ${row.pnl >= 0 ? '+' : ''}₹${Math.round(row.pnl)}` : ''}
              </span>
            </li>
          ))}
        </ul>
      ) : null}
      {(classroom?.activity || []).length ? (
        <ul className="lab-activity">
          {(classroom?.activity || []).slice(0, 6).map((line) => <li key={line}>{line}</li>)}
        </ul>
      ) : null}
    </section>
  )
}

export function BacktestLabView({ runControl, setActive }: ExperienceViewProps) {
  const [lab, setLab] = useState<BacktestLabPayload | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [note, setNote] = useState('')

  const load = () => {
    fetchBacktestLab()
      .then((next) => { setLab(next); setError('') })
      .catch((reason: Error) => setError(reason.message || 'Lab unread'))
  }

  const running = Boolean(lab?.running || busy)

  useEffect(() => {
    load()
    const t = window.setInterval(load, running ? 2_000 : 8_000)
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

  const arm = async () => {
    setBusy(true)
    try {
      const alloc = Number(lab?.classroom?.allocation || 0)
      const out = await armPaperAutopilot(alloc >= 5000 ? alloc : 25_000)
      setNote(out.message)
      load()
    } catch (reason) {
      setNote(reason instanceof Error ? reason.message : 'Paper arm failed')
    } finally {
      setBusy(false)
    }
  }

  const feed = async () => {
    setBusy(true)
    try {
      const out = await feedPaperClassroom()
      setNote(out.message)
      load()
    } catch (reason) {
      setNote(reason instanceof Error ? reason.message : 'Feed failed')
    } finally {
      setBusy(false)
    }
  }

  const disarm = async () => {
    setBusy(true)
    try {
      await disarmAutopilot()
      setNote('Paper classroom disarmed')
      load()
    } finally {
      setBusy(false)
    }
  }

  const lesson = lab?.lesson
  const board = lab?.scoreboard
  const done = Number(lab?.progress || 0)
  const total = Number(lab?.total || 0)
  const pct = total > 0
    ? Math.min(100, Math.round((100 * done) / total))
    : (running ? 8 : (lab?.actionable ? 100 : 0))
  const cta = running
    ? `${lesson?.cta_running || 'Practicing…'}${total ? ` ${done}/${total}` : ''}`
    : (lesson?.cta || 'Run the practice test')
  const pulse = lab?.pulse
  const keepN = board?.keep?.length || 0
  const skipN = board?.skip?.length || 0
  const quietN = board?.quiet?.length || 0

  const lanes: Array<{ key: 'keep' | 'skip' | 'quiet'; title: string; empty: string; rows: NonNullable<BacktestLabPayload['scoreboard']>['keep'] }> = [
    { key: 'keep', title: 'Passed — keep using', empty: 'No signal has earned this yet. That is honest.', rows: board?.keep || [] },
    { key: 'skip', title: 'Failed — skip next scan', empty: 'No proven loser on file. Nothing to demote.', rows: board?.skip || [] },
    { key: 'quiet', title: 'Too few tries — stay quiet', empty: 'Under 30 practice trades we do not brag.', rows: board?.quiet || [] },
  ]

  return (
    <section className="reco-light trade-desk lab-alive">
      <nav className="reco-crumb" aria-label="Breadcrumb">
        <button type="button" onClick={() => setActive('Ready Trades')}>Trade</button>
        <span>/</span>
        <strong>Lab</strong>
        <span>/</span>
        <button type="button" onClick={() => setActive('Live Journey')}>Journey</button>
      </nav>
      <header className="reco-hero">
        <div className={`reco-hero-icon lab-pulse-icon ${pulse?.tone || 'idle'}`}>L</div>
        <div>
          <h2>{lesson?.title || 'What is a backtest?'}</h2>
          <p>{lesson?.plain || 'A practice test on official NSE history. It never places an order.'}</p>
        </div>
        <div className={`lab-pulse-pill ${pulse?.tone || 'idle'}`}>
          <i />
          <strong>{pulse?.label || 'Classroom idle'}</strong>
          <span>{pulse?.hint || 'Run the practice test, then arm paper.'}</span>
        </div>
      </header>
      <p className="trade-lede">{lesson?.now}</p>
      <ol className="lab-loop" aria-label="Learning loop">
        {(lab?.loop || []).map((node) => (
          <li key={node.id} className={labLoopTone(node.state)}>
            <b>{node.n}</b>
            <strong>{node.title}</strong>
            <em>{node.state}</em>
            <p>{node.detail}</p>
          </li>
        ))}
      </ol>
      <div className="trade-actions">
        <button type="button" className="reco-primary" disabled={running} onClick={() => void run()}>
          {cta}
        </button>
        <button type="button" className="reco-ghost" disabled={busy} onClick={() => void arm()}>
          {lab?.classroom?.armed ? 'Re-arm paper classroom' : 'Start paper classroom'}
        </button>
        <span className="trade-note">{lab?.evidence_note || 'no measured backtest yet'}</span>
        {lab?.live_locked ? <span className="trade-note">Live locked.</span> : null}
      </div>
      <div className="lab-progress-wrap">
        <div className="lab-progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct}>
          <span style={{ width: `${Math.max(running ? 4 : 0, pct)}%` }} />
        </div>
        <p className="lab-progress-label">
          {running
            ? `Testing ${lab?.current || 'the next stock'} · ${done}/${total || '—'}`
            : (lab?.actionable ? 'Practice test on file' : 'No practice test on file yet')}
        </p>
      </div>
      {error ? <p className="stock-peek-note">{error}</p> : null}
      <section className="trade-section">
        <h3>Scoreboard</h3>
        <div className="lab-tally">
          <article className="is-pass"><span>Keep</span><strong>{keepN}</strong></article>
          <article className="is-lock"><span>Skip</span><strong>{skipN}</strong></article>
          <article className="is-wait"><span>Quiet</span><strong>{quietN}</strong></article>
        </div>
        <p className="trade-lede">{board?.headline || 'No practice test on file yet.'}</p>
        {lesson?.r_plain ? <p className="trade-lede">{lesson.r_plain}</p> : null}
        {lab?.learning?.plain ? <p className={`trade-learning ${lab.learning.applied ? 'is-on' : ''}`}>{lab.learning.plain}</p> : null}
        <div className="lab-scoreboard">
          {lanes.map((lane) => (
            <article key={lane.key} className={`lab-lane ${labKidTone(lane.key)}`}>
              <h4>{lane.title}</h4>
              <em>{lane.rows.length}</em>
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
              ) : <p className="trade-note">{lane.empty}</p>}
            </article>
          ))}
        </div>
      </section>
      <ClassroomBook
        classroom={lab?.classroom}
        onArm={() => void arm()}
        onDisarm={() => void disarm()}
        onFeed={() => void feed()}
        busy={busy}
        note={note}
      />
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

export function LiveJourneyView({ setActive }: ExperienceViewProps) {
  const [journey, setJourney] = useState<LiveJourneyPayload | null>(null)
  const [alloc, setAlloc] = useState('25000')
  const [note, setNote] = useState('')
  const [busy, setBusy] = useState(false)
  const armedPoll = Boolean(journey?.autopilot.armed)

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
    const t = window.setInterval(load, armedPoll ? 5_000 : 12_000)
    return () => window.clearInterval(t)
  }, [armedPoll])

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

  const feed = async () => {
    setBusy(true)
    try {
      const out = await feedPaperClassroom()
      setNote(out.message)
      load()
    } catch (reason) {
      setNote(reason instanceof Error ? reason.message : 'Feed failed')
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
  const classroom = journey?.classroom

  return (
    <section className="reco-light trade-desk lab-alive">
      <nav className="reco-crumb" aria-label="Breadcrumb">
        <button type="button" onClick={() => setActive('Ready Trades')}>Trade</button>
        <span>/</span>
        <button type="button" onClick={() => setActive('Backtest Lab')}>Lab</button>
        <span>/</span>
        <strong>Journey</strong>
      </nav>
      <header className="reco-hero">
        <div className={`reco-hero-icon lab-pulse-icon ${armed ? 'live' : 'idle'}`}>J</div>
        <div>
          <h2>Paper → live journey</h2>
          <p>
            {journey?.rung.label || 'Paper is production. Live is earned.'}
            {' '}This page can arm paper. It cannot arm live.
          </p>
        </div>
        <div className={`lab-pulse-pill ${armed ? 'live' : 'idle'}`}>
          <i />
          <strong>{armed ? (classroom?.in_window ? 'Paper in session' : 'Armed · waiting') : 'Disarmed'}</strong>
          <span>{journey?.next_action || classroom?.next_action || 'Arm paper to auto-take Ready tickets.'}</span>
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
          <span>Open now</span>
          <strong>{classroom?.open_n ?? journey?.autopilot.open_trades ?? 0}</strong>
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
      <ClassroomBook
        classroom={classroom}
        onArm={() => void arm()}
        onDisarm={() => void disarm()}
        onFeed={() => void feed()}
        busy={busy}
        note={note}
      />
      <section className="trade-section">
        <h3>Arm paper autopilot</h3>
        <p className="trade-lede">
          {journey?.autopilot.headline || 'Default is disarmed. Allocation compounds from closed paper P&L.'}
          {' '}Arming immediately feeds the last Ready/BUY tickets. Paper window is 09:30–15:20 IST.
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
            {armed ? 'Re-arm + feed' : 'Arm paper'}
          </button>
          <button type="button" className="reco-ghost" disabled={busy || !armed} onClick={() => void feed()}>
            Feed Ready tickets
          </button>
          <button type="button" className="reco-ghost" disabled={busy || !armed} onClick={() => void disarm()}>
            Disarm
          </button>
        </div>
        <p className="trade-lede">
          Expectancy {stats.expectancy_r != null ? `${Number(stats.expectancy_r) >= 0 ? '+' : ''}${stats.expectancy_r}R` : '—'}
          {' · '}PF {stats.profit_factor ?? '—'}
          {' · '}n {stats.n ?? 0}
          {' · '}seen today {journey?.autopilot.considered_today ?? 0}
        </p>
        {journey?.scaling?.reason ? <p className="trade-note">{journey.scaling.action}: {journey.scaling.reason}</p> : null}
      </section>
      {journey?.disclaimer ? <p className="trade-disclaimer">{journey.disclaimer}</p> : null}
    </section>
  )
}
