import { useEffect, useState } from 'react'
import { fetchEvolutionLab } from './productApi'
import { EmptyState } from './designSystem'
import {
  formatR,
  incrementalTone,
  sampleFloorNote,
} from './evolutionLabModel'
import type { EvolutionLabBoard, EvolutionPolicyRow } from './evolutionLabModel'
import './evolutionLab.css'

/**
 * Champion vs Challenger: which decision process actually deserves PAPER
 * authority, asked scientifically rather than assumed. Only the Champion
 * row ever reached real PAPER execution; every Challenger row is a frozen
 * shadow decision graded against the same realized market path.
 */

function ScorecardRow({ row, isChampion }: { row: EvolutionPolicyRow; isChampion: boolean }) {
  const paired = row.paired_vs_champion
  const tone = paired ? incrementalTone(paired.incremental_expectancy_R) : 'neutral'
  return (
    <tr className={`evolution-lab__row evolution-lab__row--${tone}`}>
      <td>
        <div className="evolution-lab__policy-id">
          {row.policy_id}
          {isChampion ? <span className="evolution-lab__badge">CHAMPION</span> : null}
        </div>
        <div className="evolution-lab__hypothesis">{row.hypothesis}</div>
        <div className="evolution-lab__paired-note">
          Controls PAPER decisions: {row.controls_paper_decisions ? 'YES' : 'NO'}
        </div>
      </td>
      <td>
        <div>{row.status}</div>
        {row.promotion_evaluation?.status ? (
          <div className="evolution-lab__paired-note">
            {row.promotion_evaluation.status}
          </div>
        ) : null}
      </td>
      <td>{row.scorecard.decision_snapshots}</td>
      <td>{row.scorecard.selected_trades}</td>
      <td>{formatR(row.scorecard.expectancy_R)}</td>
      <td>{formatR(row.scorecard.max_drawdown_R)}</td>
      <td>
        <div>{formatR(row.historical_prior?.selected_expectancy_R)}</div>
        <div className="evolution-lab__paired-note">
          {row.historical_prior?.observations ?? 0} historical · prior only
        </div>
      </td>
      <td>{row.scorecard.missed_winners}</td>
      <td>{row.scorecard.avoided_losers}</td>
      <td>
        {paired ? (
          <>
            <div className={`evolution-lab__incremental evolution-lab__incremental--${tone}`}>
              {formatR(paired.incremental_expectancy_R)}
            </div>
            <div className="evolution-lab__paired-note">{sampleFloorNote(paired.paired_snapshots)}</div>
          </>
        ) : (
          <span className="evolution-lab__none">—</span>
        )}
      </td>
    </tr>
  )
}

const DOMAINS = ['EQUITY', 'FNO_UNDERLYING', 'FNO_CONTRACT'] as const

export function EvolutionLabView() {
  const [domain, setDomain] = useState<string>('EQUITY')
  const [board, setBoard] = useState<EvolutionLabBoard | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false
    setBoard(null)
    setError('')
    fetchEvolutionLab(domain)
      .then((payload) => { if (!cancelled) setBoard(payload) })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err))
      })
    return () => { cancelled = true }
  }, [domain])

  if (error && !board) {
    return <EmptyState title="Evolution Lab" detail={`Could not load the tournament board: ${error}`} />
  }
  if (!board) {
    return <EmptyState title="Evolution Lab" detail="Loading…" />
  }

  return (
    <section className="evolution-lab">
      <header className="evolution-lab__header">
        <div className="evolution-lab__domain-tabs">
          {DOMAINS.map((item) => (
            <button
              key={item}
              type="button"
              onClick={() => setDomain(item)}
              aria-pressed={domain === item}
            >
              {item === 'EQUITY' ? 'Equity' : item === 'FNO_UNDERLYING' ? 'F&O Underlying' : 'F&O Contract'}
            </button>
          ))}
        </div>
        <p className="evolution-lab__safety">
          PAPER only — live_locked={String(board.live_locked)}, live_execution_authorized={String(board.live_execution_authorized)}.
          {' '}Automatic promotion is {board.auto_promotion_enabled ? 'enabled' : 'disabled'}; only an explicit, deliberate
          promotion can change the Champion.
        </p>
      </header>

      {!board.champion ? (
        <EmptyState
          title="No Champion yet"
          detail="The Evolution Engine bootstraps its first Champion automatically on the next PAPER cycle."
        />
      ) : (
        <div className="evolution-lab__table-wrap">
          <table className="evolution-lab__table">
            <thead>
              <tr>
                <th>Policy</th><th>Status</th><th>Snapshots</th><th>Selected</th>
                <th>Expectancy</th><th>Max DD</th><th>Historical prior</th><th>Missed winners</th><th>Avoided losers</th>
                <th>Incremental vs Champion</th>
              </tr>
            </thead>
            <tbody>
              <ScorecardRow row={board.champion} isChampion />
              {board.challenger_leaderboard.map((row) => (
                <ScorecardRow key={row.policy_id} row={row} isChampion={false} />
              ))}
            </tbody>
          </table>
          {board.challenger_leaderboard.length === 0 ? (
            <p className="evolution-lab__none">No active Challengers right now.</p>
          ) : null}
        </div>
      )}

      <div className="evolution-lab__events">
        <h3>Recent promotions / status changes</h3>
        {board.recent_events.length === 0 ? (
          <p className="evolution-lab__none">No lifecycle events yet.</p>
        ) : (
          <ul>
            {board.recent_events.map((event, idx) => (
              <li key={`${event.policy_id}-${event.at}-${idx}`}>
                <span className="evolution-lab__event-policy">{event.policy_id}</span>
                {' → '}
                <span className="evolution-lab__event-status">{event.status}</span>
                {event.reason ? <span className="evolution-lab__event-reason"> — {event.reason}</span> : null}
              </li>
            ))}
          </ul>
        )}
      </div>

      <footer className="evolution-lab__footer">
        <p>{board.retired_count} retired/rejected policies kept for history, excluded from the leaderboard above.</p>
      </footer>
    </section>
  )
}
