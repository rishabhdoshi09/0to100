import { useEffect, useState } from 'react'
import { fetchForwardEvidence } from './productApi'
import { EmptyState } from './designSystem'
import {
  attributionWarning,
  distributionBars,
  formatPct,
  formatR,
  progressToFloor,
  stateTone,
} from './forwardEvidenceModel'
import type { ForwardEvidenceBoard, GroupRow } from './forwardEvidenceModel'
import './forwardEvidence.css'

/**
 * Has the desk earned any market evidence yet?
 *
 * The screen is built to be as readable when the answer is "none" as when it
 * is "here is the edge". The empty state is stated in words and never wears
 * the healthy colour, because a row of zeros reads like a measurement.
 */

function GroupTable({ rows, field, label, minSample }: {
  rows: GroupRow[]
  field: 'setup' | 'regime' | 'sector'
  label: string
  minSample: number
}) {
  if (rows.length === 0) {
    return (
      <div className="forward-evidence__group">
        <h3>{label}</h3>
        <p className="forward-evidence__none">Nothing settled yet.</p>
      </div>
    )
  }
  return (
    <div className="forward-evidence__group">
      <h3>{label}</h3>
      <table>
        <thead>
          <tr>
            <th>{label}</th><th>Trades</th><th>W/L</th><th>Expectancy</th><th />
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={String(row[field])}
                className={row.usable_for_ranking ? 'is-usable' : 'is-thin'}>
              <td>{String(row[field])}</td>
              <td>{row.count}</td>
              <td>{row.wins}/{row.losses}</td>
              <td>{formatR(row.expectancy_R)}</td>
              <td className="forward-evidence__floor">
                {progressToFloor(row.count, minSample)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function ForwardEvidenceView() {
  const [board, setBoard] = useState<ForwardEvidenceBoard | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false
    fetchForwardEvidence()
      .then((payload) => { if (!cancelled) setBoard(payload) })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err))
      })
    return () => { cancelled = true }
  }, [])

  if (error) {
    return (
      <EmptyState
        title="Forward evidence"
        detail={`Could not load the evidence board: ${error}`}
      />
    )
  }
  if (!board) {
    return <EmptyState title="Forward evidence" detail="Loading…" />
  }

  const bars = distributionBars(board.r_distribution)
  const warning = attributionWarning(board)

  return (
    <section className="forward-evidence">
      <header className={`forward-evidence__state ${stateTone(board.state)}`}>
        <h2>{board.state.replace(/_/g, ' ')}</h2>
        <p>{board.headline}</p>
        <p className="forward-evidence__class">
          Counting {board.evidence_class} only · {board.min_sample} settled
          trades before a context may affect ranking
        </p>
      </header>

      {warning ? <p className="forward-evidence__warning">{warning}</p> : null}

      <div className="forward-evidence__groups">
        <GroupTable rows={board.by_setup} field="setup" label="Setup"
                    minSample={board.min_sample} />
        <GroupTable rows={board.by_regime} field="regime" label="Regime"
                    minSample={board.min_sample} />
        <GroupTable rows={board.by_sector} field="sector" label="Sector"
                    minSample={board.min_sample} />
      </div>

      <div className="forward-evidence__distribution">
        <h3>Where outcomes landed</h3>
        {board.settled_trades === 0 ? (
          <p className="forward-evidence__none">
            No settled outcomes to distribute.
          </p>
        ) : (
          <ul>
            {bars.map((bar) => (
              <li key={bar.bucket}>
                <span className="forward-evidence__bucket">{bar.bucket}</span>
                <span className="forward-evidence__bar"
                      style={{ width: `${Math.round(bar.share * 100)}%` }} />
                <span className="forward-evidence__count">{bar.count}</span>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="forward-evidence__unresolved">
        <h3>Still running ({board.unresolved_count})</h3>
        {board.unresolved.length === 0 ? (
          <p className="forward-evidence__none">No open paper positions.</p>
        ) : (
          <table>
            <thead>
              <tr><th>Symbol</th><th>Entered</th><th>Bars</th><th>Attributable</th></tr>
            </thead>
            <tbody>
              {board.unresolved.map((row) => (
                <tr key={`${row.symbol}-${row.entry_date}`}>
                  <td>{row.symbol}</td>
                  <td>{row.entry_date}</td>
                  <td>{row.bars_held}</td>
                  <td>{row.attributable ? 'yes' : 'no'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <footer className="forward-evidence__footer">
        <p>
          Replay cells: {board.non_market_evidence.historical_replay_cells} ·
          fixture cells: {board.non_market_evidence.test_fixture_cells}
        </p>
        <p>{board.non_market_evidence.note}</p>
      </footer>
    </section>
  )
}
