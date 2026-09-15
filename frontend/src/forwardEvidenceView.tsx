import { useEffect, useMemo, useState } from 'react'
import {
  fetchForwardEvidence,
  simulatePastDecisions,
  verifyForwardSoakNow,
} from './productApi'
import { EmptyState } from './designSystem'
import {
  attributionWarning,
  distributionBars,
  formatPct,
  formatR,
  progressToFloor,
  stateTone,
} from './forwardEvidenceModel'
import type { EvidenceCell, ForwardEvidenceBoard, GroupRow, UnresolvedRow } from './forwardEvidenceModel'
import './forwardEvidence.css'

/**
 * Has the desk earned any market evidence yet?
 *
 * The screen is built to be as readable when the answer is "none" as when it
 * is "here is the edge". Historical simulation remains explicitly diagnostic:
 * it can help inspect rules, but it can never increment real forward evidence.
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

function ContextExplorer({ cells, unresolved, query }: {
  cells: EvidenceCell[]
  unresolved: UnresolvedRow[]
  query: string
}) {
  const needle = query.trim().toLowerCase()
  const cellRows = useMemo(() => {
    const rows = needle
      ? cells.filter((row) => row.context_key.toLowerCase().includes(needle))
      : cells
    return rows.slice(0, 40)
  }, [cells, needle])
  const unresolvedRows = useMemo(() => {
    const rows = needle
      ? unresolved.filter((row) => (
        row.symbol.toLowerCase().includes(needle)
        || row.context_key.toLowerCase().includes(needle)
        || row.decision_id.toLowerCase().includes(needle)
      ))
      : unresolved
    return rows.slice(0, 40)
  }, [unresolved, needle])

  return (
    <div className="forward-evidence__explorer">
      <div>
        <h3>Context cells</h3>
        {cellRows.length === 0 ? (
          <p className="forward-evidence__none">No matching settled context cells.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Context</th><th>N</th><th>W/L</th><th>EV</th><th>Wilson</th><th>Calibration</th><th>Ranking</th>
              </tr>
            </thead>
            <tbody>
              {cellRows.map((row) => (
                <tr key={row.context_key} className={row.usable_for_ranking ? 'is-usable' : 'is-thin'}>
                  <td className="forward-evidence__context-key">{row.context_key}</td>
                  <td>{row.count}</td>
                  <td>{row.wins}/{row.losses}</td>
                  <td>{formatR(row.expectancy_R)}</td>
                  <td>{formatPct(row.wilson_lower_bound)}</td>
                  <td>{formatPct(row.calibration_gap)}</td>
                  <td>{row.usable_for_ranking ? 'yes' : 'no'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div>
        <h3>Open / unresolved</h3>
        {unresolvedRows.length === 0 ? (
          <p className="forward-evidence__none">No matching open paper evidence.</p>
        ) : (
          <table>
            <thead>
              <tr><th>Symbol</th><th>Decision</th><th>Context</th><th>Entered</th><th>Bars</th><th>Attributable</th></tr>
            </thead>
            <tbody>
              {unresolvedRows.map((row) => (
                <tr key={`${row.decision_id}-${row.symbol}-${row.entry_date}`}>
                  <td>{row.symbol}</td>
                  <td className="forward-evidence__decision-id">{row.decision_id}</td>
                  <td className="forward-evidence__context-key">{row.context_key}</td>
                  <td>{row.entry_date}</td>
                  <td>{row.bars_held}</td>
                  <td>{row.attributable ? 'yes' : 'no'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

export function ForwardEvidenceView() {
  const [board, setBoard] = useState<ForwardEvidenceBoard | null>(null)
  const [error, setError] = useState('')
  const [query, setQuery] = useState('')
  const [busy, setBusy] = useState<'refresh' | 'verify' | 'simulate' | ''>('')
  const [actionMessage, setActionMessage] = useState('')

  const load = async () => {
    setError('')
    const payload = await fetchForwardEvidence()
    setBoard(payload)
    return payload
  }

  useEffect(() => {
    let cancelled = false
    fetchForwardEvidence()
      .then((payload) => { if (!cancelled) setBoard(payload) })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err))
      })
    return () => { cancelled = true }
  }, [])

  const refresh = async () => {
    setBusy('refresh')
    setActionMessage('Refreshing persisted forward evidence…')
    try {
      const next = await load()
      setActionMessage(`Refreshed · ${next.settled_trades} settled · ${next.unresolved_count} open`)
    } catch (err) {
      setActionMessage(err instanceof Error ? err.message : 'Forward evidence refresh failed')
    } finally {
      setBusy('')
    }
  }

  const verifySoak = async () => {
    setBusy('verify')
    setActionMessage('Re-evaluating forward-soak gates from persisted evidence…')
    try {
      const result = await verifyForwardSoakNow()
      await load()
      setActionMessage(
        `Forward soak: ${result.FORWARD_SOAK_STATUS} · ${result.real_forward_observations} real observations · ${result.settled_trades} settled`,
      )
    } catch (err) {
      setActionMessage(err instanceof Error ? err.message : 'Forward-soak verification failed')
    } finally {
      setBusy('')
    }
  }

  const simulate = async () => {
    setBusy('simulate')
    setActionMessage('Running historical decision simulation…')
    try {
      const result = await simulatePastDecisions()
      const tested = Number(result.decisions_tested || 0)
      const status = result.status || (result.accepted ? 'ACCEPTED' : 'COMPLETE')
      setActionMessage(
        `Historical simulation ${status} · ${tested} decisions tested. Diagnostic only — it does not count as forward market evidence.`,
      )
    } catch (err) {
      setActionMessage(err instanceof Error ? err.message : 'Historical simulation failed')
    } finally {
      setBusy('')
    }
  }

  if (error && !board) {
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
      <div className="forward-evidence__toolbar" aria-label="Forward evidence operator controls">
        <div className="forward-evidence__actions">
          <button type="button" disabled={Boolean(busy)} onClick={() => void refresh()}>
            {busy === 'refresh' ? 'Refreshing…' : 'Refresh evidence'}
          </button>
          <button type="button" disabled={Boolean(busy)} onClick={() => void verifySoak()}>
            {busy === 'verify' ? 'Verifying…' : 'Re-evaluate forward soak'}
          </button>
          <button type="button" disabled={Boolean(busy)} onClick={() => void simulate()}>
            {busy === 'simulate' ? 'Simulating…' : 'Simulate past decisions'}
          </button>
        </div>
        <label>
          Inspect symbol / context / decision
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="e.g. RELIANCE, VCP, decision id"
          />
        </label>
        <small>Simulation is replay evidence only and never increments real forward evidence.</small>
        {actionMessage ? <p role="status">{actionMessage}</p> : null}
      </div>

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

      <ContextExplorer cells={board.cells} unresolved={board.unresolved} query={query} />

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
          test-fixture rows excluded: {board.non_market_evidence.test_fixture_cells}
        </p>
        <p>{board.non_market_evidence.note}</p>
      </footer>
    </section>
  )
}
