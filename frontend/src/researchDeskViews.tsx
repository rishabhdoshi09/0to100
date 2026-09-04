import { useEffect, useState } from 'react'
import {
  fetchDecisionSimulator,
  fetchForwardSoak,
  fetchLearningDashboard,
  fetchResearchStatus,
  fetchScanAudit,
  fetchStrategyCatalog,
  fetchSystemHealthContract,
  simulatePastDecision,
  simulatePastDecisions,
  type DecisionSimulatorReport,
  type PastDecisionSimulation,
  type ForwardSoakScoreboard,
  type HealthLane,
  type LearningDashboard,
  type ResearchStatus,
  type ScanAuditPayload,
  type StrategyCatalog,
  type SystemHealthContract,
} from './productApi'
import { originalVsSimulated, simulationUiState, displayHonest } from './pastDecisionSimulation'
import { pageHealth, pageStatusLabel } from './pageRequest'
import { Panel } from './components'
import type { ViewProps } from './views'

function laneTone(status: string): string {
  const s = (status || '').toUpperCase()
  if (s === 'HEALTHY') return 'positive'
  if (s === 'BROKEN') return 'negative'
  return ''
}

function ParityBadge({ value }: { value?: string }) {
  return <strong>{value === 'UNVERIFIED' ? 'BACKTEST PARITY: UNVERIFIED' : (value || 'UNVERIFIED')}</strong>
}

export function StrategiesView() {
  const [data, setData] = useState<StrategyCatalog | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    fetchStrategyCatalog()
      .then(setData)
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : 'Catalog unavailable'))
  }, [])
  const ensemble = data?.ensemble
  return (
    <section className="workspace-view">
      <div className="reco-how">
        <div className="qt-eyebrow">Production methods</div>
        <p>
          These are the checks that rank today&apos;s Recommendations. Paper StrategySpec rows are research-only
          and never attached as if they were this ensemble&apos;s backtest.
        </p>
      </div>
      {error ? <div className="api-warning">{error}</div> : null}
      <Panel title="QT_RECO_ENSEMBLE" subtitle={ensemble ? `v${ensemble.strategy_version} · hash ${ensemble.rules_hash}` : 'Loading'}>
        {ensemble ? (
          <div className="key-value-list">
            <div><span>Parity</span><ParityBadge value={ensemble.backtest_parity} /></div>
            <div><span>Universe</span><strong>{ensemble.universe || '—'}</strong></div>
            <div><span>Hold</span><strong>{ensemble.intended_holding_period || '—'}</strong></div>
            <div><span>Active</span><strong>{ensemble.active ? 'yes' : 'no'}</strong></div>
          </div>
        ) : <div className="empty-row">Waiting for catalog…</div>}
        {ensemble?.backtest_parity_detail ? <p className="panel-copy">{ensemble.backtest_parity_detail}</p> : null}
      </Panel>
      <Panel title="METHOD CHECKS" subtitle="Each method has its own id and hash. Unknown is not a fail.">
        {(data?.methods || []).map((method) => (
          <div className="insight" key={method.strategy_id}>
            <i className="cyan" />
            <div>
              <strong>{method.label}</strong>
              <span>{method.strategy_id} v{method.strategy_version} · {method.rules_hash} · {method.backtest_parity}</span>
            </div>
          </div>
        ))}
      </Panel>
      <Panel title="RELATED SCANNER CALIBRATION" subtitle="Not recommendation parity">
        <p className="panel-copy">{data?.related_signal_calibration?.detail || 'Calibration file not read yet.'}</p>
        <p className="panel-copy">Parity: {data?.related_signal_calibration?.parity || 'UNVERIFIED'}</p>
      </Panel>
      <Panel title="RESEARCH-ONLY STRATEGIES" subtitle="Paper / autonomy registry snapshot if one exists">
        {(data?.research_only || []).length === 0 ? (
          <div className="empty-row">No paper strategy snapshot on disk. Missing stays missing — none are invented.</div>
        ) : (data?.research_only || []).map((row) => (
          <div className="insight" key={row.strategy_id}>
            <i className="amber" />
            <div>
              <strong>{row.label}</strong>
              <span>{row.strategy_id} · {row.role} · {row.backtest_parity}</span>
            </div>
          </div>
        ))}
      </Panel>
    </section>
  )
}

function metricText(value: number | null | undefined, fallback = 'INSUFFICIENT EVIDENCE'): string {
  if (value === null || value === undefined) return fallback
  return String(value)
}

export function LearningJournalView() {
  const [data, setData] = useState<ResearchStatus | null>(null)
  const [learning, setLearning] = useState<LearningDashboard | null>(null)
  const [soak, setSoak] = useState<ForwardSoakScoreboard | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  useEffect(() => {
    let alive = true
    setLoading(true)
    Promise.allSettled([
      fetchResearchStatus(),
      fetchLearningDashboard(),
      fetchForwardSoak(),
    ]).then(([status, dash, soakRow]) => {
      if (!alive) return
      if (status.status === 'fulfilled') setData(status.value)
      else setError(status.reason instanceof Error ? status.reason.message : 'Research status unavailable')
      if (dash.status === 'fulfilled') setLearning(dash.value)
      if (soakRow.status === 'fulfilled') setSoak(soakRow.value)
    }).finally(() => { if (alive) setLoading(false) })
    return () => { alive = false }
  }, [])
  const journal = data?.decision_journal
  const recent = learning?.recent_learning
  const board = soak || learning?.forward_soak || null
  const evidenceLabel = board?.insufficient_evidence ? 'INSUFFICIENT EVIDENCE' : (board?.evidence_label || 'INSUFFICIENT EVIDENCE')
  return (
    <section className="workspace-view">
      <div className="reco-how">
        <div className="qt-eyebrow">Learning / Decision Journal</div>
        <p>{data?.disclaimer || 'Measurable evidence only. Empty is a valid state.'}</p>
      </div>
      {error ? <div className="api-warning">{error}</div> : null}
      {loading ? <p className="panel-copy">Loading learning journal…</p> : null}
      <Panel title="FORWARD EVIDENCE SCOREBOARD" subtitle={board?.FORWARD_SOAK_STATUS || (loading ? 'Loading' : 'NOT_STARTED')}>
        {loading && !board ? (
          <div className="empty-row">Loading forward evidence…</div>
        ) : !board ? (
          <div className="empty-row">Forward soak scoreboard unavailable. Missing stays missing.</div>
        ) : (
          <>
            <div className="fact-grid">
              <div><span>Real forward observations</span><strong>{board.real_forward_observations}</strong></div>
              <div><span>Paper trades taken</span><strong>{board.paper_trades_taken}</strong></div>
              <div><span>Settled trades</span><strong>{board.settled_trades}</strong></div>
              <div><span>Rejected candidates settled</span><strong>{board.rejected_candidates_settled}</strong></div>
              <div><span>Missed winners</span><strong>{board.missed_winners}</strong></div>
              <div><span>Avoided losers</span><strong>{board.avoided_losers}</strong></div>
              <div><span>Good waits</span><strong>{board.good_waits}</strong></div>
              <div><span>Gross expectancy</span><strong>{metricText(board.gross_expectancy)}</strong></div>
              <div><span>Execution-adjusted expectancy</span><strong>{metricText(board.execution_adjusted_expectancy)}</strong></div>
              <div><span>Execution coverage</span><strong>{board.execution_adjusted_coverage_pct == null ? evidenceLabel : `${board.execution_adjusted_coverage_pct}%`}</strong></div>
              <div><span>Current drawdown</span><strong>{metricText(board.current_drawdown)}</strong></div>
              <div><span>Win rate</span><strong>{metricText(board.win_rate)}</strong></div>
              <div><span>Average win</span><strong>{metricText(board.average_win)}</strong></div>
              <div><span>Average loss</span><strong>{metricText(board.average_loss)}</strong></div>
              <div><span>Active policies</span><strong>{board.active_policies}</strong></div>
              <div><span>Eligible policies</span><strong>{board.eligible_policies}</strong></div>
              <div><span>Challengers under evaluation</span><strong>{board.challengers_under_evaluation}</strong></div>
              <div><span>Live locked</span><strong>yes</strong></div>
            </div>
            <p className="panel-copy">{board.soak_detail || board.note || evidenceLabel}</p>
            {Object.keys(board.setup_level_evidence || {}).length ? (
              <p className="panel-copy">
                Setup-level evidence:{' '}
                {Object.entries(board.setup_level_evidence || {}).map(([key, row]) => `${key} n=${row.n} ${row.evidence}`).join(' · ') || 'none'}
              </p>
            ) : null}
            {Object.keys(board.regime_level_evidence || {}).length ? (
              <p className="panel-copy">
                Regime-level evidence:{' '}
                {Object.entries(board.regime_level_evidence || {}).map(([key, row]) => `${key} n=${row.n} ${row.evidence}`).join(' · ') || 'none'}
              </p>
            ) : null}
            {Object.keys(board.sector_level_evidence || {}).length ? (
              <p className="panel-copy">
                Sector-level evidence:{' '}
                {Object.entries(board.sector_level_evidence || {}).map(([key, row]) => `${key} n=${row.n} ${row.evidence}`).join(' · ') || 'none'}
              </p>
            ) : null}
            {(board.promotion_blockers?.components || []).length ? (
              <p className="panel-copy">
                Promotion blockers:{' '}
                {(board.promotion_blockers?.components || []).map((row) => (
                  `${row.component} ${row.decision || 'KEEP_SHADOW'}${(row.blockers || []).length ? ` (${(row.blockers || []).join(', ')})` : ''}`
                )).join(' · ')}
              </p>
            ) : null}
          </>
        )}
      </Panel>
      <Panel title="WHAT IS MEASURABLE NOW" subtitle={data?.learning_status || 'UNKNOWN'}>
        {(data?.headlines || []).map((line) => <p className="panel-copy" key={line}>{line}</p>)}
      </Panel>
      <Panel title="PRODUCTION POLICIES" subtitle="Versioned evidence overlays. They never invent a BUY.">
        {(learning?.active || []).length === 0 ? (
          <div className="empty-row">No ACTIVE policies yet. INSUFFICIENT EVIDENCE is the honest state.</div>
        ) : (learning?.active || []).map((policy) => (
          <div className="insight" key={`${policy.policy_id}-${policy.version || 0}`}>
            <i className="cyan" />
            <div>
              <strong>{policy.policy_id}</strong>
              <span>
                {policy.production_status} · n={policy.sample_size ?? 0} · edge {policy.expectancy_difference_R ?? 0}R · {policy.confidence || 'UNKNOWN'}
              </span>
            </div>
          </div>
        ))}
      </Panel>
      <Panel title="POLICIES UNDER OBSERVATION" subtitle="INSUFFICIENT EVIDENCE until sample floors. One trade cannot move production.">
        {(learning?.observing || []).length === 0 ? (
          <div className="empty-row">No hypotheses under observation.</div>
        ) : (learning?.observing || []).slice(0, 12).map((policy) => (
          <div className="insight" key={`${policy.policy_id}-obs-${policy.version || 0}`}>
            <i className="amber" />
            <div>
              <strong>{policy.policy_id}</strong>
              <span>
                {policy.production_status} · n={policy.sample_size ?? 0} · {policy.confidence || 'INSUFFICIENT EVIDENCE'}
              </span>
            </div>
          </div>
        ))}
      </Panel>
      <Panel title="REJECTED HYPOTHESES" subtitle="No measurable edge, or demoted. Not deleted.">
        {(learning?.rejected_hypotheses || []).length === 0 ? (
          <div className="empty-row">No rejected hypotheses yet.</div>
        ) : (learning?.rejected_hypotheses || []).slice(0, 12).map((policy) => (
          <div className="insight" key={`${policy.policy_id}-rej-${policy.version || 0}`}>
            <i className="amber" />
            <div>
              <strong>{policy.policy_id}</strong>
              <span>{policy.production_status} · n={policy.sample_size ?? 0} · no production effect</span>
            </div>
          </div>
        ))}
      </Panel>
      <Panel title="WHY BOT TOOK / DID NOT TAKE" subtitle="Deterministic evidence. An LLM must not manufacture this.">
        {(learning?.explanations?.taken || []).length === 0 && (learning?.explanations?.rejected || []).length === 0 ? (
          <div className="empty-row">No autopilot explanations yet. Missing stays missing.</div>
        ) : (
          <>
            {(learning?.explanations?.taken || []).map((row) => (
              <div className="insight" key={`took-${row.symbol}`}>
                <i className="green" />
                <div>
                  <strong>{row.symbol} · {row.title || 'WHY BOT TOOK THIS'}</strong>
                  <span>{(row.plus || []).join(' · ')}{(row.minus || []).length ? ` — ${(row.minus || []).join(' · ')}` : ''}</span>
                </div>
              </div>
            ))}
            {(learning?.explanations?.rejected || []).slice(0, 8).map((row) => (
              <div className="insight" key={`skip-${row.symbol}-${row.reason_code}`}>
                <i className="amber" />
                <div>
                  <strong>{row.symbol} · {row.reason_code || row.title}</strong>
                  <span>{row.action || (row.minus || []).join(' · ') || 'Rejected with a machine-readable reason'}</span>
                </div>
              </div>
            ))}
          </>
        )}
      </Panel>
      <Panel title="RECENT LEARNING" subtitle="Taken, rejected, and counterfactual classifications — not P&L from skipped names">
        <div className="fact-grid">
          <div><span>Taken fills</span><strong>{recent?.taken_fills ?? 0}</strong></div>
          <div><span>Correct rejects</span><strong>{recent?.correct_rejects ?? 0}</strong></div>
          <div><span>Missed winners</span><strong>{recent?.missed_winners ?? 0}</strong></div>
          <div><span>Avoided losers</span><strong>{recent?.avoided_losers ?? 0}</strong></div>
          <div><span>Good waits</span><strong>{recent?.good_waits ?? 0}</strong></div>
          <div><span>Live locked</span><strong>yes</strong></div>
        </div>
        <p className="panel-copy">{learning?.note || ''}</p>
      </Panel>
      <Panel title="SETTLED / REJECTED" subtitle="Paper taken vs skipped, plus latest scan decisions">
        <div className="fact-grid">
          <div><span>Paper closed</span><strong>{data?.paper.closed_trades ?? 0}</strong></div>
          <div><span>Taken (last feed)</span><strong>{data?.paper.taken.length ?? 0}</strong></div>
          <div><span>Skipped (last feed)</span><strong>{data?.paper.skipped.length ?? 0}</strong></div>
          <div><span>Surfaced journal</span><strong>{journal?.counts?.surfaced_history ?? 0}</strong></div>
          <div><span>Latest scan rows</span><strong>{journal?.counts?.latest_scan_decisions ?? 0}</strong></div>
          <div><span>Tracked sample</span><strong>{journal?.performance?.sample_size ?? 0}</strong></div>
        </div>
        <p className="panel-copy">{journal?.performance?.sample_note || journal?.note || ''}</p>
      </Panel>
      <Panel title="RECENT DECISIONS" subtitle="Surfaced recommendations and names the scan checked but did not qualify">
        {(journal?.entries || []).length === 0 ? (
          <div className="empty-row">No journal rows yet.</div>
        ) : (journal?.entries || []).slice(0, 24).map((row, index) => (
          <div className="insight" key={`${row.symbol}-${row.kind}-${index}`}>
            <i className={row.kind === 'SURFACED' ? 'green' : 'amber'} />
            <div>
              <strong>{row.symbol}</strong>
              <span>{row.kind} · {row.decision} · {row.reason}</span>
            </div>
          </div>
        ))}
      </Panel>
    </section>
  )
}

export function CoverageView() {
  const [data, setData] = useState<ScanAuditPayload | null>(null)
  const [query, setQuery] = useState('')
  const [lookup, setLookup] = useState<ScanAuditPayload | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [lookupBusy, setLookupBusy] = useState(false)
  useEffect(() => {
    let alive = true
    setLoading(true)
    fetchScanAudit('', 80)
      .then((payload) => { if (alive) setData(payload) })
      .catch((reason: unknown) => { if (alive) setError(reason instanceof Error ? reason.message : 'Coverage unavailable') })
      .finally(() => { if (alive) setLoading(false) })
    return () => { alive = false }
  }, [])
  const summary = data?.summary || {}
  const inspect = async () => {
    const clean = query.trim().toUpperCase()
    if (!clean) return
    setLookupBusy(true)
    try {
      setLookup(await fetchScanAudit(clean, 1))
      setError('')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Lookup failed')
    } finally {
      setLookupBusy(false)
    }
  }
  return (
    <section className="workspace-view">
      <div className="reco-how">
        <div className="qt-eyebrow">Scan coverage</div>
        <p>Requested, checked, qualified, no-setup, excluded, and failed are separate. Missing names can be inspected.</p>
      </div>
      {error ? <div className="api-warning">{error}</div> : null}
      {loading && !data ? <p className="panel-copy">Loading scan coverage…</p> : null}
      <div className="fact-grid">
        {['requested', 'checked', 'qualified', 'no_setup', 'policy_excluded', 'data_unavailable', 'analysis_errors'].map((key) => (
          <div key={key}>
            <span>{key.replace(/_/g, ' ')}</span>
            <strong>{String(summary[key] ?? '—')}</strong>
          </div>
        ))}
      </div>
      <Panel title="INSPECT A TICKER" subtitle="Was it requested, checked, qualified, or missing?">
        <div className="inline-actions">
          <input
            aria-label="Coverage symbol"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => { if (event.key === 'Enter') void inspect() }}
            placeholder="TCS"
          />
          <button type="button" disabled={lookupBusy} onClick={() => void inspect()}>{lookupBusy ? 'Looking…' : 'Look up'}</button>
        </div>
        {lookup?.result ? (
          <p className="panel-copy">
            {lookup.result.symbol}: {lookup.result.status} — {lookup.result.reason || lookup.result.error || 'No extra reason'}
          </p>
        ) : lookup && lookup.found === false ? (
          <p className="panel-copy">{lookup.symbol} was not in the latest scan audit. That is missing, not a fail.</p>
        ) : null}
      </Panel>
      <Panel title="LATEST AUDIT ROWS" subtitle={`${data?.total ?? 0} symbols in the ledger`}>
        {(data?.rows || []).slice(0, 40).map((row) => (
          <div className="insight" key={row.symbol}>
            <i className={row.status === 'QUALIFIED' ? 'green' : 'amber'} />
            <div>
              <strong>{row.symbol}</strong>
              <span>{row.status} · {row.reason || row.error || '—'}</span>
            </div>
          </div>
        ))}
      </Panel>
    </section>
  )
}

function HealthLanes({ contract }: { contract: SystemHealthContract | null }) {
  if (!contract) return <div className="empty-row">Health contract not loaded.</div>
  return (
    <>
      <p className="panel-copy">{contract.note}</p>
      <div className="fact-grid">
        {Object.entries(contract.counts).map(([key, value]) => (
          <div key={key}><span>{key}</span><strong>{value}</strong></div>
        ))}
      </div>
      {(contract.lanes || []).map((lane: HealthLane) => (
        <div className="insight" key={lane.key}>
          <i className={lane.status === 'HEALTHY' ? 'green' : lane.status === 'BROKEN' ? 'amber' : 'cyan'} />
          <div>
            <strong className={laneTone(lane.status)}>{lane.label}: {lane.status}</strong>
            <span>{lane.detail}{lane.as_of ? ` · ${lane.as_of}` : ''}</span>
          </div>
        </div>
      ))}
    </>
  )
}

export function SystemHealthView({ dashboard, runControl }: ViewProps) {
  const [contract, setContract] = useState<SystemHealthContract | null>(null)
  const [healthError, setHealthError] = useState('')
  const [healthLoading, setHealthLoading] = useState(true)
  const loadContract = () => {
    setHealthLoading(true)
    fetchSystemHealthContract()
      .then((payload) => { setContract(payload); setHealthError('') })
      .catch((reason: unknown) => {
        setContract(null)
        setHealthError(reason instanceof Error ? reason.message : 'Health contract failed')
      })
      .finally(() => setHealthLoading(false))
  }
  useEffect(() => { loadContract() }, [dashboard.generated_at])
  const a = dashboard.autonomy
  return (
    <section className="workspace-view">
      <div className="inline-actions">
        <button type="button" onClick={() => void runControl('RUN_SCAN_NOW')}>Start market scan</button>
        <button type="button" onClick={() => void runControl('RUN_CYCLE_NOW')}>Request paper cycle</button>
        <button type="button" onClick={() => void runControl('REFRESH_DATA_NOW')}>Prepare market data</button>
        <button type="button" onClick={() => void runControl(a.new_paper_entries ? 'PAUSE_NEW_PAPER_ENTRIES' : 'RESUME_NEW_PAPER_ENTRIES')}>
          {a.new_paper_entries ? 'Pause entries' : 'Resume entries'}
        </button>
      </div>
      <Panel title="WHY NO TRADE TODAY" subtitle="Selection authority — not the autonomy badge">
        {contract?.why_no_trade?.available ? (
          <>
            <p className="panel-copy">{contract.why_no_trade.headline}</p>
            <div className="fact-grid">
              <div><span>Decision</span><strong>{contract.why_no_trade.decision}</strong></div>
              <div><span>Taken</span><strong>{(contract.why_no_trade.taken || []).length}</strong></div>
              <div><span>Rejected</span><strong>{(contract.why_no_trade.rejections || []).length}</strong></div>
            </div>
            {(contract.why_no_trade.reasons || []).length ? (
              <p className="muted">Reasons: {(contract.why_no_trade.reasons || []).join(' · ')}</p>
            ) : null}
          </>
        ) : (
          <div className="empty-row">No paper-autopilot cycle recorded yet. Missing stays missing.</div>
        )}
      </Panel>
      <Panel title="INDEPENDENT HEALTH LANES" subtitle="No collapsed green light. Paper execution is its own lane.">
        {(() => {
          const page = pageHealth({
            page: 'System Health',
            loading: healthLoading,
            data: contract,
            error: healthError,
          })
          return (
            <p className="panel-copy">
              Page: {pageStatusLabel(page.status)}
              {page.lastError ? ` · ${page.lastError}` : ''}
              {page.loadingMs ? ` · ${Math.round(page.loadingMs / 1000)}s` : ''}
            </p>
          )
        })()}
        {healthError ? (
          <div className="empty-row">
            {healthError}
            {' '}
            <button type="button" className="secondary" onClick={() => loadContract()}>Retry</button>
          </div>
        ) : healthLoading && !contract ? (
          <div className="empty-row">Loading health contract…</div>
        ) : (
          <HealthLanes contract={contract} />
        )}
      </Panel>
    </section>
  )
}

export function ProductionBacktestView({ dashboard, setActive }: ViewProps) {
  const [catalog, setCatalog] = useState<StrategyCatalog | null>(null)
  const [catalogError, setCatalogError] = useState('')
  const [catalogLoading, setCatalogLoading] = useState(true)
  const [sim, setSim] = useState<DecisionSimulatorReport | null>(null)
  const [simError, setSimError] = useState('')
  const [simBusy, setSimBusy] = useState(false)
  const [openDecision, setOpenDecision] = useState(0)
  const [caseSim, setCaseSim] = useState<PastDecisionSimulation | null>(null)
  const [caseError, setCaseError] = useState('')
  const [caseBusy, setCaseBusy] = useState(false)
  const [caseSymbol, setCaseSymbol] = useState('')
  const [caseAsOf, setCaseAsOf] = useState('')
  const [caseAlt, setCaseAlt] = useState('BUY')
  useEffect(() => {
    setCatalogLoading(true)
    fetchStrategyCatalog()
      .then((payload) => { setCatalog(payload); setCatalogError('') })
      .catch((reason: unknown) => setCatalogError(reason instanceof Error ? reason.message : 'Catalog unavailable'))
      .finally(() => setCatalogLoading(false))
    fetchDecisionSimulator()
      .then((payload) => { setSim(payload); setSimError('') })
      .catch((reason: unknown) => setSimError(reason instanceof Error ? reason.message : 'Simulator unavailable'))
  }, [])
  const pollSim = async (seed?: DecisionSimulatorReport) => {
    let latest = seed
    for (let i = 0; i < 60; i += 1) {
      if (!latest || latest.status === 'RUNNING' || latest.accepted) {
        await new Promise((resolve) => window.setTimeout(resolve, 1500))
        latest = await fetchDecisionSimulator()
        setSim(latest)
        continue
      }
      break
    }
    return latest
  }
  const runSim = () => {
    setSimBusy(true)
    simulatePastDecisions()
      .then((payload) => { setSim(payload); setSimError(''); return pollSim(payload) })
      .then((payload) => { if (payload) setSim(payload) })
      .catch((reason: unknown) => setSimError(reason instanceof Error ? reason.message : 'Simulator failed'))
      .finally(() => setSimBusy(false))
  }
  const runCase = (symbol: string, as_of: string, alternative?: string) => {
    const name = symbol.trim().toUpperCase()
    const day = as_of.trim().slice(0, 10)
    if (!name || !day) {
      setCaseError('Symbol and historical date are required. No sample decision is invented.')
      return
    }
    setCaseBusy(true)
    setCaseError('')
    simulatePastDecision({ symbol: name, as_of: day, alternative: alternative || caseAlt })
      .then((payload) => { setCaseSim(payload); setCaseError(payload.error || '') })
      .catch((reason: unknown) => {
        setCaseSim(null)
        setCaseError(reason instanceof Error ? reason.message : 'Simulation failed')
      })
      .finally(() => setCaseBusy(false))
  }
  const caseState = simulationUiState(caseSim, caseError && !caseSim ? caseError : '')
  const caseView = caseSim ? originalVsSimulated(caseSim) : null
  const feed = dashboard.paper.learning?.self_feed || {}
  return (
    <section className="workspace-view">
      <div className="reco-how">
        <div className="qt-eyebrow">Backtests connected to production</div>
        <p>
          Only the live recommendation ensemble is shown as production. If the same rules_hash was not
          evaluated, the page says BACKTEST PARITY: UNVERIFIED. Paper diary rows below are outcomes, not a substitute backtest.
        </p>
      </div>
      <Panel title="HISTORICAL REPLAY" subtitle="Production scanner + evaluate_candidate · BACKTEST · never writes REAL_FORWARD_MARKET">
        {simError ? (
          <p className="panel-copy">
            {simError}
            {' '}
            <button type="button" className="secondary" onClick={runSim}>Retry</button>
          </p>
        ) : null}
        <p className="panel-copy">
          {sim?.status || 'NO RUN'}
          {sim?.period_start ? ` · Period ${sim.period_start} → ${sim.period_end}` : ''}
          {sim?.sessions_total ? ` · Sessions ${sim.sessions_done ?? 0}/${sim.sessions_total}` : ''}
        </p>
        {sim?.simple ? <p className="panel-copy">{sim.simple}</p> : <p className="panel-copy">No historical replay yet. This button runs the production decision path on official past sessions.</p>}
        <p className="panel-copy">{sim?.engine || ''}</p>
        <div className="fact-grid">
          <div><span>Trading sessions</span><strong>{sim?.trading_sessions ?? '—'}</strong></div>
          <div><span>Universe observations</span><strong>{sim?.universe_observations ?? '—'}</strong></div>
          <div><span>Stocks evaluated</span><strong>{sim?.stocks_evaluated ?? '—'}</strong></div>
          <div><span>Decision candidates</span><strong>{sim?.decision_candidates ?? sim?.decisions_tested ?? '—'}</strong></div>
          <div><span>BUY</span><strong>{sim?.BUY ?? sim?.would_take ?? '—'}</strong></div>
          <div><span>WAIT</span><strong>{sim?.WAIT ?? sim?.waited ?? '—'}</strong></div>
          <div><span>AVOID</span><strong>{sim?.AVOID ?? '—'}</strong></div>
          <div><span>REJECT</span><strong>{sim?.REJECT ?? sim?.rejected ?? '—'}</strong></div>
          <div><span>Outcomes matured</span><strong>{sim?.outcomes_matured ?? '—'}</strong></div>
          <div><span>Correct rejections</span><strong>{sim?.correct_rejections ?? '—'}</strong></div>
          <div><span>Missed winners</span><strong>{sim?.missed_winners ?? '—'}</strong></div>
          <div><span>Open / unresolved</span><strong>{sim?.open_unresolved ?? '—'}</strong></div>
        </div>
        <div className="inline-actions" style={{ padding: '12px' }}>
          <button type="button" disabled={simBusy} onClick={runSim}>{simBusy ? (sim?.message || 'Replaying…') : 'Simulate past decisions'}</button>
        </div>
        <p className="panel-copy">{sim?.note || 'Later prices are used only for outcome classification.'}</p>
        <Panel title="ONE PAST DECISION" subtitle="Original decision vs simulated alternative · PIT at T · subsequent bars only for outcome">
          <p className="panel-copy">Replay one persisted QuantTerm decision. Missing journal rows stay UNAVAILABLE. No sample data is substituted.</p>
          <div className="inline-actions" style={{ padding: '12px', gap: 8 }}>
            <input value={caseSymbol} onChange={(event) => setCaseSymbol(event.target.value)} placeholder="Symbol" aria-label="Historical symbol" />
            <input value={caseAsOf} onChange={(event) => setCaseAsOf(event.target.value)} placeholder="YYYY-MM-DD" aria-label="Historical date" />
            <select value={caseAlt} onChange={(event) => setCaseAlt(event.target.value)} aria-label="Counterfactual action">
              <option value="BUY">Simulate BUY</option>
              <option value="WAIT">Simulate WAIT</option>
              <option value="AVOID">Simulate AVOID</option>
            </select>
            <button type="button" disabled={caseBusy} onClick={() => runCase(caseSymbol, caseAsOf, caseAlt)}>
              {caseBusy ? 'Simulating…' : 'Simulate Past Decision'}
            </button>
          </div>
          {caseBusy ? <p className="panel-copy">Loading point-in-time replay…</p> : null}
          {caseState === 'error' ? <p className="panel-copy">{caseError}</p> : null}
          {caseState === 'failed' ? <p className="panel-copy">{caseSim?.error || caseError || 'Simulation failed'}</p> : null}
          {caseState === 'unavailable' ? <p className="panel-copy">{caseSim?.error || 'No persisted historical decision. Nothing was invented.'}</p> : null}
          {caseSim && caseView ? (
            <>
              <div className="fact-grid">
                <div><span>Original Decision</span><strong>{caseView.originalAction}</strong></div>
                <div><span>Simulated Alternative</span><strong>{caseView.simulatedAction}</strong></div>
                <div><span>Timestamp</span><strong>{displayHonest(caseSim.historical_timestamp)}</strong></div>
                <div><span>Reason</span><strong>{displayHonest(caseSim.original?.reason_code)}</strong></div>
              </div>
              <p className="panel-copy"><strong>{caseView.evidenceLabel}</strong></p>
              <p className="panel-copy">
                Close at T: {displayHonest(caseSim.evidence_at_t?.close)}
                {' · '}max bar {displayHonest(caseSim.evidence_at_t?.max_bar_date)}
                {caseView.lookahead ? ' · LOOKAHEAD FLAG' : ' · no future bars in the decision'}
              </p>
              <p className="panel-copy">
                Financials: {caseSim.evidence_at_t?.financials?.available ? 'available at T' : displayHonest(caseSim.evidence_at_t?.financials?.status)}
                {' · '}Research: {caseSim.evidence_at_t?.research?.available ? 'available at T' : displayHonest(caseSim.evidence_at_t?.research?.status)}
                {' · '}News: {displayHonest(caseSim.evidence_at_t?.news_status)}
              </p>
              <p className="panel-copy"><strong>{caseView.outcomeLabel}</strong></p>
              <div className="fact-grid">
                <div><span>Actual path</span><strong>{displayHonest((caseSim.subsequent_outcome?.actual as { status?: string } | undefined)?.status)}</strong></div>
                <div><span>Simulated path</span><strong>{displayHonest((caseSim.subsequent_outcome?.simulated as { status?: string } | undefined)?.status)}</strong></div>
                <div><span>Simulated MFE</span><strong>{displayHonest((caseSim.subsequent_outcome?.simulated as { mfe_pct?: unknown } | undefined)?.mfe_pct)}</strong></div>
                <div><span>Simulated MAE</span><strong>{displayHonest((caseSim.subsequent_outcome?.simulated as { mae_pct?: unknown } | undefined)?.mae_pct)}</strong></div>
                <div><span>Simulated return</span><strong>{displayHonest(caseSim.comparison?.simulated_return_pct)}</strong></div>
                <div><span>Return delta</span><strong>{displayHonest(caseSim.comparison?.return_delta_pct)}</strong></div>
              </div>
              <p className="panel-copy">{displayHonest((caseSim.subsequent_outcome?.simulated as { methodology?: string } | undefined)?.methodology, '')}</p>
              {(caseSim.warnings || []).length ? <p className="panel-copy">Warnings: {(caseSim.warnings || []).join(' · ')}</p> : null}
              {caseSim.error && caseState === 'ready' ? <p className="panel-copy">{caseSim.error}</p> : null}
            </>
          ) : null}
        </Panel>
        {(sim?.decisions || sim?.rows || []).slice(0, 12).map((row, index) => (
          <article key={`${row.as_of}-${row.symbol}-${index}`} className="requirement-card" style={{ marginTop: 8 }}>
            <button type="button" className="secondary" onClick={() => setOpenDecision(index)}>
              {row.as_of} · {row.symbol} · {row.decision} · {row.classification || 'UNRESOLVED'}
            </button>
            {openDecision === index ? (
              <div>
                <p className="panel-copy">Decision: {row.decision} · {row.reason_code}</p>
                <p className="panel-copy">Reasons: {(row.reasons || []).join(' · ') || '—'}</p>
                <p className="panel-copy">
                  Data available at decision date: {row.pit?.max_bar_date || row.as_of || 'unknown'}
                  {row.pit?.future_evidence_used ? ' · LOOKAHEAD FLAG' : ' · no future bars'}
                </p>
                <p className="panel-copy">
                  Subsequent outcome: {row.forward_return_pct == null ? 'unresolved' : `${row.forward_return_pct}%`}
                  {' · '}{row.classification || row.outcome_status || 'INCONCLUSIVE'}
                </p>
                {(row.pit?.degraded || []).length ? <p className="panel-copy">Degraded: {(row.pit?.degraded || []).join(' · ')}</p> : null}
                {row.symbol && row.as_of ? (
                  <button
                    type="button"
                    className="secondary"
                    disabled={caseBusy}
                    onClick={() => {
                      setCaseSymbol(row.symbol || '')
                      setCaseAsOf(row.as_of || '')
                      runCase(row.symbol || '', row.as_of || '')
                    }}
                  >
                    Simulate this decision
                  </button>
                ) : null}
              </div>
            ) : null}
          </article>
        ))}
      </Panel>
      <Panel title="PRODUCTION ENSEMBLE" subtitle={catalog?.ensemble.strategy_id || 'QT_RECO_ENSEMBLE'}>
        {catalogLoading && !catalog ? <p className="panel-copy">Waiting for catalog…</p> : null}
        {catalogError ? <p className="panel-copy">{catalogError}</p> : null}
        <p className="panel-copy">
          <ParityBadge value={catalog?.ensemble.backtest_parity} />
        </p>
        <p className="panel-copy">{catalog?.ensemble.backtest_parity_detail}</p>
        <p className="panel-copy">{catalog?.related_signal_calibration?.detail}</p>
      </Panel>
      <Panel title="PAPER DIARY" subtitle="Does not change today's BUY list">
        <div className="fact-grid">
          <div><span>Taken</span><strong>{(feed.taken || []).length}</strong></div>
          <div><span>Skipped</span><strong>{(feed.skipped || []).length}</strong></div>
          <div><span>Candidate tests</span><strong>{(feed.candidate_tests || []).length}</strong></div>
        </div>
        <div className="inline-actions" style={{ padding: '12px' }}>
          <button type="button" onClick={() => setActive('Learning')}>Open Learning / Decision Journal</button>
          <button type="button" onClick={() => setActive('Paper Portfolio')}>Open Portfolio</button>
        </div>
      </Panel>
    </section>
  )
}
