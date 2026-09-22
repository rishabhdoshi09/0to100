import { useEffect, useState } from 'react'
import {
  fetchDecisionSimulator,
  fetchForwardSoak,
  fetchLearningDashboard,
  fetchAutonomousLearning,
  fetchResearchStatus,
  fetchResearchDirector,
  fetchScanAudit,
  fetchStrategyCatalog,
  fetchSystemHealthContract,
  runHistoricalReplayNow,
  setAutonomousLearning,
  simulatePastDecision,
  simulatePastDecisions,
  type AutonomousLearningDashboard,
  type DecisionSimulatorReport,
  type PastDecisionSimulation,
  type ForwardSoakScoreboard,
  type HealthLane,
  type LearningDashboard,
  type ResearchStatus,
  type ResearchDirectorStatus,
  type ScanAuditPayload,
  type StrategyCatalog,
  type SystemHealthContract,
} from './productApi'
import { compactDateTime } from './format'
import { sendControl } from './api'
import { originalVsSimulated, simulationUiState, displayHonest } from './pastDecisionSimulation'
import { pageHealth, pageStatusLabel } from './pageRequest'
import { Panel } from './components'
import type { ViewProps } from './views'

// A lane whose acquisition was refused or errored must not read as neutral.
// BLOCKED/FAILED/BROKEN are problems the operator has to see; MISSING, STALE,
// PARTIAL and WAITING are honest gaps, not faults.
const LANE_BAD = new Set(['BROKEN', 'FAILED', 'BLOCKED'])
const LANE_GAP = new Set(['MISSING', 'STALE', 'PARTIAL', 'WAITING'])

export function laneTone(status: string): string {
  const s = (status || '').toUpperCase()
  if (s === 'HEALTHY') return 'positive'
  if (LANE_BAD.has(s)) return 'negative'
  return ''
}

export function laneDot(status: string): string {
  const s = (status || '').toUpperCase()
  if (s === 'HEALTHY') return 'green'
  if (LANE_BAD.has(s)) return 'amber'
  if (LANE_GAP.has(s)) return 'cyan'
  return 'cyan'
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

export function liveSafetyLabel(state?: {
  live_locked?: boolean | null
  live_lock_verified?: boolean
} | null): string {
  if (!state || state.live_lock_verified !== true) return 'UNVERIFIED'
  if (state.live_locked === true) return 'VERIFIED / LOCKED'
  if (state.live_locked === false) return 'VERIFIED / UNLOCKED'
  return 'UNVERIFIED'
}

function countValue(counts: Record<string, number> | undefined, key: string): string {
  if (!counts || counts[key] === undefined || counts[key] === null) return 'No data'
  return String(counts[key])
}

function AutonomousLearningPanel({
  data,
  onRefresh,
}: {
  data: AutonomousLearningDashboard | null
  onRefresh: () => void
}) {
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const apply = (enabled?: boolean, mode?: string) => {
    setBusy(mode || (enabled === false ? 'off' : 'on'))
    setError('')
    setAutonomousLearning(enabled, mode)
      .then(() => onRefresh())
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : 'Control failed'))
      .finally(() => setBusy(''))
  }
  const runReplay = () => {
    setBusy('replay')
    setError('')
    runHistoricalReplayNow()
      .then(() => onRefresh())
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : 'Replay failed'))
      .finally(() => setBusy(''))
  }
  const runLearning = () => {
    setBusy('learning')
    setError('')
    sendControl('RUN_LEARNING_NOW')
      .then(() => onRefresh())
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : 'Learning cycle failed'))
      .finally(() => setBusy(''))
  }
  const counts = data?.counts || {}
  const champion = data?.champion || {}
  const challenger = data?.challenger || {}
  return (
    <Panel title="AUTONOMOUS LEARNING" subtitle={data?.enabled ? `ON · ${data.mode || 'AUTO'}` : 'OFF'}>
      {!data ? (
        <div className="empty-row">Autonomous learning state has not been loaded. Missing stays missing.</div>
      ) : (
        <>
          <p className="panel-copy">{data.note || 'Replay evidence never counts as forward paper evidence. Live money stays locked.'}</p>
          {error ? <p className="panel-copy">{error}</p> : null}
          <div className="inline-actions" style={{ padding: '12px', gap: 8 }}>
            <button type="button" disabled={!!busy} onClick={() => apply(true)}>{data.enabled ? 'ON' : 'Turn ON'}</button>
            <button type="button" className="secondary" disabled={!!busy} onClick={() => apply(false)}>Turn OFF</button>
            {['AUTO', 'FORWARD_PAPER', 'HISTORICAL_REPLAY', 'PAUSED'].map((mode) => (
              <button key={mode} type="button" className={data.mode === mode ? '' : 'secondary'} disabled={!!busy} onClick={() => apply(true, mode)}>
                {mode.replaceAll('_', ' ')}
              </button>
            ))}
          </div>
          <div className="fact-grid">
            <div><span>Mode</span><strong>{data.mode || 'No data'}</strong></div>
            <div><span>Current activity</span><strong>{data.current_activity || data.activity || 'idle'}</strong></div>
            <div><span>Evidence lane</span><strong>{data.evidence_lane || 'NONE'}</strong></div>
            <div><span>Market</span><strong>{data.market_closed ? 'Closed · replay lane' : 'Open · forward paper lane'}</strong></div>
            <div><span>Historical decisions simulated</span><strong>{countValue(counts, 'historical_decisions_simulated')}</strong></div>
            <div><span>Forward paper decisions</span><strong>{countValue(counts, 'forward_paper_decisions')}</strong></div>
            <div><span>Paper trades opened</span><strong>{countValue(counts, 'paper_trades_opened')}</strong></div>
            <div><span>Paper trades settled</span><strong>{countValue(counts, 'paper_trades_settled')}</strong></div>
            <div><span>Correct rejects</span><strong>{countValue(counts, 'correct_rejects')}</strong></div>
            <div><span>Avoided losers</span><strong>{countValue(counts, 'avoided_losers')}</strong></div>
            <div><span>Missed winners</span><strong>{countValue(counts, 'missed_winners')}</strong></div>
            <div><span>Good waits</span><strong>{countValue(counts, 'good_waits')}</strong></div>
            <div><span>False positives</span><strong>{countValue(counts, 'false_positives')}</strong></div>
            <div><span>False negatives</span><strong>{countValue(counts, 'false_negatives')}</strong></div>
            <div><span>Challengers under evaluation</span><strong>{countValue(counts, 'challenger_policies_under_evaluation')}</strong></div>
            <div><span>Active policies</span><strong>{countValue(counts, 'active_policies')}</strong></div>
            <div><span>Rejected policies</span><strong>{countValue(counts, 'rejected_policies')}</strong></div>
            <div><span>Forward evidence</span><strong>{countValue(counts, 'forward_evidence_count')}</strong></div>
            <div><span>Replay evidence</span><strong>{countValue(counts, 'replay_evidence_count')}</strong></div>
          </div>
          <p className="panel-copy">
            Champion: {String(champion.strategy_id || champion.status || 'unavailable')}
            {champion.version ? ` v${String(champion.version)}` : ''}
            {champion.rules_hash ? ` · ${String(champion.rules_hash)}` : ''}
          </p>
          <p className="panel-copy">
            Challenger: {String(challenger.challenger_id || challenger.status || 'none')}
            {data.promotion_eligible ? ' · promotion eligible' : ` · blocked: ${data.promotion_blocked_reason || 'no data'}`}
          </p>
          <p className="panel-copy">Last learning cycle: {data.last_learning_cycle ? compactDateTime(data.last_learning_cycle) : 'No learning cycle has been recorded.'}</p>
          <p className="panel-copy">Next: {data.next_learning_action || 'No next action recorded.'}</p>
          <p className="panel-copy">
            Latest persisted evidence: {String(data.latest_persisted_evidence?.replay_status || 'NONE')}
            {data.latest_persisted_evidence?.replay_period ? ` · ${String(data.latest_persisted_evidence.replay_period)}` : ''}
          </p>
          {(data.missing || []).map((line) => <p className="panel-copy" key={line}>{line}</p>)}
          <div className="inline-actions" style={{ padding: '12px', gap: 8 }}>
            <button type="button" disabled={!!busy} onClick={runReplay}>{busy === 'replay' ? 'Starting replay…' : 'Run historical replay'}</button>
            <button type="button" className="secondary" disabled={!!busy} onClick={runLearning}>{busy === 'learning' ? 'Queueing…' : 'Run learning cycle'}</button>
            <button type="button" className="secondary" disabled={!!busy} onClick={onRefresh}>Refresh</button>
          </div>
        </>
      )}
    </Panel>
  )
}

function ResearchDirectorPanel({ data }: { data: ResearchDirectorStatus | null }) {
  if (!data) {
    return (
      <Panel title="RESEARCH DIRECTOR" subtitle="NO PERSISTED DIRECTOR STATUS">
        <div className="empty-row">Research Director state is unavailable. QuantTerm does not invent a research question.</div>
      </Panel>
    )
  }

  const req = data.evidence_request || {
    request_id: '',
    status: 'NONE',
    gap_kind: '',
    evidence_origin: '',
    allowed_lanes: [],
    current_samples: 0,
    target_samples: 0,
    sample_deficit: 0,
    missing_metrics: [],
    acquisition_tasks: [],
    stop_conditions: [],
  }
  const batch = data.next_evidence_batch || {
    batch_id: '',
    phase: 'IDLE',
    sessions: [],
    selection_policy: '',
    selection_objective: '',
  }
  const delta = data.learning_delta
  const signals = data.signals || {}
  const learned = data.challengers?.learned || {}
  const dossier = learned.promotion_dossier || {}
  const governor = data.resource_governor || {}
  const stateTruth = data.state_truth || {}
  const batchWindow = batch.sessions?.length
    ? `${batch.sessions[0]} → ${batch.sessions[batch.sessions.length - 1]} · ${batch.sessions.length} sessions`
    : 'No targeted batch selected'
  const calibration = data.calibration?.snapshot_id || 'NO SNAPSHOT'
  const requestStatus = req.request_id ? req.status || 'OPEN' : 'NO OPEN REQUEST'
  const changeToday = delta?.measurable_change_today ? 'YES — MEASURABLE' : 'NO MEASURABLE CHANGE'

  return (
    <Panel title="RESEARCH DIRECTOR" subtitle={data.research_phase || 'UNKNOWN'}>
      <div className="fact-grid">
        <div><span>Current activity</span><strong>{data.current_activity || 'UNKNOWN'}</strong></div>
        <div><span>Policy state</span><strong>{data.policy_state || 'UNKNOWN'}</strong></div>
        <div><span>Evidence request</span><strong>{requestStatus}</strong></div>
        <div><span>Evidence lane</span><strong>{req.evidence_origin || 'NONE'}</strong></div>
        <div><span>Samples</span><strong>{req.request_id ? `${delta?.sample_count ?? req.current_samples}/${delta?.target_samples ?? req.target_samples}` : 'NO REQUEST'}</strong></div>
        <div><span>Deficit</span><strong>{req.request_id ? String(delta?.sample_deficit ?? req.sample_deficit) : '—'}</strong></div>
        <div><span>Calibration</span><strong>{calibration}</strong></div>
        <div><span>Signal registry</span><strong>{signals.registry_version || 'NO SNAPSHOT'}</strong></div>
        <div><span>Scanner / forward calibrated</span><strong>{signals.scanner_catalog ?? '—'} / {signals.forward_calibrated ?? '—'}</strong></div>
        <div><span>Learned challenger</span><strong>{learned.status || 'NONE'}</strong></div>
        <div><span>Forward evidence</span><strong>{data.forward_evidence?.status || 'NOT_STARTED'}</strong></div>
        <div><span>Wiser today?</span><strong>{changeToday}</strong></div>
      </div>

      <p className="panel-copy"><strong>Current research question:</strong> {data.current_question || 'No unresolved question persisted.'}</p>
      <p className="panel-copy"><strong>Why this evidence next:</strong> {batch.selection_objective || 'No targeted selection rationale persisted.'}</p>
      <p className="panel-copy">
        <strong>Next batch:</strong> {batchWindow}
        {batch.selection_policy ? ` · ${batch.selection_policy}` : ''}
        {batch.outcome_blind_selection === true ? ' · outcome-blind selection' : ''}
      </p>
      <p className="panel-copy"><strong>Next action:</strong> {data.next_action || 'No explicit next action persisted.'}</p>

      <div className="fact-grid">
        <div><span>Last-batch evidence added</span><strong>{delta?.last_batch_evidence_added ?? 0}</strong></div>
        <div><span>Stagnant batches</span><strong>{delta?.stagnant_batches ?? 0}</strong></div>
        <div><span>Metrics resolved</span><strong>{delta?.resolved_metrics?.length ?? 0}</strong></div>
        <div><span>Metrics unresolved</span><strong>{delta?.unresolved_metrics?.length ?? 0}</strong></div>
        <div><span>Knowledge validated (1d)</span><strong>{delta?.knowledge_validated_1d ?? 0}</strong></div>
        <div><span>Knowledge retired (1d)</span><strong>{delta?.knowledge_retired_1d ?? 0}</strong></div>
      </div>

      {(req.missing_metrics || []).length ? (
        <p className="panel-copy"><strong>Missing metrics:</strong> {req.missing_metrics.join(' · ')}</p>
      ) : null}
      {(req.acquisition_tasks || []).length ? (
        <p className="panel-copy"><strong>Evidence acquisition:</strong> {req.acquisition_tasks.join(' · ')}</p>
      ) : null}
      {(delta?.unresolved_metrics || []).length ? (
        <p className="panel-copy"><strong>Still unresolved:</strong> {delta.unresolved_metrics.join(' · ')}</p>
      ) : null}

      <p className="panel-copy">
        <strong>Challenger dossier:</strong>{' '}
        {String(dossier.decision || 'NO DOSSIER')}
        {learned.model_version ? ` · ${learned.model_version}` : ''}
        {learned.trained_n != null ? ` · trained n=${learned.trained_n}` : ''}
        {learned.real_forward_n != null ? ` · forward n=${learned.real_forward_n}` : ''}
      </p>
      <p className="panel-copy">
        <strong>Resource governor:</strong>{' '}
        {String(governor.decision || 'NO STATUS')}
        {governor.reason ? ` · ${String(governor.reason)}` : ''}
      </p>
      {stateTruth.mismatch ? (
        <div className="api-warning">
          Runtime-state mismatch: {stateTruth.reason || 'persisted policy state does not match durable activity truth'}
        </div>
      ) : null}
      {(data.blockers || []).length ? (
        <div className="api-warning">Research blockers: {data.blockers.join(' · ')}</div>
      ) : null}
      <p className="panel-copy">{data.truth_note || delta?.note || ''}</p>
      <p className="panel-copy"><strong>Live execution:</strong> {liveSafetyLabel(data)}</p>
    </Panel>
  )
}

export function LearningJournalView() {
  const [data, setData] = useState<ResearchStatus | null>(null)
  const [learning, setLearning] = useState<LearningDashboard | null>(null)
  const [soak, setSoak] = useState<ForwardSoakScoreboard | null>(null)
  const [autoLearn, setAutoLearn] = useState<AutonomousLearningDashboard | null>(null)
  const [director, setDirector] = useState<ResearchDirectorStatus | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const refreshLearning = () => {
    fetchAutonomousLearning().then(setAutoLearn).catch(() => undefined)
    fetchLearningDashboard().then(setLearning).catch(() => undefined)
    fetchResearchDirector().then(setDirector).catch(() => undefined)
  }
  useEffect(() => {
    let alive = true
    setLoading(true)
    Promise.allSettled([
      fetchResearchStatus(),
      fetchLearningDashboard(),
      fetchForwardSoak(),
      fetchAutonomousLearning(),
      fetchResearchDirector(),
    ]).then(([status, dash, soakRow, autoRow, directorRow]) => {
      if (!alive) return
      if (status.status === 'fulfilled') setData(status.value)
      else setError(status.reason instanceof Error ? status.reason.message : 'Research status unavailable')
      if (dash.status === 'fulfilled') setLearning(dash.value)
      if (soakRow.status === 'fulfilled') setSoak(soakRow.value)
      if (autoRow.status === 'fulfilled') setAutoLearn(autoRow.value)
      if (directorRow.status === 'fulfilled') setDirector(directorRow.value)
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
      <ResearchDirectorPanel data={director} />
      <AutonomousLearningPanel data={autoLearn || learning?.autonomous_learning || null} onRefresh={refreshLearning} />
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
              <div><span>Live lock</span><strong>{liveSafetyLabel(board)}</strong></div>
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
          <div><span>Live lock</span><strong>{liveSafetyLabel(learning)}</strong></div>
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
          <i className={laneDot(lane.status)} />
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
        <button type="button" onClick={() => void runControl('RUN_CYCLE_NOW')}>Run paper evaluation now</button>
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
  const [caseDecisionId, setCaseDecisionId] = useState('')
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
  const runCase = (symbol: string, as_of: string, alternative?: string, decisionId?: string) => {
    const name = symbol.trim().toUpperCase()
    const day = as_of.trim().slice(0, 10)
    const id = (decisionId || caseDecisionId).trim()
    if (!name || !day) {
      setCaseError('Symbol and historical date are required. No sample decision is invented.')
      return
    }
    setCaseBusy(true)
    setCaseError('')
    simulatePastDecision({ symbol: name, as_of: day, alternative: alternative || caseAlt, decision_id: id || undefined })
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
      <Panel title="HISTORICAL REPLAY" subtitle={`${sim?.engine || 'Production scanner + evaluate_candidate'} · ${sim?.provenance || 'HISTORICAL_REPLAY'} · never writes REAL_FORWARD_MARKET`}>
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
        {sim?.simple ? <p className="panel-copy">{sim.simple}</p> : <p className="panel-copy">Decision Simulation has not been approved for this startup yet. Start it once after reviewing the current best-trade shortlist; the same production thesis is then replayed on historical sessions and used for present paper decisions.</p>}
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
          <button type="button" disabled={simBusy} onClick={runSim}>{simBusy ? (sim?.message || 'Starting simulation…') : 'Start Decision Simulation'}</button>
        </div>
        <p className="panel-copy">{sim?.note || 'Later prices are used only for outcome classification.'}</p>
        <Panel title="ONE PAST DECISION" subtitle="Original decision vs simulated alternative · PIT at T · subsequent bars only for outcome">
          <p className="panel-copy">Replay one persisted QuantTerm decision. Missing journal rows stay UNAVAILABLE. No sample data is substituted.</p>
          <div className="inline-actions" style={{ padding: '12px', gap: 8 }}>
            <input value={caseSymbol} onChange={(event) => setCaseSymbol(event.target.value)} placeholder="Symbol" aria-label="Historical symbol" />
            <input value={caseAsOf} onChange={(event) => setCaseAsOf(event.target.value)} placeholder="YYYY-MM-DD" aria-label="Historical date" />
            <input value={caseDecisionId} onChange={(event) => setCaseDecisionId(event.target.value)} placeholder="decision_id if several that day" aria-label="Decision id" />
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
          {caseState === 'ambiguous' ? (
            <div>
              <p className="panel-copy">{caseSim?.error || 'Multiple persisted decisions match. Select the exact decision_id.'}</p>
              {(caseSim?.matches || []).map((match) => (
                <button
                  key={match.decision_id || `${match.as_of}-${match.decision}`}
                  type="button"
                  className="secondary"
                  disabled={caseBusy}
                  onClick={() => {
                    setCaseDecisionId(match.decision_id || '')
                    runCase(match.symbol || caseSymbol, match.as_of || caseAsOf, caseAlt, match.decision_id)
                  }}
                >
                  {match.decision_id} · {match.decision} · {match.reason_code}
                </button>
              ))}
            </div>
          ) : null}
          {caseSim && caseView ? (
            <>
              <div className="fact-grid">
                <div><span>Original Decision</span><strong>{caseView.originalAction}</strong></div>
                <div><span>Simulated Alternative</span><strong>{caseView.simulatedAction}</strong></div>
                <div><span>Timestamp</span><strong>{displayHonest(caseSim.historical_timestamp)}</strong></div>
                <div><span>Reason</span><strong>{displayHonest(caseSim.original?.reason_code)}</strong></div>
                <div><span>Original entry</span><strong>{displayHonest(caseSim.original?.entry)}</strong></div>
                <div><span>Original entry source</span><strong>{displayHonest(caseSim.original?.entry_source)}</strong></div>
                <div><span>Simulated entry source</span><strong>{displayHonest(caseSim.simulated?.entry_source)}</strong></div>
                <div><span>PIT status</span><strong>{displayHonest(caseSim.pit_status)}</strong></div>
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
                <div className="fact-grid">
                  <div><span>Symbol</span><strong>{row.symbol || 'unavailable'}</strong></div>
                  <div><span>Simulation date</span><strong>{row.as_of || 'unavailable'}</strong></div>
                  <div><span>Decision timestamp</span><strong>{row.decision_timestamp ? compactDateTime(row.decision_timestamp) : (row.as_of || 'unavailable')}</strong></div>
                  <div><span>Data cutoff</span><strong>{row.data_cutoff || row.pit?.max_bar_date || row.as_of || 'unavailable'}</strong></div>
                  <div><span>Market regime</span><strong>{row.regime || 'unavailable'}</strong></div>
                  <div><span>Stock setup</span><strong>{row.setup || row.tier || 'unavailable'}</strong></div>
                  <div><span>Decision</span><strong>{row.decision || 'unavailable'}</strong></div>
                  <div><span>Entry</span><strong>{row.entry == null ? 'unavailable' : String(row.entry)}</strong></div>
                  <div><span>Stop</span><strong>{row.stop == null ? 'unavailable' : String(row.stop)}</strong></div>
                  <div><span>Target</span><strong>{row.target == null ? 'unavailable' : String(row.target)}</strong></div>
                  <div><span>MFE</span><strong>{row.mfe_pct == null ? 'unresolved' : `${row.mfe_pct}%`}</strong></div>
                  <div><span>MAE</span><strong>{row.mae_pct == null ? 'unresolved' : `${row.mae_pct}%`}</strong></div>
                  <div><span>Realized / simulated return</span><strong>{row.forward_return_pct == null ? 'unresolved' : `${row.forward_return_pct}%`}</strong></div>
                  <div><span>Classification</span><strong>{row.classification || row.outcome_status || 'INCONCLUSIVE'}</strong></div>
                </div>
                <p className="panel-copy">Reasons: {(row.reasons || []).join(' · ') || 'No reasons were persisted.'}</p>
                <p className="panel-copy">Rejection reasons: {(row.rejection_reasons || []).join(' · ') || row.reason_code || 'None recorded.'}</p>
                <p className="panel-copy">
                  Data available at decision date: {row.pit?.max_bar_date || row.as_of || 'unknown'}
                  {row.pit?.future_evidence_used ? ' · LOOKAHEAD FLAG' : ' · no future bars'}
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
                      setCaseDecisionId(row.decision_id || '')
                      runCase(row.symbol || '', row.as_of || '', undefined, row.decision_id)
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
