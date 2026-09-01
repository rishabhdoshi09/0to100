import { useEffect, useState } from 'react'
import {
  fetchResearchStatus,
  fetchScanAudit,
  fetchStrategyCatalog,
  fetchSystemHealthContract,
  type HealthLane,
  type ResearchStatus,
  type ScanAuditPayload,
  type StrategyCatalog,
  type SystemHealthContract,
} from './productApi'
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

export function LearningJournalView() {
  const [data, setData] = useState<ResearchStatus | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    fetchResearchStatus()
      .then(setData)
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : 'Research status unavailable'))
  }, [])
  const journal = data?.decision_journal
  return (
    <section className="workspace-view">
      <div className="reco-how">
        <div className="qt-eyebrow">Learning / Decision Journal</div>
        <p>{data?.disclaimer || 'Measurable evidence only. Empty is a valid state.'}</p>
      </div>
      {error ? <div className="api-warning">{error}</div> : null}
      <Panel title="WHAT IS MEASURABLE NOW" subtitle={data?.learning_status || 'UNKNOWN'}>
        {(data?.headlines || []).map((line) => <p className="panel-copy" key={line}>{line}</p>)}
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
  useEffect(() => {
    fetchScanAudit('', 80)
      .then(setData)
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : 'Coverage unavailable'))
  }, [])
  const summary = data?.summary || {}
  const inspect = async () => {
    const clean = query.trim().toUpperCase()
    if (!clean) return
    try {
      setLookup(await fetchScanAudit(clean, 1))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Lookup failed')
    }
  }
  return (
    <section className="workspace-view">
      <div className="reco-how">
        <div className="qt-eyebrow">Scan coverage</div>
        <p>Requested, checked, qualified, no-setup, excluded, and failed are separate. Missing names can be inspected.</p>
      </div>
      {error ? <div className="api-warning">{error}</div> : null}
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
          <button type="button" onClick={() => void inspect()}>Look up</button>
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
  useEffect(() => {
    fetchSystemHealthContract().then(setContract).catch(() => setContract(null))
  }, [dashboard.generated_at])
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
      <Panel title="INDEPENDENT HEALTH LANES" subtitle="No collapsed green light">
        <HealthLanes contract={contract} />
      </Panel>
    </section>
  )
}

export function ProductionBacktestView({ dashboard, setActive }: ViewProps) {
  const [catalog, setCatalog] = useState<StrategyCatalog | null>(null)
  useEffect(() => {
    fetchStrategyCatalog().then(setCatalog).catch(() => setCatalog(null))
  }, [])
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
      <Panel title="PRODUCTION ENSEMBLE" subtitle={catalog?.ensemble.strategy_id || 'QT_RECO_ENSEMBLE'}>
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
