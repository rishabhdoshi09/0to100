import { useEffect, useMemo, useState } from 'react'
import { EmptyState } from './designSystem'
import { EvidenceList, MetricCard, Panel } from './components'

export type StrategyBacktest = {
  status?: string
  reason?: string
  experiment_id?: string | null
  experiment_status?: string | null
  evaluated_at?: string | null
  evidence?: Record<string, unknown> | null
}

export type ProductionStrategy = {
  strategy_id: string
  version: number
  name: string
  category_id: string
  status: string
  holding_period: string
  universe: string
  rules_hash: string
  entry_logic: string[]
  exit_logic: string[]
  risk_assumptions: string[]
  evidence_requirements: string[]
  backtest?: StrategyBacktest
}

type StrategyRegistry = {
  schema_version: number
  production_strategy_count: number
  verified_backtest_parity_count: number
  unverified_backtest_parity_count: number
  strategies: ProductionStrategy[]
  invariant: string
}

type ResearchStatus = {
  schema_version: number
  generated_at: string
  state: string
  production: {
    active_strategies: number
    verified_backtest_parity: number
    unverified_backtest_parity: number
    strategies: ProductionStrategy[]
  }
  experiments: {
    awaiting_validation: number
    promoted: number
    rejected: number
    recently_rejected: string[]
  }
  learning: {
    beliefs_active: number
    beliefs_watch: number
    beliefs_retired: number
    promoted_this_week: number
    retired_this_week: number
    net_knowledge_gain: number | null
    avg_evidence_per_active_belief: number | null
    calibration: Record<string, unknown>
  }
  edge_health: {
    tracked_signals: number
    durable: number
    decaying: number
    dead: number
    recovering: number
    signals_in_drift: string[]
  }
  decisions: {
    surfaced_history: number
    latest_scan_decisions: number
    settled_sample_size: number
    hit_rate_pct?: number | null
    expectancy_pct?: number | null
    average_gain_pct?: number | null
    average_loss_pct?: number | null
    max_drawdown_pct?: number | null
    performance_claim_allowed: boolean
    performance_label: string
  }
  data: {
    total_observations: number
    on_current_schema: boolean
    thin_features: string[]
    stale_values: number
    impossible_values: number
  }
  research_debt: Record<string, unknown>
  blockers: string[]
  invariant: string
}

const readJson = async <T,>(response: Response): Promise<T> => {
  if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`)
  return response.json() as Promise<T>
}

const metric = (evidence: Record<string, unknown> | null | undefined, keys: string[]): string => {
  if (!evidence) return '—'
  for (const key of keys) {
    const value = evidence[key]
    if (value !== null && value !== undefined && value !== '') return String(value)
  }
  return '—'
}

const parityTone = (status?: string) => status === 'VERIFIED' ? 'positive' : 'negative'

export function ResearchBacktestView() {
  const [registry, setRegistry] = useState<StrategyRegistry | null>(null)
  const [research, setResearch] = useState<ResearchStatus | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  const load = async () => {
    setLoading(true)
    try {
      const [nextRegistry, nextResearch] = await Promise.all([
        fetch('/api/strategy-registry', { headers: { Accept: 'application/json' } }).then((r) => readJson<StrategyRegistry>(r)),
        fetch('/api/research-status', { headers: { Accept: 'application/json' } }).then((r) => readJson<ResearchStatus>(r)),
      ])
      setRegistry(nextRegistry)
      setResearch(nextResearch)
      setError('')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Research status is unavailable')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load()
    const timer = window.setInterval(() => void load(), 30_000)
    return () => window.clearInterval(timer)
  }, [])

  const strategies = useMemo(() => registry?.strategies || research?.production.strategies || [], [registry, research])
  const d = research?.decisions

  return (
    <section className="workspace-view">
      {error && <div className="api-warning">{error}</div>}
      <div className="reco-how">
        <div className="qt-eyebrow">Research OS · exact production parity</div>
        <p>
          This is the canonical bridge between today&apos;s recommendation lanes and historical research. Backtest numbers appear only when the exact production strategy version and rules hash match an evaluated experiment. Unmatched history stays UNVERIFIED rather than decorating a live recommendation.
        </p>
      </div>

      <div className="view-metrics">
        <MetricCard label="PRODUCTION STRATEGIES" value={String(registry?.production_strategy_count ?? research?.production.active_strategies ?? 0)} detail="Current recommendation lanes" />
        <MetricCard label="BACKTEST PARITY" value={`${registry?.verified_backtest_parity_count ?? research?.production.verified_backtest_parity ?? 0} VERIFIED`} detail={`${registry?.unverified_backtest_parity_count ?? research?.production.unverified_backtest_parity ?? 0} unverified`} tone={(registry?.unverified_backtest_parity_count ?? 1) === 0 ? 'green' : 'amber'} />
        <MetricCard label="EXPERIMENTS PENDING" value={String(research?.experiments.awaiting_validation ?? 0)} detail={`${research?.experiments.promoted ?? 0} promoted · ${research?.experiments.rejected ?? 0} rejected`} tone="purple" />
        <MetricCard label="SETTLED DECISIONS" value={String(d?.settled_sample_size ?? 0)} detail={d?.performance_claim_allowed ? `Performance evidence available · ${d.performance_label}` : 'No measured performance claim yet'} tone="cyan" />
      </div>

      <Panel title="ACTIVE PRODUCTION STRATEGIES" subtitle="Exact identity first; performance only after version + rules-hash parity">
        {loading && strategies.length === 0 ? <div className="empty-row">Loading strategy registry…</div> : null}
        {!loading && strategies.length === 0 ? <EmptyState title="No canonical production strategies registered" detail="Recommendations cannot claim backtest support until a production strategy identity exists." /> : null}
        {strategies.map((strategy) => {
          const backtest = strategy.backtest || {}
          const evidence = backtest.evidence || null
          return (
            <article className="requirement-card" key={strategy.strategy_id}>
              <div className="requirement-head">
                <div>
                  <h3>{strategy.name}</h3>
                  <p>{strategy.strategy_id} v{strategy.version} · {strategy.holding_period} · {strategy.universe}</p>
                </div>
                <strong className={parityTone(backtest.status)}>BACKTEST PARITY: {backtest.status || 'UNVERIFIED'}</strong>
              </div>
              <div className="requirement-meta">
                <span>Rules hash <strong>{strategy.rules_hash}</strong></span>
                <span>Experiment <strong>{backtest.experiment_id || 'none'}</strong></span>
                <span>Sample <strong>{metric(evidence, ['n_trades', 'sample_size', 'n'])}</strong></span>
                <span>Expectancy <strong>{metric(evidence, ['net_expectancy_R', 'expectancy_R', 'expectancy_pct'])}</strong></span>
                <span>Max drawdown <strong>{metric(evidence, ['max_drawdown', 'max_drawdown_pct'])}</strong></span>
              </div>
              <p className="panel-copy">{backtest.reason || 'No parity statement recorded.'}</p>
              <details>
                <summary>Production rules</summary>
                <EvidenceList title="Entry / qualification" items={strategy.entry_logic || []} tone="green" />
                <EvidenceList title="Risk assumptions" items={strategy.risk_assumptions || []} tone="red" />
                <EvidenceList title="Required evidence" items={strategy.evidence_requirements || []} tone="cyan" />
              </details>
            </article>
          )
        })}
      </Panel>

      <div className="automation-grid">
        <Panel title="RESEARCH PIPELINE" subtitle="Measured experiments, not an AI-learning claim">
          <div className="fact-grid">
            <div><span>Awaiting validation</span><strong>{research?.experiments.awaiting_validation ?? 0}</strong></div>
            <div><span>Promoted</span><strong>{research?.experiments.promoted ?? 0}</strong></div>
            <div><span>Rejected</span><strong>{research?.experiments.rejected ?? 0}</strong></div>
            <div><span>Active beliefs</span><strong>{research?.learning.beliefs_active ?? 0}</strong></div>
            <div><span>Watch beliefs</span><strong>{research?.learning.beliefs_watch ?? 0}</strong></div>
            <div><span>Retired beliefs</span><strong>{research?.learning.beliefs_retired ?? 0}</strong></div>
          </div>
          <EvidenceList title="Recently rejected hypotheses" items={research?.experiments.recently_rejected || []} tone="red" />
        </Panel>

        <Panel title="EDGE & DRIFT" subtitle="What the system can actually measure today">
          <div className="fact-grid">
            <div><span>Tracked signals</span><strong>{research?.edge_health.tracked_signals ?? 0}</strong></div>
            <div><span>Durable</span><strong>{research?.edge_health.durable ?? 0}</strong></div>
            <div><span>Decaying</span><strong>{research?.edge_health.decaying ?? 0}</strong></div>
            <div><span>Dead</span><strong>{research?.edge_health.dead ?? 0}</strong></div>
            <div><span>Recovering</span><strong>{research?.edge_health.recovering ?? 0}</strong></div>
            <div><span>Observations</span><strong>{research?.data.total_observations ?? 0}</strong></div>
          </div>
          <EvidenceList title="Signals in drift" items={research?.edge_health.signals_in_drift || []} tone="red" />
        </Panel>

        <Panel title="DECISION JOURNAL PERFORMANCE" subtitle="Tracked/paper outcomes only; not broker-verified live P&L">
          {d?.performance_claim_allowed ? (
            <div className="fact-grid">
              <div><span>Sample</span><strong>{d.settled_sample_size}</strong></div>
              <div><span>Hit rate</span><strong>{d.hit_rate_pct == null ? '—' : `${d.hit_rate_pct}%`}</strong></div>
              <div><span>Expectancy</span><strong>{d.expectancy_pct == null ? '—' : `${d.expectancy_pct}%`}</strong></div>
              <div><span>Average gain</span><strong>{d.average_gain_pct == null ? '—' : `${d.average_gain_pct}%`}</strong></div>
              <div><span>Average loss</span><strong>{d.average_loss_pct == null ? '—' : `${d.average_loss_pct}%`}</strong></div>
              <div><span>Max drawdown</span><strong>{d.max_drawdown_pct == null ? '—' : `${d.max_drawdown_pct}%`}</strong></div>
            </div>
          ) : <EmptyState title="No settled performance sample" detail="QuantTerm will not show a hit-rate or expectancy claim until tracked decisions have actually settled." />}
        </Panel>

        <Panel title="WHAT NEEDS ATTENTION" subtitle={research?.state || 'UNKNOWN'}>
          <EvidenceList title="Current blockers" items={research?.blockers || []} tone={(research?.blockers || []).length ? 'red' : 'green'} />
          <p className="panel-copy">{registry?.invariant || research?.invariant || ''}</p>
          <div className="inline-actions"><button type="button" onClick={() => void load()}>Refresh research status</button></div>
        </Panel>
      </div>
    </section>
  )
}
