import { useEffect, useMemo, useRef, useState } from 'react'
import { ChartWorkspace, Panel } from './components'
import { deskStartupReason } from './deskStartupState'
import { money, pct, relativeAge, words } from './format'
import {
  addWatchlistItem,
  bootstrapProduct,
  fetchCompareWorkspace,
  fetchProductReadiness,
  fetchRadarHome,
  fetchScannerWorkspace,
  fetchTradePlan,
  fetchWatchlist,
  verifyForwardSoakNow,
  fetchDecisionSimulator,
  simulatePastDecisions,
  type HomeAction,
  type HomeOperatingSystem,
  removeWatchlistItem,
  type CompareWorkspace,
  type DeskPipeline,
  type DeskPipelineStep,
  type ProductReadiness,
  type RadarHome,
  type ScannerWorkspaceRow,
  type TradePlan,
  type WatchlistPayload,
} from './productApi'
import { RiskLensCard } from './productViews'
import type { ControlName } from './types'
import { LiveScanBanner, type ExperienceViewProps } from './experience'
import { keepRicher, markInvestigate, recall, remember } from './sessionMemory'
import { SystemLaneInspector, SystemLaneStrip } from './homeSystemInspector'
import type { CheckSystemSnapshot, SystemLane } from './backendControlPlane'
import { DailyWrapList, magazineWrapLines } from './dailyWrap'
import {
  bestSetupsFromRadar,
  dashCell,
  scannerEmptyHint,
  scannerFallbackRows,
  scannerMetaFromDashboard,
} from './scannerFallback'

type RadarRow = ScannerWorkspaceRow & {
  breakout_state?: string
  momentum_state?: string
  setup_label?: string
  freshness?: string
  change_5d_pct?: number
  relative_strength?: number
  risk_label?: string
  reason?: string
  company?: string
  classification?: string
  combined_score?: number
  why?: string
  sepa_score?: number
  sepa_passed?: number
  sepa_total?: number
  sepa_verdict?: string
  breakout_grade?: string
  breakout_conviction?: number
  breakout_quality?: number
  fundamental_score?: number
  sniper_candidate?: boolean
  volume_ratio?: number
  rsi?: number
  tech_source?: string
  price_tag?: string
  best_among_why?: string
  sepa_used?: boolean
  funds_used?: boolean
  rank?: number
  action_badge?: string
  risk_tier?: string
  entry?: number | null
  target?: number | null
  stop?: number | null
  cmp?: number | null
  upside_from_entry_pct?: number | null
  upside_to_target_pct?: number | null
  best_of_best_score?: number
  best_of_best_parts?: { sepa?: number; funds?: number; tape?: number; composite?: number }
}

const breakoutLabel: Record<string, string> = {
  confirmed_breakout: 'Confirmed',
  near_breakout: 'Near breakout',
  breakout_under_observation: 'Under observation',
  breakout_without_volume: 'No volume confirm',
  insufficient_confirmation: 'Needs confirmation',
  extended_after_breakout: 'Extended',
  failed_breakout: 'Failed',
  failed_or_extended: 'Failed / extended',
  insufficient_data: 'Insufficient data',
  not_in_breakout_lane: '—',
}

const momentumLabel: Record<string, string> = {
  strong_actionable: 'Strong · actionable',
  strong_but_extended: 'Strong · extended',
  steady_leadership: 'Steady leadership',
  improving: 'Improving',
  weakening: 'Weakening',
  high_volatility_momentum: 'High-vol momentum',
  insufficient_history: 'Short history',
  watch_momentum: 'Watch',
  not_momentum: '—',
}

const DESK_PIPELINE_FALLBACK: DeskPipelineStep[] = [
  { id: 'prices', title: 'Official prices', page: 'Home', why: 'Download bhavcopy history so charts and the market scan have bars.', state: 'waiting' },
  { id: 'scan', title: 'Market scan', page: 'Home', why: 'One whole-market scan for Home, Scanner and recommendation setups.', state: 'waiting' },
  { id: 'long_term', title: 'Long-term / funds', page: 'Recommendations', why: 'Fundamentals for Best Among and Wealth Builders.', state: 'waiting' },
  { id: 'news', title: 'Market reports', page: 'Market Reports', why: 'Street pulse and news for Market Reports.', state: 'waiting' },
  { id: 'investigate', title: 'Investigate acquire', page: 'Stock Intelligence', why: 'Download filings and fundamentals for shortlisted names, then Investigate reads the files.', state: 'waiting' },
]

const PIPELINE_STATE_LABEL: Record<string, string> = {
  ready: 'Ready',
  running: 'Downloading',
  queued: 'Queued',
  waiting: 'Waiting',
  failed: 'Failed',
  skipped_failed: 'Skipped',
}

function DeskPipelineStrip({ pipeline }: { pipeline?: DeskPipeline | null }) {
  const steps = pipeline?.steps?.length ? pipeline.steps : DESK_PIPELINE_FALLBACK
  return (
    <div className="radar-pipeline-wrap">
      <div className="radar-pipeline" aria-label="Desk downloads, one at a time">
        {steps.map((step, index) => {
          const state = step.state || 'waiting'
          const active = state === 'running' || state === 'queued'
          return (
            <div key={step.id} className={`radar-pipeline-step is-${state}${active ? ' is-active' : ''}`}>
              <span>{index + 1}. {step.page}</span>
              <strong>{step.title}</strong>
              <small>{PIPELINE_STATE_LABEL[state] || state}{step.why && active ? ` · ${step.why}` : ''}</small>
            </div>
          )
        })}
      </div>
      {pipeline?.message ? <p className="radar-pipeline-msg">{pipeline.message}</p> : (
        <p className="radar-pipeline-msg">Downloads run one at a time: prices, then the shared market scan, then long-term funds, then market reports. Scan Now still refreshes the scan on demand.</p>
      )}
    </div>
  )
}

function BestSniperPanel({
  best,
  sniperCount,
  onSelect,
}: {
  best: RadarRow | null | undefined
  sniperCount: number
  onSelect: (symbol: string) => void
}) {
  if (best) {
    const volOk = best.volume_ratio == null || Number(best.volume_ratio) >= 1
    return (
      <div className="radar-best-breakout">
        <Panel
          title={`BEST TECHNICAL BREAKOUT · ${best.symbol}`}
          subtitle={
            [
              sniperCount > 0 ? `${sniperCount} sniper candidate${sniperCount === 1 ? '' : 's'}` : null,
              best.breakout_grade ? `Grade ${best.breakout_grade}` : null,
              best.rsi != null
                ? `RSI ${Math.round(Number(best.rsi))}${best.tech_source === 'live' || best.price_tag === 'LIVE' ? ' LIVE' : ' EOD'}`
                : null,
              best.volume_ratio != null
                ? `Vol ${Number(best.volume_ratio).toFixed(1)}×${volOk ? '' : ' THIN'}`
                : null,
            ].filter(Boolean).join(' · ') || 'Volume floor · not chasing · RSI ≤82 — tape only, not SEPA'
          }
        >
          <button
            type="button"
            className="radar-best-pick-btn"
            onClick={() => onSelect(String(best.symbol || ''))}
          >
            Score {best.score ?? '—'}
            {best.breakout_quality != null
              ? ` · Quality ${Number(best.breakout_quality).toFixed(0)}`
              : ''}
            {' · '}
            {breakoutLabel[String(best.breakout_state || '')]
              || words(String(best.breakout_state || best.status || ''))}
          </button>
        </Panel>
      </div>
    )
  }
  return (
    <div className="radar-best-breakout radar-best-empty">
      <Panel
        title="BEST TECHNICAL BREAKOUT"
        subtitle="Volume floor · not extended · RSI ≤82 — tape only, SEPA is not used here"
      >
        <p className="radar-empty-li">
          {sniperCount === 0
            ? 'No sniper breakouts yet — thin volume / extended names stay out.'
            : 'Sniper pool has names but none ranked as technical best.'}
        </p>
      </Panel>
    </div>
  )
}

function RankingLegend({
  legend,
}: {
  legend?: RadarHome['ranking_legend']
}) {
  const items = [
    { key: 'best_among_breakouts', title: 'Best among the best', body: legend?.best_among_breakouts || 'Sniper plus SEPA overlay and/or long-term funds. 0.45 SEPA · 0.30 funds · 0.25 tape.' },
    { key: 'best_setups', title: 'SEPA overlay', body: legend?.best_setups || 'SEPA 7-rule overlay ≥40/100. Not a buy.' },
    { key: 'best_technical_breakout', title: 'Tape only', body: legend?.best_technical_breakout || 'Sniper tape rank. SEPA is not used here.' },
  ]
  return (
    <div className="radar-rank-legend">
      {items.map((item) => (
        <div key={item.key}>
          <span>{item.title}</span>
          <p>{item.body}</p>
        </div>
      ))}
    </div>
  )
}

function BestOfBestHero({
  row,
  note,
  onSelect,
}: {
  row: RadarRow | null | undefined
  note?: string
  onSelect: (symbol: string) => void
}) {
  if (!row) {
    return (
      <div className="radar-bob-hero radar-best-empty">
        <Panel title="BEST AMONG THE BEST" subtitle="Sniper plus SEPA overlay ≥40 and/or usable long-term funds. Not a buy.">
          <p className="radar-empty-li">
            {note || 'No sniper has a second screen yet. Tape lane below is independent.'}
          </p>
        </Panel>
      </div>
    )
  }
  const parts = row.best_of_best_parts || {}
  const upside = row.upside_to_target_pct ?? row.upside_from_entry_pct
  const risk = String(row.risk_tier || 'Medium').toLowerCase()
  return (
    <article className="radar-bob-hero">
      <button type="button" className="radar-bob-hit" onClick={() => onSelect(String(row.symbol || ''))}>
        <div className="radar-bob-row1">
          <span className="reco-buy is-watch">Candidate</span>
          <span className="reco-opp">Best among the best</span>
          <span className={`reco-risk-chip ${risk}`}>{row.risk_tier || 'Medium'} Risk</span>
        </div>
        <h3>{row.company && row.company !== row.symbol ? row.company : row.symbol}</h3>
        <p className="radar-bob-sub">
          {row.symbol}
          {row.sepa_score != null ? ` · SEPA ${row.sepa_score}/100` : ''}
          {row.best_among_why ? ` · ${row.best_among_why}` : ''}
        </p>
        <div className="radar-bob-kpis">
          <div><span>Entry</span><strong>{row.entry != null ? money(row.entry, 2) : '—'}</strong></div>
          <div><span>Target</span><strong>{row.target != null ? money(row.target, 2) : '—'}</strong></div>
          <div><span>Stop</span><strong>{row.stop != null ? money(row.stop, 2) : '—'}</strong></div>
          <div>
            <span>Upside to target</span>
            <strong>{upside != null ? pct(upside) : '—'}</strong>
          </div>
        </div>
        <div className="radar-bob-weights">
          <span>SEPA {parts.sepa ?? '—'}</span>
          <span>Funds {parts.funds ?? '—'}</span>
          <span>Tape {parts.tape != null ? Number(parts.tape).toFixed(0) : '—'}</span>
          <span>Score {row.best_of_best_score ?? '—'}</span>
        </div>
      </button>
      {note ? <p className="radar-rank-note">{note}</p> : null}
    </article>
  )
}

function TopStocksList({
  rows,
  selected,
  onSelect,
}: {
  rows: RadarRow[]
  selected: string
  onSelect: (symbol: string) => void
}) {
  return (
    <section className="radar-top-stocks">
      <header>
        <span>TOP STOCKS</span>
        <strong>{rows.length ? `${rows.length} ranked` : 'no second-screen names'}</strong>
      </header>
      <p className="radar-rank-note">
        Numbered by independent confirms, then 0.45·SEPA + 0.30·funds + 0.25·tape. Not sorted by today’s %. Not a buy.
      </p>
      {rows.length === 0 ? (
        <p className="radar-empty-li">Nothing cleared sniper plus a second screen on this scan.</p>
      ) : (
        <ol>
          {rows.map((row) => {
            const change = row.change_5d_pct
            const up = change != null && Number(change) >= 0
            return (
              <li key={row.symbol}>
                <button
                  type="button"
                  className={selected === row.symbol ? 'active' : ''}
                  onClick={() => onSelect(String(row.symbol || ''))}
                >
                  <em>{row.rank ?? ''}</em>
                  <b>{row.symbol}</b>
                  <span className="radar-top-meta">
                    {row.sepa_score != null ? `SEPA ${row.sepa_score}` : 'SEPA —'}
                    {row.funds_used && row.fundamental_score != null ? ` · fund ${Math.round(Number(row.fundamental_score))}` : ''}
                  </span>
                  <strong className="radar-top-px">{row.price != null || row.cmp != null ? money((row.price ?? row.cmp) as number) : '—'}</strong>
                  <span className={`radar-top-chg ${change == null ? '' : up ? 'up' : 'down'}`}>
                    {change == null ? '—' : pct(Number(change))}
                  </span>
                </button>
              </li>
            )
          })}
        </ol>
      )}
    </section>
  )
}

function thinVolume(row: RadarRow): boolean {
  const vol = Number(row.volume_ratio)
  return Number.isFinite(vol) && vol > 0 && vol < 1
}

function DenseTable({
  rows,
  selected,
  onSelect,
  depth,
  mode,
  emptyHint,
}: {
  rows: RadarRow[]
  selected: string
  onSelect: (symbol: string) => void
  depth: ExperienceViewProps['depth']
  mode: string
  emptyHint: string
}) {
  const [sortKey, setSortKey] = useState('score')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')

  const sorted = useMemo(() => {
    const copy = [...rows]
    copy.sort((a, b) => {
      const av = (a as Record<string, unknown>)[sortKey]
      const bv = (b as Record<string, unknown>)[sortKey]
      const an = typeof av === 'number' ? av : String(av ?? '')
      const bn = typeof bv === 'number' ? bv : String(bv ?? '')
      if (typeof an === 'number' && typeof bn === 'number') {
        return sortDir === 'asc' ? an - bn : bn - an
      }
      return sortDir === 'asc'
        ? String(an).localeCompare(String(bn))
        : String(bn).localeCompare(String(an))
    })
    return copy
  }, [rows, sortKey, sortDir])

  const toggleSort = (key: string) => {
    if (sortKey === key) setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    else { setSortKey(key); setSortDir('desc') }
  }

  const cols = mode === 'Long-Term'
    ? ['symbol', 'classification', 'combined_score', 'sector', 'coverage_pct', 'risk_label']
    : depth === 'professional'
      ? ['symbol', 'price', 'setup_label', 'sector', 'decision', 'entry', 'stop', 'target', 'why']
      : ['symbol', 'price', 'setup_label', 'sector', 'decision', 'entry', 'stop', 'target', 'why']

  return (
    <div className="radar-table-wrap">
      <table className="radar-table">
        <thead>
          <tr>
            {cols.map((col) => (
              <th key={col} onClick={() => toggleSort(col)}>{words(col.replace(/_/g, ' '))}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.length === 0 && (
            <tr><td colSpan={cols.length} className="radar-empty">{emptyHint}</td></tr>
          )}
          {sorted.map((row) => (
            <tr key={row.symbol} className={selected === row.symbol ? 'selected' : ''} onClick={() => onSelect(row.symbol)}>
              {cols.map((col) => {
                const raw = (row as Record<string, unknown>)[col]
                let cell: string
                if (col === 'breakout_state') {
                  const key = dashCell(raw)
                  cell = key === '—' ? '—' : (breakoutLabel[key] || words(key))
                } else if (col === 'momentum_state') {
                  const key = dashCell(raw)
                  cell = key === '—' ? '—' : (momentumLabel[key] || words(key))
                }                 else if (col === 'price' || col === 'entry' || col === 'stop' || col === 'target') cell = money(raw as number)
                else if (col === 'change_5d_pct') cell = pct(raw as number)
                else if (col === 'decision') cell = dashCell(raw || (row as RadarRow).decision)
                else if (col === 'combined_score' || col === 'relative_strength') cell = raw != null && raw !== '' ? String(raw) : '—'
                else cell = dashCell(raw)
                return <td key={col}>{cell}</td>
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function HomeOsCard({
  os,
  depth,
  busy,
  onAction,
  onOpenPage,
  setSelected,
}: {
  os: HomeOperatingSystem
  depth: string
  busy: boolean
  onAction: (action: HomeAction) => void
  onOpenPage?: (page: string) => void
  setSelected?: (symbol: string) => void
}) {
  const system = (os.system || {}) as Record<string, SystemLane>
  const [openLane, setOpenLane] = useState<string | null>(null)
  const checkAction = os.check_system?.action || { id: 'CHECK_SYSTEM', control: 'CHECK_SYSTEM', label: 'Check system', kind: 'refresh' }
  const selectLane = (id: string) => setOpenLane(id || null)
  return (
    <section className={`home-os-card state-${(os.state || '').toLowerCase()}`}>
      <div className="home-os-hero">
        <span>WHAT SHOULD I DO?</span>
        {os.runtime?.lifecycle ? (
          <p className="panel-copy">
            {os.runtime.lifecycle}
            {` · ${deskStartupReason({
              lifecycle: os.runtime.lifecycle,
              reason: os.runtime.reason,
              reasons: os.runtime.reasons,
              components: os.runtime.components,
              state: os.runtime.lifecycle === 'READY' ? 'READY' : 'STARTING',
            })}`}
          </p>
        ) : null}
        <h2>{os.headline}</h2>
        <p>{os.subtext}</p>
        <div className="home-os-now-next">
          <div><span>Now</span><strong>{os.now || '—'}</strong></div>
          <div><span>Next</span><strong>{os.next || '—'}</strong></div>
        </div>
        {os.progress?.total ? (
          <p className="panel-copy">
            {os.progress.label || 'Working'}
            {os.progress.current != null ? ` · ${os.progress.current} / ${os.progress.total}` : ''}
          </p>
        ) : null}
        <div className="home-os-actions">
          {os.primary_action ? (
            <button type="button" disabled={busy} onClick={() => onAction(os.primary_action as HomeAction)}>
              {os.primary_action.label}
            </button>
          ) : (
            <em>Nothing to click. Leave it running.</em>
          )}
          {(os.secondary_actions || []).map((action) => (
            <button key={action.label} type="button" className="secondary" disabled={busy} onClick={() => onAction(action)}>
              {action.label}
            </button>
          ))}
        </div>
        {os.primary_action?.kind === 'instruction' && os.primary_action.instruction ? (
          <p className="panel-copy">{os.primary_action.instruction}</p>
        ) : null}
        <div className="home-os-quick" aria-label="Open the next desk page">
          {[
            ['Opportunities', 'Recommendations'],
            ['Stock Intelligence', 'Stock Intelligence'],
            ['Portfolio', 'Portfolio'],
            ['Learning', 'Learning'],
          ].map(([label, page]) => (
            <button
              key={page}
              type="button"
              className="home-os-quick-link"
              onClick={() => onOpenPage?.(page)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
      <div className="home-os-grid">
        <div>
          <span>SYSTEM</span>
          <strong>{os.headline}</strong>
          <small>{os.subtext}</small>
        </div>
        <div>
          <span>AUTONOMY</span>
          <strong>{os.runtime?.lifecycle || os.now || os.state}</strong>
          <small>{os.next || os.runtime?.reason || 'Leave it running'}</small>
        </div>
        <div>
          <span>NEEDS YOU</span>
          <strong>{os.need_me ? 'Yes' : 'No'}</strong>
          <small>
            {os.need_me
              ? (os.primary_action?.instruction || os.primary_action?.label || 'Operator action required')
              : 'No genuine operator intervention'}
          </small>
        </div>
        <div>
          <span>LIVE MONEY</span>
          <strong>Locked</strong>
          <small>Paper only. No live buy button.</small>
        </div>
      </div>
      {(os.recent_activity || []).length ? (
        <div className="home-os-past">
          <span>WHAT QUANTTERM DID</span>
          <strong>{os.now || 'Recent automatic work'}</strong>
          <ul className="home-os-activity">
            {(os.recent_activity || []).slice(0, 8).map((row, index) => (
              <li key={`${row.text}-${index}`}>
                {row.at ? <time>{row.at.slice(11, 16) || row.at.slice(0, 10)}</time> : <time>—</time>}
                <span>{row.text}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {(os.opportunities || []).length ? (
        <div className="home-os-opps">
          <span>OPPORTUNITIES</span>
          {(os.opportunities || []).slice(0, 6).map((row, index) => {
            const symbol = String(row.found || '').split(/\s+/)[0]
            return (
            <div key={`${row.found}-${index}`}>
              <span>{row.label || row.action || 'WAIT / research'}</span>
              <strong>{row.found}</strong>
              <small>{depth === 'professional' ? (row.technical || row.meaning) : row.meaning}</small>
              {symbol && onOpenPage ? (
                <button
                  type="button"
                  className="home-os-inspect-link"
                  onClick={() => {
                    setSelected?.(symbol)
                    onOpenPage('Stock Intelligence')
                  }}
                >
                  Research
                </button>
              ) : null}
            </div>
            )
          })}
        </div>
      ) : (
        <div className="home-os-past">
          <span>OPPORTUNITIES</span>
          <strong>None ready</strong>
          <small>No BUY / READY names. WAIT and research stay in the committee journal.</small>
        </div>
      )}
      <div className="home-os-grid">
        <div>
          <span>PORTFOLIO</span>
          <strong>{os.paper_bot?.paused ? 'PAUSED' : 'ON'}</strong>
          <small>
            {os.observe_only ? 'Observe only today · paper still runs · ' : ''}
            {os.paper_bot?.positions_open ?? 0} open · heat {String(os.paper_bot?.risk_used ?? 'n/a')}
            {os.paper_bot?.why ? ` · ${os.paper_bot.why}` : ''}
          </small>
        </div>
        <div>
          <span>LEARNING</span>
          <strong>{os.learning?.insufficient_evidence ? 'Too early to judge' : (os.learning?.simple || 'Collecting')}</strong>
          <small>
            {depth === 'professional'
              ? `REAL_FORWARD_N ${os.learning?.real_forward_n ?? 0} · coverage ${os.learning?.execution_adjusted_coverage_pct ?? 'n/a'}`
              : os.learning?.simple}
          </small>
        </div>
        <div>
          <span>TODAY</span>
          <strong>{os.today?.market_open ? 'Market open' : 'Market closed'}</strong>
          <small>{os.today?.market_mood || os.today?.market_phase || '—'}</small>
        </div>
        <div>
          <span>PAPER BOT</span>
          <strong>{os.paper_bot?.positions_open ?? 0} open</strong>
          <small>{os.paper_bot?.todays_entries ?? 0} entries · {os.paper_bot?.exits ?? 0} exits</small>
        </div>
      </div>
      {os.past_decisions?.available ? (
        <div className="home-os-past">
          <span>PAST DECISION TEST</span>
          <strong>{os.past_decisions.simple || `${os.past_decisions.decisions_tested || 0} historical decisions tested`}</strong>
          <small>
            {os.past_decisions.filters_helped?.length ? `Helped: ${os.past_decisions.filters_helped.join(', ')}` : 'BACKTEST only — not promotion evidence'}
            {depth === 'professional' && os.past_decisions.would_take != null
              ? ` · take ${os.past_decisions.would_take} · reject ${os.past_decisions.rejected ?? 0} · correct ${os.past_decisions.correct_rejections ?? 0} · missed ${os.past_decisions.missed_winners ?? 0}`
              : ''}
          </small>
        </div>
      ) : null}
      <div className="home-os-system-head">
        <span>SYSTEM</span>
        <button
          type="button"
          className="home-os-check"
          disabled={busy}
          onClick={() => {
            setOpenLane('check_system')
            onAction(checkAction)
          }}
        >
          Check system
        </button>
      </div>
      <SystemLaneStrip
        system={system}
        selected={openLane && openLane !== 'check_system' ? openLane : null}
        onSelect={selectLane}
      />
      {openLane ? (
        <SystemLaneInspector
          laneId={openLane}
          lane={openLane === 'check_system' ? undefined : system[openLane]}
          depth={depth}
          busy={busy}
          liveLocked={os.live_locked !== false}
          checkSystem={os.check_system as CheckSystemSnapshot | undefined}
          system={system}
          onAction={onAction}
          onOpenPage={onOpenPage}
          onClose={() => setOpenLane(null)}
        />
      ) : null}
      {os.yesterday ? (
        <p className="panel-copy">
          Yesterday:
          {os.yesterday.scan ? ' scan' : ' scan pending'}
          {os.yesterday.paper_decisions ? ' · paper decisions' : ' · paper pending'}
          {os.yesterday.settlement_pending ? ' · settlement pending' : os.yesterday.settlement ? ' · settlement' : ''}
          {os.yesterday.learning ? ' · learning' : ''}
          {os.yesterday.forward_evidence ? ' · forward evidence' : ''}
        </p>
      ) : null}
      <details className="home-os-why">
        <summary>Why?</summary>
        <p>{os.four_questions?.what}</p>
        <p>{os.four_questions?.found}</p>
        <p>{os.four_questions?.meaning}</p>
        <p>{os.four_questions?.action}</p>
      </details>
      {depth === 'professional' ? (
        <details className="home-os-why">
          <summary>Technical details</summary>
          <p>state={os.state} · soak={os.learning?.forward_soak_status} · live_locked={String(os.live_locked)}</p>
          <p>
            Expected session: {os.today?.expected_session || os.history_freshness?.expected_latest_completed_session || '—'}
          </p>
          <p>
            Available session: {os.today?.available_session || os.history_freshness?.available_session || '—'}
          </p>
          <p>
            stale sessions={os.today?.stale_sessions ?? os.history_freshness?.stale_sessions ?? 'n/a'}
            {' · '}
            reason={os.today?.history_reason_code || os.history_freshness?.reason_code || '—'}
          </p>
          <p>verify: {JSON.stringify(os.verify?.lanes || {})}</p>
        </details>
      ) : null}
    </section>
  )
}

export function RadarHomeView(props: ExperienceViewProps & {
  onCompare: (symbol: string) => void
  onWatchlist: (symbol: string) => void
}) {
  const { dashboard, selected, setSelected, bars, setActive, depth, marketScan, longTermScan, runControl, onCompare, onWatchlist } = props
  const [radar, setRadar] = useState<RadarHome | null>(() => recall<RadarHome>('radar-home') ?? null)
  const [plan, setPlan] = useState<TradePlan | null>(null)
  const [readiness, setReadiness] = useState<ProductReadiness | null>(() => recall<ProductReadiness>('product-readiness') ?? null)
  const [bootstrapBusy, setBootstrapBusy] = useState(false)
  const [deskNote, setDeskNote] = useState('')
  const [radarNote, setRadarNote] = useState('')
  const autoBootRef = useRef(false)
  const radarInFlight = useRef(false)

  useEffect(() => {
    let alive = true
    const load = () => {
      if (radarInFlight.current) return
      radarInFlight.current = true
      if (!recall('radar-home')) setRadarNote('Refreshing home workspace…')
      fetchRadarHome()
        .then((payload) => {
          const kept = keepRicher('radar-home', payload, (row) => {
            const counts = (row.counts?.breakouts || 0) + (row.counts?.momentum || 0) + (row.counts?.long_term_picks || 0)
            return counts === 0 && !(row.best_setups || []).length && !row.best_breakout
          })
          if (alive) {
            setRadar(kept)
            setRadarNote('')
          }
        })
        .catch((reason: unknown) => {
          if (!alive) return
          if (!recall('radar-home')) setRadar(null)
          setRadarNote(reason instanceof Error ? reason.message : 'Home workspace timed out. Dashboard below still works.')
        })
        .finally(() => { radarInFlight.current = false })
      fetchProductReadiness()
        .then((payload) => {
          remember('product-readiness', payload)
          if (alive) setReadiness(payload)
        })
        .catch(() => undefined)
    }
    load()
    const watching = marketScan.isActive
      || Boolean(dashboard.scan_progress?.active)
      || (dashboard.operations.active || []).some((item) => (
        item.kind === 'DATA_PREPARE'
        || item.kind === 'FNO_REFRESH'
        || item.kind === 'MARKET_SCAN'
        || item.kind === 'LONG_TERM_SCAN'
        || item.kind === 'LONG_TERM_REFRESH'
        || item.kind === 'NEWS_REFRESH'
      ))
    const timer = window.setInterval(load, watching ? 4000 : 20_000)
    return () => { alive = false; window.clearInterval(timer) }
  }, [dashboard.scan.scanned_at, dashboard.long_term.scanned_at, dashboard.generated_at, dashboard.scan_progress?.updated_at, dashboard.operations.active, marketScan.isActive])

  useEffect(() => {
    if (!selected) { setPlan(null); return }
    let alive = true
    fetchTradePlan(selected)
      .then((payload) => { if (alive) setPlan(payload) })
      .catch(() => { if (alive) setPlan(null) })
    return () => { alive = false }
  }, [selected, dashboard.scan.scanned_at, dashboard.generated_at])

  const scanAt = radar?.scan_scanned_at || dashboard.scan.scanned_at || ''
  const brokerStatus = String(radar?.home_os?.broker?.status || '').toUpperCase()
  const kiteOk = brokerStatus === 'READY'
  const kiteLoginOptional = Boolean(radar?.home_os?.broker?.login_required)
  const telegram = radar?.telegram || dashboard.autonomy.telegram
  const telegramOn = Boolean(telegram?.configured)
  const telegramWarn = telegramOn && telegram?.state !== 'live' && telegram?.state !== 'scan_sent'
  const breakoutRows = ((radar?.lanes.breakouts?.length
    ? radar.lanes.breakouts
    : scannerFallbackRows('Breakouts', dashboard)) || []) as RadarRow[]
  const momentumRows = ((radar?.lanes.momentum?.length
    ? radar.lanes.momentum
    : scannerFallbackRows('Momentum', dashboard)) || []) as RadarRow[]
  const longTermRows = ((radar?.lanes.long_term_picks?.length
    ? radar.lanes.long_term_picks
    : scannerFallbackRows('Long-Term', dashboard)) || []) as RadarRow[]
  const breakoutCount = radar?.counts.breakouts || breakoutRows.length
  const momentumCount = radar?.counts.momentum || momentumRows.length
  const longTermCount = radar?.counts.long_term_picks || longTermRows.length
  const emptyDesk = !scanAt && dashboard.scan.records.length === 0 && longTermRows.length === 0
  const readinessScore = readiness?.score ?? 0
  const needsBootstrap = emptyDesk || readinessScore < 70 || !dashboard.data.ready

  const runBootstrap = async () => {
    setBootstrapBusy(true)
    setDeskNote('Starting the next desk download…')
    try {
      const result = await bootstrapProduct()
      setReadiness(result.readiness)
      setDeskNote(result.message || (result.queued_kind
        ? `Queued ${result.queued_kind} — the next step waits until this finishes.`
        : 'Desk data is current.'))
    } catch (reason) {
      setDeskNote(reason instanceof Error ? reason.message : 'Could not start data lanes')
    } finally {
      setBootstrapBusy(false)
      window.setTimeout(() => setDeskNote(''), 6000)
    }
  }

  useEffect(() => {
    if (autoBootRef.current) return
    const cached = recall<RadarHome>('radar-home')
    const cachedCount = (cached?.counts.breakouts || 0) + (cached?.counts.momentum || 0) + (cached?.counts.long_term_picks || 0)
    if (cachedCount > 0 || dashboard.scan.scanned_at) {
      autoBootRef.current = true
      return
    }
    const deskNeedsWork = emptyDesk || !dashboard.data.ready
    if (!deskNeedsWork) return
    autoBootRef.current = true
    void runBootstrap()
  }, [emptyDesk, dashboard.data.ready, dashboard.scan.scanned_at])

  const laneCard = (title: string, rows: RadarRow[], count: number, qualityHint?: number) => (
    <section className="radar-lane-card">
      <header>
        <span>{title}</span>
        <strong>
          {count}
          {qualityHint != null ? ` · ${qualityHint} sniper` : ''}
        </strong>
      </header>
      <ul>
        {rows.slice(0, 6).map((item) => {
          const thin = thinVolume(item)
          return (
          <li key={item.symbol}>
            <button
              type="button"
              className={[selected === item.symbol ? 'active' : '', thin ? 'thin-volume' : ''].filter(Boolean).join(' ')}
              onClick={() => { setSelected(item.symbol); markInvestigate(item.symbol); setActive('Stock Intelligence') }}
            >
              <b>
                {item.symbol}
                {item.sniper_candidate ? <em className="sniper-tag"> SNIPER</em> : null}
                {thin ? <em className="thin-tag"> THIN VOL</em> : null}
              </b>
              <span>{thin ? 'No volume confirm' : (item.setup_label || item.status)}</span>
              <small>
                {item.sector}
                {item.volume_ratio != null ? ` · ${Number(item.volume_ratio).toFixed(1)}×` : ''}
                {item.rsi != null ? ` · RSI ${Math.round(Number(item.rsi))}` : ''}
                {' · '}
                {item.reason?.slice(0, 36) || '—'}
              </small>
            </button>
          </li>
          )
        })}
        {rows.length === 0 && (
          <li className="radar-empty-li">
            Preparing official history and scan…
          </li>
        )}
      </ul>
    </section>
  )

  const row = breakoutRows.find((r) => r.symbol === selected)
    || momentumRows.find((r) => r.symbol === selected)
    || longTermRows.find((r) => r.symbol === selected)

  const homeOs = radar?.home_os || null
  const runHomeAction = (action: HomeAction) => {
    if (action.kind === 'instruction') return
    const control = String(action.control || '')
    if (control === 'CHECK_SYSTEM' || action.kind === 'refresh') {
      void fetchRadarHome().then((payload) => setRadar(payload)).catch(() => undefined)
      return
    }
    if (control === 'RUN_SCAN_NOW') {
      void marketScan.start()
      return
    }
    if (control === 'VERIFY_FORWARD_SOAK') {
      void verifyForwardSoakNow().then(() => { void fetchRadarHome().then(setRadar) }).catch(() => undefined)
      return
    }
    if (control === 'SIMULATE_PAST_DECISIONS') {
      setDeskNote('Historical replay starting…')
      void simulatePastDecisions()
        .then(async (payload) => {
          remember('decision-simulator', payload)
          let latest = payload
          for (let i = 0; i < 40 && (latest.status === 'RUNNING' || latest.accepted); i += 1) {
            setDeskNote(latest.message || latest.simple || `Historical replay running · ${latest.sessions_done || 0}/${latest.sessions_total || '?'}`)
            await new Promise((resolve) => window.setTimeout(resolve, 1500))
            latest = await fetchDecisionSimulator()
          }
          remember('decision-simulator', latest)
          setDeskNote(latest.simple || latest.message || 'Historical replay finished.')
          void fetchRadarHome().then(setRadar)
        })
        .catch((reason: unknown) => {
          setDeskNote(reason instanceof Error ? reason.message : 'Historical replay failed')
        })
      return
    }
    if (control) void runControl(control as ControlName)
  }

  return (
    <section className="radar-home">
      {homeOs ? (
        <HomeOsCard
          os={homeOs}
          depth={depth}
          busy={marketScan.isBusy || bootstrapBusy}
          onAction={runHomeAction}
          onOpenPage={setActive}
          setSelected={setSelected}
        />
      ) : (
        <p className="panel-copy">{radarNote || 'Loading Home operating system… Dashboard below still works.'}</p>
      )}
      {radarNote && homeOs ? <p className="panel-copy">{radarNote}</p> : null}
      <header className="radar-hero">
        <div>
          <span>MARKET DESK</span>
          <h2>{radar?.market_health || dashboard.market.health}</h2>
          <p>{dashboard.market.summary}</p>
        </div>
        <div className="radar-hero-actions">
          {needsBootstrap && (
            <button type="button" disabled={bootstrapBusy} onClick={() => void runBootstrap()}>
              {bootstrapBusy ? 'Preparing…' : 'Refresh desk'}
            </button>
          )}
          <button type="button" disabled={marketScan.isBusy} onClick={() => void marketScan.start()}>
            {marketScan.isBusy
              ? `Scanning… ${marketScan.percent != null ? `${marketScan.percent}%` : ''}${marketScan.etaLine ? ` · ETA ${marketScan.etaLine}` : ''}`
              : 'Scan now'}
          </button>
        </div>
      </header>

      <DeskPipelineStrip pipeline={radar?.desk_pipeline} />

      <div className={`radar-desk-strip ${telegramWarn ? 'telegram-warn' : ''}`}>
        <div>
          <span>SCAN</span>
          <strong>{relativeAge(scanAt)}</strong>
          <small>{scanAt ? 'one saved scan · Home and Scanner share it' : 'scan queued only if the last file is stale'}</small>
        </div>
        <div>
          <span>PRICE DATA</span>
          <strong>{dashboard.data.bhavcopy.latest_date || 'Preparing…'}</strong>
          <small>{dashboard.data.ready ? 'official bhavcopy ready' : 'preparing official bhavcopy'}</small>
        </div>
        <div>
          <span>ZERODHA</span>
          <strong>{kiteOk ? 'READY' : kiteLoginOptional ? 'LOGIN OPTIONAL' : 'CHECKING'}</strong>
          <small>
            {kiteOk
              ? 'broker-dependent quotes and paper capability available'
              : kiteLoginOptional
                ? 'Log in only when you want broker-dependent capability. Core research keeps running.'
                : 'Broker state comes from backend readiness; core research does not depend on it.'}
          </small>
        </div>
        <div>
          <span>TELEGRAM</span>
          <strong>{telegramOn ? (telegram?.headline || 'CONNECTED') : 'OFF'}</strong>
          <small>
            {telegram?.detail
              || (telegramOn
                ? 'Setups and breakout watches send after each market scan'
                : 'Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env')}
          </small>
        </div>
      </div>
      {deskNote ? <p className="radar-desk-note">{deskNote}</p> : null}

      <div className="radar-market-strip">
        <div><span>NIFTY 1D</span><strong>{pct(radar?.nifty_change_1d ?? dashboard.market.nifty_change_1d)}</strong></div>
        <div><span>BREADTH</span><strong>{radar?.breadth || dashboard.market.breadth}</strong></div>
        <div><span>VIX</span><strong>{radar?.vix ?? dashboard.market.vix ?? '—'}</strong></div>
        <div><span>LEADERS</span><strong>{(radar?.leaders || dashboard.market.leaders).slice(0, 3).join(', ') || '—'}</strong></div>
        <div><span>SCAN AGE</span><strong>{relativeAge(scanAt)}</strong></div>
        <div><span>STANCE</span><strong>{dashboard.market.trade_stance?.split(';')[0] || '—'}</strong></div>
      </div>

      <DailyWrapList
        lines={magazineWrapLines(dashboard.daily_wrap, dashboard)}
        onSymbol={(symbol) => { setSelected(symbol); markInvestigate(symbol); setActive('Stock Intelligence') }}
      />

      <LiveScanBanner scan={marketScan} depth={depth} label="Shared market scan" />
      {longTermScan.isActive || longTermScan.notice ? (
        <LiveScanBanner scan={longTermScan} depth={depth} label="Funds refresh" />
      ) : null}
      <p className="radar-scan-share">
        {radar?.scan_shared_note
          || 'One scan fills Home, Scanner, Recommendations and long-term. Scan Now jumps the queue and walks every category once.'}
      </p>

      <RankingLegend legend={radar?.ranking_legend} />

      <BestOfBestHero
        row={(radar?.best_of_best?.[0] || radar?.best_among_fundamentals) as RadarRow | null | undefined}
        note={radar?.best_among_note}
        onSelect={setSelected}
      />
      <TopStocksList
        rows={((radar?.best_of_best && radar.best_of_best.length
          ? radar.best_of_best
          : radar?.best_among_fundamentals
            ? [radar.best_among_fundamentals]
            : []) as RadarRow[])}
        selected={selected}
        onSelect={setSelected}
      />

      <BestSniperPanel
        best={radar?.best_breakout as RadarRow | null | undefined}
        sniperCount={radar?.counts.sniper_breakouts || radar?.sniper_candidates?.length || 0}
        onSelect={setSelected}
      />

      {(radar?.sniper_candidates?.length || 0) > 0 && (
        <section className="radar-sniper-pool">
          <header>
            <span>SNIPER BREAKOUT CANDIDATES</span>
            <strong>
              {radar?.sniper_candidates?.length}
              {telegramOn && !telegram?.live_ticks ? ' · confirms need live ticks' : ''}
            </strong>
          </header>
          <ul>
            {(radar?.sniper_candidates || []).slice(0, 8).map((item) => (
              <li key={item.symbol}>
                <button
                  type="button"
                  className={selected === item.symbol ? 'active' : ''}
                  onClick={() => { setSelected(item.symbol); markInvestigate(item.symbol); setActive('Stock Intelligence') }}
                >
                  <b>{item.symbol}</b>
                  <span>
                    {(item as RadarRow).breakout_grade ? `G${(item as RadarRow).breakout_grade}` : '—'}
                    {item.volume_ratio != null ? ` · ${Number(item.volume_ratio).toFixed(1)}×` : ''}
                    {(item as RadarRow).rsi != null
                      ? ` · RSI ${Math.round(Number((item as RadarRow).rsi))}${(item as RadarRow).tech_source === 'live' || (item as RadarRow).price_tag === 'LIVE' ? ' LIVE' : ''}`
                      : ''}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      <div className="radar-three-lanes">
        {laneCard(
          'Breakouts',
          breakoutRows,
          breakoutCount,
          radar?.counts.sniper_breakouts,
        )}
        {laneCard('Momentum', momentumRows, momentumCount)}
        {laneCard('Long-Term Picks', longTermRows, longTermCount)}
      </div>

      <div className="radar-workspace">
        <Panel title={`CHART · ${selected || 'SELECT STOCK'}`} subtitle={`Official history · ${dashboard.data.bhavcopy.latest_date || '—'}`}>
          <ChartWorkspace symbol={selected} bars={bars} row={row} />
        </Panel>
        <Panel title="DECISION PREVIEW" subtitle="Risk before reward · read-only">
          {selected ? (
            <div className="radar-decision-preview">
              <p><strong>{(row as RadarRow)?.reason || plan?.summary || 'Select a stock from a lane above.'}</strong></p>
              {plan?.entry != null && <div>Entry zone: {money(plan.entry)}</div>}
              {plan?.stop != null && <div>Invalidation: {money(plan.stop)}</div>}
              {plan?.target != null && <div>Target: {money(plan.target)}</div>}
              <RiskLensCard plan={plan} />
              <div className="radar-action-row">
                <button type="button" onClick={() => { markInvestigate(selected); setActive('Stock Intelligence') }}>Investigate</button>
                <button type="button" onClick={() => setActive('Stock Intelligence')}>Full research</button>
                <button type="button" onClick={() => onCompare(selected)}>Compare</button>
                <button type="button" onClick={() => onWatchlist(selected)}>Watchlist</button>
              </div>
            </div>
          ) : (
            <p className="radar-empty-li">Pick one name above — then check ₹ risk here before opening research.</p>
          )}
        </Panel>
      </div>
    </section>
  )
}

export function MarketScannerView(props: ExperienceViewProps & { onCompare: (symbol: string) => void }) {
  const { dashboard, selected, setSelected, bars, setActive, depth, marketScan, longTermScan, onCompare } = props
  const [tab, setTab] = useState<'Best Setups' | 'Breakouts' | 'Momentum' | 'Long-Term'>('Best Setups')
  const [rows, setRows] = useState<RadarRow[]>(() => {
    const cached = recall<RadarRow[]>('scanner:Best Setups')
    if (cached?.length) return cached
    return scannerFallbackRows('Best Setups', dashboard) as RadarRow[]
  })
  const [meta, setMeta] = useState(() => {
    const cached = recall<{ scanned_at: string; universe: number }>('scanner-meta:Best Setups')
    if (cached?.universe || cached?.scanned_at) return cached
    return scannerMetaFromDashboard('Best Setups', dashboard)
  })
  const [search, setSearch] = useState('')
  const [sector, setSector] = useState('All')
  const [excludeChase, setExcludeChase] = useState(true)

  const activeScan = marketScan
  const hasScan = Boolean(dashboard.scan.scanned_at || dashboard.scan.records.length || meta.universe || meta.scanned_at)

  useEffect(() => {
    const seed = scannerFallbackRows(tab, dashboard) as RadarRow[]
    const seedMeta = scannerMetaFromDashboard(tab, dashboard)
    const cachedRows = recall<RadarRow[]>(`scanner:${tab}`)
    const cachedMeta = recall<{ scanned_at: string; universe: number }>(`scanner-meta:${tab}`)
    const opening = (cachedRows?.length ? cachedRows : seed)
    if (opening.length) setRows(opening)
    const openingMeta = (cachedMeta?.universe || cachedMeta?.scanned_at) ? cachedMeta : seedMeta
    if (openingMeta.universe || openingMeta.scanned_at) setMeta(openingMeta)

    const apply = (next: RadarRow[], nextMeta: { scanned_at: string; universe: number }) => {
      const kept = keepRicher(`scanner:${tab}`, next, (items) => items.length === 0)
      const metaToKeep = nextMeta.scanned_at || nextMeta.universe
        ? nextMeta
        : (recall<{ scanned_at: string; universe: number }>(`scanner-meta:${tab}`) || openingMeta)
      remember(`scanner-meta:${tab}`, metaToKeep)
      setRows(kept)
      setMeta(metaToKeep)
    }

    if (tab === 'Best Setups') {
      fetchRadarHome()
        .then((result) => {
          const next = bestSetupsFromRadar(result, dashboard) as RadarRow[]
          apply(next, {
            scanned_at: result.scan_scanned_at || seedMeta.scanned_at,
            universe: result.universe_size || seedMeta.universe,
          })
        })
        .catch(() => apply(opening, openingMeta))
      return
    }
    fetchScannerWorkspace(tab)
      .then((result) => {
        const next = (result.rows?.length ? result.rows : seed) as RadarRow[]
        apply(next, {
          scanned_at: result.scanned_at || seedMeta.scanned_at,
          universe: result.universe_size || seedMeta.universe,
        })
      })
      .catch(() => apply(opening, openingMeta))
  }, [
    tab,
    dashboard.scan.scanned_at,
    dashboard.long_term.scanned_at,
    dashboard.scan.records.length,
    dashboard.long_term.records.length,
    dashboard.scan.universe_size,
    marketScan.succeeded,
    longTermScan.succeeded,
  ])

  const sectors = useMemo(() => [...new Set(rows.map((r) => r.sector).filter(Boolean))].sort(), [rows])
  const filtered = rows.filter((row) => {
    const q = search.trim().toUpperCase()
    if (q && !row.symbol.includes(q) && !String(row.company || '').toUpperCase().includes(q)) return false
    if (sector !== 'All' && row.sector !== sector) return false
    if (excludeChase && row.chase_risk) return false
    return true
  })

  const selectedRow = filtered.find((r) => r.symbol === selected) || rows.find((r) => r.symbol === selected)

  return (
    <section className="market-scanner">
      <header className="scanner-command-bar">
        <div>
          <span>MARKET SCANNER</span>
          <h2>Breakouts, momentum, SEPA and long-term</h2>
          <p>{filtered.length} matches · universe {meta.universe.toLocaleString('en-IN')} · one scan fills every tab · {meta.scanned_at || '—'}</p>
        </div>
        <div>
          <button type="button" disabled={marketScan.isBusy} onClick={() => void marketScan.start()}>
            {marketScan.isBusy
              ? `Scanning… ${marketScan.percent != null ? `${marketScan.percent}%` : ''}${marketScan.etaLine ? ` · ETA ${marketScan.etaLine}` : ''}`
              : 'Scan now'}
          </button>
          {tab === 'Long-Term' ? (
            <button type="button" disabled={longTermScan.isBusy} onClick={() => void longTermScan.start()}>
              {longTermScan.isBusy ? 'Refreshing funds…' : 'Refresh funds'}
            </button>
          ) : null}
        </div>
      </header>

      <LiveScanBanner scan={activeScan} depth={depth} label="Shared market scan" />
      {tab === 'Long-Term' && (longTermScan.isActive || longTermScan.notice) ? (
        <LiveScanBanner scan={longTermScan} depth={depth} label="Funds refresh" />
      ) : null}
      <p className="radar-scan-share">
        One scan fills Home, Scanner (all tabs), Recommendations and long-term. Scan Now jumps the queue and walks every category once. Refresh funds only reloads Screener snapshots.
      </p>

      <div className="radar-tab-row">
        {(['Best Setups', 'Breakouts', 'Momentum', 'Long-Term'] as const).map((item) => (
          <button key={item} type="button" className={tab === item ? 'active' : ''} onClick={() => setTab(item)}>{item}</button>
        ))}
      </div>

      <div className="scanner-filter-row">
        <label>Search<input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Symbol" /></label>
        <label>Sector<select value={sector} onChange={(e) => setSector(e.target.value)}><option>All</option>{sectors.map((s) => <option key={s}>{s}</option>)}</select></label>
        <label className="scanner-check"><input type="checkbox" checked={excludeChase} onChange={(e) => setExcludeChase(e.target.checked)} /> Hide extended</label>
      </div>

      <div className="scanner-workspace-grid">
        <Panel title={`${tab.toUpperCase()} · ${filtered.length}`} subtitle={tab === 'Best Setups' ? 'SEPA 7-rule overlay ≥40/100 on the saved scan — not a buy' : 'Sorted from the same persisted scan Home uses'}>
          <DenseTable
            rows={filtered}
            selected={selected}
            onSelect={(symbol) => { setSelected(symbol); markInvestigate(symbol); setActive('Stock Intelligence') }}
            depth={depth}
            mode={tab}
            emptyHint={scannerEmptyHint(rows.length, filtered.length, hasScan)}
          />
        </Panel>
        <div className="scanner-detail-column">
          <Panel title={`CHART · ${selected || '—'}`}><ChartWorkspace symbol={selected} bars={bars} row={selectedRow} /></Panel>
          <Panel title="ACTIONS">
            <div className="radar-action-row">
              <button type="button" disabled={!selected} onClick={() => { if (selected) { markInvestigate(selected); setActive('Stock Intelligence') } }}>Investigate</button>
              <button type="button" disabled={!selected} onClick={() => setActive('Stock Intelligence')}>Stock Intelligence</button>
              <button type="button" disabled={!selected} onClick={() => selected && onCompare(selected)}>Compare</button>
            </div>
          </Panel>
        </div>
      </div>
    </section>
  )
}

export function CompareView({ symbols, setSymbols, setActive, setSelected, seedSymbols = [] }: {
  symbols: string[]
  setSymbols: (s: string[]) => void
  setActive: (page: string) => void
  setSelected: (s: string) => void
  seedSymbols?: string[]
}) {
  const [data, setData] = useState<CompareWorkspace | null>(null)
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const seededRef = useRef(false)

  useEffect(() => {
    if (seededRef.current || symbols.length) return
    const unique = [...new Set(seedSymbols.map((item) => item.trim().toUpperCase()).filter(Boolean))].slice(0, 3)
    if (unique.length < 2) return
    seededRef.current = true
    setSymbols(unique)
  }, [seedSymbols, setSymbols, symbols.length])

  useEffect(() => {
    if (symbols.length === 0) { setData(null); return }
    setLoading(true)
    fetchCompareWorkspace(symbols)
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false))
  }, [symbols.join(',')])

  const addSymbol = () => {
    const sym = input.trim().toUpperCase()
    if (!sym || symbols.includes(sym)) return
    if (symbols.length >= 5) return
    setSymbols([...symbols, sym])
    setInput('')
  }

  return (
    <section className="compare-view">
      <header className="radar-hero">
        <div><span>COMPARE</span><h2>Side-by-side fundamentals and market state</h2><p>{data?.disclaimer || 'Add up to 5 NSE symbols.'}</p></div>
      </header>
      <div className="compare-chips">
        {symbols.map((sym) => (
          <button key={sym} type="button" className="compare-chip" onClick={() => { setSelected(sym); markInvestigate(sym); setActive('Stock Intelligence') }}>{sym}</button>
        ))}
        <input value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && addSymbol()} placeholder="Add symbol" />
        <button type="button" onClick={addSymbol}>Add</button>
        <button type="button" onClick={() => setSymbols([])}>Clear</button>
      </div>
      {loading && <p>Loading comparison…</p>}
      {data && (
        <div className="compare-grid">
          {Object.entries(data.section_labels).map(([key, label]) => (
            <Panel key={key} title={label.toUpperCase()}>
              <table className="radar-table compare-table">
                <thead><tr><th>Metric</th>{data.rows.map((r) => <th key={r.symbol}>{r.symbol}</th>)}</tr></thead>
                <tbody>
                  {(data.rows[0]?.sections[key] || []).map((_, idx) => (
                    <tr key={idx}>
                      <td>{data.rows[0]?.sections[key]?.[idx]?.label}</td>
                      {data.rows.map((row) => {
                        const m = row.sections[key]?.[idx]
                        return <td key={row.symbol}>{m?.available ? `${m.value}${m.unit ? ` ${m.unit}` : ''}` : '—'}</td>
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </Panel>
          ))}
        </div>
      )}
    </section>
  )
}

export function WatchlistView({ setActive, setSelected, onCompare, selected = '' }: {
  setActive: (page: string) => void
  setSelected: (s: string) => void
  onCompare: (symbol: string) => void
  selected?: string
}) {
  const [payload, setPayload] = useState<WatchlistPayload | null>(() => recall<WatchlistPayload>('watchlist') ?? null)
  const [symbol, setSymbol] = useState('')
  const [notes, setNotes] = useState('')
  const [busy, setBusy] = useState(false)

  const reload = () => fetchWatchlist()
    .then((next) => { remember('watchlist', next); setPayload(next) })
    .catch(() => { if (!recall('watchlist')) setPayload(null) })

  useEffect(() => { void reload() }, [])

  const add = async (explicit?: string) => {
    const sym = (explicit || symbol).trim().toUpperCase()
    if (!sym) return
    setBusy(true)
    try {
      await addWatchlistItem({ symbol: sym, notes: notes || (explicit ? 'From current stock' : '') })
      setSymbol('')
      setNotes('')
      await reload()
    } finally { setBusy(false) }
  }

  return (
    <section className="watchlist-view">
      <header className="radar-hero">
        <div><span>WATCHLIST</span><h2>Track names you want to investigate</h2><p>Personal list — not a second alerts engine.</p></div>
      </header>
      <div className="watchlist-add">
        <input value={symbol} onChange={(e) => setSymbol(e.target.value)} placeholder="NSE symbol" />
        <input value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="Why watching" />
        <button type="button" disabled={busy} onClick={() => void add()}>Add</button>
        {selected ? (
          <button type="button" disabled={busy} onClick={() => void add(selected)}>
            Add {selected}
          </button>
        ) : null}
      </div>
      <table className="radar-table">
        <thead><tr><th>Symbol</th><th>Added</th><th>Setup</th><th>Notes</th><th>Actions</th></tr></thead>
        <tbody>
          {(payload?.items || []).map((item) => (
            <tr key={item.id}>
              <td><button type="button" onClick={() => { setSelected(item.symbol); markInvestigate(item.symbol); setActive('Stock Intelligence') }}>{item.symbol}</button></td>
              <td>{item.added_date}</td>
              <td>{String((item.snapshot as RadarRow)?.setup_label || item.snapshot?.status || '—')}</td>
              <td>{item.notes || '—'}</td>
              <td className="radar-action-row">
                <button type="button" onClick={() => onCompare(item.symbol)}>Compare</button>
                <button type="button" onClick={() => void removeWatchlistItem(item.id).then(reload)}>Remove</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {payload?.count === 0 && <p className="radar-empty-li">No watchlist items yet.</p>}
    </section>
  )
}
