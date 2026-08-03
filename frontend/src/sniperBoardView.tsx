import { useCallback, useEffect, useMemo, useState } from 'react'
import { EvidenceList, MetricCard, Panel } from './components'
import { compactDateTime, money, pct, score, words } from './format'
import { LiveScanBanner } from './experience'
import { fetchSniperBoard } from './api'
import type { ScanRunnerHandle } from './scanRunner'
import type { SniperBoardHit, SniperBoardPayload, SniperEvalRecord } from './types'

function verdictTone(verdict: string | undefined): string {
  const v = String(verdict || '').toUpperCase()
  if (v === 'PRIORITY') return 'positive'
  if (v === 'CANDIDATE') return ''
  if (v === 'AVOID' || v === 'INCOMPLETE' || v === 'WATCH' || v === 'WEAK') return 'negative'
  return ''
}

function edgeCell(edge: number | null | undefined): string {
  if (edge == null || !Number.isFinite(edge)) return '—'
  return `${edge >= 0 ? '+' : ''}${edge.toFixed(2)}R`
}

export function ConfirmedBreakoutsView({
  selected,
  setSelected,
  setActive,
  evalScan,
}: {
  selected: string
  setSelected: (symbol: string) => void
  setActive: (page: string) => void
  evalScan: ScanRunnerHandle
}) {
  const [board, setBoard] = useState<SniperBoardPayload | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    try {
      const payload = await fetchSniperBoard()
      setBoard(payload)
      setError(null)
      const ranked = payload.evaluation_records || []
      if (!selected && ranked[0]?.symbol) setSelected(ranked[0].symbol)
      else if (!selected && payload.hits?.[0]?.symbol) setSelected(payload.hits[0].symbol)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Sniper board unavailable')
    } finally {
      setLoading(false)
    }
  }, [selected, setSelected])

  useEffect(() => {
    void load()
  }, [load])

  useEffect(() => {
    if (!evalScan.isActive && !evalScan.succeeded) return
    void load()
  }, [evalScan.isActive, evalScan.succeeded, evalScan.operation?.status, load])

  const records = board?.evaluation_records || []
  const hits = board?.hits || []
  const summary = board?.evaluation_summary || {}
  const current = useMemo(
    () => records.find((row) => row.symbol === selected) || null,
    [records, selected],
  )
  const currentHit = useMemo(
    () => hits.find((row) => row.symbol === selected) || hits[0] || null,
    [hits, selected],
  )

  return (
    <section className="enhanced-long-term professional">
      <header className="long-term-hero">
        <div>
          <span>CONFIRMED BREAKOUTS</span>
          <h2>Sniper hits ranked for tomorrow-watch and longer-horizon research</h2>
          <p>
            Live pivot confirms land here first. Evaluate ranks only that list — momentum,
            fundamentals, and measured backtest edge — without inventing missing evidence.
          </p>
        </div>
        <button type="button" disabled={evalScan.isBusy || hits.length === 0} onClick={() => void evalScan.start()}>
          {evalScan.isBusy ? 'Evaluating…' : 'Evaluate board'}
        </button>
      </header>

      <LiveScanBanner scan={evalScan} depth="professional" label="Sniper-board evaluation" />

      {error ? <div className="empty-row">{error}</div> : null}
      {loading && !board ? <div className="empty-row">Loading confirmed breakouts…</div> : null}

      <div className="long-term-metric-strip">
        <MetricCard label="CONFIRMED HITS" value={String(board?.hit_count ?? hits.length)} detail="Durable sniper confirms (not Telegram-only)" tone="cyan" />
        <MetricCard label="PRIORITY" value={String(summary.priority ?? 0)} detail="Best composite evidence on the board" tone="green" />
        <MetricCard label="CANDIDATES" value={String(summary.candidate ?? 0)} detail="Usable for tomorrow-watch consideration" />
        <MetricCard label="WITH EDGE" value={String(summary.with_measured_edge ?? 0)} detail="Signal combo has actionable backtest edge" tone="amber" />
        <MetricCard label="FUNDAMENTALS" value={String(summary.with_fundamentals ?? 0)} detail="Coverage ≥ 35% on focused long-term screen" tone="purple" />
      </div>

      <p className="no-fake-performance" style={{ marginTop: 0 }}>
        <strong>Research ranking — not a buy ticket.</strong>
        {' '}
        {board?.honesty || 'Confirmed sniper hits only. Missing scan, fundamentals, or edge stays missing.'}
        {' '}
        Last board update {compactDateTime(board?.updated_at)}.
        {board?.evaluated_at ? ` Last evaluation ${compactDateTime(board.evaluated_at)}.` : ' Evaluation not run yet.'}
      </p>

      <div className="long-term-workspace-grid">
        <Panel
          title={`RANKED SHORTLIST · ${records.length}`}
          subtitle={records.length ? 'Priority → candidate → watch → avoid' : 'Run Evaluate board after sniper confirms land'}
        >
          <EvalTable rows={records} selected={selected} onSelect={setSelected} />
        </Panel>

        <div className="long-term-detail-column">
          <Panel title={`VERDICT · ${selected || 'SELECT STOCK'}`}>
            {current ? (
              <div className="quality-overlay-grid">
                <div><span>Verdict</span><strong className={verdictTone(current.verdict)}>{words(current.verdict)}</strong></div>
                <div><span>Rank score</span><strong>{score(current.rank_score)}</strong></div>
                <div><span>Measured edge</span><strong className={(current.edge_r || 0) >= 0 ? 'positive' : 'negative'}>{edgeCell(current.edge_r)}</strong></div>
                <div><span>Momentum</span><strong>{score(current.momentum_score)}</strong></div>
                <div><span>Fundamentals</span><strong>{score(current.fundamental_score)}</strong></div>
                <div><span>Coverage</span><strong>{current.fundamental_coverage == null ? '—' : `${(Number(current.fundamental_coverage) * 100).toFixed(0)}%`}</strong></div>
                <div><span>Breakout quality</span><strong>{score(current.breakout_quality)}</strong></div>
                <div><span>Consider for</span><strong>{(current.consider_for || []).map(words).join(' · ') || '—'}</strong></div>
              </div>
            ) : (
              <div className="empty-row">Select a ranked symbol, or run evaluation once hits exist.</div>
            )}
          </Panel>

          <Panel title="EVIDENCE">
            <div className="evidence-grid">
              <EvidenceList title="Why it ranks" items={current?.reasons} tone="green" />
              <EvidenceList title="Risks / gaps" items={current?.risks} tone="red" />
            </div>
          </Panel>

          <Panel title={`CONFIRM EVENT · ${currentHit?.symbol || '—'}`}>
            {currentHit ? (
              <div className="quality-overlay-grid">
                <div><span>Trigger</span><strong>{money(currentHit.trigger)}</strong></div>
                <div><span>Confirm LTP</span><strong>{money(currentHit.ltp)}</strong></div>
                <div><span>Held</span><strong>{currentHit.held_s != null ? `${currentHit.held_s}s` : '—'}</strong></div>
                <div><span>Volume pace</span><strong>{currentHit.vol_pace == null ? '—' : `${currentHit.vol_pace.toFixed(1)}×`}</strong></div>
                <div><span>Stop</span><strong>{currentHit.stop == null ? '—' : money(currentHit.stop)}</strong></div>
                <div><span>Target</span><strong>{currentHit.target == null ? '—' : money(currentHit.target)}</strong></div>
                <div><span>Session</span><strong>{currentHit.session_date || '—'}</strong></div>
                <div><span>5D mom</span><strong className={(current?.momentum_5d || 0) >= 0 ? 'positive' : 'negative'}>{pct(current?.momentum_5d)}</strong></div>
              </div>
            ) : (
              <div className="empty-row">No confirm event yet — sniper must fire during market hours.</div>
            )}
          </Panel>

          <Panel title="NEXT ACTIONS">
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, padding: '4px 2px 8px' }}>
              <button type="button" disabled={!selected} onClick={() => setActive('Stock Intelligence')}>Open stock intelligence</button>
              <button type="button" disabled={!selected} onClick={() => setActive('Long-Term Picks')}>Open long-term research</button>
              <button type="button" onClick={() => void load()}>Refresh board</button>
            </div>
          </Panel>
        </div>
      </div>

      <Panel title={`RAW CONFIRM LOG · ${hits.length}`} subtitle="One confirm per symbol per session · newest first">
        <HitTable rows={hits} selected={selected} onSelect={setSelected} />
      </Panel>
    </section>
  )
}

function EvalTable({
  rows,
  selected,
  onSelect,
}: {
  rows: SniperEvalRecord[]
  selected: string
  onSelect: (symbol: string) => void
}) {
  return (
    <div className="table-shell">
      <div className="table-head">
        <span>#</span><span>STOCK</span><span>VERDICT</span><span>RANK</span><span>EDGE</span><span>MOM</span><span>FUND</span>
      </div>
      {rows.length === 0 && <div className="empty-row">No evaluation yet — click Evaluate board after confirms arrive.</div>}
      {rows.map((row, index) => (
        <button
          key={`${row.symbol}-${index}`}
          type="button"
          className={selected === row.symbol ? 'table-row selected' : 'table-row'}
          onClick={() => onSelect(row.symbol)}
        >
          <span>{index + 1}</span>
          <strong>{row.symbol}</strong>
          <span className={verdictTone(row.verdict)}>{words(row.verdict)}</span>
          <span className="score-cell">{score(row.rank_score)}</span>
          <span className={(row.edge_r || 0) >= 0 ? 'positive' : 'negative'}>{edgeCell(row.edge_r)}</span>
          <span>{score(row.momentum_score)}</span>
          <span>{score(row.fundamental_score)}</span>
        </button>
      ))}
    </div>
  )
}

function HitTable({
  rows,
  selected,
  onSelect,
}: {
  rows: SniperBoardHit[]
  selected: string
  onSelect: (symbol: string) => void
}) {
  return (
    <div className="table-shell">
      <div className="table-head">
        <span>#</span><span>STOCK</span><span>TRIGGER</span><span>LTP</span><span>HELD</span><span>PACE</span><span>SESSION</span>
      </div>
      {rows.length === 0 && <div className="empty-row">Waiting for live sniper confirms during market hours.</div>}
      {rows.map((row, index) => (
        <button
          key={`${row.symbol}-${row.session_date}-${index}`}
          type="button"
          className={selected === row.symbol ? 'table-row selected' : 'table-row'}
          onClick={() => onSelect(row.symbol)}
        >
          <span>{index + 1}</span>
          <strong>{row.symbol}</strong>
          <span>{money(row.trigger)}</span>
          <span>{money(row.ltp)}</span>
          <span>{row.held_s != null ? `${row.held_s}s` : '—'}</span>
          <span>{row.vol_pace == null ? '—' : `${row.vol_pace.toFixed(1)}×`}</span>
          <span>{row.session_date || '—'}</span>
        </button>
      ))}
    </div>
  )
}
