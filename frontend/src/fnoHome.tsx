import type { ControlName, DashboardPayload } from './types'
import { Panel } from './components'
import { compactDateTime, words } from './format'

const price = (value: unknown) => value != null && Number.isFinite(Number(value))
  ? `₹${Number(value).toLocaleString('en-IN', { maximumFractionDigits: 2 })}` : 'unavailable'

export function FnoHomePanel({ dashboard, setActive, setSelected, runControl }: {
  dashboard: DashboardPayload
  setActive: (page: string) => void
  setSelected: (symbol: string) => void
  runControl: (control: ControlName) => Promise<void>
}) {
  const desk = dashboard.fno.desk
  const active = dashboard.operations.active.find(row => row.kind === 'FNO_REFRESH' || row.kind === 'DATA_PREPARE')
  const candidates = desk?.candidates || []
  const positions = desk?.open_positions || []
  const select = (symbol: string) => { setSelected(symbol); setActive('F&O Desk') }
  return <section className="home-fno-desk" data-testid="home-fno-desk">
    <Panel title={`F&O PAPER · ${candidates.length} CANDIDATES · ${positions.length} OPEN`}
      subtitle="Simulated trading · candidates and executed paper positions shown separately">
      <div className="inline-actions">
        <button type="button" onClick={() => setActive('F&O Desk')}>Open F&O desk</button>
        <button type="button" disabled={Boolean(active)} onClick={() => void runControl('REFRESH_FNO_NOW')}>Refresh F&O scan</button>
      </div>
      {active && <p role="status">F&O scan {words(active.status)} · {words(active.stage || '')} · {active.message || 'Preparing current scan'}</p>}
      <p>Scan: {desk?.as_of || 'not yet completed'} · {words(desk?.status || 'NOT_RUN')}
        {desk?.generated_at ? ` · saved ${compactDateTime(new Date(desk.generated_at * 1000).toISOString())}` : ''}
      </p>
      {candidates.length === 0 && <p className="panel-copy">No current eligible F&O candidate · {words(desk?.reason || dashboard.fno.directional?.reason || dashboard.fno.directional?.code || 'Waiting for F&O scan completion')}.</p>}
      {(desk?.blockers || []).length > 0 && <p>Gates: {desk?.blockers.map(row => `${words(row.reason)} (${row.count})`).join(' · ')}</p>}
      <div className="exclusion-list">
        {candidates.map(row => {
          const contract = row.selected_contract || {}
          const plan = contract.trade_plan || {}
          const open = positions.some(pos => pos.option_symbol === contract.symbol)
          return <button type="button" key={contract.symbol} onClick={() => select(row.symbol)}>
            <strong>{row.symbol} · {row.direction} · {contract.option_type} {contract.strike}</strong>
            <span>{contract.symbol} · expiry {contract.expiry} · premium {price(contract.premium)}</span>
            <small>Paper plan · entry {price(plan.entry)} · stop {price(plan.stop)} · target {price(plan.target)}</small>
            <small>Setup {row.setup?.score ?? '—'}/100 · option {contract.score ?? '—'}/100 · quality scores, not win probabilities</small>
            <small>{open ? 'Matching contract is OPEN in paper book' : 'Candidate only · awaiting entry window and paper risk gates'}</small>
          </button>
        })}
      </div>
      <h3>Open F&O paper positions</h3>
      {!desk?.paper_available && <p>Paper ledger unavailable{desk?.paper_error ? ` · ${desk.paper_error}` : ''}.</p>}
      {desk?.paper_available && positions.length === 0 && <p>No open F&O paper position.</p>}
      <div className="exclusion-list">{positions.slice(0, 5).map(pos => <button type="button" key={pos.trade_id} onClick={() => select(pos.underlying || '')}>
        <strong>PAPER OPEN · {pos.underlying} · {pos.option_symbol}</strong>
        <span>Quantity {pos.quantity ?? '—'} · entry {price(pos.entry_price)} · stop {price(pos.trailing_stop_price ?? pos.stop_price)} · target {price(pos.target_price)}</span>
        <small>Opened {compactDateTime(pos.opened_at || '')} · bars held {pos.bars_held ?? 0}</small>
      </button>)}</div>
      {(desk?.recent_closed_trades || []).length > 0 && <>
        <h3>Recent F&O paper exits</h3>
        <div className="exclusion-list">{desk?.recent_closed_trades.slice(0, 3).map((trade, index) => <div key={String(trade.trade_id || index)}>
          <strong>PAPER CLOSED · {String(trade.underlying || '')} · {String(trade.option_symbol || '')}</strong>
          <span>Exit {price(trade.exit_price)} · net P&amp;L {price(trade.net_pnl)} · {words(String(trade.exit_reason || ''))}</span>
        </div>)}</div>
      </>}
    </Panel>
  </section>
}
