import { money, pct } from './format'
import type { QuoteHeartbeatPayload, QuoteTick } from './api'

function TickCell({ label, tick }: { label: string; tick?: QuoteTick }) {
  if (!tick || tick.price == null) {
    return (
      <div className="live-tick-cell muted">
        <span>{label}</span>
        <strong>—</strong>
        <small>no tick</small>
      </div>
    )
  }
  const up = (tick.chg_pct || 0) >= 0
  return (
    <div className={`live-tick-cell ${up ? 'up' : 'down'}`}>
      <span>{label}</span>
      <strong>{money(tick.price, tick.price >= 100 ? 1 : 2)}</strong>
      <small>
        {pct(tick.chg_pct)}
        {tick.age_s != null ? ` · ${Math.round(Number(tick.age_s))}s` : ''}
        {tick.source ? ` · ${tick.source}` : ''}
      </small>
    </div>
  )
}

export function LivePriceStrip({
  payload,
  focusSymbol,
}: {
  payload: QuoteHeartbeatPayload | null
  focusSymbol?: string
}) {
  if (!payload) {
    return (
      <div className="live-price-strip">
        <div className="live-tick-cell muted">
          <span>LIVE</span>
          <strong>…</strong>
          <small>connecting</small>
        </div>
      </div>
    )
  }
  const quotes = payload.quotes || {}
  const nifty = quotes.NIFTY
  const bank = quotes.BANKNIFTY
  const focus = focusSymbol ? quotes[focusSymbol.toUpperCase()] : undefined
  const mode = !payload.session_open
    ? 'SESSION CLOSED'
    : payload.streaming
      ? 'KITE STREAM'
      : 'LIVE REST'
  return (
    <div className="live-price-strip" title={payload.honesty || 'Live LTP heartbeat'}>
      <div className={`live-tick-mode ${payload.session_open ? 'on' : 'off'}`}>
        <i />
        <span>{mode}</span>
      </div>
      <TickCell label="NIFTY" tick={nifty} />
      <TickCell label="BANKNIFTY" tick={bank} />
      {focusSymbol ? <TickCell label={focusSymbol.toUpperCase()} tick={focus} /> : null}
    </div>
  )
}
