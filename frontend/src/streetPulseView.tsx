import { useEffect, useState } from 'react'
import { EmptyState, MetricCell, StatusBadge } from './designSystem'
import { EvidenceList, MetricCard, Panel } from './components'
import { words } from './format'
import { fetchStreetPulse, sendStreetPulseTelegram, type StreetPulsePayload } from './api'

type Props = {
  setSelected?: (symbol: string) => void
  setActive?: (page: string) => void
  onOpenPdf?: () => void
}

function stanceTone(stance?: string): 'green' | 'amber' | 'purple' | 'cyan' {
  const s = String(stance || '').toUpperCase()
  if (s === 'SUPPORTIVE') return 'green'
  if (s === 'HOSTILE' || s === 'CAUTION') return 'amber'
  if (s === 'NEUTRAL') return 'purple'
  return 'cyan'
}

function StockChip({
  symbol,
  detail,
  onOpen,
}: {
  symbol?: string
  detail?: string
  onOpen?: (symbol: string) => void
}) {
  if (!symbol) return null
  return (
    <button
      type="button"
      className="inline-actions"
      style={{ display: 'inline-flex', gap: 8, alignItems: 'baseline', marginRight: 8, marginBottom: 6 }}
      onClick={() => onOpen?.(symbol)}
    >
      <strong>{symbol}</strong>
      {detail ? <small>{detail}</small> : null}
    </button>
  )
}

export function StreetPulseView({ setSelected, setActive, onOpenPdf }: Props) {
  const [pulse, setPulse] = useState<StreetPulsePayload | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [token, setToken] = useState(0)
  const [sending, setSending] = useState(false)
  const [sendNote, setSendNote] = useState('')

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError('')
    fetchStreetPulse(token > 0)
      .then((payload) => {
        if (!cancelled) setPulse(payload)
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message || 'Street pulse failed')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [token])

  const openSymbol = (symbol: string) => {
    setSelected?.(symbol)
    setActive?.('Stock Intelligence')
  }

  const sendTelegram = async () => {
    setSending(true)
    setSendNote('')
    try {
      const result = await sendStreetPulseTelegram(true)
      if (result.sent) {
        setSendNote(`Sent to Telegram${result.date ? ` · ${result.date}` : ''}`)
      } else if (result.configured === false) {
        setSendNote(result.error || 'Telegram not configured — set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID')
      } else {
        setSendNote(result.error || 'Telegram send failed')
      }
    } catch (err) {
      setSendNote(err instanceof Error ? err.message : 'Telegram send failed')
    } finally {
      setSending(false)
    }
  }

  const stance = pulse?.snapshot?.options_stance?.stance
  const sectors = pulse?.sectors

  return (
    <section className="workspace-view">
      <div className="inline-actions" style={{ marginBottom: 12 }}>
        <button type="button" disabled={loading} onClick={() => setToken((n) => n + 1)}>
          {loading ? 'Building pulse…' : 'Rebuild pulse'}
        </button>
        <button type="button" disabled={sending || loading} onClick={() => void sendTelegram()}>
          {sending ? 'Sending…' : 'Send to Telegram'}
        </button>
        {onOpenPdf ? (
          <button type="button" onClick={onOpenPdf}>
            Open PDF
          </button>
        ) : null}
      </div>
      {sendNote ? <p className="panel-copy">{sendNote}</p> : null}

      {loading && !pulse && <div className="large-empty">Assembling Daily Street Pulse from scan, bhav, options and news…</div>}
      {error && <EmptyState title="Pulse unavailable" detail={error} />}

      {pulse && (
        <>
          <header className="stock-workspace-hero" style={{ marginBottom: 16 }}>
            <div>
              <span>Research digest · not a buy desk</span>
              <h2>{pulse.title || 'Daily Street Pulse'}</h2>
              <p>
                {pulse.date || '—'} · {pulse.scanned || 0} scanned · source {pulse.scan_source || '—'}
                {pulse.scan_as_of ? ` · as of ${pulse.scan_as_of}` : ''}
              </p>
            </div>
            <div className="stock-workspace-state">
              <span>{words(stance || 'Incomplete')}</span>
              <strong>{(pulse.gaps || []).length}</strong>
              <small>gaps disclosed</small>
            </div>
          </header>

          <p className="panel-copy">{pulse.honesty || 'Paper-first research digest.'}</p>

          <div className="metric-grid">
            <MetricCard
              label="OPTIONS STANCE"
              value={words(stance || '—')}
              detail={pulse.snapshot?.options_stance?.headline || 'NIFTY positioning read'}
              tone={stanceTone(stance)}
            />
            <MetricCard
              label="SCAN SOURCE"
              value={words(pulse.scan_source || 'Unavailable')}
              detail={pulse.scan_as_of || 'No durable scan stamp'}
            />
            <MetricCard
              label="SECTOR HEAT"
              value={sectors?.available ? `${(sectors.leaders || [])[0]?.sector || '—'}` : '—'}
              detail={
                sectors?.available && (sectors.leaders || [])[0]
                  ? `1D ${(sectors.leaders || [])[0].chg_1d ?? '—'}%`
                  : sectors?.message || 'Unavailable'
              }
              tone="purple"
            />
            <MetricCard
              label="STATUS"
              value={pulse.available ? 'Ready' : 'Incomplete'}
              detail="Places orders: no"
              tone={pulse.available ? 'green' : 'amber'}
            />
          </div>

          <Panel title="COVER TAKEAWAYS" subtitle="What the system can defend from today's stores">
            {(pulse.takeaways || []).length === 0 && <EmptyState title="No takeaways yet" detail="Rebuild after a market scan." />}
            <ul className="plain-list">
              {(pulse.takeaways || []).map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </Panel>

          <div className="stock-context-grid">
            <Panel title="MARKET SNAPSHOT" subtitle={pulse.snapshot?.commentary || 'Index + regime context'}>
              {(pulse.snapshot?.indices || []).map((idx) => (
                <MetricCell
                  key={idx.name}
                  label={idx.name || 'Index'}
                  value={`${idx.price ?? '—'} · ${idx.chg_pct == null ? '—' : `${idx.chg_pct > 0 ? '+' : ''}${idx.chg_pct}%`}`}
                />
              ))}
              {pulse.snapshot?.options_stance?.honesty && (
                <p className="panel-copy">{pulse.snapshot.options_stance.honesty}</p>
              )}
            </Panel>

            <Panel title="SECTOR HEAT" subtitle="Bhav-averaged sector moves · packs only">
              {!sectors?.available && <EmptyState title="Sector heat unavailable" detail={sectors?.message} />}
              {(sectors?.leaders || []).slice(0, 5).map((row) => (
                <div key={`lead-${row.sector}`} className="fact-grid" style={{ marginBottom: 6 }}>
                  <div><span>Leader</span><strong>{row.sector}</strong></div>
                  <div><span>1D</span><strong>{row.chg_1d}%</strong></div>
                  <div><span>5D</span><strong>{row.chg_5d}%</strong></div>
                </div>
              ))}
              {(sectors?.laggards || []).slice(0, 3).map((row) => (
                <div key={`lag-${row.sector}`} className="fact-grid" style={{ marginBottom: 6 }}>
                  <div><span>Laggard</span><strong>{row.sector}</strong></div>
                  <div><span>1D</span><strong>{row.chg_1d}%</strong></div>
                </div>
              ))}
            </Panel>
          </div>

          <div className="stock-context-grid">
            <Panel title="BUZZING" subtitle="Biggest move × relative volume in scan">
              {!pulse.buzzing && <EmptyState title="No buzz name" detail="No ≥3% move with ≥2× volume in the scan." />}
              {pulse.buzzing && (
                <>
                  <StockChip
                    symbol={pulse.buzzing.symbol}
                    detail={`${pulse.buzzing.change_pct ?? '—'}% · ${pulse.buzzing.volume_ratio ?? '—'}× vol`}
                    onOpen={openSymbol}
                  />
                  <p className="panel-copy">{pulse.buzzing.note || pulse.buzzing.why}</p>
                </>
              )}
            </Panel>
            <Panel title="GAINING STRENGTH" subtitle="Closest pre-breakout / accumulation">
              {!pulse.strength && <EmptyState title="No strength candidate" />}
              {pulse.strength && (
                <>
                  <StockChip
                    symbol={pulse.strength.symbol}
                    detail={
                      pulse.strength.pivot_distance_pct == null
                        ? pulse.strength.status
                        : `${pulse.strength.pivot_distance_pct}% from pivot`
                    }
                    onOpen={openSymbol}
                  />
                  <p className="panel-copy">{pulse.strength.why || (pulse.strength.reasons || [])[0]}</p>
                </>
              )}
            </Panel>
            <Panel title="LOSING MOMENTUM" subtitle="Liquid breakdown under 50-day average">
              {!pulse.weak && <EmptyState title="No breakdown highlight" />}
              {pulse.weak && (
                <>
                  <StockChip
                    symbol={pulse.weak.symbol}
                    detail={pulse.weak.chg_5d == null ? undefined : `${pulse.weak.chg_5d}% / 5d`}
                    onOpen={openSymbol}
                  />
                  <p className="panel-copy">{pulse.weak.note}</p>
                </>
              )}
            </Panel>
          </div>

          <div className="stock-context-grid">
            <Panel title="GAINERS / LOSERS" subtitle="Liquid bhav session movers">
              <div className="evidence-grid">
                <EvidenceList
                  title="Gainers"
                  tone="green"
                  items={(pulse.gainers || []).map((r) => `${r.symbol}: ${r.chg_pct}%`)}
                />
                <EvidenceList
                  title="Losers"
                  tone="red"
                  items={(pulse.losers || []).map((r) => `${r.symbol}: ${r.chg_pct}%`)}
                />
              </div>
            </Panel>
            <Panel title="RELATIVE STRENGTH" subtitle="Durable scan-score leaders (proxy, not a separate RS model)">
              {(pulse.relative_strength || []).length === 0 && <EmptyState title="No RS leaders" />}
              {(pulse.relative_strength || []).map((row) => (
                <div key={row.symbol} style={{ marginBottom: 8 }}>
                  <StockChip symbol={row.symbol} detail={`score ${row.score ?? '—'}`} onOpen={openSymbol} />
                  <small>{row.why}</small>
                </div>
              ))}
            </Panel>
          </div>

          <div className="stock-context-grid">
            <Panel title="BREAKOUTS TODAY" subtitle="Sniper confirms first, then scan breakouts">
              {(pulse.breakouts_today || []).length === 0 && <EmptyState title="No confirmed breakouts" />}
              {(pulse.breakouts_today || []).map((row) => (
                <div key={`brk-${row.symbol}`} style={{ marginBottom: 8 }}>
                  <StockChip symbol={row.symbol} detail={row.status || 'breakout'} onOpen={openSymbol} />
                  <small>{row.why || (row.reasons || [])[0]}</small>
                </div>
              ))}
            </Panel>
            <Panel title="TOMORROW WATCH" subtitle="Near-pivot pre-breakout names">
              {(pulse.breakouts_tomorrow || []).length === 0 && <EmptyState title="No near-pivot watch" />}
              {(pulse.breakouts_tomorrow || []).map((row) => (
                <div key={`watch-${row.symbol}`} style={{ marginBottom: 8 }}>
                  <StockChip
                    symbol={row.symbol}
                    detail={
                      row.pivot_distance_pct == null ? 'pre-breakout' : `${row.pivot_distance_pct}% to pivot`
                    }
                    onOpen={openSymbol}
                  />
                </div>
              ))}
            </Panel>
          </div>

          <div className="stock-context-grid">
            <Panel title="GLOBAL / US CUES" subtitle="Soft-fail — omitted when feeds are down">
              {(pulse.global_cues || []).length === 0 && <EmptyState title="Cues unavailable" />}
              {(pulse.global_cues || []).map((cue) => (
                <MetricCell
                  key={cue.name}
                  label={cue.name || '—'}
                  value={`${cue.price ?? '—'} · ${cue.chg_pct == null ? '—' : `${cue.chg_pct}%`}`}
                  hint={cue.source}
                />
              ))}
            </Panel>
            <Panel title="TOP UPDATES" subtitle="Curator/fetcher headlines only">
              <EvidenceList title="Headlines" items={pulse.headlines || []} tone="cyan" />
            </Panel>
          </div>

          {(pulse.gaps || []).length > 0 && (
            <Panel title="GAPS DISCLOSED" subtitle="Missing evidence stays missing">
              <EvidenceList title="Unavailable / incomplete" items={pulse.gaps} tone="red" />
              <StatusBadge status="INCOMPLETE" />
            </Panel>
          )}
        </>
      )}
    </section>
  )
}
