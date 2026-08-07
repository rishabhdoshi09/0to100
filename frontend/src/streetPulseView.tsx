import { useEffect, useState } from 'react'
import { EmptyState, MetricCell, StatusBadge } from './designSystem'
import { EvidenceList, MetricCard, Panel } from './components'
import { words } from './format'
import {
  clearWrapOverride,
  fetchStreetPulse,
  notifyHoldingsDesk,
  notifyMarketDecisionBrief,
  notifyWrapOfTheDay,
  rebuildMarketDecisionBrief,
  rebuildWrapOfTheDay,
  runHoldingsDesk,
  saveWrapOfTheDay,
  sendStreetPulseTelegram,
  type HoldingsDeskPayload,
  type MarketDecisionBriefPayload,
  type StreetPulsePayload,
} from './api'

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
  const [wrapText, setWrapText] = useState('')
  const [wrapBusy, setWrapBusy] = useState('')
  const [wrapNote, setWrapNote] = useState('')
  const [showOverride, setShowOverride] = useState(false)
  const [deskBusy, setDeskBusy] = useState('')
  const [deskNote, setDeskNote] = useState('')
  const [desk, setDesk] = useState<HoldingsDeskPayload | null>(null)
  const [brief, setBrief] = useState<MarketDecisionBriefPayload | null>(null)
  const [briefBusy, setBriefBusy] = useState('')
  const [briefNote, setBriefNote] = useState('')

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError('')
    fetchStreetPulse(token > 0)
      .then((payload) => {
        if (!cancelled) {
          setPulse(payload)
          if (payload.holdings_desk) setDesk(payload.holdings_desk)
          if (payload.market_decision_brief) setBrief(payload.market_decision_brief)
          if (payload.wrap_of_the_day?.override) {
            const bullets = payload.wrap_of_the_day?.bullets || []
            setWrapText(bullets.map((b, i) => `${i + 1}) ${b}`).join('\n\n'))
            setShowOverride(true)
          }
        }
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

  const rebuildBrief = async () => {
    setBriefBusy('rebuild')
    setBriefNote('')
    try {
      const next = await rebuildMarketDecisionBrief()
      setBrief(next)
      setBriefNote(
        next.available
          ? next.message || 'Market Decision Brief rebuilt from live stores'
          : next.message || 'Brief incomplete — missing sections stay missing',
      )
    } catch (err) {
      setBriefNote(err instanceof Error ? err.message : 'Brief rebuild failed')
    } finally {
      setBriefBusy('')
    }
  }

  const sendBriefTelegram = async () => {
    setBriefBusy('notify')
    setBriefNote('')
    try {
      const result = await notifyMarketDecisionBrief()
      setBriefNote(
        result.telegram?.sent
          ? 'Market Decision Brief sent to Telegram'
          : result.telegram?.reason || 'Telegram send failed — check bot token / chat id',
      )
    } catch (err) {
      setBriefNote(err instanceof Error ? err.message : 'Telegram notify failed')
    } finally {
      setBriefBusy('')
    }
  }

  const rebuildWrap = async () => {
    setWrapBusy('rebuild')
    setWrapNote('')
    try {
      const wrap = await rebuildWrapOfTheDay()
      setWrapNote(
        wrap.available
          ? `Rebuilt ${wrap.bullets?.length || 0} system wrap bullet(s) from Pulse stores`
          : wrap.message || 'Wrap rebuild produced no bullets — run a market scan first',
      )
      setToken((n) => n + 1)
    } catch (err) {
      setWrapNote(err instanceof Error ? err.message : 'Wrap rebuild failed')
    } finally {
      setWrapBusy('')
    }
  }

  const saveWrap = async (notify: boolean) => {
    setWrapBusy(notify ? 'notify' : 'save')
    setWrapNote('')
    try {
      const saved = await saveWrapOfTheDay({ text: wrapText, notify, source: 'override' })
      if (!saved.available) {
        setWrapNote(saved.message || 'Override not saved')
        return
      }
      const tg = saved.telegram?.sent
        ? ' · Telegram sent'
        : saved.telegram?.reason
          ? ` · Telegram: ${saved.telegram.reason}`
          : ''
      setWrapNote(`Saved override · ${saved.bullets?.length || 0} bullet(s)${tg}`)
      setToken((n) => n + 1)
    } catch (err) {
      setWrapNote(err instanceof Error ? err.message : 'Override save failed')
    } finally {
      setWrapBusy('')
    }
  }

  const clearOverride = async () => {
    setWrapBusy('clear')
    setWrapNote('')
    try {
      const wrap = await clearWrapOverride()
      setWrapText('')
      setShowOverride(false)
      setWrapNote(
        wrap.available
          ? `Override cleared · restored ${wrap.bullets?.length || 0} system bullet(s)`
          : 'Override cleared',
      )
      setToken((n) => n + 1)
    } catch (err) {
      setWrapNote(err instanceof Error ? err.message : 'Clear override failed')
    } finally {
      setWrapBusy('')
    }
  }

  const sendWrapTelegram = async () => {
    setWrapBusy('notify')
    setWrapNote('')
    try {
      const result = await notifyWrapOfTheDay()
      setWrapNote(
        result.telegram?.sent
          ? `Wrap sent to Telegram · ${result.count ?? 0} bullet(s)`
          : result.telegram?.reason || 'Telegram not sent — check TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in .env',
      )
    } catch (err) {
      setWrapNote(err instanceof Error ? err.message : 'Telegram notify failed')
    } finally {
      setWrapBusy('')
    }
  }

  const runDesk = async (notify: boolean) => {
    setDeskBusy(notify ? 'notify' : 'run')
    setDeskNote('')
    try {
      const scored = await runHoldingsDesk(notify)
      setDesk(scored)
      const tg = scored.telegram?.sent
        ? ' · Telegram sent'
        : scored.telegram?.reason
          ? ` · Telegram: ${scored.telegram.reason}`
          : ''
      setDeskNote(
        scored.available
          ? `${scored.message || `Scored ${scored.holdings_count || 0} holding(s)`}${tg}`
          : scored.message || 'Holdings desk empty — sync Zerodha holdings first',
      )
    } catch (err) {
      setDeskNote(err instanceof Error ? err.message : 'Holdings desk failed')
    } finally {
      setDeskBusy('')
    }
  }

  const sendDeskTelegram = async () => {
    setDeskBusy('notify')
    setDeskNote('')
    try {
      const result = await notifyHoldingsDesk()
      setDeskNote(
        result.telegram?.sent
          ? `Holdings desk sent to Telegram · ${result.count ?? 0} name(s)`
          : result.telegram?.reason || 'Telegram not sent — check TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in .env',
      )
    } catch (err) {
      setDeskNote(err instanceof Error ? err.message : 'Holdings desk Telegram failed')
    } finally {
      setDeskBusy('')
    }
  }

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
        setSendNote(result.error || 'Telegram not configured — set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env')
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

          <Panel
            title="3 THINGS THAT WILL DECIDE THE MARKET"
            subtitle={
              (brief || pulse.market_decision_brief)?.available
                ? `${(brief || pulse.market_decision_brief)?.message || 'Retail morning desk'} · beat sell-side with honesty`
                : 'Gift / global / options zones + long-term fund picks + scanner tech picks — rebuild when you want fresh prints'
            }
          >
            {((brief || pulse.market_decision_brief)?.deciders || []).length === 0 ? (
              <EmptyState
                title="Market Decision Brief not built yet"
                detail="Tap Rebuild brief to compose Gift/premarket, macro prints, options zones, and fund/tech picks from real stores."
              />
            ) : (
              <div style={{ display: 'grid', gap: 14 }}>
                {((brief || pulse.market_decision_brief)?.deciders || []).map((decider, idx) => (
                  <div key={decider.key || decider.title || idx}>
                    <div style={{ display: 'flex', gap: 8, alignItems: 'baseline', flexWrap: 'wrap' }}>
                      <strong>
                        {idx + 1}. {decider.title || 'Decider'}
                      </strong>
                      <StatusBadge status={decider.available ? 'READY' : 'INCOMPLETE'} />
                      {decider.gift_hard ? <StatusBadge status="GIFT_HARD" /> : null}
                      {decider.stale ? <StatusBadge status="STALE_CACHE" /> : null}
                    </div>
                    {decider.headline ? <p className="panel-copy">{decider.headline}</p> : null}
                    {(decider.bullets || []).length > 0 ? (
                      <ul className="plain-list">
                        {(decider.bullets || []).slice(0, 5).map((item) => (
                          <li key={item}>{item}</li>
                        ))}
                      </ul>
                    ) : null}
                  </div>
                ))}

                <div>
                  <strong>Fundamental / long-term picks</strong>
                  <p className="panel-copy">
                    {(brief || pulse.market_decision_brief)?.fundamental_picks?.subtitle
                      || 'Multi-month research · prior-high watch when history exists · not broker targets'}
                  </p>
                  {((brief || pulse.market_decision_brief)?.fundamental_picks?.rows || []).length === 0 ? (
                    <p className="panel-copy">
                      {(brief || pulse.market_decision_brief)?.fundamental_picks?.message
                        || 'No long-term picks yet — run Long-Term scan.'}
                    </p>
                  ) : (
                    <div style={{ display: 'grid', gap: 8 }}>
                      {((brief || pulse.market_decision_brief)?.fundamental_picks?.rows || []).map((row) => (
                        <div key={row.symbol} className="fact-grid" style={{ alignItems: 'start' }}>
                          <div>
                            <StockChip
                              symbol={row.symbol}
                              detail={
                                row.upside_to_prior_high_pct != null
                                  ? `prior-high ~${row.upside_to_prior_high_pct}% · ${row.verdict || 'LONG_TERM'}`
                                  : row.verdict || 'LONG_TERM'
                              }
                              onOpen={openSymbol}
                            />
                            <p className="panel-copy">{row.thesis || row.note}</p>
                          </div>
                          <div>
                            <MetricCell label="PRICE" value={row.price != null ? `₹${row.price}` : '—'} />
                            <MetricCell
                              label="PRIOR-HIGH WATCH"
                              value={row.target_watch != null ? `₹${row.target_watch}` : '—'}
                              hint="Official history — not a sell-side TP"
                            />
                            <MetricCell label="SCORE" value={row.score != null ? String(row.score) : '—'} />
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div>
                  <strong>Technical picks</strong>
                  <p className="panel-copy">
                    {(brief || pulse.market_decision_brief)?.technical_picks?.subtitle
                      || 'Short-term research from last whole-market scan'}
                  </p>
                  {((brief || pulse.market_decision_brief)?.technical_picks?.rows || []).length === 0 ? (
                    <p className="panel-copy">
                      {(brief || pulse.market_decision_brief)?.technical_picks?.message
                        || 'No scanner setups yet — run Scan now.'}
                    </p>
                  ) : (
                    <div style={{ display: 'grid', gap: 8 }}>
                      {((brief || pulse.market_decision_brief)?.technical_picks?.rows || []).map((row) => (
                        <div key={row.symbol} className="fact-grid" style={{ alignItems: 'start' }}>
                          <div>
                            <StockChip
                              symbol={row.symbol}
                              detail={
                                row.upside_pct != null
                                  ? `${row.status || row.verdict || 'WATCH'} · ~${row.upside_pct}% to target`
                                  : row.status || row.verdict || 'WATCH'
                              }
                              onOpen={openSymbol}
                            />
                            <p className="panel-copy">{row.why || (row.signals || []).slice(0, 2).join(' · ')}</p>
                          </div>
                          <div>
                            <MetricCell label="ENTRY" value={row.entry != null ? `₹${row.entry}` : '—'} />
                            <MetricCell label="STOP" value={row.stop != null ? `₹${row.stop}` : '—'} />
                            <MetricCell label="TARGET" value={row.target != null ? `₹${row.target}` : '—'} />
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}

            {((brief || pulse.market_decision_brief)?.gaps || []).length > 0 ? (
              <p className="panel-copy">
                Gaps: {((brief || pulse.market_decision_brief)?.gaps || []).slice(0, 4).join(' · ')}
              </p>
            ) : null}
            {((brief || pulse.market_decision_brief)?.why_better || []).length > 0 ? (
              <EvidenceList
                title="Why this beats a broker morning note"
                items={((brief || pulse.market_decision_brief)?.why_better || []).slice(0, 4)}
              />
            ) : null}

            <div className="inline-actions" style={{ marginTop: 8, flexWrap: 'wrap', gap: 8 }}>
              <button type="button" disabled={!!briefBusy} onClick={() => void rebuildBrief()}>
                {briefBusy === 'rebuild' ? 'Composing Gift + globals + levels…' : 'Rebuild brief'}
              </button>
              <button
                type="button"
                disabled={!!briefBusy || !(brief || pulse.market_decision_brief)?.available}
                onClick={() => void sendBriefTelegram()}
              >
                {briefBusy === 'notify' ? 'Sending…' : 'Send brief to Telegram'}
              </button>
            </div>
            {briefNote ? <p className="panel-copy">{briefNote}</p> : null}
            <p className="panel-copy">
              {(brief || pulse.market_decision_brief)?.honesty
                || (brief || pulse.market_decision_brief)?.competitor_note
                || 'Research brief · not a buy ticket · paper-first'}
            </p>
          </Panel>

          <Panel
            title="WRAP OF THE DAY"
            subtitle={
              pulse.wrap_of_the_day?.available
                ? `${pulse.wrap_of_the_day.override ? 'User override' : 'News-led system wrap'} · ${pulse.wrap_of_the_day.date || 'today'} · ${(pulse.day_stories || []).length} day stories`
                : 'Built from Moneycontrol/ET/Mint/BS/CNBC/Google News day stories + tape — missing stays missing'
            }
          >
            {(pulse.wrap_of_the_day?.bullets || []).length > 0 ? (
              <ol className="plain-list">
                {(pulse.wrap_of_the_day?.bullets || []).map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ol>
            ) : (
              <EmptyState
                title="Wrap not ready yet"
                detail={pulse.wrap_of_the_day?.message || 'Tap Rebuild wrap (refreshes market news) or run Refresh news on System Health.'}
              />
            )}
            {(pulse.day_stories || []).length > 0 ? (
              <p className="panel-copy">
                Sources today:{' '}
                {Array.from(new Set((pulse.day_stories || []).map((s) => s.source).filter(Boolean))).slice(0, 6).join(' · ')
                  || 'curator'}
              </p>
            ) : null}
            {(pulse.wrap_of_the_day?.gaps || []).length > 0 ? (
              <p className="panel-copy">Gaps: {(pulse.wrap_of_the_day?.gaps || []).slice(0, 3).join(' · ')}</p>
            ) : null}
            <div className="inline-actions" style={{ marginTop: 8, flexWrap: 'wrap', gap: 8 }}>
              <button type="button" disabled={!!wrapBusy} onClick={() => void rebuildWrap()}>
                {wrapBusy === 'rebuild' ? 'Refreshing news + wrap…' : 'Rebuild wrap'}
              </button>
              <button
                type="button"
                disabled={!!wrapBusy || !(pulse.wrap_of_the_day?.available)}
                onClick={() => void sendWrapTelegram()}
              >
                {wrapBusy === 'notify' ? 'Sending…' : 'Send wrap to Telegram'}
              </button>
              {pulse.wrap_of_the_day?.override ? (
                <button type="button" disabled={!!wrapBusy} onClick={() => void clearOverride()}>
                  {wrapBusy === 'clear' ? 'Clearing…' : 'Clear override'}
                </button>
              ) : (
                <button type="button" onClick={() => setShowOverride((v) => !v)}>
                  {showOverride ? 'Hide override' : 'Optional override'}
                </button>
              )}
            </div>
            {showOverride ? (
              <div style={{ marginTop: 12 }}>
                <p className="panel-copy">Optional only — system wrap is the default. Paste replaces today’s auto wrap until cleared.</p>
                <textarea
                  aria-label="Optional Wrap of the Day override"
                  placeholder={"1) ...\n2) ..."}
                  value={wrapText}
                  onChange={(event) => setWrapText(event.target.value)}
                  rows={5}
                  style={{ width: '100%', fontFamily: 'inherit' }}
                />
                <div className="inline-actions" style={{ marginTop: 8, flexWrap: 'wrap', gap: 8 }}>
                  <button type="button" disabled={!!wrapBusy || !wrapText.trim()} onClick={() => void saveWrap(false)}>
                    {wrapBusy === 'save' ? 'Saving…' : 'Save override'}
                  </button>
                  <button type="button" disabled={!!wrapBusy || !wrapText.trim()} onClick={() => void saveWrap(true)}>
                    Save override + Telegram
                  </button>
                </div>
              </div>
            ) : null}
            {wrapNote ? <p className="panel-copy">{wrapNote}</p> : null}
            <p className="panel-copy">{pulse.wrap_of_the_day?.honesty || pulse.honesty}</p>
          </Panel>

          <Panel
            title="HOLDINGS DESK"
            subtitle={
              (desk || pulse.holdings_desk)?.available
                ? `${(desk || pulse.holdings_desk)?.holdings_count || 0} Zerodha holding(s) · fund → tech → news → research verdict`
                : 'Track demat book: fundamentals → technicals → news good/bad → buy/sell/hold-style watch'
            }
          >
            {((desk || pulse.holdings_desk)?.market_flows?.bias_label
              || (desk || pulse.holdings_desk)?.market_flows?.bias) ? (
              <p className="panel-copy">
                Market FII/DII:{' '}
                <strong>
                  {(desk || pulse.holdings_desk)?.market_flows?.bias_label
                    || (desk || pulse.holdings_desk)?.market_flows?.bias}
                </strong>
                {(desk || pulse.holdings_desk)?.market_flows?.as_of
                  ? ` · ${(desk || pulse.holdings_desk)?.market_flows?.as_of}`
                  : ''}
                {(desk || pulse.holdings_desk)?.market_flows?.bias_note
                  ? ` — ${(desk || pulse.holdings_desk)?.market_flows?.bias_note}`
                  : ''}
              </p>
            ) : null}
            {((desk || pulse.holdings_desk)?.rows || []).length === 0 ? (
              <EmptyState
                title="Holdings desk not scored yet"
                detail={(desk || pulse.holdings_desk)?.message || 'Sync Zerodha holdings on My Holdings, then Run holdings desk. Analyse first — Telegram is a separate step.'}
              />
            ) : (
              <div style={{ display: 'grid', gap: 12 }}>
                {((desk || pulse.holdings_desk)?.rows || []).map((row) => (
                  <div key={row.symbol || row.tradingsymbol} className="fact-grid" style={{ alignItems: 'start' }}>
                    <div>
                      <StockChip
                        symbol={row.symbol || row.tradingsymbol}
                        detail={
                          row.vs_entry_pct == null
                            ? `${row.quantity ?? '—'} sh · ${row.horizon || 'SHORT_TERM'}`
                            : `${row.quantity ?? '—'} sh · vs avg ${row.vs_entry_pct > 0 ? '+' : ''}${row.vs_entry_pct}% · ${row.horizon || 'SHORT_TERM'}`
                        }
                        onOpen={openSymbol}
                      />
                      <div style={{ marginTop: 4, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                        <StatusBadge status={String(row.suggestion || row.stance || 'INCOMPLETE')} />
                        <StatusBadge status={`NEWS_${String(row.news?.bias || 'NONE')}`} />
                        <StatusBadge status={String(row.horizon || 'SHORT_TERM')} />
                      </div>
                      <p className="panel-copy" style={{ marginTop: 6 }}>{row.thesis}</p>
                      <p className="panel-copy">{row.fund_brief || row.fundamentals?.brief || row.fundamentals?.note}</p>
                      {(row.suggestions || []).slice(0, 3).map((tip) => (
                        <p key={tip} className="panel-copy">{tip}</p>
                      ))}
                    </div>
                    <div>
                      <MetricCell
                        label="TECHNICALS"
                        value={words(row.technicals?.status_label || row.technicals?.severity || '—')}
                      />
                      <MetricCell
                        label="FUNDAMENTALS"
                        value={words(row.fundamentals?.severity || '—')}
                      />
                      <MetricCell
                        label="RANGE"
                        value={
                          row.price_plan?.range_low != null || row.price_plan?.range_high != null
                            ? `₹${row.price_plan?.range_low ?? '—'}–₹${row.price_plan?.range_high ?? '—'}`
                            : '—'
                        }
                      />
                      <MetricCell
                        label="STOP / TARGET"
                        value={
                          row.price_plan?.stop_watch != null || row.price_plan?.target_watch != null
                            ? `₹${row.price_plan?.stop_watch ?? '—'} → ₹${row.price_plan?.target_watch ?? '—'}`
                            : '—'
                        }
                        hint={row.price_plan?.target_note || row.price_plan?.note}
                      />
                      <p className="panel-copy">{row.news?.label || row.news?.headlines?.[0]?.title || 'No recent symbol news'}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
            <div className="inline-actions" style={{ marginTop: 8, flexWrap: 'wrap', gap: 8 }}>
              <button type="button" disabled={!!deskBusy} onClick={() => void runDesk(false)}>
                {deskBusy === 'run' ? 'Analysing holdings…' : 'Analyse holdings desk'}
              </button>
              <button
                type="button"
                disabled={!!deskBusy || !((desk || pulse.holdings_desk)?.available)}
                onClick={() => void sendDeskTelegram()}
              >
                {deskBusy === 'notify' ? 'Sending…' : 'Send analysed desk to Telegram'}
              </button>
            </div>
            {deskNote ? <p className="panel-copy">{deskNote}</p> : null}
            <p className="panel-copy">
              {(desk || pulse.holdings_desk)?.honesty
                || 'Research suggestions only — BUY/SELL/HOLD labels are watches, never live orders. Analyse before Telegram.'}
            </p>
          </Panel>

          <Panel title="COVER TAKEAWAYS" subtitle="Same as Wrap of the Day when composed · else store takeaways">
            {(pulse.takeaways || []).length === 0 && (
              <EmptyState title="No takeaways yet" detail="Rebuild pulse after a market scan." />
            )}
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
