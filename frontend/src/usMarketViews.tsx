import { useCallback, useEffect, useMemo, useState } from 'react'
import { sendControl } from './api'
import { MetricCard, Panel } from './components'
import {
  fetchUsDashboard,
  fetchUsStock,
  type UsDashboard,
  type UsScanRecord,
  type UsStockWorkspace,
} from './productApi'
import type { ControlName } from './types'

function usd(value: number | null | undefined): string {
  if (value == null || Number.isNaN(Number(value))) return '—'
  return `$${Number(value).toLocaleString('en-US', { maximumFractionDigits: 2 })}`
}

export function UsMarketHome({
  setActive,
  setSelected,
}: {
  setActive: (page: string) => void
  setSelected: (symbol: string) => void
}) {
  const [dash, setDash] = useState<UsDashboard | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState('')

  const load = useCallback(async () => {
    try {
      const payload = await fetchUsDashboard()
      setDash(payload)
      setError('')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'US dashboard unavailable')
    }
  }, [])

  useEffect(() => {
    void load()
    const timer = window.setInterval(() => void load(), 15_000)
    return () => window.clearInterval(timer)
  }, [load])

  const run = async (control: ControlName, label: string) => {
    setBusy(label)
    try {
      await sendControl(control)
      await load()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'US control failed')
    } finally {
      setBusy('')
    }
  }

  const readiness = dash?.readiness
  const overview = dash?.overview
  const scan = dash?.scan
  const top = (scan?.records || []).slice(0, 8)

  return (
    <section className="workspace-view us-market-view">
      <div className="feature-purpose">
        <strong>US Market · retail plane</strong>
        <p>
          NASDAQ Trader listings + Yahoo Finance daily bars. Same setup engine as NSE, S&P relative strength,
          paper autopilot only — never a live US broker order.
        </p>
      </div>
      <div className="inline-actions">
        <button type="button" disabled={!!busy} onClick={() => void run('RUN_US_DATA_PREPARE_NOW', 'Preparing US history…')}>
          {busy.startsWith('Preparing') ? busy : 'Prepare US history'}
        </button>
        <button type="button" disabled={!!busy} onClick={() => void run('RUN_US_SCAN_NOW', 'Scanning US…')}>
          {busy.startsWith('Scanning') ? busy : 'Scan US market (S&P 500)'}
        </button>
        <button type="button" onClick={() => setActive('US Scanner')}>Open US Scanner</button>
        <button type="button" onClick={() => void load()}>Reload</button>
      </div>
      {error ? <div className="large-empty">{error}</div> : null}
      <div className="view-metrics">
        <MetricCard
          label="SESSION"
          value={overview?.session_label || '—'}
          detail={overview?.timezone || 'America/New_York'}
          tone={overview?.session_open ? 'green' : 'amber'}
        />
        <MetricCard
          label="READINESS"
          value={String(readiness?.state || '—')}
          detail={readiness?.recommended_action || '—'}
          tone={readiness?.state === 'READY' ? 'green' : 'amber'}
        />
        <MetricCard
          label="US SETUPS"
          value={String(scan?.summary?.with_any_setup ?? '—')}
          detail={`Scope ${scan?.scope || '—'} · ${scan?.universe_size || 0} names`}
          tone="purple"
        />
        <MetricCard
          label="HISTORY CACHE"
          value={String(readiness?.history?.symbols ?? '—')}
          detail={`Latest ${readiness?.history?.latest_date || '—'} · Yahoo EOD`}
        />
      </div>
      {dash?.honesty ? <p className="edu-honesty">{dash.honesty}</p> : null}
      <div className="us-index-strip">
        {(overview?.indices || []).map((idx) => (
          <div key={idx.symbol}>
            <span>{idx.label}</span>
            <strong>{idx.available ? usd(idx.price) : '—'}</strong>
          </div>
        ))}
      </div>
      <div className="us-home-grid">
        <Panel title="TOP US SETUPS" subtitle={scan?.scanned_at ? `As of ${scan.scanned_at}` : 'No US scan persisted yet'}>
          {!top.length && <div className="large-empty">Prepare history, run US scan, then setups appear here.</div>}
          <ul className="us-setup-list">
            {top.map((row) => (
              <li key={row.symbol}>
                <button
                  type="button"
                  onClick={() => {
                    setSelected(row.symbol)
                    setActive('US Stock')
                  }}
                >
                  <b>{row.symbol}</b>
                  <span>{row.verdict || row.status}</span>
                  <small>{usd(row.price)} · score {row.score ?? '—'}</small>
                </button>
              </li>
            ))}
          </ul>
        </Panel>
        <Panel title="DATA LANES" subtitle="Honest US readiness — missing stays missing">
          <div className="us-lane-list">
            {(readiness?.lanes || []).map((lane) => (
              <div key={lane.key}>
                <strong>{lane.label}</strong>
                <span className={lane.available ? 'positive' : 'negative'}>{lane.status}</span>
                <small>{lane.details}</small>
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </section>
  )
}

export function UsScannerView({
  setActive,
  setSelected,
}: {
  setActive: (page: string) => void
  setSelected: (symbol: string) => void
}) {
  const [dash, setDash] = useState<UsDashboard | null>(null)
  const [filter, setFilter] = useState<'All' | 'Ready' | 'Breakout' | 'Momentum'>('All')
  const [busy, setBusy] = useState(false)

  const load = useCallback(async () => {
    setDash(await fetchUsDashboard())
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  const rows = useMemo(() => {
    const all = dash?.scan?.records || []
    if (filter === 'Ready') return all.filter((r) => r.status === 'Ready to trade')
    if (filter === 'Breakout') return all.filter((r) => (r.signals || []).some((s) => String(s).includes('BREAKOUT')))
    if (filter === 'Momentum') return all.filter((r) => (r.signals || []).includes('MOMENTUM'))
    return all
  }, [dash, filter])

  const startScan = async () => {
    setBusy(true)
    try {
      await sendControl('RUN_US_SCAN_NOW')
      await load()
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="workspace-view us-market-view">
      <div className="feature-purpose">
        <strong>US Scanner</strong>
        <p>Liquid S&P 500 scope by default · quality floor ($5 / ~$10M turnover) · no US options overlay.</p>
      </div>
      <div className="inline-actions">
        <button type="button" disabled={busy} onClick={() => void startScan()}>
          {busy ? 'Scan queued…' : 'Scan US now'}
        </button>
        <button type="button" onClick={() => setActive('US Market')}>US Home</button>
      </div>
      <div className="mode-tabs">
        {(['All', 'Ready', 'Breakout', 'Momentum'] as const).map((key) => (
          <button key={key} type="button" className={filter === key ? 'active' : ''} onClick={() => setFilter(key)}>
            {key}
          </button>
        ))}
      </div>
      <Panel title={`US SETUPS · ${rows.length}`} subtitle={dash?.scan?.honesty || 'Yahoo daily bars'}>
        <div className="us-scan-table">
          <div className="us-scan-head">
            <span>SYMBOL</span><span>STATUS</span><span>PRICE</span><span>SCORE</span><span>ENTRY</span><span>STOP</span><span>TARGET</span>
          </div>
          {!rows.length && <div className="large-empty">No US setups in this filter. Run a scan after history prepare.</div>}
          {rows.map((row: UsScanRecord) => (
            <button
              key={row.symbol}
              type="button"
              className="us-scan-row"
              onClick={() => {
                setSelected(row.symbol)
                setActive('US Stock')
              }}
            >
              <strong>{row.symbol}</strong>
              <span>{row.status || row.verdict}</span>
              <span>{usd(row.price)}</span>
              <span>{row.score ?? '—'}</span>
              <span>{usd(row.entry)}</span>
              <span>{usd(row.stop)}</span>
              <span>{usd(row.target)}</span>
            </button>
          ))}
        </div>
      </Panel>
    </section>
  )
}

export function UsStockView({
  symbol,
  setSymbol,
}: {
  symbol: string
  setSymbol: (symbol: string) => void
}) {
  const [query, setQuery] = useState(symbol || 'AAPL')
  const [workspace, setWorkspace] = useState<UsStockWorkspace | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!symbol) return
    setQuery(symbol)
    fetchUsStock(symbol)
      .then(setWorkspace)
      .catch((reason) => setError(reason instanceof Error ? reason.message : 'US stock unavailable'))
  }, [symbol])

  const open = () => {
    const next = query.trim().toUpperCase()
    if (next) setSymbol(next)
  }

  const row = workspace?.scan_row
  const last = workspace?.bars?.length ? workspace.bars[workspace.bars.length - 1] : null

  return (
    <section className="workspace-view us-market-view">
      <div className="inline-actions">
        <input
          className="inline-search"
          value={query}
          onChange={(event) => setQuery(event.target.value.toUpperCase())}
          onKeyDown={(event) => { if (event.key === 'Enter') open() }}
          placeholder="US ticker e.g. AAPL"
          aria-label="US symbol"
        />
        <button type="button" onClick={open}>Open</button>
      </div>
      {error ? <div className="large-empty">{error}</div> : null}
      <div className="view-metrics">
        <MetricCard label="SYMBOL" value={workspace?.symbol || symbol || '—'} detail={workspace?.company || 'US equity'} />
        <MetricCard label="LAST CLOSE" value={last ? usd(last.close) : '—'} detail={last?.time || 'Yahoo EOD'} tone="green" />
        <MetricCard label="SETUP" value={String(row?.verdict || '—')} detail={row?.status || 'No scan row'} tone="purple" />
        <MetricCard
          label="R:R"
          value={
            row?.entry && row?.stop && row?.target
              ? ((row.target - row.entry) / Math.max(0.01, row.entry - row.stop)).toFixed(1)
              : '—'
          }
          detail="From last US scan"
        />
      </div>
      {workspace?.honesty ? <p className="edu-honesty">{workspace.honesty}</p> : null}
      <div className="us-home-grid">
        <Panel title={`CHART · ${workspace?.symbol || '—'}`} subtitle={`${workspace?.bars?.length || 0} daily bars · ${workspace?.history_source || 'yfinance'}`}>
          {!workspace?.bars?.length && <div className="large-empty">No US history yet — run Prepare US history.</div>}
          {!!workspace?.bars?.length && (
            <div className="us-bar-spark">
              {workspace.bars.slice(-60).map((bar) => (
                <i
                  key={bar.time}
                  title={`${bar.time} ${usd(bar.close)}`}
                  style={{ height: `${Math.max(8, Math.min(100, (bar.close / (last?.close || bar.close)) * 48))}px` }}
                />
              ))}
            </div>
          )}
          {row && (
            <div className="us-setup-meta">
              <div>Entry {usd(row.entry)}</div>
              <div>Stop {usd(row.stop)}</div>
              <div>Target {usd(row.target)}</div>
              <div>Score {row.score ?? '—'}</div>
            </div>
          )}
        </Panel>
        <Panel title="WHAT IS MISSING" subtitle="Retail honesty — unavailable is unavailable">
          <div className="us-lane-list">
            <div>
              <strong>Fundamentals</strong>
              <span className="negative">NOT WIRED</span>
              <small>{workspace?.fundamentals?.message}</small>
            </div>
            <div>
              <strong>US options</strong>
              <span className="negative">NOT AVAILABLE</span>
              <small>{workspace?.options?.message}</small>
            </div>
            <div>
              <strong>Live broker</strong>
              <span className="negative">PAPER ONLY</span>
              <small>US autopilot journals paper trades — no Alpaca/live adapter yet.</small>
            </div>
          </div>
        </Panel>
      </div>
    </section>
  )
}
