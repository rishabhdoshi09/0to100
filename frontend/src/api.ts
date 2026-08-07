import type {
  ChartBar,
  ControlName,
  DashboardPayload,
  OperationRecord,
  OptionsChainPayload,
  OptionsEodHistoryPayload,
  SniperBoardPayload,
} from './types'

const json = async <T>(response: Response): Promise<T> => {
  if (!response.ok) {
    const body = await response.text()
    throw new Error(body || `Request failed with ${response.status}`)
  }
  return response.json() as Promise<T>
}

export const fetchDashboard = (opts?: { timeoutMs?: number }): Promise<DashboardPayload> => {
  const timeoutMs = Math.max(5_000, Number(opts?.timeoutMs) || 30_000)
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), timeoutMs)
  return fetch('/api/dashboard', {
    headers: { Accept: 'application/json' },
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        const body = await response.text()
        throw new Error(
          body?.trim()
            || `Dashboard HTTP ${response.status} — is http://127.0.0.1:8765 up?`,
        )
      }
      return response.json() as Promise<DashboardPayload>
    })
    .catch((reason) => {
      if (reason instanceof DOMException && reason.name === 'AbortError') {
        throw new Error(
          `Dashboard timed out after ${Math.round(timeoutMs / 1000)}s. Check Terminal API on :8765 and market-ops worker; then Retry.`,
        )
      }
      if (reason instanceof TypeError) {
        throw new Error(
          'Cannot reach /api/dashboard (proxy → :8765). Run: bash scripts/stop_quantterm.sh && bash scripts/run_quantterm_low_power.sh',
        )
      }
      throw reason
    })
    .finally(() => window.clearTimeout(timer))
}

export const fetchChart = (symbol: string): Promise<{ symbol: string; bars: ChartBar[] }> =>
  fetch(`/api/chart/${encodeURIComponent(symbol)}`, {
    headers: { Accept: 'application/json' },
  }).then((response) => json<{ symbol: string; bars: ChartBar[] }>(response))

export const fetchOperation = (operationId: string): Promise<OperationRecord> =>
  fetch(`/api/operations/${encodeURIComponent(operationId)}`, {
    headers: { Accept: 'application/json' },
  }).then((response) => json<OperationRecord>(response))

export const fetchMarketOptions = (symbol: string, force = false): Promise<OptionsChainPayload> =>
  fetch(`/api/market/options/${encodeURIComponent(symbol)}?force=${force ? 'true' : 'false'}`, {
    headers: { Accept: 'application/json' },
  }).then((response) => json<OptionsChainPayload>(response))

export const fetchOptionsEodHistory = (
  symbol: string,
  days = 14,
): Promise<OptionsEodHistoryPayload> =>
  fetch(`/api/market/options/${encodeURIComponent(symbol)}/history?days=${days}`, {
    headers: { Accept: 'application/json' },
  }).then((response) => json<OptionsEodHistoryPayload>(response))

export const fetchOperationsPayload = (): Promise<DashboardPayload['operations']> =>
  fetch('/api/operations', { headers: { Accept: 'application/json' } })
    .then((response) => json<DashboardPayload['operations']>(response))

export const fetchSniperBoard = (): Promise<SniperBoardPayload> =>
  fetch('/api/sniper-board', { headers: { Accept: 'application/json' } })
    .then((response) => json<SniperBoardPayload>(response))

export type StreetPulseStock = {
  symbol?: string
  company?: string
  price?: number | null
  change_pct?: number | null
  chg_pct?: number | null
  chg_5d?: number | null
  volume_ratio?: number | null
  score?: number | null
  entry?: number | null
  stop?: number | null
  target?: number | null
  pivot_distance_pct?: number | null
  status?: string
  verdict?: string
  note?: string
  why?: string
  reasons?: string[]
  signals?: string[]
  chase_risk?: boolean
}

export type StreetPulsePayload = {
  available: boolean
  report_type?: string
  title?: string
  date?: string
  generated_at?: string
  takeaways?: string[]
  snapshot?: {
    indices?: Array<{ name?: string; price?: number; chg_pct?: number }>
    commentary?: string
    regime?: string
    options_stance?: {
      stance?: string
      score?: number | null
      confidence?: number
      headline?: string
      honesty?: string
      consider_for?: string[]
    }
    options?: Record<string, unknown>
  }
  sectors?: {
    available?: boolean
    leaders?: Array<{ sector?: string; chg_1d?: number; chg_5d?: number; members?: number }>
    laggards?: Array<{ sector?: string; chg_1d?: number; chg_5d?: number; members?: number }>
    message?: string
  }
  gainers?: StreetPulseStock[]
  losers?: StreetPulseStock[]
  buzzing?: StreetPulseStock | null
  strength?: StreetPulseStock | null
  weak?: StreetPulseStock | null
  relative_strength?: StreetPulseStock[]
  breakouts_today?: StreetPulseStock[]
  breakouts_tomorrow?: StreetPulseStock[]
  global_cues?: Array<{ name?: string; price?: number; chg_pct?: number; source?: string }>
  headlines?: string[]
  day_stories?: Array<{
    headline?: string
    wrap_line?: string
    source?: string
    event_type?: string
    mentioned_symbols?: string[]
    wrap_score?: number
  }>
  wrap_of_the_day?: WrapOfTheDayPayload
  holdings_desk?: HoldingsDeskPayload
  scanned?: number
  scan_as_of?: string
  scan_source?: string
  gaps?: string[]
  places_orders?: boolean
  live_locked?: boolean
  signal_desk?: boolean
  honesty?: string
  disclaimer?: string
  error?: string
}

export type HoldingsDeskRow = {
  tradingsymbol?: string
  symbol?: string
  quantity?: number
  average_price?: number | null
  last_price?: number | null
  pnl?: number | null
  pnl_pct?: number | null
  vs_entry_pct?: number | null
  horizon?: string
  stance?: string
  suggestion?: string
  confidence?: number
  thesis?: string
  fund_brief?: string
  suggestions?: string[]
  price_plan?: {
    horizon?: string
    range_low?: number | null
    range_high?: number | null
    stop_watch?: number | null
    target_watch?: number | null
    target_note?: string
    reward_risk?: number | null
    note?: string
  }
  fundamentals?: {
    available?: boolean
    severity?: string
    status?: string
    ratios?: Record<string, number | null | undefined>
    flags?: Array<{ severity?: string; code?: string; text?: string }>
    brief?: string
    note?: string
  }
  technicals?: {
    available?: boolean
    severity?: string
    status_label?: string
    risk_score?: number | null
    warnings?: Array<{ severity?: string; code?: string; text?: string } | string>
    structure?: Record<string, unknown>
    as_of?: string
  }
  news?: {
    available?: boolean
    bias?: string
    label?: string
    positive?: number
    negative?: number
    headlines?: Array<{ title?: string; tone?: string; source?: string }>
  }
  places_orders?: boolean
  honesty?: string
}

export type HoldingsDeskPayload = {
  available: boolean
  title?: string
  generated_at?: string
  holdings_count?: number
  rows?: HoldingsDeskRow[]
  summary?: Record<string, number>
  market_flows?: {
    available?: boolean
    bias?: string
    bias_label?: string
    bias_note?: string
    fii_net_cr?: number | null
    dii_net_cr?: number | null
    as_of?: string
  }
  message?: string
  places_orders?: boolean
  honesty?: string
  telegram?: { sent?: boolean; reason?: string; count?: number; configured?: boolean }
}

export type WrapOfTheDayPayload = {
  available: boolean
  date?: string
  title?: string
  bullets?: string[]
  source?: string
  auto?: boolean
  override?: boolean
  raw_text?: string
  updated_at?: string
  gaps?: string[]
  message?: string
  places_orders?: boolean
  honesty?: string
  telegram?: { sent?: boolean; reason?: string; count?: number; date?: string; source?: string }
}

export type QuoteTick = {
  symbol: string
  price: number
  chg_pct?: number | null
  volume?: number | null
  high?: number | null
  low?: number | null
  age_s?: number | null
  source?: string
  streaming?: boolean
}

export type QuoteHeartbeatPayload = {
  available: boolean
  session_open?: boolean
  streaming?: boolean
  watching?: number
  requested?: string[]
  missing?: string[]
  quotes?: Record<string, QuoteTick>
  rows?: QuoteTick[]
  sources?: string[]
  max_age_s?: number | null
  honesty?: string
  error?: string
}

export const fetchQuoteHeartbeat = (symbols: string[], limit = 40): Promise<QuoteHeartbeatPayload> => {
  const qs = encodeURIComponent(symbols.filter(Boolean).join(','))
  return fetch(`/api/quotes/heartbeat?symbols=${qs}&limit=${limit}`, {
    headers: { Accept: 'application/json' },
  }).then((response) => json<QuoteHeartbeatPayload>(response))
}

export const fetchStreetPulse = (force = false): Promise<StreetPulsePayload> =>
  fetch(`/api/street-pulse?force=${force ? 'true' : 'false'}`, {
    headers: { Accept: 'application/json' },
  }).then((response) => json<StreetPulsePayload>(response))

export const sendStreetPulseTelegram = (force = true): Promise<{
  sent: boolean
  configured?: boolean
  date?: string
  error?: string | null
  places_orders?: boolean
}> =>
  fetch(`/api/street-pulse/telegram?force=${force ? 'true' : 'false'}`, {
    method: 'POST',
    headers: { Accept: 'application/json' },
  }).then((response) => json<{
    sent: boolean
    configured?: boolean
    date?: string
    error?: string | null
    places_orders?: boolean
  }>(response))

export const fetchWrapOfTheDay = (): Promise<WrapOfTheDayPayload> =>
  fetch('/api/wrap-of-the-day', { headers: { Accept: 'application/json' } })
    .then((response) => json<WrapOfTheDayPayload>(response))

export const saveWrapOfTheDay = (body: {
  text?: string
  bullets?: string[]
  date?: string
  notify?: boolean
  source?: string
}): Promise<WrapOfTheDayPayload> =>
  fetch('/api/wrap-of-the-day', {
    method: 'POST',
    headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then((response) => json<WrapOfTheDayPayload>(response))

export const notifyWrapOfTheDay = (): Promise<{
  accepted: boolean
  telegram?: { sent?: boolean; reason?: string; count?: number; date?: string; source?: string }
  available?: boolean
  count?: number
  date?: string
  source?: string
}> =>
  fetch('/api/wrap-of-the-day/notify', {
    method: 'POST',
    headers: { Accept: 'application/json' },
  }).then((response) => json(response))

export const rebuildWrapOfTheDay = (): Promise<WrapOfTheDayPayload> =>
  fetch('/api/wrap-of-the-day/rebuild', {
    method: 'POST',
    headers: { Accept: 'application/json' },
  }).then((response) => json<WrapOfTheDayPayload>(response))

export const clearWrapOverride = (): Promise<WrapOfTheDayPayload> =>
  fetch('/api/wrap-of-the-day/clear-override', {
    method: 'POST',
    headers: { Accept: 'application/json' },
  }).then((response) => json<WrapOfTheDayPayload>(response))

export const fetchHoldingsDesk = (): Promise<HoldingsDeskPayload> =>
  fetch('/api/holdings-desk', { headers: { Accept: 'application/json' } })
    .then((response) => json<HoldingsDeskPayload>(response))

export const runHoldingsDesk = (notify = false): Promise<HoldingsDeskPayload> =>
  fetch('/api/holdings-desk/run', {
    method: 'POST',
    headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
    body: JSON.stringify({ notify }),
  }).then((response) => json<HoldingsDeskPayload>(response))

export const notifyHoldingsDesk = (): Promise<{
  accepted: boolean
  telegram?: { sent?: boolean; reason?: string; count?: number; configured?: boolean }
  available?: boolean
  count?: number
}> =>
  fetch('/api/holdings-desk/notify', {
    method: 'POST',
    headers: { Accept: 'application/json' },
  }).then((response) => json(response))

export const sendControl = (
  control: ControlName,
): Promise<{
  accepted: boolean
  control: string
  control_id?: string
  operation_id?: string
  operation_status?: string
  created?: boolean
}> =>
  fetch(`/api/controls/${control}`, {
    method: 'POST',
    headers: { Accept: 'application/json' },
  }).then((response) => json<{
    accepted: boolean
    control: string
    control_id?: string
    operation_id?: string
    operation_status?: string
    created?: boolean
  }>(response))
