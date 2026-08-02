import type { ControlName, ConvictionRecord, LongTermRecord, NewsArticle, ScanRecord } from './types'

export type ProductLane = {
  key: string
  label: string
  meaning: string
  status: 'FRESH' | 'STALE' | 'MISSING' | 'UNKNOWN_DATE' | string
  available: boolean
  as_of: string
  age_seconds: number | null
  max_age_seconds: number
  weight: number
  earned_weight: number
  action: string
  details: string
}

export type ProductReadiness = {
  schema_version: number
  generated_at: string
  score: number
  state: 'READY' | 'PARTIAL' | 'INCOMPLETE' | 'EMPTY' | string
  summary: string
  lanes: ProductLane[]
  blockers: string[]
  recommended_action: string
}

export type IntelligenceMetric = {
  key: string
  label: string
  value: number | string | null
  unit: string
  meaning: string
  interpretation: string
}

export type IntelligenceSource = {
  name: string
  available: boolean
  status: string
  as_of: string
  age_days: number | null
  max_age_days: number
  meaning: string
}

export type StockWorkspace = {
  schema_version: number
  generated_at: string
  symbol: string
  company: string
  sector: string
  state: string
  summary: string
  confidence_pct: number
  gaps: string[]
  technical: {
    available: boolean
    latest_date?: string
    close?: number
    ema20?: number | null
    ema50?: number | null
    ema200?: number | null
    rsi14?: number | null
    atr14?: number | null
    atr_pct?: number | null
    high_52w?: number | null
    low_52w?: number | null
    from_high_pct?: number | null
    volume_ratio?: number | null
    trend: string
    trend_explanation: string
    metrics: IntelligenceMetric[]
  }
  fundamentals: {
    available: boolean
    coverage_pct: number
    score?: number | null
    classification?: string
    quality_factors: string[]
    risk_flags: string[]
    metrics: IntelligenceMetric[]
    key_ratios?: Array<{ name: string; value: string }>
    company_about: string
    fetched_at: string
    section_as_of: Record<string, string>
  }
  peers?: {
    available: boolean
    sector: string
    screener_table: Array<Record<string, unknown>>
    sector_peers: Array<{
      symbol: string
      company: string
      score: number
      status: string
      sector: string
    }>
    average_pe?: number | null
    peer_pe_sample_count?: number
    pe_vs_peer_avg?: number | null
    stock_pe?: number | null
    peer_pe_note?: string
    note?: string
  }
  scanner: ScanRecord
  long_term: LongTermRecord
  news: NewsArticle[]
  fno: Record<string, unknown>
  sources: IntelligenceSource[]
  next_actions: Array<{ control: ControlName | 'REFRESH_STOCK_FUNDAMENTALS'; label: string }>
}

export type CommandCenterWorkspace = {
  generated_at: string
  market_health: string
  market_summary: string
  trade_stance: string
  breadth: string
  scan_universe: number
  momentum_count: number
  ready_count: number
  near_breakout_count: number
  long_term_count: number
  fundamental_coverage_pct: number
  paper_capital: number
  paper_equity: number
  paper_return_pct: number
  open_position_count: number
  open_risk: number
  autonomy_running: boolean
  autonomy_state: string
  heartbeat_ist: string
  top_setups: ScanRecord[]
  top_long_term: LongTermRecord[]
  insights: string[]
}

export type ScannerWorkspaceRow = ScanRecord
  & Partial<ConvictionRecord>
  & Partial<LongTermRecord>
  & { _source?: string }

export type ScannerWorkspace = {
  generated_at: string
  mode: string
  source: string
  scanned_at: string
  universe_size: number
  rows: ScannerWorkspaceRow[]
}

const json = async <T>(response: Response): Promise<T> => {
  if (!response.ok) {
    const body = await response.text()
    throw new Error(body || `Request failed with ${response.status}`)
  }
  return response.json() as Promise<T>
}

export const fetchProductReadiness = (): Promise<ProductReadiness> =>
  fetch('/api/product-readiness', { headers: { Accept: 'application/json' } })
    .then((response) => json<ProductReadiness>(response))

export const bootstrapProduct = (): Promise<{
  accepted: boolean
  message: string
  operations: Array<{ kind: string; operation_id: string; status: string; created: boolean }>
  readiness: ProductReadiness
}> => fetch('/api/product-bootstrap', {
  method: 'POST',
  headers: { Accept: 'application/json' },
}).then((response) => json(response))

export const fetchStockIntelligence = (symbol: string): Promise<StockWorkspace> =>
  fetch(`/api/stock-intelligence/${encodeURIComponent(symbol)}`, {
    headers: { Accept: 'application/json' },
  }).then((response) => json<StockWorkspace>(response))

export type TradePlan = {
  available: boolean
  symbol: string
  message?: string
  tradeable?: boolean
  reason?: string
  entry?: number
  stop?: number
  target?: number | null
  qty?: number
  invested?: number
  rupee_risk?: number
  capped?: boolean
  pct_of_capital?: number
  risk_pct_of_capital?: number
  reward_risk?: number | null
  invalidation_pct?: number
  suggested_risk_pct?: number
  open_risk_pct_before?: number | null
  open_risk_pct_after?: number | null
  heat_verdict?: string
  heat_warnings?: string[]
  correlation_status?: string
  correlated_with?: string[]
  market_health?: string
  market_risk_factor?: number
  capital?: number
  summary?: string
}

// Read-only risk-first plan for a scanned candidate. Never places an order.
export const fetchTradePlan = (symbol: string): Promise<TradePlan> =>
  fetch(`/api/trade-plan/${encodeURIComponent(symbol)}`, {
    headers: { Accept: 'application/json' },
  }).then((response) => json<TradePlan>(response))

export const fetchStockFundamentals = (
  symbol: string,
  force = false,
): Promise<{
  accepted: boolean
  symbol: string
  sections: Record<string, number | boolean>
  workspace: StockWorkspace
}> =>
  fetch(
    `/api/stock-intelligence/${encodeURIComponent(symbol)}/fetch-fundamentals?force=${force ? 'true' : 'false'}`,
    {
      method: 'POST',
      headers: { Accept: 'application/json' },
    },
  ).then((response) => json(response))

export const refreshStockFundamentals = (symbol: string): Promise<{
  accepted: boolean
  symbol: string
  sections: Record<string, number | boolean>
  workspace: StockWorkspace
}> => fetchStockFundamentals(symbol, true)

export const fetchCommandCenterWorkspace = (): Promise<CommandCenterWorkspace> =>
  fetch('/api/command-center-workspace', { headers: { Accept: 'application/json' } })
    .then((response) => json<CommandCenterWorkspace>(response))

export const fetchScannerWorkspace = (mode: string): Promise<ScannerWorkspace> =>
  fetch(`/api/scanner-workspace/${encodeURIComponent(mode)}`, {
    headers: { Accept: 'application/json' },
  }).then((response) => json<ScannerWorkspace>(response))

export type RadarHome = {
  generated_at: string
  market_session: string
  market_health: string
  breadth: string
  nifty_change_1d: number
  vix: number
  leaders: string[]
  laggards: string[]
  scan_scanned_at: string
  long_term_scanned_at: string
  universe_size: number
  lanes: {
    breakouts: ScannerWorkspaceRow[]
    momentum: ScannerWorkspaceRow[]
    long_term_picks: ScannerWorkspaceRow[]
  }
  counts: { breakouts: number; momentum: number; long_term_picks: number }
}

export const fetchRadarHome = (): Promise<RadarHome> =>
  fetch('/api/radar-home', { headers: { Accept: 'application/json' } })
    .then((response) => json<RadarHome>(response))

export type CompareMetric = {
  label: string
  value: unknown
  unit: string
  source: string
  available: boolean
}

export type CompareRow = {
  symbol: string
  company: string
  sector: string
  available: boolean
  error?: string
  confidence_pct?: number
  sections: Record<string, CompareMetric[]>
}

export type CompareWorkspace = {
  schema_version: number
  generated_at: string
  symbols: string[]
  rows: CompareRow[]
  section_labels: Record<string, string>
  disclaimer: string
}

export const fetchCompareWorkspace = (symbols: string[]): Promise<CompareWorkspace> =>
  fetch(`/api/compare?symbols=${encodeURIComponent(symbols.join(','))}`, {
    headers: { Accept: 'application/json' },
  }).then((response) => json<CompareWorkspace>(response))

export type WatchlistItem = {
  id: number
  symbol: string
  added_date: string
  notes?: string
  buy_zone_low?: number | null
  buy_zone_high?: number | null
  target_price?: number | null
  stop_price?: number | null
  added_price?: number | null
  snapshot?: ScannerWorkspaceRow & Record<string, unknown>
}

export type WatchlistPayload = {
  generated_at: string
  items: WatchlistItem[]
  count: number
}

export const fetchWatchlist = (): Promise<WatchlistPayload> =>
  fetch('/api/watchlist', { headers: { Accept: 'application/json' } })
    .then((response) => json<WatchlistPayload>(response))

export const addWatchlistItem = (body: {
  symbol: string
  notes?: string
  buy_zone_low?: number
  buy_zone_high?: number
  target_price?: number
  stop_price?: number
}): Promise<{ accepted: boolean; item: WatchlistItem }> =>
  fetch('/api/watchlist', {
    method: 'POST',
    headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then((response) => json(response))

export const removeWatchlistItem = (rowId: number): Promise<{ accepted: boolean }> =>
  fetch(`/api/watchlist/${rowId}`, { method: 'DELETE', headers: { Accept: 'application/json' } })
    .then((response) => json(response))

export type SymbolRatioRow = {
  key: string
  label: string
  value: number | null
  formula?: string
  period?: string
  scope?: string
  missing_reason?: string
  quality_status?: string
}

export const fetchSymbolRatios = (symbol: string): Promise<{ symbol: string; ratios: SymbolRatioRow[] }> =>
  fetch(`/api/data/ratios/${encodeURIComponent(symbol)}`, { headers: { Accept: 'application/json' } })
    .then((response) => json(response))
