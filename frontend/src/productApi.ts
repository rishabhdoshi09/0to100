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
  retail_research_checklist?: RetailResearchChecklist
}

export type RetailChecklistItem = {
  key: string
  label: string
  status: string
  why_it_matters: string
  next_action: string
  evidence: string
}

export type RetailResearchChecklist = {
  schema_version: number
  summary: string
  ready_count: number
  gap_count: number
  items: RetailChecklistItem[]
  gaps: RetailChecklistItem[]
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
  growth_outlook?: {
    available: boolean
    symbol?: string
    company?: string
    sector?: string
    title?: string
    thesis?: { label?: string; engines?: string[]; text?: string }
    claims?: Array<{
      key: string
      label: string
      value?: number | string | null
      unit?: string
      source?: string
      as_of?: string
      status?: string
      note?: string
    }>
    sections?: Array<{ id: string; title: string; body: string }>
    guidance?: Array<{
      kind?: string
      event_date?: string
      speaker?: string
      topic?: string
      commentary?: string
      guidance_metric?: string
      guidance_value?: string
      guidance_period?: string
      source_url?: string
    }>
    technical?: {
      available?: boolean
      price?: number | null
      trend?: string
      trend_explanation?: string
      as_of?: string
    }
    gaps?: string[]
    summary?: string
    honesty?: string
    places_orders?: boolean
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
    peer_rank?: number
    total_peers?: number
    peer_rank_sector?: string
    peer_rank_score?: number
    peer_rank_verdict?: string
    sector_leader?: boolean
    peer_rank_note?: string
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
    try {
      const parsed = JSON.parse(body) as { detail?: string }
      if (typeof parsed.detail === 'string' && parsed.detail.trim()) {
        throw new Error(parsed.detail)
      }
    } catch {
      // not JSON — use raw body below
    }
    throw new Error(body.trim() || `Request failed with ${response.status}`)
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
  effective_bets_before?: number | null
  effective_bets_after?: number | null
  round_trip_cost_pct?: number | null
  cost_drag_r?: number | null
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

export type SymbolDirectoryRow = {
  symbol: string
  name: string
}

export type SymbolDirectory = {
  schema_version: number
  query: string
  limit: number
  universe_size: number
  count: number
  symbols: SymbolDirectoryRow[]
  letter_coverage?: string[]
  truncated?: boolean
  holdings_pinned?: number
  source: string
  note?: string
}

/** Full NSE equity directory for search — not limited to scan setups. */
export const fetchSymbolDirectory = (opts?: { q?: string; limit?: number }): Promise<SymbolDirectory> => {
  const params = new URLSearchParams()
  if (opts?.q) params.set('q', opts.q)
  // limit=0 → API returns the complete A→Z universe on empty query
  if (opts?.limit != null) params.set('limit', String(opts.limit))
  else if (!opts?.q) params.set('limit', '0')
  const q = params.toString()
  return fetch(`/api/symbols${q ? `?${q}` : ''}`, {
    headers: { Accept: 'application/json' },
  }).then((response) => json<SymbolDirectory>(response))
}

export type PreTradeVerdict = 'GO' | 'CAUTION' | 'NO_GO'

export type PreTrade = {
  schema_version: number
  symbol: string
  available: boolean
  verdict: PreTradeVerdict
  meaning: string
  tradeable: boolean
  blockers: string[]
  warnings: string[]
  plan: TradePlan
  plan_summary: string
  cost_drag_r: number | null
  round_trip_cost_pct?: number | null
  correlation: {
    status: string
    correlated_with: string[]
    effective_bets_before?: number | null
    effective_bets_after?: number | null
    n_positions: number
    n_bets: number
    message: string
  }
  market_throttle: {
    health: string
    market_risk_factor: number | null
    suggested_risk_pct?: number
    trade_stance: string
  }
  data_gaps: Array<{
    key?: string
    label?: string
    status?: string
    next_action?: string
  }>
  paper_snapshot: {
    open_positions: number
    capital?: number
    open_risk_pct?: number | null
  }
  scan: {
    available: boolean
    verdict?: string
    score?: number
    signals?: string[]
    edge_r?: number | null
    entry?: number
    stop?: number
    target?: number | null
  }
  measured_edge_r?: number | null
  learning?: {
    signal_backtest_actionable?: boolean
    evidence_note?: string
    as_of?: string | null
    n_symbols_tested?: number | null
  }
  read_only: boolean
  places_orders: boolean
  honesty: string
}

/** Compose plan + book + market + data gaps into GO/CAUTION/NO_GO. Never places orders. */
export const fetchPreTrade = (symbol: string): Promise<PreTrade> =>
  fetch(`/api/pre-trade/${encodeURIComponent(symbol)}`, {
    headers: { Accept: 'application/json' },
  }).then((response) => json<PreTrade>(response))

export type BookCorrelation = {
  available?: boolean
  n_positions: number
  n_bets: number
  clusters: string[][]
  biggest: string[] | null
  message?: string
}

export const fetchBookCorrelation = (): Promise<BookCorrelation> =>
  fetch('/api/book-correlation', { headers: { Accept: 'application/json' } })
    .then((response) => json<BookCorrelation>(response))

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

export type BuyHealthWarning = {
  severity: string
  code: string
  text: string
}

export type BuyBookHealth = {
  available: boolean
  severity?: string
  status_label?: string
  price?: number | null
  eod_close?: number | null
  live_price?: number | null
  price_source?: string
  as_of?: string
  warnings?: BuyHealthWarning[]
  risk_score?: number
  supports?: { swing_20d?: number | null; swing_60d?: number | null }
  averages?: { ema20?: number | null; ema50?: number | null; ema200?: number | null }
  structure?: Record<string, unknown>
  vs_entry_pct?: number | null
  honesty?: string
}

export type BuyBookItem = {
  id: string
  symbol: string
  entry_price?: number | null
  stop_price?: number | null
  quantity?: number | null
  notes?: string
  status?: string
  added_at?: string
  updated_at?: string
  health?: BuyBookHealth
  severity?: string
  status_label?: string
  price?: number | null
  vs_entry_pct?: number | null
  chg_1d_pct?: number | null
  chg_5d_pct?: number | null
  est_pnl?: number | null
  result_label?: string
}

export type BuyBookResults = {
  with_entry: number
  missing_entry: number
  up: number
  down: number
  flat: number
  avg_vs_entry_pct?: number | null
  est_pnl_total?: number | null
  honesty?: string
}

export type BuyBookPayload = {
  available: boolean
  generated_at?: string
  summary?: {
    total: number
    critical: number
    warn: number
    info: number
    good: number
    unknown: number
  }
  results?: BuyBookResults
  items: BuyBookItem[]
  places_orders?: boolean
  honesty?: string
}

export const fetchBuyBook = (opts?: { fresh?: boolean }): Promise<BuyBookPayload> => {
  const qs = opts?.fresh ? '?fresh=1' : ''
  return fetch(`/api/buy-book${qs}`, { headers: { Accept: 'application/json' } })
    .then((response) => json<BuyBookPayload>(response))
}

export const fetchBuyBookSymbols = (): Promise<{ symbols: string[]; updated_at?: string | null }> =>
  fetch('/api/buy-book/symbols', { headers: { Accept: 'application/json' } })
    .then((response) => json(response))

export const addBuyBookItem = (body: {
  symbol: string
  entry_price?: number
  stop_price?: number
  quantity?: number
  notes?: string
}): Promise<{ accepted: boolean; item: BuyBookItem }> =>
  fetch('/api/buy-book', {
    method: 'POST',
    headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then((response) => json(response))

export const removeBuyBookItem = (itemId: string): Promise<{ accepted: boolean }> =>
  fetch(`/api/buy-book/${encodeURIComponent(itemId)}`, {
    method: 'DELETE',
    headers: { Accept: 'application/json' },
  }).then((response) => json(response))

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

export type InstitutionalDomain = {
  key: string
  label: string
  status: string
  summary: string
  evidence: string[]
  blockers: string[]
  next_action: string
}

export type InstitutionalReadiness = {
  schema_version: number
  generated_at: string
  system_state: string
  summary: string
  domains: InstitutionalDomain[]
  deployment: Record<string, { status: string; allowed: boolean; blockers?: string[] }>
  hard_blockers: string[]
}

export type ServiceProjection = {
  available: boolean
  message?: string
  summary?: Record<string, unknown>
  mode?: string
  latest?: Record<string, unknown>
  certified_for_live?: boolean
  enabled?: boolean
  running?: boolean
}

export type InstitutionalStack = {
  readiness: InstitutionalReadiness | { available: false; message?: string }
  oms: ServiceProjection
  risk_governor: ServiceProjection
  reconciliation: ServiceProjection
  protection: ServiceProjection
  tca: ServiceProjection
  broker_observer: ServiceProjection & { snapshots?: ServiceProjection }
}

export type DataProviderRow = {
  name: string
  capabilities: string[]
  priority: number
  authentication_status: string
  coverage_note: string
  freshness_note: string
  status: string
}

export type DataProvidersPayload = {
  generated_at: string
  providers: DataProviderRow[]
}

export type DataJobRow = {
  id: string
  label: string
  control: string | null
  description: string
  trigger: string
}

export type DataJobsPayload = {
  generated_at: string
  bhavcopy: Record<string, unknown>
  jobs: DataJobRow[]
}

export type SymbolCoverageRow = {
  symbol: string
  identity: string
  price_history: string
  fundamentals: string
  long_term_eligible: string
  reasons: Record<string, string>
}

export type DataCoveragePayload = {
  generated_at: string
  symbol?: string
  audited?: number
  status_counts?: Record<string, number>
  symbols?: SymbolCoverageRow[]
  remediation_queue?: Array<{ action: string; symbol: string; reason: string }>
  coverage?: Record<string, unknown>
}

const safeJson = async <T>(path: string, fallback: T): Promise<T> => {
  try {
    const response = await fetch(path, { headers: { Accept: 'application/json' } })
    if (!response.ok) return fallback
    return (await response.json()) as T
  } catch {
    return fallback
  }
}

export const fetchInstitutionalReadiness = (): Promise<InstitutionalReadiness> =>
  fetch('/api/institutional-readiness', { headers: { Accept: 'application/json' } })
    .then((response) => json<InstitutionalReadiness>(response))

export const fetchInstitutionalStack = async (): Promise<InstitutionalStack> => {
  const [readiness, oms, risk_governor, reconciliation, protection, tca, broker_observer] =
    await Promise.all([
      safeJson<InstitutionalReadiness | { available: false; message?: string }>(
        '/api/institutional-readiness',
        { available: false, message: 'Institutional readiness unavailable' },
      ),
      safeJson<ServiceProjection>('/api/oms', { available: false }),
      safeJson<ServiceProjection>('/api/risk-governor', { available: false }),
      safeJson<ServiceProjection>('/api/reconciliation', { available: false }),
      safeJson<ServiceProjection>('/api/protection', { available: false }),
      safeJson<ServiceProjection>('/api/tca', { available: false }),
      safeJson<ServiceProjection & { snapshots?: ServiceProjection }>('/api/broker-observer', {
        available: false,
      }),
    ])
  return {
    readiness,
    oms,
    risk_governor,
    reconciliation,
    protection,
    tca,
    broker_observer,
  }
}

export const fetchDataProviders = (): Promise<DataProvidersPayload> =>
  fetch('/api/data/providers', { headers: { Accept: 'application/json' } })
    .then((response) => json<DataProvidersPayload>(response))

export const fetchDataJobs = (): Promise<DataJobsPayload> =>
  fetch('/api/data/jobs', { headers: { Accept: 'application/json' } })
    .then((response) => json<DataJobsPayload>(response))

export const fetchDataCoverage = (symbol?: string): Promise<DataCoveragePayload> => {
  const query = symbol?.trim() ? `?symbol=${encodeURIComponent(symbol.trim().toUpperCase())}` : ''
  return fetch(`/api/data/coverage${query}`, { headers: { Accept: 'application/json' } })
    .then((response) => json<DataCoveragePayload>(response))
}

export type TargetPortfolioPayload = {
  available: boolean
  portfolio: Record<string, unknown>
  positions: Array<Record<string, unknown>>
  summary?: {
    current_positions: number
    target_positions: number
    executable_changes: number
    blocked_changes: number
    current_open_risk_pct: number
    pending_open_risk_pct: number
    target_open_risk_pct: number
    available_cash: number
  }
  message?: string
  error?: string
}

export const fetchTargetPortfolio = (): Promise<TargetPortfolioPayload> =>
  fetch('/api/target-portfolio', { headers: { Accept: 'application/json' } })
    .then((response) => json<TargetPortfolioPayload>(response))

export type SignalBacktestStatus = {
  running: boolean
  progress?: number
  total?: number
  has_report: boolean
  generated_at?: string
  symbols_run?: number
  universe?: {
    run?: number
    available?: number
    available_in_store?: number
    truncated?: boolean
    scope?: string
    note?: string
  }
  places_orders?: boolean
  live_locked?: boolean
}

export const fetchSignalBacktestStatus = (): Promise<SignalBacktestStatus> =>
  fetch('/api/signal-backtest', { headers: { Accept: 'application/json' } })
    .then((response) => json<SignalBacktestStatus>(response))

export type CorporateActionsStatus = {
  available: boolean
  path?: string
  symbols: number
  events: number
  research_grade: boolean
  adjustment_verified?: boolean
  gap_rate?: number | null
  verify_note?: string
  todo_path?: string
  todo_available?: boolean
  todo_gaps?: number | null
  next_action?: string
  never_invents?: boolean
  honesty?: string
  rejected_types?: { dividend?: number; invalid?: number }
}

export type EducationLens = 'MACRO' | 'MICRO' | 'POLICY' | 'DERIVATIVES' | 'CONCEPT'

export type EducationCard = {
  id: string
  lens: EducationLens | string
  kind: string
  title: string
  teach_point: string
  why_it_matters: string
  summary?: string
  level: string
  impact_score: number
  direction: string
  category?: string
  event_type?: string
  source: string
  source_tier?: number
  official: boolean
  url: string
  published_at: string
  fetched_at?: string
  symbols: string[]
  fno_symbols: string[]
  sectors?: string[]
  tags?: string[]
  corroboration_count: number
  places_orders?: boolean
  is_signal?: boolean
}

export type EducationFeed = {
  schema_version: number
  generated_at: string
  available: boolean
  honesty: string
  places_orders: boolean
  summary: {
    news_lessons: number
    macro_themes: number
    concepts: number
    by_lens: Record<string, number>
    articles_considered: number
  }
  lenses: EducationLens[]
  cards: EducationCard[]
  empty_hint?: string | null
}

export const fetchEducation = (minImpact = 40, limit = 40): Promise<EducationFeed> =>
  fetch(
    `/api/education?min_impact=${encodeURIComponent(String(minImpact))}&limit=${encodeURIComponent(String(limit))}`,
    { headers: { Accept: 'application/json' } },
  ).then((response) => json<EducationFeed>(response))

export type UsScanRecord = {
  symbol: string
  company?: string
  status?: string
  verdict?: string
  price?: number
  score?: number
  entry?: number
  stop?: number
  target?: number
  signals?: string[]
  reasons?: string[]
  chase_risk?: boolean
  fno_available?: boolean
  market?: string
  currency?: string
}

export type UsDashboard = {
  schema_version: number
  market: string
  generated_at: string
  honesty?: string
  places_orders?: boolean
  readiness: {
    state: string
    score: number
    recommended_action: string
    universe_size: number
    history: {
      ready?: boolean
      symbols?: number
      latest_date?: string
      source?: string
    }
    lanes: Array<{
      key: string
      label: string
      status: string
      available: boolean
      details: string
      action?: string
    }>
  }
  overview: {
    session_open: boolean
    session_label: string
    timezone?: string
    currency: string
    indices: Array<{ symbol: string; label: string; price?: number | null; available: boolean }>
  }
  scan: {
    available: boolean
    scanned_at: string
    scope: string
    universe_size: number
    summary: Record<string, number>
    records: UsScanRecord[]
    honesty?: string
  }
  paper?: Record<string, unknown>
}

export type UsStockWorkspace = {
  available: boolean
  symbol: string
  company?: string
  bars: Array<{ time: string; open: number; high: number; low: number; close: number; volume: number }>
  history_source?: string
  scan_row?: UsScanRecord | null
  fundamentals?: { available: boolean; message?: string }
  options?: { available: boolean; message?: string }
  honesty?: string
  places_orders?: boolean
}

export const fetchUsDashboard = (): Promise<UsDashboard> =>
  fetch('/api/us/dashboard', { headers: { Accept: 'application/json' } })
    .then((response) => json<UsDashboard>(response))

export const fetchUsStock = (symbol: string): Promise<UsStockWorkspace> =>
  fetch(`/api/us/stock/${encodeURIComponent(symbol)}`, { headers: { Accept: 'application/json' } })
    .then((response) => json<UsStockWorkspace>(response))

export const fetchCorporateActionsStatus = (): Promise<CorporateActionsStatus> =>
  fetch('/api/corporate-actions', { headers: { Accept: 'application/json' } })
    .then((response) => json<CorporateActionsStatus>(response))

export const exportCorporateActionGaps = (sample = 400): Promise<Record<string, unknown>> =>
  fetch(`/api/corporate-actions/from-gaps?sample=${encodeURIComponent(String(sample))}`, {
    method: 'POST',
    headers: { Accept: 'application/json' },
  }).then((response) => json<Record<string, unknown>>(response))

export const verifyCorporateActions = (sample = 80): Promise<CorporateActionsStatus> =>
  fetch(`/api/corporate-actions/verify?sample=${encodeURIComponent(String(sample))}`, {
    method: 'POST',
    headers: { Accept: 'application/json' },
  }).then((response) => json<CorporateActionsStatus>(response))

export type HoldingRow = {
  tradingsymbol: string
  research_symbol?: string
  quantity: number
  average_price: number
  last_price: number
  invested: number
  current_value: number
  pnl: number
  pnl_pct: number
  day_change?: number
  day_change_percentage?: number
  exchange?: string
  product?: string
}

export type HoldingsBook = {
  schema_version: number
  available: boolean
  updated_at?: string
  source?: string
  holdings: HoldingRow[]
  summary: {
    count: number
    invested: number
    current_value: number
    pnl: number
    pnl_pct: number
    day_pnl?: number
    day_pnl_pct?: number
  }
  message?: string
  synced?: boolean
  places_orders?: boolean
}

export const fetchHoldings = (): Promise<HoldingsBook> =>
  fetch('/api/holdings', { headers: { Accept: 'application/json' } })
    .then((response) => json<HoldingsBook>(response))

export const syncHoldings = (): Promise<HoldingsBook> =>
  fetch('/api/holdings/sync', {
    method: 'POST',
    headers: { Accept: 'application/json' },
  }).then((response) => json<HoldingsBook>(response))

export const importHoldings = (holdings: Array<Record<string, unknown>>, source = 'import'): Promise<HoldingsBook> =>
  fetch('/api/holdings/import', {
    method: 'POST',
    headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
    body: JSON.stringify({ holdings, source }),
  }).then((response) => json<HoldingsBook>(response))

export const runDataJob = (jobId: string): Promise<{
  ok: boolean
  job_id: string
  message?: string
  error?: string
  operation_id?: string
  created?: boolean
  kind?: string
  note?: string
}> =>
  fetch(`/api/data/jobs/${encodeURIComponent(jobId)}/run`, {
    method: 'POST',
    headers: { Accept: 'application/json' },
  }).then((response) => json(response))

export const refreshFiiDiiStore = (): Promise<Record<string, unknown>> =>
  fetch('/api/market/fii-dii/backfill', {
    method: 'POST',
    headers: { Accept: 'application/json' },
  }).then((response) => json(response))

export const fetchMarketInstitutional = (days = 30): Promise<Record<string, unknown>> =>
  fetch(`/api/market/institutional?days=${days}`, { headers: { Accept: 'application/json' } })
    .then((response) => json(response))
