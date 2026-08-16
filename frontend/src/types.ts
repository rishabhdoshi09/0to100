export type ScanRecord = {
  symbol: string
  company?: string
  status?: string
  verdict?: string
  price?: number
  score?: number
  momentum_5d?: number
  volume_ratio?: number
  rsi?: number
  entry?: number
  stop?: number
  target?: number
  sector?: string
  signals?: string[]
  reasons?: string[]
  chase_risk?: boolean
  edge_r?: number | null
  breakout_grade?: string
  breakout_conviction?: number
  breakout_quality?: number
  breakout_state?: string
  fundamental_score?: number
  classification?: string
  sniper_candidate?: boolean
}

export type ConvictionRecord = ScanRecord & {
  classification?: string
  conviction_score?: number
  scanner_score?: number
  risks?: string[]
}

export type LongTermRecord = {
  symbol: string
  classification?: string
  combined_score?: number
  technical_score?: number
  fundamental_score?: number
  fundamental_coverage?: number
  price?: number
  sector?: string
  timing?: string
  mom_12m_pct?: number
  from_high_pct?: number
  quality_factors?: string[]
  risk_flags?: string[]
}

export type PaperPosition = {
  symbol?: string
  entry_price?: number
  current_price?: number
  quantity?: number
  stop?: number
  target?: number
  pnl?: number
  pnl_pct?: number
  strategy?: string
  days_held?: number
  exit_reason?: string
  result_r?: number
  [key: string]: unknown
}

export type AutonomyJob = {
  job_id: string
  job_type: string
  status: string
  attempt: number
  critical?: number | boolean
  scheduled_for?: number
  started_at?: number
  finished_at?: number
  result_summary?: string
  error_code?: string
  error_message?: string
  blocked_on?: string
  blocked_reason?: string
}

export type OperationRecord = {
  operation_id: string
  kind: string
  lane: string
  status: 'PENDING' | 'RUNNING' | 'SUCCEEDED' | 'FAILED' | 'BLOCKED' | 'CANCELLED' | string
  requested_by: string
  requested_at: number
  started_at?: number | null
  finished_at?: number | null
  updated_at: number
  attempt: number
  worker_pid?: number | null
  stage: string
  message: string
  progress_current: number
  progress_total: number
  progress_pct?: number | null
  payload?: Record<string, unknown>
  result?: Record<string, unknown>
  error_code?: string
  error_message?: string
}

export type NewsArticle = {
  article_id: string
  headline: string
  summary: string
  source: string
  source_key: string
  source_tier: number
  official: boolean
  url: string
  published_at: string
  fetched_at: string
  category: string
  event_type: string
  impact_score: number
  direction: string
  why_it_matters: string
  mentioned_symbols: string[]
  fno_symbols: string[]
  sectors: string[]
  tags: string[]
  corroboration_count: number
}

export type NewsSourceHealth = {
  source_key: string
  source_name: string
  status: string
  fetched_at: string
  article_count: number
  latency_ms: number
  error: string
}

export type FnoUnderlying = {
  symbol: string
  company_name: string
  future_symbol: string
  expiry: string
  lot_size: number
  instrument_token: number
  contract_count: number
}

export type FnoExclusion = {
  underlying: string
  stage: string
  reason: string
}

export type DataReadiness = {
  ready: boolean
  snapshot: {
    ready: boolean
    snapshot_id: string
    latest_date: string
    source: string
    error?: string
  }
  bhavcopy: {
    ready: boolean
    symbols: number
    sessions: number
    latest_date: string
    csv_files: number
    csv_latest_date?: string
    cache_exists: boolean
    cache_path?: string
    bhavcopy_dir?: string
    minimum_sessions?: number
    source?: string
    required_session?: string
    is_stale?: boolean
    freshness?: string
    error?: string
  }
  kite?: {
    ok?: boolean
    status?: string
    note?: string
    nifty?: number
    chg_pct?: number
  }
  scan_saved: boolean
  scan_records: number
  long_term_saved: boolean
  long_term_records: number
  blockers: string[]
}

export type DashboardPayload = {
  generated_at: string
  market: {
    available: boolean
    health: string
    summary: string
    trade_stance: string
    breadth: string
    leaders: string[]
    laggards: string[]
    nifty_change_1d: number | null
    nifty_change_5d: number | null
    vix: number | null
    as_of?: string
    source?: string
    quote_source?: string
    technical_details?: Record<string, unknown>
  }
  scan: {
    available: boolean
    scanned_at?: string
    universe_size: number
    summary: Record<string, number>
    records: ScanRecord[]
  }
  long_term: {
    available: boolean
    scanned_at?: string
    fundamentals_source?: string
    summary: Record<string, number>
    records: LongTermRecord[]
    job?: Partial<AutonomyJob>
  }
  paper: {
    available?: boolean
    enabled: boolean
    supervisor_running: boolean
    capital: number
    equity: number
    equity_curve?: number[]
    open_risk: number
    risk_per_trade_pct: number
    max_positions: number
    open_positions: PaperPosition[]
    closed_trades: PaperPosition[]
    refusals?: Array<Record<string, unknown> | unknown[]>
    last_cycle?: Record<string, unknown>
    last_error?: string
  }
  autonomy: {
    available?: boolean
    running: boolean
    process_running?: boolean
    state: string
    plain_state: string
    explanation: string
    heartbeat_ist: string
    scheduler_owner_pid?: number | string | null
    active_job?: Record<string, unknown>
    new_entry_capability?: 'allowed' | 'limited' | 'blocked' | 'read_only'
    existing_exit_capability?: 'allowed' | 'limited' | 'blocked' | 'read_only'
    research_capability?: 'allowed' | 'limited' | 'blocked' | 'read_only'
    new_paper_entries: boolean
    existing_exits?: boolean
    research_enabled?: boolean
    capability_notes?: string[]
    active_failures?: string[]
    recent_dialogue: Array<Record<string, unknown>>
    recent_transitions?: Array<Record<string, unknown>>
    jobs: Record<string, number>
    jobs_recent?: AutonomyJob[]
    owner_state?: Record<string, boolean>
    live_feed?: Record<string, unknown>
    last_cycle?: Record<string, unknown>
  }
  operations: {
    available: boolean
    running: boolean
    worker_pid?: number | null
    heartbeat: string
    active_lanes: Record<string, Record<string, unknown>>
    ensure_ok?: boolean
    ensure_error?: string
    counts: Record<string, number>
    active: OperationRecord[]
    recent: OperationRecord[]
    latest: Record<string, OperationRecord>
    error?: string
  }
  news: {
    available: boolean
    stats: Record<string, number>
    articles: NewsArticle[]
    source_health: NewsSourceHealth[]
    latest_refresh?: Partial<OperationRecord>
    error?: string
  }
  fno: {
    available: boolean
    generated_at?: number | null
    source: string
    total_instrument_rows?: number
    total_future_contracts?: number
    index_future_contracts?: number
    unique_stock_underlyings?: number
    mapped_underlyings: number
    underlyings: FnoUnderlying[]
    exclusions: FnoExclusion[]
    cache_mtime?: number | null
    error?: string
  }
  institutional?: InstitutionalFlowsPayload
  data: DataReadiness
  conviction: ConvictionRecord[]
}

export type InstitutionalFlowsPayload = {
  available: boolean
  cash?: {
    available?: boolean
    sessions?: number
    history?: Array<{
      date: string
      fii_net: number
      dii_net: number
      fii_buy?: number
      fii_sell?: number
      dii_buy?: number
      dii_sell?: number
    }>
    today?: Record<string, number | string>
    totals?: { fii_net_cr?: number; dii_net_cr?: number; combined_net_cr?: number }
    fii_streak?: number
    dii_streak?: number
    bias?: string
    note?: string
  }
  derivatives?: Record<string, number | null>
  bulk_deals?: Array<Record<string, unknown>>
  bulk_buy_symbols?: string[]
  nifty_options?: OptionsChainPayload
  insight?: string
  generated_at?: string
  error?: string
}

export type OptionsChainPayload = {
  available: boolean
  symbol?: string
  expiry?: string
  pcr?: number
  max_pain?: number
  bias?: string
  note?: string
  atm_iv?: number
  iv_rank?: number
  spot?: number | null
  total_ce_oi?: number
  total_pe_oi?: number
  strike_count?: number
  top_call_oi?: Array<{ strike: number; ce_oi: number; ce_coi?: number }>
  top_put_oi?: Array<{ strike: number; pe_oi: number; pe_coi?: number }>
  chain?: Array<Record<string, number>>
  message?: string
  greeks_available?: boolean
  signal_desk?: boolean
  honesty?: string
}

export type OptionsEodHistoryPayload = {
  available: boolean
  symbol: string
  days: number
  rows: Array<{
    symbol: string
    as_of: string
    expiry: string
    pcr?: number | null
    max_pain?: number | null
    atm_iv?: number | null
    spot?: number | null
    strike_count?: number
    source?: string
    captured_at?: string
  }>
  store?: Record<string, unknown>
  message?: string
}

export type ChartBar = {
  time: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export type ControlName =
  | 'RUN_SCAN_NOW'
  | 'RUN_LONG_TERM_SCAN_NOW'
  | 'REFRESH_LONG_TERM_NOW'
  | 'REFRESH_NEWS_NOW'
  | 'REFRESH_FNO_NOW'
  | 'RUN_CYCLE_NOW'
  | 'REFRESH_DATA_NOW'
  | 'RUN_FULL_UNIVERSE_BACKTEST_NOW'
  | 'PAUSE_NEW_PAPER_ENTRIES'
  | 'RESUME_NEW_PAPER_ENTRIES'
