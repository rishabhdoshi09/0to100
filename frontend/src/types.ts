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
  why?: string
  sepa_score?: number
  sepa_max?: number
  sepa_passed?: number
  sepa_total?: number
  sepa_verdict?: string
  sepa_headline?: string
  sepa_advice?: string
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
  priority?: number
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
    error?: string
    current?: boolean
    expected_latest_completed_session?: string
    available_session?: string
    stale_sessions?: number | null
    reason_code?: string
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
    nifty_price?: number | null
    technical_details?: Record<string, unknown>
  }
  daily_wrap?: Array<{
    id?: string
    text: string
    source?: string
    official?: boolean
    url?: string
    symbols?: string[]
  }>
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
    learning?: {
      available?: boolean
      as_of?: string
      closed_trades?: number
      cooldown?: Array<{ symbol?: string; until?: string; reason?: string }>
      prefer?: string[]
      shadow_prefer?: string[]
      self_feed?: {
        as_of?: string
        slot?: string
        summary?: string
        taken?: Array<{ symbol?: string; strategy_id?: string; status?: string }>
        skipped?: Array<{ symbol?: string; status?: string; reason?: string }>
        sepa_best?: Array<{
          symbol?: string
          sepa_score?: number | null
          sepa_verdict?: string
          paper_status?: string
          skip_reason?: string
          not_a_buy?: boolean
        }>
        candidate_tests?: Array<{
          symbol?: string
          outcome?: string
          r_multiple?: number | null
          n_forward_bars?: number
          role?: string
          paper_status?: string
        }>
        disclaimer?: string
        live_locked?: boolean
      }
      summary?: string
      live_locked?: boolean
      disclaimer?: string
      ladder?: string
    }
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
    telegram?: {
      configured?: boolean
      state?: string
      headline?: string
      detail?: string
      scan_reason?: string
      sniper_reason?: string
      sniper_watch?: number
      live_ticks?: boolean
    }
    last_cycle?: Record<string, unknown>
  }
  scan_progress?: {
    active?: boolean
    stage?: string
    current?: number
    total?: number
    pct?: number | null
    eta_s?: number | null
    eta_label?: string
    error?: string
    updated_at?: number
  }
  operations: {
    available: boolean
    running: boolean
    worker_pid?: number | null
    heartbeat: string
    active_lanes: Record<string, Record<string, unknown>>
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
  data: DataReadiness
  conviction: ConvictionRecord[]
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
  | 'REFRESH_MARKET_REPORT_NOW'
  | 'REFRESH_FNO_NOW'
  | 'RUN_CYCLE_NOW'
  | 'REFRESH_DATA_NOW'
  | 'PAUSE_NEW_PAPER_ENTRIES'
  | 'RESUME_NEW_PAPER_ENTRIES'
