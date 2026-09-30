export type ScanRecord = {
  symbol: string
  company?: string
  status?: string
  verdict?: string
  /** null means the scanner produced no level. Never render a null as 0. */
  price?: number | null
  score?: number
  momentum_5d?: number
  volume_ratio?: number
  rsi?: number
  entry?: number | null
  stop?: number | null
  target?: number | null
  /** Deterministic geometry derived from the levels above, when they exist. */
  plan_reference_price?: number | null
  risk_per_share?: number | null
  reward_per_share?: number | null
  upside_pct?: number | null
  downside_pct?: number | null
  reward_risk?: number | null
  plan_complete?: boolean
  plan_missing?: string[]
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


export type FnoForwardEvidence = {
  context_key?: string
  context_schema_version?: string
  valid_context?: boolean
  evidence_lane?: string
  status?: 'EVIDENCE_READY' | 'ACCUMULATING' | 'COST_MODEL_REQUIRED' | 'NO_FORWARD_OUTCOMES' | 'NO_CONTEXT_KEY' | string
  observed_n?: number
  n?: number
  excluded_unpriced_costs?: number
  excluded_path_observation?: number
  excluded_other_ineligible?: number
  wins?: number
  losses?: number
  probability_claim_available?: boolean
  win_probability_pct?: number | null
  win_probability_wilson_lb_pct?: number | null
  expectancy_pct?: number | null
  conservative_ev_pct?: number | null
  avg_win_pct?: number | null
  avg_loss_pct?: number | null
  avg_mfe_pct?: number | null
  avg_mae_pct?: number | null
  false_breakout_rate_pct?: number | null
  max_drawdown_pct?: number | null
  insufficient_evidence?: boolean
  minimum_required_n?: number
  production_influence_allowed?: boolean
  coverage?: {
    valid_context?: boolean
    context_schema_version?: string
    exact_n?: number
    thesis_n?: number
    direction_oi_n?: number
    direction_n?: number
    minimum_exact_n?: number
    remaining_to_exact_min_n?: number
    research_only?: boolean
    counts_only?: boolean
    probability_claim_available?: boolean
    production_influence_allowed?: boolean
  }
}

export type FnoDirectionalCandidate = {
  symbol: string
  direction: 'LONG' | 'SHORT' | string
  decision: string
  setup?: {
    direction?: string
    score?: number
    score_is_probability?: boolean
    futures_oi_state?: string
    breakout_distance_pct?: number
    atr_pct?: number
    components?: Record<string, number>
    expected_move?: {
      lower_pct?: number
      upper_pct?: number
      mid_pct?: number
      horizon?: string
      holding_days?: number
    }
    underlying_trade_plan?: {
      entry?: number
      stop?: number | null
      target?: number | null
      invalidation_model?: string
      target_model?: string
    }
    blockers?: string[]
    reasons?: string[]
  }
  options?: {
    spot?: number
  }
  selected_contract?: {
    symbol?: string
    option_type?: 'CE' | 'PE' | string
    strike?: number
    expiry?: string
    dte?: number
    premium?: number
    bid?: number
    ask?: number
    spread_pct?: number | null
    volume?: number
    oi?: number
    iv?: number
    iv_percentile?: number | null
    iv_percentile_available?: boolean
    delta?: number
    gamma?: number
    theta_per_day?: number
    vega_per_vol_point?: number
    score?: number
    score_is_probability?: boolean
    components?: Record<string, number>
    moneyness?: string
    moneyness_pct?: number | null
    projected_return_at_expected_move_pct?: number
    context_key?: string
    trade_plan?: {
      entry?: number
      stop?: number | null
      target?: number | null
      risk_reward?: number
      underlying_invalidation?: number | null
      model?: string
    }
    raw_contract_score?: number
    learned_contract_score?: number
    contract_evidence?: {
      context_key?: string
      usable?: boolean
      count?: number
      direction?: string
      adjustment?: number
      reason?: string
      expectancy_R?: number | null
      wilson_lower_bound?: number | null
      base_context_key?: string
      used_modifier?: string | null
    }
  }
  iv_history?: {
    available?: boolean
    percentile_pct?: number | null
    prior_sessions?: number
    minimum_prior_sessions?: number
    lookback_sessions?: number
    current_iv_pct?: number | null
    source?: string
    historical_backfill?: boolean
    reason?: string
  }
  forward_evidence?: FnoForwardEvidence
  paper_only?: boolean
  live_execution_allowed?: boolean
  base_score?: number
  historical_prior?: number
  forward_adjustment?: number
  ranking_adjustment?: number
  ranking_score?: number
  ranking_evidence?: {
    usable?: boolean
    reason?: string
    status?: string
    adjustment?: number
    historical_prior?: number
    forward_adjustment?: number
    historical_note?: string
    evidence_class?: string
    context_key?: string
    count?: number
    min_sample?: number
    expectancy_R?: number | null
    wilson_lower_bound?: number | null
  }
}

export type FnoLearningEvidenceBlock = {
  available?: boolean
  evidence_class?: string
  error?: string
  last_run_at?: string
  cursor_date?: string
  coverage_complete?: boolean
  historical_simulations_completed?: number
  decisions_graded?: number
  settled?: number
  correct_rejects?: number
  missed_winners?: number
  avoided_losers?: number
  ran_away_without_entry?: number
  good_waits?: number
  flat?: number
  open_positions?: number
  forward_paper_trades?: number
  wins?: number
  losses?: number
  production_evidence_trades?: number
  evidence_cells?: number
  matured_cells?: number
  minimum_sample_for_ranking?: number
  can_affect_ranking?: boolean
  note?: string
  cells_large_enough_for_a_prior?: number
  prior_min_sample?: number
  prior_cap?: number
  richest_priors?: Array<{
    context?: Record<string, string>
    count?: number
    win_rate?: number | null
    wilson_lower_bound?: number | null
    expectancy_R?: number | null
    median_R?: number | null
    mfe_R?: number | null
    mae_R?: number | null
    prior_direction?: string
    prior_adjustment?: number
  }>
  contract_selection?: {
    evidence_cells?: number
    matured_cells?: number
    can_affect_contract_selection?: boolean
    outcome_classification_counts?: Record<string, number>
    note?: string
  }
}

export type FnoLearningImpact = {
  schema_version?: number
  available?: boolean
  error?: string
  historical?: FnoLearningEvidenceBlock
  forward?: FnoLearningEvidenceBlock
  ranking_impact?: {
    status?: string
    plain?: string
    influenced_count?: number
    forward_influenced_count?: number
    historical_influenced_count?: number
    influenced?: Array<{
      symbol?: string
      direction?: string
      base_score?: number
      historical_prior?: number
      forward_adjustment?: number
      ranking_score?: number
      adjustment?: number
      status?: string
      reason?: string
      count?: number
      expectancy_R?: number | null
      why?: string
    }>
  }
  threshold_review?: {
    min_score_to_take?: number
    below_threshold_sample?: number
    above_threshold_sample?: number
    below_threshold_would_be_winner_rate_pct?: number
    below_threshold_would_be_loser_rate_pct?: number
    above_threshold_winner_rate_pct?: number
    above_threshold_loser_rate_pct?: number
    min_sample?: number
    material_gap_pp?: number
    recommendation?: string
    reason?: string
    autonomous_change_applied?: boolean
    action_required?: string
  }
  policy?: {
    historical_and_forward_kept_in_separate_cells?: boolean
    historical_alone_can_move_ranking?: boolean
    historical_prior_is_small_and_bounded?: boolean
    forward_is_always_the_stronger_vote?: boolean
    forward_required_to_promote_past_the_historical_cap?: boolean
    contract_selection_learned_separately_from_underlying_call?: boolean
    contract_selection_requires_genuine_forward_option_trades?: boolean
    no_fabricated_historical_option_chain_data?: boolean
    live_money_affected?: boolean
  }
  live_locked?: boolean
}

export type FnoDirectionalState = {
  available?: boolean
  status?: string
  code?: string
  decision?: string | null
  candidate_count?: number
  candidates?: FnoDirectionalCandidate[]
  paper_only?: boolean
  live_execution_allowed?: boolean
  cache_mtime?: number | null
  candidate_evidence_status?: string
  candidate_evidence_error?: string
  candidate_evidence_policy?: {
    lane?: string
    minimum_fully_costed_n?: number
    context_schema_version?: string
    probability_requires_current_context_version?: boolean
    probability_requires_exact_context?: boolean
    probability_requires_complete_observed_path?: boolean
    broader_context_counts_research_only?: boolean
    paper_only?: boolean
    live_execution_allowed?: boolean
  }
}

export type FnoPaperPosition = {
  trade_id?: string
  underlying?: string
  option_symbol?: string
  option_type?: string
  strike?: number
  expiry?: string
  entry_price?: number
  stop_price?: number
  target_price?: number
  trailing_stop_price?: number
  quantity?: number
  lots?: number
  setup_score?: number
  option_score?: number
  opened_at?: string
  max_holding_sessions?: number
  bars_held?: number
  max_mark?: number
  min_mark?: number
  entry_minute_status?: string
  entry_iv_pct?: number
  entry_underlying_spot?: number
  horizon?: string
  exit_policy?: string
  context_key?: string
}

export type FnoPaperState = {
  available?: boolean
  status?: {
    schema_version?: number
    open_positions?: number
    closed_trades?: number
    production_evidence_trades?: number
    realized_pnl?: number
  }
  open_positions?: FnoPaperPosition[]
  recent_closed_trades?: Array<Record<string, unknown>>
  production_evidence_enabled?: boolean
  paper_only?: boolean
  live_execution_allowed?: boolean
  error?: string
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

export type BrokerReadiness = {
  state: 'READY' | 'LOGIN_REQUIRED' | 'SNAPSHOT_REQUIRED' | 'UNAVAILABLE' | 'CONFIG_REQUIRED' | 'NOT_READY' | 'UNKNOWN' | string
  ready: boolean
  live_data_ready: boolean
  execution_ready: boolean
  auth_ready: boolean
  login_required: boolean
  auth_status: string
  reason_code: string
  detail: string
  snapshot_id: string
}

/** When a scan RAN vs which market session it READ. Never the same fact. */
export type ScanProvenance = {
  scan_id?: string
  scan_started_at?: string
  scan_completed_at?: string
  scan_duration_s?: number | null
  market_session_date?: string
  price_data_as_of?: string
  expected_session_date?: string
  sessions_behind?: number | null
  data_freshness?: string
  data_current?: boolean
  price_source?: string
  universe_failed?: number | null
  provenance_available?: boolean
  provenance_reason?: string
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
    dashboard_record_limit?: number
    dashboard_records_shown?: number
    provenance?: ScanProvenance
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
    current_activity?: string
    activity_truth?: Record<string, unknown>
    resource_governor?: Record<string, unknown>
    operational_incidents?: {
      open_count?: number
      open?: Array<{
        incident_id?: string
        status?: string
        code?: string
        message?: string
        first_seen_at?: string
        last_seen_at?: string
        occurrence_count?: number
        recovery_action?: string
        progress?: Record<string, unknown>
        job?: Record<string, unknown>
      }>
      recent?: Array<Record<string, unknown>>
    }
    new_entry_capability?: 'allowed' | 'limited' | 'blocked' | 'read_only'
    existing_exit_capability?: 'allowed' | 'limited' | 'blocked' | 'read_only'
    research_capability?: 'allowed' | 'limited' | 'blocked' | 'read_only'
    new_paper_entries: boolean
    existing_exits?: boolean
    research_enabled?: boolean
    broker?: BrokerReadiness
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
    directional?: FnoDirectionalState
    paper?: FnoPaperState
    learning_impact?: FnoLearningImpact
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
  | 'OBSERVE_ONLY_TODAY'
  | 'CLEAR_OBSERVE_ONLY'
  | 'RUN_HISTORICAL_REPLAY'
  | 'RUN_LEARNING_NOW'