import { readJson } from './http'
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
    company_about: string
    fetched_at: string
    section_as_of: Record<string, string>
  }
  scanner: ScanRecord
  long_term: LongTermRecord
  news: NewsArticle[]
  fno: Record<string, unknown>
  sources: IntelligenceSource[]
  next_actions: Array<{ control: ControlName | 'REFRESH_STOCK_FUNDAMENTALS'; label: string }>
  case?: RecommendationCase
  decision_memory?: {
    symbol?: string
    stance?: string
    setup_quality?: { score?: number | null; label?: string }
    similar?: RecommendationCase['similar']
    why_not?: { found?: boolean; line?: string; label?: string; verdict?: string; n_observations?: number }
    trust?: { n?: number; line?: string; status?: string }
    edge?: { profile?: string; line?: string }
    places_orders?: boolean
  }
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
  & {
    _source?: string
    change_5d_pct?: number | null
    breakout_state?: string | null
    momentum_state?: string | null
    setup_label?: string | null
    relative_strength?: number | null
    risk_label?: string | null
  }

export type ScannerWorkspace = {
  generated_at: string
  mode: string
  source: string
  scanned_at: string
  universe_size: number
  rows: ScannerWorkspaceRow[]
}

const json = readJson

export const fetchProductReadiness = (): Promise<ProductReadiness> =>
  fetch('/api/product-readiness', { headers: { Accept: 'application/json' } })
    .then((response) => json<ProductReadiness>(response))

export const bootstrapProduct = (): Promise<{
  accepted: boolean
  message: string
  sequential?: boolean
  queued_kind?: string | null
  scan_reused?: boolean
  operations: Array<{ kind: string; operation_id: string; status: string; created: boolean }>
  pipeline?: DeskPipeline
  readiness: ProductReadiness
}> => fetch('/api/product-bootstrap', {
  method: 'POST',
  headers: { Accept: 'application/json' },
}).then((response) => json(response))

export const fetchStockIntelligence = (symbol: string): Promise<StockWorkspace> =>
  fetch(`/api/stock-intelligence/${encodeURIComponent(symbol)}`, {
    headers: { Accept: 'application/json' },
  }).then((response) => json<StockWorkspace>(response))

export type DueDiligenceKpi = {
  id: string
  label: string
  pillar: string
  available: boolean
  trend: string
  fact: string
  interpretation: string
  implication: string
  formula?: string
  source: string
  source_url: string
  source_date: string
  confidence: string
  table?: string
  provenance?: {
    value?: number | string | null
    period?: string
    source?: string
    source_url?: string
    retrieved_at?: string
    published_at?: string
    source_type?: string
    source_type_label?: string
    confidence?: string
    raw_reference?: string
  }
  snapshot?: {
    current: number | null
    current_period?: string
    previous?: number | null
    previous_period?: string
    year_ago?: number | null
    year_ago_period?: string
    qoq_change?: number | null
    yoy_change?: number | null
    points?: { period: string; value: number }[]
  }
  availability_state?: string
  availability_label?: string
  importance?: string
  implemented?: boolean
  reliability?: string
  reliability_label?: string
  period_policy?: string
  definition?: string
  source_count?: number
  source_consensus?: string
  agreeing_sources?: string[]
  period_type?: string
  reporting_basis?: string
}

export type DueDiligenceEvent = {
  headline: string
  published_at: string
  source: string
  url: string
  official: boolean
  verified: boolean
  event_type: string
  category?: string
  impact: string
  summary: string
  materiality?: string
  materiality_basis?: string
  original_source?: string
}

export type OptionChainSnapshot = {
  available: boolean
  acquired?: boolean
  expiry?: string | null
  spot?: number | null
  call_oi?: number | null
  put_oi?: number | null
  pcr?: number | null
  max_pain?: number | null
  atm_strike?: number | null
  atm_iv?: number | null
  top_call_oi?: { strike: number; oi: number }[]
  top_put_oi?: { strike: number; oi: number }[]
  n_strikes?: number
  source?: string
  source_url?: string
  reason?: string
  note?: string
  not_a_signal?: boolean
  places_orders?: boolean
}

export type DueDiligenceFlag = {
  id: string
  title: string
  kind: string
  fact: string
  source?: string
  source_date?: string
  url?: string
  severity?: string
  rule?: string
  triggered_value?: unknown
  threshold?: unknown
  evidence?: string
}

export type InvestigatorMatch = {
  symbol: string
  company: string
  label: string
  match?: string
}

export type DueDiligenceReport = {
  schema_version: number
  symbol: string
  company: string
  profile: {
    sector: string
    industry: string
    framework_id: string
    framework_reason: string
    business_model: string
    sub_sector?: string
    revenue_drivers: string
    about: string
    classification_note?: string
  }
  framework: { id: string; label: string; blurb: string; sub_sector?: string; business_model?: string; peer_note?: string }
  technical_context: {
    available: boolean
    scanner_status: string
    scanner_score: number | null
    sepa_score: number | null
    breakout_grade: string | null
    breakout_quality: string
    chase_risk: boolean
    detail: string
  }
  fundamental_quality: {
    score: number | null
    label: string
    coverage_pct: number
    score_coverage_pct?: number
    raw_awarded?: number
    evaluated_weight?: number
    scoring_weight?: number
    explain: string
    breakdown?: {
      explain?: string
      pillars?: { id: string; label: string; awarded: number | null; max: number; display: string; explain?: string; formula?: string }[]
    }
  }
  research_coverage?: {
    coverage_pct: number
    available_n?: number
    required_n?: number
    summary?: string
    needs_acquire?: boolean
    to_fetch?: string[]
    latest_data_refresh?: string | null
    not_a_quality_score?: boolean
    datasets?: {
      id: string
      label: string
      status: string
      required?: boolean
      present?: boolean
      age_label?: string | null
      checked_at?: string | null
    }[]
  }
  framework_audit?: {
    id?: string
    label?: string
    decision_n?: number
    implemented_n?: number
    implementation_coverage_pct?: number
    summary?: string
    decision_metrics?: {
      id?: string
      label?: string
      importance?: string
      implemented?: boolean
      reliability?: string
      reliability_label?: string
      populated?: boolean
      company_state?: string
      definition?: string
    }[]
  }
  implementation_coverage_pct?: number
  sector_kpi_label?: string
  sector_kpi_detail?: string
  decision_coverage?: {
    coverage?: number
    coverage_pct?: number
    critical_n?: number
    critical_available?: number
  }
  decision_coverage_pct?: number
  missing_evidence?: {
    id?: string
    label?: string
    importance?: string
    availability_state?: string
    reason?: string
  }[]
  critical_metrics_missing?: string[]
  confirmation_reason?: string
  confirmation_qualifier?: string
  deeper_acquire_available?: boolean
  business_trend: string
  financial_strength: string
  earnings_quality: string
  balance_sheet_quality: string
  governance_risk: string
  news_event_impact: string
  vs_technical_setup: string
  fundamental_confirmation?: string
  vs_detail: string
  strengths: string[]
  concerns: string[]
  unavailable: string[]
  what_changed: string[]
  red_flags: DueDiligenceFlag[]
  flag_groups?: {
    critical?: DueDiligenceFlag[]
    warnings?: DueDiligenceFlag[]
    monitor?: DueDiligenceFlag[]
    n_critical?: number
    n_warnings?: number
    n_monitor?: number
  }
  watch_next: string[]
  kpis: DueDiligenceKpi[]
  events: DueDiligenceEvent[]
  as_of: {
    latest_financial_period: string
    fundamentals_fetched_at: string
    fundamentals_freshness: string
    latest_material_news: string
    scan_scanned_at: string
    generated_at: string
    evidence_pack_coverage_pct?: number
    autonomy_acquired_at?: string
    latest_data_refresh?: string
  }
  extracted_guidance?: {
    tone: string
    excerpt: string
    excerpts?: string[]
    source: string
    source_url?: string
    source_date?: string
    method?: string
    not_an_llm?: boolean
  }[]
  thesis?: {
    kind: string
    not_an_llm: boolean
    text: string
    basis: string[]
  }
  autonomy?: {
    acquired_at?: string | null
    steps?: { id: string; ok?: boolean; error?: string }[]
    downloads?: { ok?: boolean; url?: string; path?: string; error?: string }[]
    still_missing?: string[]
    files_on_disk?: string[]
    option_chain?: OptionChainSnapshot | null
    not_an_llm?: boolean
  }
  evidence_pack?: {
    coverage_pct: number
    empty_note?: string
    gaps: { key: string; label: string; status: string; why: string; instructions: string; source_attached: boolean; link_label: string; link_url: string }[]
    management_commentary: { speaker: string; topic: string; commentary: string; event_date: string; source_url: string }[]
    order_book: { metric: string; fact: string; as_of: string; source_url: string }[]
    peers: { name: string; fact: string }[]
    snapshot_metrics: { id: string; label: string; available: boolean; fact: string; interpretation: string; implication: string; source: string }[]
    option_chain?: OptionChainSnapshot
    next_actions: { id: string; label: string; page?: string; control?: string; detail: string }[]
  }
  long_term_overlay?: {
    classification?: string | null
    quality_factors?: string[]
    risk_flags?: string[]
    note?: string
  }
  first_screen?: {
    company?: string
    ticker?: string
    selected_by?: string
    technical_score?: number | null
    sepa_score?: number | null
    breakout_quality?: string
    fundamental_quality?: number | null
    fundamental_quality_label?: string
    score_coverage_pct?: number
    data_coverage_pct?: number
    decision_coverage_pct?: number
    confirmation_reason?: string
    confirmation_qualifier?: string
    critical_metrics_missing?: string[]
    missing_evidence?: { id?: string; label?: string; importance?: string; reason?: string; availability_state?: string }[]
    deeper_acquire_available?: boolean
    sector_kpi_detail?: string
    fundamental_confirmation?: string
    vs_detail?: string
    business_trend?: string
    earnings_trend?: string
    balance_sheet?: string
    critical_red_flags?: number
    warnings?: number
    latest_financial_quarter?: string
    latest_data_refresh?: string
    data_freshness?: string
    research_coverage_pct?: number
    research_coverage_summary?: string
    research_coverage_needs_acquire?: boolean
    implementation_coverage_pct?: number
    implementation_coverage_summary?: string
    framework_audit_metrics?: {
      id?: string
      label?: string
      importance?: string
      implemented?: boolean
      reliability_label?: string
      company_state?: string
    }[]
    sector_kpis?: string
    sector_kpi_framework?: string
    sub_sector?: string
    business_model?: string
    improving?: string[]
    deteriorating?: string[]
    recent_material_events?: { date?: string; headline?: string; category?: string; materiality?: string; source?: string; url?: string }[]
    technical_reason?: string[]
    fundamental_evidence?: string[]
    sections?: string[]
  }
  company_snapshot?: Record<string, unknown>
  cash_flow_quality?: { applicable?: boolean; label?: string; detail?: string; flags?: DueDiligenceFlag[]; metrics?: { id: string; label: string; available: boolean; fact: string; formula?: string }[] }
  peers?: { available?: boolean; detail?: string; ranks?: { metric: string; quartile: string; formula?: string; rank?: number; n?: number; value?: number }[]; rows?: { name: string; fact?: string }[] }
  filings?: { title?: string; category?: string; url?: string; source?: string; published_at?: string }[]
  sources?: { source?: string; source_url?: string; period?: string; source_type_label?: string; retrieved_at?: string }[]
  source_conflicts?: { field?: string; status?: string; note?: string; preferred?: { value?: unknown; source?: string }; other?: { value?: unknown; source?: string } }[]
  places_orders: boolean
  disclaimer: string
  question: string
}

export const fetchDueDiligence = (symbol: string): Promise<DueDiligenceReport> =>
  fetch(`/api/due-diligence/${encodeURIComponent(symbol)}`, {
    headers: { Accept: 'application/json' },
  }).then((response) => json<DueDiligenceReport>(response))

export const fetchFrameworkAudit = (framework = ''): Promise<Record<string, unknown>> => {
  const query = framework ? `?framework=${encodeURIComponent(framework)}` : ''
  return fetch(`/api/due-diligence/framework-audit${query}`, {
    headers: { Accept: 'application/json' },
  }).then((response) => json(response))
}

export const acquireDueDiligence = (
  symbol: string,
  mode: 'missing_or_stale' | 'all' = 'missing_or_stale',
): Promise<{
  accepted: boolean
  symbol: string
  mode?: string
  acquire: Record<string, unknown>
  report: DueDiligenceReport
  places_orders: boolean
}> => fetch(`/api/due-diligence/${encodeURIComponent(symbol)}/acquire?mode=${encodeURIComponent(mode)}`, {
  method: 'POST',
  headers: { Accept: 'application/json' },
}).then((response) => json(response))

export const fetchInvestigatorSuggest = (query: string): Promise<{ query: string; matches: InvestigatorMatch[]; engine: string }> =>
  fetch(`/api/stock-investigator/suggest?q=${encodeURIComponent(query)}&limit=8`, {
    headers: { Accept: 'application/json' },
  }).then((response) => json(response))

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

export const refreshStockFundamentals = (symbol: string): Promise<{
  accepted: boolean
  symbol: string
  sections: Record<string, number | boolean>
  workspace: StockWorkspace
}> => fetch(`/api/stock-intelligence/${encodeURIComponent(symbol)}/refresh-fundamentals`, {
  method: 'POST',
  headers: { Accept: 'application/json' },
}).then((response) => json(response))

export const fetchCommandCenterWorkspace = (): Promise<CommandCenterWorkspace> =>
  fetch('/api/command-center-workspace', { headers: { Accept: 'application/json' } })
    .then((response) => json<CommandCenterWorkspace>(response))

export const fetchScannerWorkspace = (mode: string): Promise<ScannerWorkspace> =>
  fetch(`/api/scanner-workspace/${encodeURIComponent(mode)}`, {
    headers: { Accept: 'application/json' },
  }).then((response) => json<ScannerWorkspace>(response))

export type DeskPipelineStep = {
  id: string
  title: string
  page: string
  why: string
  kind?: string | null
  state: string
  latest_status?: string | null
}

export type DeskPipeline = {
  sequential: boolean
  queued_kind?: string | null
  queued_created?: boolean
  current?: {
    id: string
    title: string
    kind?: string | null
    status: string
    why: string
    page: string
  } | null
  steps: DeskPipelineStep[]
  message: string
  page?: string
  scan_reused?: boolean
  operations?: Array<{ kind?: string; operation_id?: string; status?: string; created?: boolean }>
  active_kind?: string | null
}

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
  counts: { breakouts: number; momentum: number; long_term_picks: number; sniper_breakouts?: number }
  best_breakout?: ScannerWorkspaceRow | null
  best_among_fundamentals?: ScannerWorkspaceRow | null
  best_of_best?: ScannerWorkspaceRow[]
  best_among_note?: string
  sniper_candidates?: ScannerWorkspaceRow[]
  best_setups?: ScannerWorkspaceRow[]
  best_setups_note?: string
  ranking_legend?: {
    best_setups?: string
    best_technical_breakout?: string
    best_among_breakouts?: string
  }
  scan_shared_note?: string
  sepa_rank_used?: boolean
  second_screen_counts?: {
    sniper?: number
    sepa_overlay?: number
    long_term_funds?: number
    sniper_with_second_screen?: number
  }
  scan_progress?: {
    active?: boolean
    stage?: string
    current?: number
    total?: number
    pct?: number | null
    eta_s?: number | null
    eta_label?: string
  }
  telegram?: {
    configured?: boolean
    state?: string
    headline?: string
    detail?: string
    scan_reason?: string
    sniper_reason?: string
    sniper_watch?: number
    live_ticks?: boolean
    last_error?: string
  }
  desk_pipeline?: DeskPipeline | null
}

export const fetchRadarHome = (): Promise<RadarHome> =>
  fetch('/api/radar-home', { headers: { Accept: 'application/json' } })
    .then((response) => json<RadarHome>(response))

export type RecommendationEvidencePanel = {
  sample_size?: number | null
  ev_pct?: number | null
  ev_lb_pct?: number | null
  p_win?: number | null
  confidence?: string | null
  score?: number | null
  rsi?: number | null
  volume_ratio?: number | null
  signals?: string[]
  price_tag?: string
  tech_source?: string
  fundamental_coverage?: number | null
  provenance?: string
}

export type RecommendationCard = {
  symbol: string
  company: string
  category_id: string
  category_label: string
  action_badge: string
  risk_tier: string
  risk_label: string
  setup_label: string
  sector: string
  score: number
  rsi?: number | null
  volume_ratio?: number | null
  price_tag?: string
  tech_source?: string
  reason?: string
  qualify_reason?: string
  evidence_tags?: string[]
  lifecycle: string
  upside_from_entry_pct?: number | null
  upside_to_target_pct?: number | null
  entry?: number | null
  target?: number | null
  cmp?: number | null
  source?: string
  stop?: number | null
  buy_zone_low?: number | null
  buy_zone_high?: number | null
  horizon?: string
  opportunity_label?: string
  expected_payoff?: string
  expected_payoff_detail?: string
  evidence?: string
  setup_quality?: number | null
  setup_quality_label?: string
  strategy_health?: string
  strategy_health_detail?: string
  market_support?: string
  market_support_detail?: string
  why_now?: string[]
  key_points?: string[]
  what_changes_mind?: string[]
  next_step?: string
  evidence_panel?: RecommendationEvidencePanel
  case?: RecommendationCase
  methods?: RecoMethod[]
  method_confirms?: number
  method_fails?: number
  quality_score?: number | null
  method_line?: string
  experts?: RecoExpert[]
  families?: RecoFamily[]
  family_confirms?: number
  primary_thesis?: string
  reco_tier?: string
  reco_tier_label?: string
  entry_state?: string
  stock_quality?: string
  timing?: string
  conflicts?: string[]
  deep_confirm?: boolean
  fundamental_confirmation?: string | null
  research_decision_coverage?: number | null
  research_quality_label?: string | null
}

export type RecoMethod = {
  id: string
  label: string
  status: 'pass' | 'fail' | 'unknown' | string
  detail?: string
  points?: number | null
}

export type RecoExpert = {
  id: string
  label: string
  family?: string
  family_label?: string
  status: string
  eligible?: boolean
  score?: number | null
  rank?: number | null
  thesis?: string
  horizon?: string
  evidence?: string[]
}

export type RecoFamily = {
  id: string
  label: string
  status: string
  strength?: string
  experts?: string[]
  evidence?: string[]
}

export type RecommendationCase = {
  schema_version?: number
  case_id?: string
  symbol?: string
  setup?: string
  idea?: string
  why_now?: string[]
  invalidation?: string[]
  n_similar?: number
  proven?: boolean
  verdict?: string
  memory_line?: string
  win_rate?: number | null
  expectancy_r?: number | null
  places_orders?: boolean
  stance?: string
  setup_quality?: { score?: number | null; label?: string; not_probability?: boolean }
  similar?: {
    found?: boolean
    n_similar?: number
    win_rate?: number | null
    avg_r?: number | null
    avg_mae?: number | null
    avg_mfe?: number | null
    median_hold?: number | null
    environment?: string[]
    line?: string
  }
  edge?: { setup?: string; profile?: string; line?: string }
}

export type RecommendationDesk = {
  market_support: string
  market_support_detail: string
  strategy_health: string
  strategy_health_detail: string
  live_n?: number
}

export type RecommendationCategory = {
  id: string
  label: string
  blurb: string
  icon: string
  count: number
  cards: RecommendationCard[]
  empty_detail: string
}

export type RecommendationsWorkspace = {
  schema_version: number
  generated_at: string
  scan_scanned_at: string
  long_term_scanned_at: string
  records_status: string
  same_ist_day: boolean
  cmp_note: string
  scan_meta?: {
    market_scanned_at?: string
    market_row_count?: number
    long_term_scanned_at?: string
    long_term_row_count?: number
    assigned_count?: number
    high_conviction_count?: number
    good_setup_count?: number
  }
  methods_note?: string
  ensemble?: {
    high_conviction_count: number
    good_setup_count: number
    watch_count?: number
    avoid_count?: number
    empty_high_conviction: boolean
    empty_line: string
    empty_detail?: string
  }
  desk?: RecommendationDesk
  categories: RecommendationCategory[]
  lifecycle: {
    active: RecommendationCard[]
    closed: RecommendationCard[]
    active_count: number
    closed_count: number
  }
  disclaimer: string
  error?: string
}

export const fetchRecommendationsWorkspace = (): Promise<RecommendationsWorkspace> =>
  fetch('/api/recommendations-workspace', { headers: { Accept: 'application/json' } })
    .then((response) => json<RecommendationsWorkspace>(response))

export type MarketMover = {
  symbol: string
  price?: number
  chg_pct?: number
}

export type MarketReportItem = {
  id: string
  title: string
  kind: string
  date: string
  created_at: string
  is_new: boolean
  badge?: string
  summary: string
  takeaways?: string[]
  breakouts_today?: string[]
  gainers?: MarketMover[]
  losers?: MarketMover[]
  snapshot?: {
    indices?: Array<{ name: string; price?: number; chg_pct?: number }>
    commentary?: string
  }
  as_of_ist?: string
  path?: string
}

export type DeskNoteBullet = {
  id: string
  label: string
  available: boolean
  headline: string
  summary: string
  source: string
  url: string
  official: boolean
  published_at: string
  symbols: string[]
  empty_detail: string
}

export type DeskNoteExplainer = {
  id: string
  title: string
  teach_point: string
  why_it_matters: string
  attached_to?: string
}

export type DeskNoteCompany = {
  symbol: string
  name: string
  lens: string
  watch: string[]
  risks: string[]
  available: boolean
  source_headline: string
  source_summary: string
  source: string
  url: string
  scan_status: string
  scan_reason: string
  empty_detail: string
  is_recommendation: boolean
}

export type DeskNote = {
  schema_version?: number
  generated_at?: string
  title?: string
  blurb?: string
  wrap?: DeskNoteBullet[]
  daily_wrap?: Array<{
    id?: string
    text: string
    source?: string
    official?: boolean
    url?: string
    symbols?: string[]
  }>
  wrap_sourced?: number
  wrap_empty?: number
  explainers?: DeskNoteExplainer[]
  desks?: DeskNoteCompany[]
  theme?: { id: string; title: string; body: string }
  memory?: {
    title?: string
    blurb?: string
    setups?: Array<{ setup: string; n_similar: number; proven: boolean; memory_line: string }>
    open_count?: number
    settled_count?: number
    places_orders?: boolean
  }
  decision_memory?: {
    title?: string
    blurb?: string
    shadow?: { proven?: boolean; line?: string; taken?: { n?: number }; rejected?: { n?: number }; gates?: Array<{ gate: string; line: string; verdict?: string }> }
    trust?: { n?: number; line?: string; status?: string; predicted_pct?: number | null; actual_pct?: number | null; calibration_error_pct?: number | null }
    places_orders?: boolean
  }
  disclaimer?: string
  places_orders?: boolean
  error?: string
}

export type ScanHighlights = {
  scanned_at?: string
  row_count?: number
  same_ist_day?: boolean
  ready_to_trade?: number
  breakout_symbols?: string[]
  pre_breakout_symbols?: string[]
  session_gainers?: MarketMover[]
  session_losers?: MarketMover[]
  empty_detail?: string
}

export type MarketReportsWorkspace = {
  schema_version: number
  generated_at: string
  as_of_ist?: string
  title: string
  blurb: string
  reports: MarketReportItem[]
  today_pulse: Record<string, unknown>
  desk_note?: DeskNote
  scan_highlights?: ScanHighlights
  news_meta?: { article_count?: number; available?: boolean }
  empty_detail?: string
  error: string
  disclaimer: string
}

export const fetchMarketReportsWorkspace = (): Promise<MarketReportsWorkspace> =>
  fetch('/api/market-reports-workspace', { headers: { Accept: 'application/json' } })
    .then((response) => json<MarketReportsWorkspace>(response))

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
