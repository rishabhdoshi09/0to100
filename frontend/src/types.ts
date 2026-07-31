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
  [key: string]: unknown
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
    summary: Record<string, number>
    records: LongTermRecord[]
  }
  paper: {
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
  }
  autonomy: {
    running: boolean
    state: string
    plain_state: string
    explanation: string
    heartbeat_ist: string
    new_paper_entries: boolean
    recent_dialogue: Array<Record<string, unknown>>
    jobs: Record<string, unknown>
  }
  conviction: Array<ScanRecord & {
    classification?: string
    conviction_score?: number
    risks?: string[]
  }>
}

export type ChartBar = {
  time: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}
