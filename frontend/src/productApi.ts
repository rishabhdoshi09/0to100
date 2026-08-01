import type { ControlName, LongTermRecord, NewsArticle, ScanRecord } from './types'

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

export const refreshStockFundamentals = (symbol: string): Promise<{
  accepted: boolean
  symbol: string
  sections: Record<string, number | boolean>
  workspace: StockWorkspace
}> => fetch(`/api/stock-intelligence/${encodeURIComponent(symbol)}/refresh-fundamentals`, {
  method: 'POST',
  headers: { Accept: 'application/json' },
}).then((response) => json(response))
