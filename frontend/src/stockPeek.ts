/** Honest number formatting for the Ideas/Recommendations stock popup. */

export function formatPeekValue(value: unknown, unit = ''): string {
  if (value == null || value === '') return 'Not on file'
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return 'Not on file'
    const digits = unit === 'x' || Math.abs(value) >= 100 ? 2 : 1
    const n = Math.abs(value) >= 1000
      ? value.toLocaleString('en-IN', { maximumFractionDigits: 2 })
      : value.toFixed(digits).replace(/\.0$/, '')
    if (unit === '%') return `${n}%`
    if (unit === 'x') return `${n}x`
    return unit ? `${n} ${unit}` : n
  }
  const text = String(value).trim()
  return text || 'Not on file'
}

export type PeekMetric = {
  key: string
  label: string
  value: unknown
  unit?: string
}

export function peekNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim()) {
    const n = Number(value.replace(/,/g, ''))
    return Number.isFinite(n) ? n : null
  }
  return null
}

export function peekUpsidePct(buy: unknown, target: unknown): number | null {
  const entry = peekNumber(buy)
  const tgt = peekNumber(target)
  if (entry == null || tgt == null || entry <= 0) return null
  return Math.round(((tgt / entry) - 1) * 1000) / 10
}

/** Numbers the snapshot can show from the Ideas card alone — any symbol, no fetch. */
export function snapshotFromCard(card: Record<string, unknown> | null | undefined): {
  symbol: string
  company: string
  sector: string
  buy: number | null
  stop: number | null
  target: number | null
  cmp: number | null
  change: number | null
  upside: number | null
  rsi: number | null
  volumeRatio: number | null
} {
  const rec = card || {}
  const cmp = peekNumber(rec.cmp) ?? peekNumber(rec.price) ?? peekNumber(rec.close)
  const buy = peekNumber(rec.entry) ?? peekNumber(rec.entry_price) ?? cmp
  const stop = peekNumber(rec.stop) ?? peekNumber(rec.stop_price)
  const target = peekNumber(rec.target) ?? peekNumber(rec.target_price)
  const storedUpside = peekNumber(rec.upside_from_buy_pct)
  return {
    symbol: String(rec.symbol || '').toUpperCase(),
    company: String(rec.company || rec.symbol || ''),
    sector: String(rec.sector || ''),
    buy,
    stop,
    target,
    cmp,
    change: peekNumber(rec.change_pct) ?? peekNumber(rec.chg_pct),
    upside: storedUpside ?? peekUpsidePct(buy, target),
    rsi: peekNumber(rec.rsi) ?? peekNumber(rec.rsi14),
    volumeRatio: peekNumber(rec.volume_ratio),
  }
}

export function peekPackThin(card: {
  pack_thin?: boolean
  fundamentals?: { metrics?: PeekMetric[]; key_ratios?: Array<{ name?: string; value?: unknown }> }
  ratios?: Array<{ value?: unknown }>
} | null | undefined): boolean {
  if (card?.pack_thin === true) return true
  if (card?.pack_thin === false) return false
  const metrics = filledPeekMetrics(card?.fundamentals?.metrics || [])
  const ratios = (card?.ratios || []).filter((row) => peekNumber(row.value) != null)
  const keys = (card?.fundamentals?.key_ratios || []).filter((row) => {
    const n = peekNumber(row.value)
    return n != null || (typeof row.value === 'string' && row.value.trim() !== '')
  })
  return metrics.length === 0 && ratios.length === 0 && keys.length === 0
}

export function filledPeekMetrics(metrics: PeekMetric[]): PeekMetric[] {
  return metrics.filter((item) => {
    if (peekNumber(item.value) != null) return true
    return typeof item.value === 'string' && item.value.trim() !== '' && item.value !== 'Not on file'
  })
}

export function mergePeekMetrics(primary: PeekMetric[], fallback: PeekMetric[]): PeekMetric[] {
  const byKey = new Map<string, PeekMetric>()
  for (const item of fallback) {
    if (item?.key) byKey.set(item.key, item)
  }
  for (const item of primary) {
    if (!item?.key) continue
    const n = peekNumber(item.value)
    if (n != null || !byKey.has(item.key)) byKey.set(item.key, item)
  }
  return [...byKey.values()]
}

export function orderPeekMetrics(metrics: PeekMetric[], preferred: string[]): PeekMetric[] {
  const byKey = new Map(metrics.map((item) => [item.key, item]))
  const out: PeekMetric[] = []
  for (const key of preferred) {
    const hit = byKey.get(key)
    if (hit) out.push(hit)
  }
  for (const item of metrics) {
    if (!preferred.includes(item.key)) out.push(item)
  }
  return out
}

export const PEEK_FETCH_MS = 8_000
export const PEEK_SCRAPE_MS = 22_000

export const PEEK_TECHNICAL_KEYS = [
  'close', 'change_pct', 'rsi14', 'ema20', 'ema50', 'ema200',
  'atr_pct', 'volume_ratio', 'high_52w', 'low_52w', 'from_high_pct',
]

export const PEEK_FUND_KEYS = [
  'pe', 'roe', 'roce', 'debt_to_equity', 'sales_growth_3y',
  'profit_growth_3y', 'promoter_holding',
]
