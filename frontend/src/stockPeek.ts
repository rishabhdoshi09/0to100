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

export const PEEK_TECHNICAL_KEYS = [
  'close', 'change_pct', 'rsi14', 'ema20', 'ema50', 'ema200',
  'atr_pct', 'volume_ratio', 'high_52w', 'low_52w', 'from_high_pct',
]

export const PEEK_FUND_KEYS = [
  'pe', 'roe', 'roce', 'debt_to_equity', 'sales_growth_3y',
  'profit_growth_3y', 'promoter_holding',
]
