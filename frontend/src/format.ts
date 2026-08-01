export const money = (value?: number | null, decimals = 0): string =>
  Number.isFinite(value)
    ? `₹${Number(value).toLocaleString('en-IN', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
      })}`
    : '—'

export const pct = (value?: number | null): string =>
  Number.isFinite(value)
    ? `${Number(value) >= 0 ? '+' : ''}${Number(value).toFixed(2)}%`
    : '—'

export const score = (value?: number | null): number =>
  Number.isFinite(value) ? Math.round(Number(value)) : 0

export const words = (value?: string | null): string =>
  String(value || 'Unavailable').replaceAll('_', ' ').replace(/\b\w/g, (c) => c.toUpperCase())

export const compactDateTime = (value?: string | number | null): string => {
  if (!value) return '—'
  if (typeof value === 'number') {
    const d = new Date(value * 1000)
    return Number.isNaN(d.getTime()) ? '—' : d.toLocaleString('en-IN')
  }
  const d = new Date(value)
  return Number.isNaN(d.getTime()) ? String(value).slice(0, 19) : d.toLocaleString('en-IN')
}

export const boolLabel = (value?: boolean): string => value ? 'ON' : 'OFF'
