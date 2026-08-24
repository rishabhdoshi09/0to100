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

export const relativeAge = (value?: string | number | null, nowMs = Date.now()): string => {
  if (value == null || value === '') return 'Not run'
  const d = typeof value === 'number' ? new Date(value * 1000) : new Date(value)
  if (Number.isNaN(d.getTime())) return String(value).slice(0, 19)
  const sec = Math.max(0, Math.round((nowMs - d.getTime()) / 1000))
  if (sec < 60) return `${sec}s ago`
  if (sec < 3600) return `${Math.round(sec / 60)} min ago`
  if (sec < 86400) return `${Math.round(sec / 3600)} hr ago`
  const days = Math.round(sec / 86400)
  return days === 1 ? 'yesterday' : `${days} days ago`
}

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
