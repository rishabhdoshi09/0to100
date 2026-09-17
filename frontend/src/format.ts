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

function parseInstant(value?: string | number | null): Date | null {
  if (value == null || value === '') return null
  if (typeof value === 'number') {
    const ms = Math.abs(value) > 1e12 ? value : value * 1000
    const date = new Date(ms)
    return Number.isNaN(date.getTime()) ? null : date
  }
  const text = String(value).trim()
  if (/^\d+(\.\d+)?$/.test(text)) {
    const n = Number(text)
    const ms = Math.abs(n) > 1e12 ? n : n * 1000
    const date = new Date(ms)
    return Number.isNaN(date.getTime()) ? null : date
  }
  const date = new Date(text)
  return Number.isNaN(date.getTime()) ? null : date
}

export const formatIst = (value?: string | number | null): string => {
  const date = parseInstant(value)
  if (!date) return value == null || value === '' ? '—' : String(value).slice(0, 19)
  const parts = new Intl.DateTimeFormat('en-GB', {
    timeZone: 'Asia/Kolkata',
    day: '2-digit',
    month: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  }).formatToParts(date)
  const get = (type: string) => parts.find((part) => part.type === type)?.value || ''
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
  const month = months[Math.max(0, Number(get('month')) - 1)] || get('month')
  return `${Number(get('day'))} ${month} ${get('year')} · ${get('hour')}:${get('minute')} IST`
}

export const relativeAge = (value?: string | number | null, nowMs = Date.now()): string => {
  if (value == null || value === '') return 'Not run'
  const d = parseInstant(value)
  if (!d) return String(value).slice(0, 19)
  const sec = Math.max(0, Math.round((nowMs - d.getTime()) / 1000))
  if (sec < 60) return `${sec}s ago`
  if (sec < 3600) return `${Math.round(sec / 60)} min ago`
  if (sec < 86400) return `${Math.round(sec / 3600)} hr ago`
  const days = Math.round(sec / 86400)
  return days === 1 ? 'yesterday' : `${days} days ago`
}

export const compactDateTime = (value?: string | number | null): string => formatIst(value)

export const boolLabel = (value?: boolean): string => value ? 'ON' : 'OFF'
