/** Pick today's name and describe the Home floor-path. Pure — no fetch. */

const SYMBOL_RE = /^[A-Z0-9&.-]{1,32}$/

export type PathCandidate = {
  symbol?: string | null
}

export function cleanSymbol(value: string | null | undefined): string {
  return String(value || '').trim().toUpperCase()
}

export function isUsableSymbol(value: string | null | undefined): boolean {
  const symbol = cleanSymbol(value)
  return Boolean(symbol) && SYMBOL_RE.test(symbol)
}

export function pickTodaySymbol(input: {
  selected?: string
  best?: PathCandidate | null
  visible?: PathCandidate[]
  scan?: PathCandidate[]
  query?: string
}): string {
  const ordered = [
    input.selected,
    input.best?.symbol,
    input.visible?.[0]?.symbol,
    input.scan?.[0]?.symbol,
    input.query,
  ]
  for (const item of ordered) {
    if (isUsableSymbol(item)) return cleanSymbol(item)
  }
  return ''
}

export type FloorId = 'desk' | 'options' | 'data' | 'holdings' | 'health'

export type FloorJump = {
  id: FloorId
  label: string
  page: string
  intelTab?: string
}

export const FLOOR_JUMPS: FloorJump[] = [
  { id: 'desk', label: 'Desk', page: 'Home' },
  { id: 'options', label: 'Options', page: 'Stock Intelligence', intelTab: 'Options' },
  { id: 'data', label: 'Data', page: 'Research Data' },
  { id: 'holdings', label: 'Holdings', page: 'Paper Portfolio' },
  { id: 'health', label: 'Health', page: 'System Health' },
]

export function pathButtonLabel(symbol: string): string {
  return symbol ? `Open ${symbol}'s floors` : "Open today's path"
}
