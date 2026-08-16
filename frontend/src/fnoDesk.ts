export const FNO_INDEX_UNDERLYINGS = ['NIFTY', 'BANKNIFTY', 'FINNIFTY'] as const

export type FnoIndexUnderlying = (typeof FNO_INDEX_UNDERLYINGS)[number]

export function isFnoIndex(symbol?: string): boolean {
  const clean = (symbol || '').trim().toUpperCase()
  return (FNO_INDEX_UNDERLYINGS as readonly string[]).includes(clean)
}

/** Stock floors open on the selected name. Indices stay on the F&O desk. */
export function defaultFnoFocus(selected?: string): string {
  const clean = (selected || '').trim().toUpperCase()
  if (!clean) return 'NIFTY'
  return clean
}

export function canOpenStockFromFno(symbol?: string): boolean {
  const clean = (symbol || '').trim().toUpperCase()
  return Boolean(clean) && !isFnoIndex(clean)
}
