/** Shared desk-thesis helpers — keep card taps and sector-wave copy honest. */

export function deskSymbol(value: unknown): string {
  return String(value || '').trim().toUpperCase()
}

export function sectorWaveVerdict(wave?: { wave?: string; verdict?: string } | null): 'YES' | 'NO' {
  if (wave?.verdict === 'YES' || wave?.verdict === 'NO') return wave.verdict
  return String(wave?.wave || '') === 'INFLOW' ? 'YES' : 'NO'
}

export function sectorWaveFirstLine(wave?: {
  wave?: string
  verdict?: string
  verdict_line?: string
} | null): string {
  const line = String(wave?.verdict_line || '').trim()
  if (line) return line
  return sectorWaveVerdict(wave) === 'YES'
    ? 'YES — sector money is coming in around this name.'
    : 'NO — not enough current sector evidence to claim a wave.'
}

export function filingsNeedRefresh(thesis: {
  filings_stale?: boolean
  filings_refresh_attempted?: boolean
  fundamentals?: { available?: boolean; coverage_pct?: number }
} | null | undefined): boolean {
  if (!thesis) return false
  if (thesis.filings_stale && !thesis.filings_refresh_attempted) return true
  if (!thesis.fundamentals?.available) return true
  return Number(thesis.fundamentals.coverage_pct || 0) < 40
}

export function thesisReplacesList(phone: boolean, selected: string): boolean {
  return phone && Boolean(deskSymbol(selected))
}
