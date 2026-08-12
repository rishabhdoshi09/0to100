import type { LongTermRecord } from './types'

export const MIN_LT_FUNDAMENTAL_COVERAGE = 0.5

const QUALITY_CLASSES = new Set([
  'QUALITY_COMPOUNDER',
  'GARP_CANDIDATE',
  'QUALITY_BUT_EXPENSIVE',
])

/** Actionable long-term pick: quality class with enough fundamental evidence. */
export function isLongTermPick(row: LongTermRecord | null | undefined): boolean {
  if (!row?.classification || !QUALITY_CLASSES.has(row.classification)) return false
  return Number(row.fundamental_coverage || 0) >= MIN_LT_FUNDAMENTAL_COVERAGE
}

export function longTermPicks(rows: LongTermRecord[]): LongTermRecord[] {
  return rows.filter(isLongTermPick)
}
