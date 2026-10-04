export type PaperCycleNote = {
  headline: string
  reasons: string[]
  eligibility: string
  posture: string
}

const asRecord = (value: unknown): Record<string, unknown> | null => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  return value as Record<string, unknown>
}

const asStrings = (value: unknown): string[] => {
  if (!Array.isArray(value)) return []
  return value.map((item) => String(item ?? '').trim()).filter(Boolean)
}

/** Operator line for the latest paper cycle. Empty input stays empty. */
export function paperCycleNote(
  lastCycle: Record<string, unknown> | null | undefined,
): PaperCycleNote | null {
  const cycle = asRecord(lastCycle)
  if (!cycle) return null
  const why = asRecord(cycle.why_no_trade)
  const brain = asRecord(cycle.brain)
  const eligibility = String(cycle.eligibility || (why ? why.decision : '') || '').trim()
  const posture = String(brain?.posture || '').trim()
  const headline = String(why?.headline || '').trim()
  const reasons = why ? asStrings(why.reasons) : asStrings(cycle.cycle_reasons)
  if (!headline && !reasons.length && !eligibility && !posture) return null
  return { headline, reasons, eligibility, posture }
}
