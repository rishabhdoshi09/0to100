import type { PastDecisionSimulation } from './productApi'

const HONEST = new Set([
  'UNAVAILABLE',
  'UNKNOWN',
  'NOT_ENTERED',
  'FAILED',
  'HISTORICAL_DECISION_UNAVAILABLE',
  'AMBIGUOUS_HISTORICAL_DECISION',
  'PIT_INTEGRITY_FAILED',
])

export function isHonestUnknown(value: unknown): boolean {
  if (value == null || value === '') return true
  return HONEST.has(String(value).toUpperCase())
}

export function displayHonest(value: unknown, fallback = 'UNAVAILABLE'): string {
  if (isHonestUnknown(value)) return fallback
  return String(value)
}

export function simulationUiState(result: PastDecisionSimulation | null, error = ''): 'idle' | 'error' | 'unavailable' | 'failed' | 'ambiguous' | 'ready' {
  if (error) return 'error'
  if (!result) return 'idle'
  const status = String(result.status || '').toUpperCase()
  if (status === 'FAILED' || status === 'PIT_INTEGRITY_FAILED') return 'failed'
  if (status === 'AMBIGUOUS_HISTORICAL_DECISION') return 'ambiguous'
  if (status === 'HISTORICAL_DECISION_UNAVAILABLE' || result.available === false) return 'unavailable'
  return 'ready'
}

export function originalVsSimulated(result: PastDecisionSimulation) {
  return {
    originalAction: displayHonest(result.original?.action),
    simulatedAction: displayHonest(result.simulated?.action),
    evidenceLabel: result.evidence_at_t?.label || 'Information known at decision time',
    outcomeLabel: result.subsequent_outcome?.label || 'What happened after T (not known at decision time)',
    lookahead: Boolean(result.evidence_at_t?.future_bars_used_for_decision),
  }
}
