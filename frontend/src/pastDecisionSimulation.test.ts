import { describe, expect, it } from 'vitest'
import { pastDecisionSimulationUrl } from './productApi'
import { displayHonest, isHonestUnknown, originalVsSimulated, simulationUiState } from './pastDecisionSimulation'

describe('past decision simulation contract', () => {
  it('addresses one decision on the existing simulator route', () => {
    expect(pastDecisionSimulationUrl({
      symbol: 'RELIANCE',
      as_of: '2025-07-15',
      alternative: 'BUY',
    })).toBe('/api/decision-simulator?symbol=RELIANCE&as_of=2025-07-15&alternative=BUY')
  })

  it('keeps honesty states visible', () => {
    expect(isHonestUnknown('UNAVAILABLE')).toBe(true)
    expect(isHonestUnknown('NOT_ENTERED')).toBe(true)
    expect(displayHonest(null)).toBe('UNAVAILABLE')
    expect(displayHonest('WAIT')).toBe('WAIT')
  })

  it('separates original decision from the simulated alternative', () => {
    const view = originalVsSimulated({
      original: { action: 'WAIT' },
      simulated: { action: 'BUY' },
      evidence_at_t: { label: 'Information known at decision time', future_bars_used_for_decision: false },
      subsequent_outcome: { label: 'What happened after T (not known at decision time)' },
    })
    expect(view.originalAction).toBe('WAIT')
    expect(view.simulatedAction).toBe('BUY')
    expect(view.lookahead).toBe(false)
    expect(view.evidenceLabel).toMatch(/decision time/i)
    expect(view.outcomeLabel).toMatch(/after T/i)
  })

  it('surfaces missing history and failures instead of sample data', () => {
    expect(simulationUiState(null, 'backend down')).toBe('error')
    expect(simulationUiState({ status: 'FAILED', error: 'boom' })).toBe('failed')
    expect(simulationUiState({ status: 'HISTORICAL_DECISION_UNAVAILABLE', available: false })).toBe('unavailable')
    expect(simulationUiState({ status: 'SUCCEEDED', available: true, original: { action: 'WAIT' } })).toBe('ready')
  })
})
