import { describe, expect, it } from 'vitest'
import { deskStartupLabel, deskStartupState } from './deskStartupState'

describe('desk startup state', () => {
  it('never shows PREPARING DATA when the API is dead or resources are exhausted', () => {
    expect(deskStartupState({ resourceState: 'RESOURCE_EXHAUSTED', dataReady: false })).toBe('RESOURCE_EXHAUSTED')
    expect(deskStartupLabel('RESOURCE_EXHAUSTED')).toBe('RESOURCE EXHAUSTED')
    expect(deskStartupState({ apiUnresponsive: true, dataReady: false })).toBe('API_UNRESPONSIVE')
    expect(deskStartupState({ operationStuck: true })).toBe('OPERATION_STUCK')
    expect(deskStartupState({ waitingForProvider: true })).toBe('WAITING_FOR_PROVIDER')
    expect(deskStartupState({ historyStale: true, hasSavedData: true })).toBe('HISTORY_STALE')
    expect(deskStartupState({ dataReady: true })).toBe('READY')
    expect(deskStartupState({})).toBe('PREPARING_DATA')
  })

  it('keeps saved-data usable when a provider is unavailable', () => {
    const state = deskStartupState({
      waitingForProvider: true,
      hasSavedData: true,
      dataReady: true,
    })
    expect(state).toBe('WAITING_FOR_PROVIDER')
    expect(state).not.toBe('PREPARING_DATA')
  })
})
