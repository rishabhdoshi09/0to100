import { describe, expect, it } from 'vitest'
import { deskStartupLabel, deskStartupReason, deskStartupState } from './deskStartupState'

describe('desk startup state', () => {
  it('never shows PREPARING DATA when the API is dead or resources are exhausted', () => {
    expect(deskStartupState({ resourceState: 'RESOURCE_EXHAUSTED', dataReady: false })).toBe('RESOURCE_EXHAUSTED')
    expect(deskStartupLabel('RESOURCE_EXHAUSTED')).toBe('RESOURCE EXHAUSTED')
    expect(deskStartupState({ resourceState: 'RESOURCE_UNKNOWN', dataReady: true })).toBe('RESOURCE_UNKNOWN')
    expect(deskStartupLabel('RESOURCE_UNKNOWN')).toBe('RESOURCE UNKNOWN')
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

  it('does not hide a usable saved desk behind generic PREPARING DATA when health is not READY', () => {
    expect(deskStartupState({
      lifecycle: 'STARTING',
      hasSavedData: true,
      dataReady: true,
    })).toBe('STARTING')
    expect(deskStartupLabel('STARTING')).toBe('STARTING')
    expect(deskStartupState({
      lifecycle: 'DEGRADED',
      hasSavedData: true,
      dataReady: true,
    })).toBe('DEGRADED')
    expect(deskStartupState({
      lifecycle: 'STARTING',
      hasSavedData: false,
      dataReady: false,
    })).toBe('PREPARING_DATA')
    expect(deskStartupState({
      lifecycle: 'READY',
      dataReady: true,
      hasSavedData: true,
    })).toBe('READY')
    expect(deskStartupState({
      dataReady: true,
      hasSavedData: true,
    })).toBe('READY')
  })

  it('shows the real health blocker instead of Desk is still coming up', () => {
    const reason = deskStartupReason({
      lifecycle: 'STARTING',
      reason: 'Desk is still coming up',
      reasons: [],
      components: [
        { name: 'official_history', status: 'STARTING', detail: 'HISTORY_STALE' },
        { name: 'api', status: 'READY', detail: 'Terminal API is serving' },
      ],
      state: 'STARTING',
    })
    expect(reason).toContain('official_history')
    expect(reason).toContain('HISTORY_STALE')
    expect(reason).not.toMatch(/desk is still coming up/i)
  })
})
