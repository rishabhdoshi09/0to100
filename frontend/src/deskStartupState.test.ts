import { describe, expect, it } from 'vitest'
import { deskStartupLabel, deskStartupState, effectiveDataReady } from './deskStartupState'

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

describe('header data readiness (9 Nov P0: PREPARING DATA with current history)', () => {
  it('a stale dashboard read cache defers to official history from /api/health', () => {
    expect(effectiveDataReady({ dashboardReady: false, dashboardStale: true, officialHistoryCurrent: true })).toBe(true)
    expect(effectiveDataReady({ dashboardReady: false, dashboardStale: true, officialHistoryCurrent: false })).toBe(false)
    expect(effectiveDataReady({ dashboardReady: false, dashboardStale: true, officialHistoryCurrent: null })).toBe(false)
  })
  it('a current dashboard projection keeps its own verdict', () => {
    expect(effectiveDataReady({ dashboardReady: true, dashboardStale: false, officialHistoryCurrent: false })).toBe(true)
    expect(effectiveDataReady({ dashboardReady: false, dashboardStale: false, officialHistoryCurrent: true })).toBe(false)
  })
  it('yields DATA READY rather than PREPARING DATA in the incident state', () => {
    const ready = effectiveDataReady({ dashboardReady: false, dashboardStale: true, officialHistoryCurrent: true })
    expect(deskStartupLabel(deskStartupState({ dataReady: ready, hasSavedData: true }))).toBe('DATA READY')
  })
})
