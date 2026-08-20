import { describe, expect, it } from 'vitest'
import {
  FLOOR_JUMPS,
  dataFloorCopy,
  decideNextStep,
  deskFloorCopy,
  optionsFloorCopy,
} from './homeFloorPath'

const ready = {
  dataReady: true,
  readinessScore: 90,
  scanRecords: 40,
  longTermRecords: 10,
  scanBusy: false,
  longTermBusy: false,
}

describe('decideNextStep', () => {
  it('asks a stranger to fill the desk when files or score are thin', () => {
    const step = decideNextStep({ ...ready, dataReady: false, readinessScore: 40, scanRecords: 0, longTermRecords: 0 })
    expect(step.id).toBe('fill_desk')
    expect(step.label).toBe("Fill today's desk")
    expect(step.label).not.toMatch(/RPEL|RELIANCE|ELGIEQUIP/)
  })

  it('scans when files are ready but the desk is empty', () => {
    expect(decideNextStep({ ...ready, scanRecords: 0, longTermRecords: 0 }).id).toBe('find_names')
  })

  it('adds long-term after names exist', () => {
    expect(decideNextStep({ ...ready, longTermRecords: 0 }).id).toBe('add_long_term')
  })

  it('only refreshes the picture when complementary layers exist', () => {
    expect(decideNextStep(ready).id).toBe('see_picture')
  })

  it('does not start a second job while one is running', () => {
    expect(decideNextStep({ ...ready, scanBusy: true }).id).toBe('working')
    expect(decideNextStep({ ...ready, longTermBusy: true }).label).toBe('Working…')
  })

  it('never names a stock in the next-job button', () => {
    const labels = [
      decideNextStep({ ...ready, dataReady: false, readinessScore: 10, scanRecords: 0, longTermRecords: 0 }).label,
      decideNextStep({ ...ready, scanRecords: 0, longTermRecords: 0 }).label,
      decideNextStep({ ...ready, longTermRecords: 0 }).label,
      decideNextStep(ready).label,
      decideNextStep({ ...ready, scanBusy: true }).label,
    ]
    for (const label of labels) {
      expect(label).not.toMatch(/RPEL|RELIANCE|YATHARTH|ELGIEQUIP/)
    }
  })
})

describe('today path is floor-general', () => {
  it('jumps to floors, never a stock Options tab', () => {
    expect(FLOOR_JUMPS.map((item) => item.page)).toEqual([
      'Home',
      'F&O Desk',
      'Research Data',
      'Paper Portfolio',
      'System Health',
    ])
  })

  it('desk and options copy stay universe-level', () => {
    const empty = {
      scanRecords: 0,
      lastSession: '',
      lastSessionLabel: '',
      sessionBanner: '',
      optionsEodAvailable: false,
      optionsEodSymbols: 0,
      optionsEodAsOf: '',
      dataReady: false,
    }
    expect(deskFloorCopy({ ...empty, scanRecords: 40, lastSessionLabel: 'Friday 14 Aug 2026' }).title)
      .toBe('40 names on the desk')
    expect(deskFloorCopy(empty).detail).toMatch(/does not pick a stock/)
    expect(optionsFloorCopy({ ...empty, optionsEodAvailable: true, optionsEodSymbols: 3, optionsEodAsOf: '2026-08-14' }).title)
      .toBe('3 EOD names')
    expect(dataFloorCopy(120, true).detail).toMatch(/open a stock yourself/)
  })
})
