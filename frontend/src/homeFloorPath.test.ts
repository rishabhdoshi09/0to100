import { describe, expect, it } from 'vitest'
import {
  FLOOR_JUMPS,
  PATH_BUTTON_LABEL,
  dataFloorCopy,
  deskFloorCopy,
  optionsFloorCopy,
} from './homeFloorPath'

const emptyCtx = {
  scanRecords: 0,
  lastSession: '',
  lastSessionLabel: '',
  sessionBanner: '',
  optionsEodAvailable: false,
  optionsEodSymbols: 0,
  optionsEodAsOf: '',
  dataReady: false,
}

describe('today path is floor-general', () => {
  it('never names a stock in the button or jumps', () => {
    expect(PATH_BUTTON_LABEL).toBe("Open today's path")
    expect(PATH_BUTTON_LABEL).not.toMatch(/RPEL|RELIANCE|[A-Z]{3,}/)
    expect(FLOOR_JUMPS.map((item) => item.page)).toEqual([
      'Home',
      'F&O Desk',
      'Research Data',
      'Paper Portfolio',
      'System Health',
    ])
    expect(FLOOR_JUMPS.every((item) => !('intelTab' in item))).toBe(true)
  })

  it('desk and options copy stay universe-level', () => {
    expect(deskFloorCopy({ ...emptyCtx, scanRecords: 40, lastSessionLabel: 'Friday 14 Aug 2026' }).title)
      .toBe('40 names on the desk')
    expect(deskFloorCopy(emptyCtx).detail).toMatch(/does not pick a stock/)
    expect(optionsFloorCopy({ ...emptyCtx, optionsEodAvailable: true, optionsEodSymbols: 3, optionsEodAsOf: '2026-08-14' }).title)
      .toBe('3 EOD names')
    expect(dataFloorCopy(120, true).detail).toMatch(/open a stock yourself/)
    expect(dataFloorCopy(null, true).detail).toMatch(/No stock is selected/)
  })
})
