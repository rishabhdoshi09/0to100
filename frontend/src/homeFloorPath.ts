/** Home floor-path chrome. No symbol — the click wires floors, not a stock. */

export type FloorId = 'desk' | 'options' | 'data' | 'holdings' | 'health'

export type FloorJump = {
  id: FloorId
  label: string
  page: string
}

export const FLOOR_JUMPS: FloorJump[] = [
  { id: 'desk', label: 'Desk', page: 'Home' },
  { id: 'options', label: 'Options', page: 'F&O Desk' },
  { id: 'data', label: 'Data', page: 'Research Data' },
  { id: 'holdings', label: 'Holdings', page: 'Paper Portfolio' },
  { id: 'health', label: 'Health', page: 'System Health' },
]

export const PATH_BUTTON_LABEL = "Open today's path"

export type FloorContext = {
  scanRecords: number
  lastSession: string
  lastSessionLabel: string
  sessionBanner: string
  optionsEodAvailable: boolean
  optionsEodSymbols: number
  optionsEodAsOf: string
  dataReady: boolean
}

export function deskFloorCopy(ctx: FloorContext): { title: string; detail: string } {
  if (ctx.scanRecords > 0) {
    return {
      title: `${ctx.scanRecords} names on the desk`,
      detail: ctx.lastSessionLabel
        ? `Last official session ${ctx.lastSessionLabel}`
        : 'Scan is on file — pick a name when you want a stock',
    }
  }
  return {
    title: 'Desk is empty',
    detail: 'Scan now fills lanes. This click does not pick a stock.',
  }
}

export function optionsFloorCopy(ctx: FloorContext, chainNote?: string): { title: string; detail: string } {
  if (ctx.optionsEodAvailable) {
    return {
      title: `${ctx.optionsEodSymbols} EOD names`,
      detail: ctx.optionsEodAsOf
        ? `Store as of ${ctx.optionsEodAsOf} · index job + names you open later`
        : 'EOD store has snapshots · not a live Greek stream',
    }
  }
  return {
    title: 'Options floor is empty',
    detail: chainNote || 'Nightly index capture has not landed. Opening a stock later queues it — this click does not pick one.',
  }
}

export function dataFloorCopy(audited: number | null, ready: boolean): { title: string; detail: string } {
  if (audited != null && audited > 0) {
    return {
      title: `${audited} names audited`,
      detail: 'File layer for the universe — open a stock yourself if you need one name',
    }
  }
  return {
    title: ready ? 'File layer reachable' : 'Evidence offline',
    detail: 'Same-origin /evidence. No stock is selected by this click.',
  }
}
