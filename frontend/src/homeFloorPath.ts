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

export const PATH_BUTTON_LABEL = "Fill today's desk"

export type NextStepId = 'working' | 'fill_desk' | 'find_names' | 'add_long_term' | 'see_picture'

export type NextStep = {
  id: NextStepId
  label: string
  why: string
  resultHint: string
}

export function decideNextStep(input: {
  dataReady: boolean
  readinessScore: number
  scanRecords: number
  longTermRecords: number
  scanBusy: boolean
  longTermBusy: boolean
}): NextStep {
  if (input.scanBusy || input.longTermBusy) {
    return {
      id: 'working',
      label: 'Working…',
      why: 'Stay on this page. Files and names are filling by themselves.',
      resultHint: 'Results appear below when the job finishes. You do not need the sidebar.',
    }
  }
  if (!input.dataReady || input.readinessScore < 70) {
    return {
      id: 'fill_desk',
      label: "Fill today's desk",
      why: 'One click prepares official prices, news, a market scan and long-term research.',
      resultHint: 'The desk then shows names. Options, Data, Holdings and Health fill in next to it.',
    }
  }
  if (input.scanRecords <= 0) {
    return {
      id: 'find_names',
      label: "Find today's names",
      why: 'Price files are ready. This click scans the market and puts names on the desk.',
      resultHint: 'Breakout and momentum lists appear below. No stock is pre-picked.',
    }
  }
  if (input.longTermRecords <= 0) {
    return {
      id: 'add_long_term',
      label: 'Add quality research',
      why: 'Technical names are in. This click adds the quality-and-valuation layer.',
      resultHint: 'Long-term picks join the desk. Still not a buy order.',
    }
  }
  return {
    id: 'see_picture',
    label: "Refresh today's picture",
    why: 'The desk already has names. This click re-reads the other floors — it does not pick a stock.',
    resultHint: 'Empty floors stay empty and honest.',
  }
}

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
