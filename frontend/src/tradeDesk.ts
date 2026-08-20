/** Trade hub helpers — Ready / Lab / Journey. No ticker hard-wires. */

export function readyLaneLabel(lane: string): string {
  if (lane === 'prime') return 'Prime'
  if (lane === 'actionable') return 'Ticket'
  return lane || 'Watch'
}

export function journeyTone(status: string): string {
  const s = (status || '').toUpperCase()
  if (s === 'PASS') return 'is-pass'
  if (s === 'BLOCK') return 'is-block'
  if (s === 'LOCKED') return 'is-lock'
  return 'is-wait'
}

export function labStatusTone(status: string): string {
  const s = (status || '').toUpperCase()
  if (s === 'READY') return 'is-pass'
  if (s === 'RUNNING') return 'is-wait'
  if (s === 'MISSING' || s === 'PARTIAL' || s === 'THIN') return 'is-wait'
  if (s === 'NONE') return 'is-pass'
  return 'is-wait'
}
