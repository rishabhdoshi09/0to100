/** Trade hub helpers — Ready / Lab / Journey. No ticker hard-wires. */

export function readyLaneLabel(lane: string): string {
  if (lane === 'stage2') return 'Stage 2'
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

export function labKidLane(lane: string): string {
  const s = (lane || '').toLowerCase()
  if (s === 'keep') return 'Passed'
  if (s === 'skip') return 'Failed'
  return 'Too few tries'
}

export function labKidTone(lane: string): string {
  const s = (lane || '').toLowerCase()
  if (s === 'keep') return 'is-pass'
  if (s === 'skip') return 'is-lock'
  return 'is-wait'
}

export function labLoopTone(state: string): string {
  const s = (state || '').toUpperCase()
  if (s === 'READY' || s === 'LIVE' || s === 'ARMED') return 'is-live'
  if (s === 'RUN') return 'is-run'
  if (s === 'IDLE') return 'is-idle'
  return 'is-wait'
}

export function labStatusTone(status: string): string {
  const s = (status || '').toUpperCase()
  if (s === 'READY' || s === 'LIVE' || s === 'ARMED' || s === 'NONE') return 'is-pass'
  if (s === 'RUNNING') return 'is-wait'
  if (s === 'MISSING' || s === 'PARTIAL' || s === 'THIN' || s === 'WAIT' || s === 'IDLE') return 'is-wait'
  return 'is-wait'
}
