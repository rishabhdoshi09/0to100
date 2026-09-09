export type PollGate = {
  tryEnter: () => boolean
  succeed: () => void
  fail: (baseMs?: number, maxMs?: number) => void
  inFlight: () => boolean
  backoffUntil: () => number
}

export function createPollGate(): PollGate {
  let busy = false
  let failures = 0
  let until = 0
  return {
    tryEnter() {
      if (busy) return false
      if (Date.now() < until) return false
      busy = true
      return true
    },
    succeed() {
      busy = false
      failures = 0
      until = 0
    },
    fail(baseMs = 2000, maxMs = 60_000) {
      busy = false
      failures += 1
      const exp = Math.min(5, failures)
      until = Date.now() + Math.min(maxMs, baseMs * 2 ** exp)
    },
    inFlight() {
      return busy
    },
    backoffUntil() {
      return until
    },
  }
}

const inflight = new Map<string, Promise<unknown>>()

export function dedupeInFlight<T>(key: string, start: () => Promise<T>): Promise<T> {
  const existing = inflight.get(key)
  if (existing) return existing as Promise<T>
  const pending = start().finally(() => {
    if (inflight.get(key) === pending) inflight.delete(key)
  })
  inflight.set(key, pending)
  return pending
}

export function resetDedupeForTests() {
  inflight.clear()
}
