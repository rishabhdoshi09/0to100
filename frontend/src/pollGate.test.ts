import { describe, expect, it } from 'vitest'
import { createPollGate, dedupeInFlight, resetDedupeForTests } from './pollGate'

describe('poll gate', () => {
  it('refuses a second enter while a request is in flight', () => {
    const gate = createPollGate()
    expect(gate.tryEnter()).toBe(true)
    expect(gate.tryEnter()).toBe(false)
    gate.succeed()
    expect(gate.tryEnter()).toBe(true)
  })

  it('backs off after failure instead of tight-looping', () => {
    const gate = createPollGate()
    expect(gate.tryEnter()).toBe(true)
    gate.fail(10_000, 60_000)
    expect(gate.tryEnter()).toBe(false)
    expect(gate.backoffUntil()).toBeGreaterThan(Date.now())
  })

  it('dedupes in-flight fetches for the same resource', async () => {
    resetDedupeForTests()
    let starts = 0
    const start = () => {
      starts += 1
      return new Promise<string>((resolve) => setTimeout(() => resolve('ok'), 20))
    }
    const [a, b] = await Promise.all([
      dedupeInFlight('GET /api/dashboard', start),
      dedupeInFlight('GET /api/dashboard', start),
    ])
    expect(a).toBe('ok')
    expect(b).toBe('ok')
    expect(starts).toBe(1)
  })
})
