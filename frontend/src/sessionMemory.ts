/** In-session + sessionStorage last-good memory for desk continuity.
 *
 * Backend truth remains authoritative. This cache exists only so a transient API
 * disconnect or process restart does not turn an already-loaded desk into an
 * empty screen on browser reload. Fresh successful responses replace it; richer
 * previous data beats an empty/degraded refresh.
 */

const memory = new Map<string, unknown>()

export function remember<T>(key: string, value: T): T {
  memory.set(key, value)
  writeSessionJson(`qt:${key}`, value)
  return value
}

export function recall<T>(key: string): T | undefined {
  if (memory.has(key)) return memory.get(key) as T
  const stored = readSessionJson<T>(`qt:${key}`)
  if (stored != null) {
    memory.set(key, stored)
    return stored
  }
  return undefined
}

/** Open Stock Intelligence on the Investigate tab for this symbol. */
export function markInvestigate(symbol: string): string {
  return remember('stock-investigate', String(symbol || '').toUpperCase())
}

export function wantsInvestigate(symbol: string): boolean {
  const want = recall<string>('stock-investigate')
  return Boolean(want && symbol && want.toUpperCase() === String(symbol).toUpperCase())
}

export function keepRicher<T>(key: string, next: T, isEmpty: (value: T) => boolean): T {
  const prev = recall<T>(key)
  if (prev !== undefined && isEmpty(next) && !isEmpty(prev)) return prev
  return remember(key, next)
}

/**
 * Historically these helpers were RAM-only because large desks could approach
 * browser quota. That made reload destructive. They now use sessionStorage when
 * possible and degrade to RAM-only only when the browser refuses the write.
 */
export function recallMemory<T>(key: string): T | undefined {
  if (memory.has(key)) return memory.get(key) as T
  const stored = readSessionJson<T>(`qt:${key}`)
  if (stored != null) {
    memory.set(key, stored)
    return stored
  }
  return undefined
}

export function keepRicherMemory<T>(key: string, next: T, isEmpty: (value: T) => boolean): T {
  const prev = recallMemory<T>(key)
  if (prev !== undefined && isEmpty(next) && !isEmpty(prev)) return prev
  memory.set(key, next)
  // Best effort. writeSessionJson already handles quota/private-mode failures;
  // the live in-memory desk remains intact even if persistence is unavailable.
  writeSessionJson(`qt:${key}`, next)
  return next
}

export function readSessionJson<T>(key: string): T | null {
  try {
    const raw = window.sessionStorage.getItem(key)
    if (!raw) return null
    return JSON.parse(raw) as T
  } catch {
    return null
  }
}

export function writeSessionJson(key: string, value: unknown): boolean {
  try {
    window.sessionStorage.setItem(key, JSON.stringify(value))
    return true
  } catch {
    if (key.startsWith('qt:') || key === 'quantterm-dashboard') {
      try {
        // Drop the large legacy dashboard first, not the currently visible desk.
        window.sessionStorage.removeItem('quantterm-dashboard')
        window.sessionStorage.setItem(key, JSON.stringify(value))
        return true
      } catch {
        return false
      }
    }
    return false
  }
}

export type DeskNav = {
  active: string
  selected: string
  compare: string[]
}

export function readDeskNav(): DeskNav {
  return readSessionJson<DeskNav>('quantterm-nav') || { active: 'Home', selected: '', compare: [] }
}

export function writeDeskNav(nav: DeskNav): void {
  writeSessionJson('quantterm-nav', nav)
}
