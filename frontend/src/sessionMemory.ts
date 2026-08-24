/** In-session memory so tab changes and reloads do not flash empty desks. */

const memory = new Map<string, unknown>()

export function remember<T>(key: string, value: T): T {
  memory.set(key, value)
  return value
}

export function recall<T>(key: string): T | undefined {
  if (!memory.has(key)) return undefined
  return memory.get(key) as T
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

export function writeSessionJson(key: string, value: unknown): void {
  try {
    window.sessionStorage.setItem(key, JSON.stringify(value))
  } catch {
    /* quota / private mode */
  }
}
