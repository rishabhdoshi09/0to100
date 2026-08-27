/** In-session + sessionStorage memory so tab changes and reloads keep the last desk. */

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
  remember('stock-tab', { symbol: String(symbol || '').toUpperCase(), tab: 'Investigate' })
  return remember('stock-investigate', String(symbol || '').toUpperCase())
}

/** Open Stock Intelligence on the Analyser tab for this symbol. */
export function markAnalyser(symbol: string): string {
  remember('stock-tab', { symbol: String(symbol || '').toUpperCase(), tab: 'Analyser' })
  return String(symbol || '').toUpperCase()
}

export function wantedStockTab(symbol: string): string {
  const want = recall<{ symbol?: string; tab?: string }>('stock-tab')
  if (want && symbol && String(want.symbol || '').toUpperCase() === String(symbol).toUpperCase()) {
    return String(want.tab || 'Analyser')
  }
  return 'Analyser'
}

export function rememberRecentSymbol(symbol: string): string[] {
  const clean = String(symbol || '').toUpperCase()
  if (!clean) return recall<string[]>('stock-recents') || []
  const prev = (recall<string[]>('stock-recents') || []).filter((item) => item !== clean)
  return remember('stock-recents', [clean, ...prev].slice(0, 6))
}

export function recentSymbols(): string[] {
  return recall<string[]>('stock-recents') || []
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

/** In-memory only. Do not sessionStorage large desks (recommendations, reports). */
export function recallMemory<T>(key: string): T | undefined {
  if (memory.has(key)) return memory.get(key) as T
  return undefined
}

export function keepRicherMemory<T>(key: string, next: T, isEmpty: (value: T) => boolean): T {
  const prev = recallMemory<T>(key)
  if (prev !== undefined && isEmpty(next) && !isEmpty(prev)) return prev
  memory.set(key, next)
  try {
    window.sessionStorage.removeItem(`qt:${key}`)
  } catch {
    /* quota / private mode */
  }
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
