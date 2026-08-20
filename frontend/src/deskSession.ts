/** Last-look desk session + payload cache so a reload does not wipe the user's place. */

export const SESSION_KEY = 'quantterm-desk-session'
export const CACHE_PREFIX = 'quantterm-cache:'
export const MAX_CACHE_CHARS = 3_200_000

export const DESK_PAGES = new Set([
  'Home',
  'Command Center',
  'Market Scanner',
  'Recommendations',
  'Market Reports',
  'Stock Intelligence',
  'Long-Term Picks',
  'Compare',
  'Watchlist',
  'Market Overview',
  'News & Events',
  'Education',
  'Research Data',
  'F&O Desk',
  'Paper Portfolio',
  'System Health',
  'Scanner',
  'Long-Term',
  'Portfolio',
  'Market Internals',
  'Automation',
])

/** Only these pages may remember a symbol. Ideas/Home never pin a favorite. */
export const STOCK_FOCUS_PAGES = new Set([
  'Stock Intelligence',
  'F&O Desk',
  'Research Data',
  'Compare',
])

export type DeskSession = {
  active: string
  selected: string
  compareSymbols: string[]
  intelTab?: string
  ideasCategory?: string
  ideasLifecycle?: 'Active' | 'Closed'
  scannerTab?: string
  homeLane?: string
  updatedAt: number
}

export function isNseSymbol(value: string): boolean {
  return /^[A-Z0-9&.-]{1,32}$/.test(String(value || '').trim().toUpperCase())
}

export function pinnedSymbol(active: string, selected: string): string {
  if (!STOCK_FOCUS_PAGES.has(active)) return ''
  return isNseSymbol(selected) ? selected.trim().toUpperCase() : ''
}

function storage(): Storage | null {
  try {
    const store = globalThis.localStorage
    if (!store) return null
    return store
  } catch {
    return null
  }
}

export function loadDeskSession(): DeskSession | null {
  const raw = storage()?.getItem(SESSION_KEY)
  if (!raw) return null
  try {
    const parsed = JSON.parse(raw) as Partial<DeskSession>
    const active = DESK_PAGES.has(String(parsed.active || '')) ? String(parsed.active) : 'Home'
    const rawSelected = isNseSymbol(String(parsed.selected || ''))
      ? String(parsed.selected).trim().toUpperCase()
      : ''
    const selected = pinnedSymbol(active, rawSelected)
    const compareSymbols = Array.isArray(parsed.compareSymbols)
      ? parsed.compareSymbols.filter((item) => isNseSymbol(String(item))).map((item) => String(item).toUpperCase()).slice(0, 5)
      : []
    const intelTab = parsed.intelTab ? String(parsed.intelTab) : undefined
    const ideasCategory = parsed.ideasCategory ? String(parsed.ideasCategory) : undefined
    const ideasLifecycle = parsed.ideasLifecycle === 'Closed' ? 'Closed' : parsed.ideasLifecycle === 'Active' ? 'Active' : undefined
    const scannerTab = parsed.scannerTab ? String(parsed.scannerTab) : undefined
    const homeLane = parsed.homeLane ? String(parsed.homeLane) : undefined
    if (active === 'Stock Intelligence' && !selected) {
      return {
        active: 'Recommendations',
        selected: '',
        compareSymbols,
        intelTab: undefined,
        ideasCategory,
        ideasLifecycle,
        scannerTab,
        homeLane,
        updatedAt: Number(parsed.updatedAt) || Date.now(),
      }
    }
    return {
      active,
      selected,
      compareSymbols,
      intelTab: active === 'Stock Intelligence' ? intelTab : undefined,
      ideasCategory,
      ideasLifecycle,
      scannerTab,
      homeLane,
      updatedAt: Number(parsed.updatedAt) || Date.now(),
    }
  } catch {
    return null
  }
}

export function patchDeskSession(partial: Partial<DeskSession>): void {
  const prev = loadDeskSession() || {
    active: 'Home',
    selected: '',
    compareSymbols: [],
    updatedAt: Date.now(),
  }
  const next: DeskSession = {
    ...prev,
    ...partial,
    compareSymbols: Array.isArray(partial.compareSymbols) ? partial.compareSymbols : prev.compareSymbols,
    selected: pinnedSymbol(
      String(partial.active ?? prev.active),
      String(partial.selected ?? prev.selected ?? ''),
    ),
    updatedAt: Date.now(),
  }
  try {
    storage()?.setItem(SESSION_KEY, JSON.stringify(next))
  } catch {
    evictCaches()
    try {
      storage()?.setItem(SESSION_KEY, JSON.stringify(next))
    } catch {
      /* quota still full — keep the desk usable */
    }
  }
}

export function cacheKey(name: string): string {
  return `${CACHE_PREFIX}${name}`
}

export function loadCachedJson<T>(name: string): T | null {
  const raw = storage()?.getItem(cacheKey(name))
  if (!raw) return null
  try {
    return JSON.parse(raw) as T
  } catch {
    return null
  }
}

export function saveCachedJson(name: string, value: unknown): boolean {
  let json: string
  try {
    json = JSON.stringify(value)
  } catch {
    return false
  }
  if (json.length > MAX_CACHE_CHARS) return false
  try {
    storage()?.setItem(cacheKey(name), json)
    return true
  } catch {
    evictCaches()
    try {
      storage()?.setItem(cacheKey(name), json)
      return true
    } catch {
      return false
    }
  }
}

export function stashDashboard(payload: Record<string, unknown>): boolean {
  const operations = (payload.operations || {}) as Record<string, unknown>
  return saveCachedJson('dashboard', {
    ...payload,
    operations: { ...operations, active: [] },
  })
}

export function evictCaches(): void {
  const store = storage()
  if (!store) return
  const keys: string[] = []
  for (let i = 0; i < store.length; i += 1) {
    const key = store.key(i)
    if (key && key.startsWith(CACHE_PREFIX)) keys.push(key)
  }
  keys.forEach((key) => store.removeItem(key))
}
