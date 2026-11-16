/**
 * Terminal command line: `<TICKER> <FUNCTION>` or `<FUNCTION>`, Enter to go.
 *
 * Professionals drive a terminal from the keyboard: type a security and a
 * function mnemonic, never hunt through menus. `/` focuses the line from
 * anywhere on the desk; Escape leaves it. A bare symbol keeps the existing
 * behaviour and opens Stock Intelligence.
 */

export type CommandFunction = {
  code: string
  page: string
  needsSymbol: boolean
  label: string
}

/** Mnemonics; several follow the conventions terminal users already know. */
export const COMMAND_FUNCTIONS: CommandFunction[] = [
  { code: 'DES', page: 'Stock Intelligence', needsSymbol: true, label: 'Security overview, chart and evidence' },
  { code: 'GP', page: 'Stock Intelligence', needsSymbol: true, label: 'Price chart (same workspace as DES)' },
  { code: 'DD', page: 'Stock Investigator', needsSymbol: true, label: 'Due diligence / fundamentals' },
  { code: 'FA', page: 'Stock Investigator', needsSymbol: true, label: 'Financial analysis (due diligence)' },
  { code: 'WHY', page: 'Why This Decision', needsSymbol: true, label: 'Why the desk decided what it did' },
  { code: 'DATA', page: 'Research Data', needsSymbol: true, label: 'Raw research data' },
  { code: 'CMP', page: 'Compare', needsSymbol: false, label: 'Compare securities' },
  { code: 'HOME', page: 'Home', needsSymbol: false, label: 'Home: BUY only, market desk' },
  { code: 'SCAN', page: 'Market Scanner', needsSymbol: false, label: 'Whole-market scanner' },
  { code: 'RECO', page: 'Recommendations', needsSymbol: false, label: 'Recommendations' },
  { code: 'N', page: 'News & Events', needsSymbol: false, label: 'News and events' },
  { code: 'NEWS', page: 'News & Events', needsSymbol: false, label: 'News and events' },
  { code: 'RPT', page: 'Market Reports', needsSymbol: false, label: 'Daily market reports' },
  { code: 'MKT', page: 'Market Overview', needsSymbol: false, label: 'Market overview / internals' },
  { code: 'LT', page: 'Long-Term Picks', needsSymbol: false, label: 'Long-term picks' },
  { code: 'FNO', page: 'F&O Desk', needsSymbol: false, label: 'Futures & options desk' },
  { code: 'PORT', page: 'Paper Portfolio', needsSymbol: false, label: 'Paper portfolio' },
  { code: 'W', page: 'Watchlist', needsSymbol: false, label: 'Watchlist' },
  { code: 'EVID', page: 'Forward Evidence', needsSymbol: false, label: 'Forward evidence' },
  { code: 'BT', page: 'Backtest', needsSymbol: false, label: 'Backtests' },
  { code: 'EVO', page: 'Evolution Lab', needsSymbol: false, label: 'Strategy evolution lab' },
  { code: 'LEARN', page: 'Learning', needsSymbol: false, label: 'Learning journal' },
  { code: 'STRAT', page: 'Strategies', needsSymbol: false, label: 'Strategy catalog' },
  { code: 'COV', page: 'Coverage', needsSymbol: false, label: 'Scan coverage' },
  { code: 'SYS', page: 'System Health', needsSymbol: false, label: 'System health / automation' },
  { code: 'HELP', page: 'Education', needsSymbol: false, label: 'How the desk works' },
]

const BY_CODE = new Map(COMMAND_FUNCTIONS.map((fn) => [fn.code, fn]))
const SYMBOL = /^[A-Z0-9&.-]{1,32}$/

export type ParsedCommand =
  | { kind: 'page'; page: string; symbol: string; code: string }
  | { kind: 'symbol'; symbol: string }
  | { kind: 'error'; message: string }

/** Resolve a typed symbol against the desk's known symbols (exact, then prefix). */
export function resolveSymbol(raw: string, known: readonly string[]): string {
  const clean = raw.toUpperCase()
  return known.find((s) => s === clean) || known.find((s) => s.startsWith(clean)) || clean
}

export function parseCommand(input: string, known: readonly string[] = []): ParsedCommand {
  const tokens = input.trim().toUpperCase().split(/\s+/).filter(Boolean)
  if (!tokens.length) return { kind: 'error', message: 'Type a symbol or a function, e.g. RELIANCE DD or FNO' }
  const [first, second] = tokens
  if (tokens.length > 2) return { kind: 'error', message: 'Use <SYMBOL> <FUNCTION>, e.g. TCS WHY' }

  if (tokens.length === 1) {
    const fn = BY_CODE.get(first)
    if (fn && !fn.needsSymbol) return { kind: 'page', page: fn.page, symbol: '', code: fn.code }
    if (fn && fn.needsSymbol) return { kind: 'error', message: `${fn.code} needs a symbol first, e.g. RELIANCE ${fn.code}` }
    if (!SYMBOL.test(first)) return { kind: 'error', message: 'Use a valid NSE symbol such as RELIANCE, TCS or HDFCBANK' }
    return { kind: 'symbol', symbol: resolveSymbol(first, known) }
  }

  const fn = BY_CODE.get(second)
  if (!fn) return { kind: 'error', message: `Unknown function ${second}. Try DES, DD, WHY, DATA or HELP.` }
  if (!SYMBOL.test(first)) return { kind: 'error', message: `${first} is not a valid NSE symbol` }
  return { kind: 'page', page: fn.page, symbol: resolveSymbol(first, known), code: fn.code }
}

/** True when `/` should focus the command line (not while typing elsewhere). */
export function shouldFocusCommandLine(key: string, target: EventTarget | null): boolean {
  if (key !== '/') return false
  const el = target as HTMLElement | null
  const tag = (el?.tagName || '').toUpperCase()
  return !(tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el?.isContentEditable)
}
