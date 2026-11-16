import { describe, expect, it } from 'vitest'
import { COMMAND_FUNCTIONS, parseCommand, resolveSymbol, shouldFocusCommandLine } from './commandLine'

const KNOWN = ['RELIANCE', 'RELINFRA', 'TCS', 'HDFCBANK']

describe('terminal command line', () => {
  it('a bare symbol opens the stock, resolving prefixes like before', () => {
    expect(parseCommand('reliance', KNOWN)).toEqual({ kind: 'symbol', symbol: 'RELIANCE' })
    expect(parseCommand('RELI', KNOWN)).toEqual({ kind: 'symbol', symbol: 'RELIANCE' })
    expect(parseCommand('NEWCO', KNOWN)).toEqual({ kind: 'symbol', symbol: 'NEWCO' })
  })

  it('<SYMBOL> <FUNCTION> routes to the function page with the symbol selected', () => {
    expect(parseCommand('tcs why', KNOWN)).toEqual({ kind: 'page', page: 'Why This Decision', symbol: 'TCS', code: 'WHY' })
    expect(parseCommand('  RELIANCE   DD ', KNOWN)).toMatchObject({ page: 'Stock Investigator', symbol: 'RELIANCE' })
    expect(parseCommand('HDFCBANK GP', KNOWN)).toMatchObject({ page: 'Stock Intelligence' })
  })

  it('a page code alone navigates; symbol functions ask for a symbol', () => {
    expect(parseCommand('fno', KNOWN)).toEqual({ kind: 'page', page: 'F&O Desk', symbol: '', code: 'FNO' })
    expect(parseCommand('PORT', KNOWN)).toMatchObject({ page: 'Paper Portfolio' })
    expect(parseCommand('DD', KNOWN)).toMatchObject({ kind: 'error' })
  })

  it('rejects junk with a helpful message instead of guessing', () => {
    expect(parseCommand('', KNOWN)).toMatchObject({ kind: 'error' })
    expect(parseCommand('TCS XYZ', KNOWN)).toMatchObject({ kind: 'error' })
    expect(parseCommand('a b c', KNOWN)).toMatchObject({ kind: 'error' })
    expect(parseCommand('rel$', KNOWN)).toMatchObject({ kind: 'error' })
  })

  it('every function points at a real desk page', () => {
    const pages = new Set([
      'Home', 'Market Scanner', 'Recommendations', 'Market Reports', 'Stock Intelligence', 'Stock Investigator',
      'Long-Term Picks', 'Compare', 'Watchlist', 'Market Overview', 'News & Events', 'Education',
      'Why This Decision', 'Forward Evidence', 'Evolution Lab', 'Research Data', 'Strategies', 'Backtest',
      'Learning', 'Coverage', 'F&O Desk', 'Paper Portfolio', 'System Health',
    ])
    for (const fn of COMMAND_FUNCTIONS) expect(pages.has(fn.page), fn.code).toBe(true)
    expect(new Set(COMMAND_FUNCTIONS.map((fn) => fn.code)).size).toBe(COMMAND_FUNCTIONS.length)
  })

  it('`/` focuses the line only when not already typing', () => {
    expect(shouldFocusCommandLine('/', { tagName: 'DIV' } as unknown as EventTarget)).toBe(true)
    expect(shouldFocusCommandLine('/', { tagName: 'INPUT' } as unknown as EventTarget)).toBe(false)
    expect(shouldFocusCommandLine('/', { tagName: 'TEXTAREA' } as unknown as EventTarget)).toBe(false)
    expect(shouldFocusCommandLine('a', null)).toBe(false)
    expect(resolveSymbol('tc', KNOWN)).toBe('TCS')
  })
})
