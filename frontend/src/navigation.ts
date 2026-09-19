export type NavRow = readonly [icon: string, route: string, label: string]

export type NavGroup = {
  label: string
  items: readonly NavRow[]
}

/**
 * Product navigation has five operator workspaces.
 *
 * Internal route names intentionally stay stable while the UI uses clearer
 * workspace labels. This lets us simplify the product without breaking saved
 * navigation state, deep links, or existing view components.
 */
export const WORKSPACE_NAV: readonly NavRow[] = [
  ['⌂', 'Home', 'Today'],
  ['▣', 'Recommendations', 'Opportunities'],
  ['✎', 'Learning', 'Research'],
  ['▣', 'Paper Portfolio', 'Portfolio'],
  ['⚙', 'System Health', 'System'],
] as const

/**
 * Secondary tools remain available, but they are not peers of the five
 * workspaces. A tool answers a focused question inside a workspace.
 */
export const TOOL_GROUPS: readonly NavGroup[] = [
  {
    label: 'Market tools',
    items: [
      ['◎', 'Market Scanner', 'Scanner'],
      ['◉', 'Stock Intelligence', 'Company'],
      ['▤', 'Market Reports', 'Reports'],
      ['★', 'Watchlist', 'Watchlist'],
      ['⇔', 'Compare', 'Compare'],
    ],
  },
  {
    label: 'Research tools',
    items: [
      ['?', 'Why This Decision', 'Decision'],
      ['∑', 'Forward Evidence', 'Evidence'],
      ['⌬', 'Strategies', 'Strategies'],
      ['🧪', 'Backtest', 'Backtest'],
      ['▤', 'Research Data', 'Data'],
      ['◎', 'Coverage', 'Coverage'],
    ],
  },
] as const

/**
 * Historical route names are normalized at the navigation boundary only.
 * Business code must never depend on display labels.
 */
export const ROUTE_ALIAS: Readonly<Record<string, string>> = {
  'Command Center': 'Home',
  Market: 'Home',
  Today: 'Home',
  Desk: 'Home',

  Opportunities: 'Recommendations',

  Research: 'Learning',

  Portfolio: 'Paper Portfolio',

  System: 'System Health',
  Automation: 'System Health',
  Health: 'System Health',

  Scanner: 'Market Scanner',
  Setups: 'Market Scanner',
  Reports: 'Market Reports',
  'Long-Term': 'Long-Term Picks',
  'Market Internals': 'Market Overview',

  'Stock Investigator': 'Stock Intelligence',
  'Company Intelligence': 'Stock Intelligence',

  Backtests: 'Backtest',
  Data: 'Research Data',
  Why: 'Why This Decision',
  Evidence: 'Forward Evidence',
}

export function canonicalNavRoute(route: string): string {
  return ROUTE_ALIAS[route] || route
}

const TOOL_ROUTES = new Set(
  TOOL_GROUPS.flatMap((group) => group.items.map(([, route]) => route)),
)

export function isToolRoute(route: string): boolean {
  return TOOL_ROUTES.has(canonicalNavRoute(route))
}
