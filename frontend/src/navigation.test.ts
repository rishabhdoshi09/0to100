import { describe, expect, it } from 'vitest'

import {
  TOOL_GROUPS,
  WORKSPACE_NAV,
  canonicalNavRoute,
  isToolRoute,
  pageMeta,
} from './navigation'

describe('product navigation contract', () => {
  it('has exactly five operator workspaces', () => {
    expect(WORKSPACE_NAV.map(([, , label]) => label)).toEqual([
      'Today',
      'Opportunities',
      'Research',
      'Portfolio',
      'System',
    ])
  })

  it('keeps workspace routes unique and separate from secondary tools', () => {
    const workspaces = WORKSPACE_NAV.map(([, route]) => route)
    const tools = TOOL_GROUPS.flatMap((group) => group.items.map(([, route]) => route))

    expect(new Set(workspaces).size).toBe(workspaces.length)
    expect(new Set(tools).size).toBe(tools.length)
    expect(tools.filter((route) => workspaces.includes(route))).toEqual([])
  })

  it('normalizes both old aliases and the new workspace language', () => {
    expect(canonicalNavRoute('Today')).toBe('Home')
    expect(canonicalNavRoute('Opportunities')).toBe('Recommendations')
    expect(canonicalNavRoute('Research')).toBe('Learning')
    expect(canonicalNavRoute('Portfolio')).toBe('Paper Portfolio')
    expect(canonicalNavRoute('System')).toBe('System Health')
    expect(canonicalNavRoute('Scanner')).toBe('Market Scanner')
  })

  it('recognizes tools after alias normalization', () => {
    expect(isToolRoute('Scanner')).toBe(true)
    expect(isToolRoute('Why')).toBe(true)
    expect(isToolRoute('Today')).toBe(false)
  })

  it('uses the same language for aliases and canonical routes', () => {
    expect(pageMeta('Today')).toEqual(pageMeta('Home'))
    expect(pageMeta('Research').title).toBe('Research')
    expect(pageMeta('System').title).toBe('System')
  })
})
