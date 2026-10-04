import { describe, expect, it } from 'vitest'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { DecisionWhyPanel } from './decisionWhy'

describe('committee and research decision transparency', () => {
  it('shows a current WAIT and execution blocker beside the research BUY assessment', () => {
    const html = renderToStaticMarkup(createElement(DecisionWhyPanel, { why: {
      available: true, symbol: 'AAA', state: 'BUY', headline: 'AAA: BUY on VCP',
      committee: { decision: 'WAIT', reason: 'WAIT_EVIDENCE', execution_state: 'NOT_ELIGIBLE' },
      sections: [],
    } }))
    expect(html).toContain('Current committee: WAIT')
    expect(html).toContain('WAIT_EVIDENCE')
    expect(html).toContain('execution NOT_ELIGIBLE')
    expect(html).toContain('Research assessment: AAA: BUY on VCP')
  })
})
