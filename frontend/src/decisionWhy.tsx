import type { ReactNode } from 'react'
import { EmptyState } from './designSystem'

/**
 * WHY THIS DECISION.
 *
 * Rendered straight from the deterministic decision record the API returns.
 * Nothing on this screen is generated prose: if a section has no content the
 * screen says so, because "we could not obtain the fundamentals" is one of the
 * more useful things it can tell a trader, and a screen that hides empty
 * sections hides exactly that.
 */

import {
  fieldLabel,
  gapSummary,
  isRenderable,
  presentFields,
  renderValue,
  sectionTone,
  visibleSections,
} from './decisionWhyModel'
import type { DecisionSection, DecisionWhy } from './decisionWhyModel'

export type { DecisionEvidenceItem, DecisionSection, DecisionWhy } from './decisionWhyModel'


function SectionBody({ section }: { section: DecisionSection }): ReactNode {
  if (!section.available) {
    return <p className="decision-why__unavailable">{section.text || 'Not available'}</p>
  }
  if (section.items.length > 0) {
    return (
      <ul className="decision-why__items">
        {section.items.map((item) => (
          <li key={item.id} className="decision-why__item">
            <span className="decision-why__item-label">{item.label}</span>
            {item.detail ? <span className="decision-why__item-detail">{item.detail}</span> : null}
            <span className="decision-why__item-meta">
              {item.source ? <span title="source">{item.source}</span> : null}
              {item.as_of ? <span title="as of">{item.as_of}</span> : null}
              {item.evidence_class ? (
                <span className="decision-why__class" title="evidence class">
                  {item.evidence_class}
                </span>
              ) : null}
            </span>
          </li>
        ))}
      </ul>
    )
  }
  const fieldKeys = presentFields(section.fields)
  if (fieldKeys.length > 0) {
    return (
      <dl className="decision-why__fields">
        {fieldKeys.map((key) => (
          <div key={key} className="decision-why__field">
            <dt>{fieldLabel(key)}</dt>
            <dd>{renderValue(section.fields[key])}</dd>
          </div>
        ))}
      </dl>
    )
  }
  return <p className="decision-why__value">{renderValue(section.value)}</p>
}


export function DecisionWhyPanel({ why }: { why: DecisionWhy | null }): ReactNode {
  if (!why) {
    return <EmptyState title="Why this decision" detail="Nothing loaded yet." />
  }
  if (!isRenderable(why)) {
    return (
      <EmptyState
        title={`Why ${why.symbol || 'this decision'}`}
        detail={why.reason || 'The desk has not decided anything about this name.'}
      />
    )
  }

  const sections = visibleSections(why)
  const gaps = gapSummary(why)

  return (
    <section className="decision-why">
      <header className="decision-why__header">
        <h2>Why this decision</h2>
        <p className="decision-why__headline">{why.headline}</p>
        {why.ranking_explanation ? (
          <p className="decision-why__ranking">{why.ranking_explanation}</p>
        ) : null}
        {why.scan_scanned_at ? (
          <p className="decision-why__scan">From the scan of {why.scan_scanned_at}</p>
        ) : null}
      </header>

      {gaps ? <p className="decision-why__gaps">{gaps}</p> : null}

      <div className="decision-why__sections">
        {sections.map((section) => (
          <article
            key={section.title}
            className={`decision-why__section ${sectionTone(section)}`.trim()}
          >
            <h3>{section.title}</h3>
            <SectionBody section={section} />
          </article>
        ))}
      </div>

      <footer className="decision-why__footer">
        <p>{why.note}</p>
        <p className="decision-why__authority">{why.authority}</p>
      </footer>
    </section>
  )
}
