/**
 * Presentation logic for WHY THIS DECISION, kept out of the component so it
 * can be tested the way the rest of this codebase tests its logic.
 *
 * The rule these functions encode: a section with nothing in it is still shown,
 * and shown as missing. The screen's job is to make an evidence gap visible,
 * not to tidy it away.
 */

export type DecisionEvidenceItem = {
  id: string
  label: string
  detail: string
  value: unknown
  source: string
  as_of: string
  evidence_class: string
}

export type DecisionSection = {
  title: string
  available: boolean
  value: unknown
  text: string
  items: DecisionEvidenceItem[]
  fields: Record<string, unknown>
}

export type DecisionWhy = {
  schema_version?: number
  available: boolean
  symbol: string
  reason?: string
  state?: string
  headline?: string
  authority?: string
  note?: string
  sections?: DecisionSection[]
  ranking?: {
    base_score: number
    evidence_adjustment: number
    ranking_score: number
    evidence: Record<string, unknown>
  }
  ranking_explanation?: string
  unfilled_sections?: string[]
  scan_scanned_at?: string
  versions?: Record<string, unknown>
}

export const CONFLICT_TITLES = new Set(['Conflicting evidence'])
export const GAP_TITLES = new Set(['Missing evidence'])

export const UNAVAILABLE_TEXT = 'Not available'

export function renderValue(value: unknown): string {
  if (value === null || value === undefined || value === '') return '—'
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return '—'
    return Number.isInteger(value) ? String(value) : value.toFixed(2)
  }
  if (typeof value === 'boolean') return value ? 'yes' : 'no'
  return String(value)
}

export function sectionTone(section: DecisionSection): string {
  if (!section.available) return 'decision-why__section--unavailable'
  if (CONFLICT_TITLES.has(section.title)) return 'decision-why__section--conflict'
  if (GAP_TITLES.has(section.title)) return 'decision-why__section--gap'
  return ''
}

/** Field keys worth rendering: a null is a gap, not a zero. */
export function presentFields(fields: Record<string, unknown> | undefined): string[] {
  if (!fields) return []
  return Object.keys(fields).filter(
    (key) => fields[key] !== null && fields[key] !== undefined && fields[key] !== '',
  )
}

export function fieldLabel(key: string): string {
  return key.replace(/_/g, ' ')
}

/** One line naming what the desk could not fill, or empty when it filled everything. */
export function gapSummary(why: DecisionWhy | null): string {
  const gaps = why?.unfilled_sections || []
  if (gaps.length === 0) return ''
  const noun = gaps.length === 1 ? 'section' : 'sections'
  return `${gaps.length} ${noun} the desk could not fill: ${gaps.join(', ')}`
}

/** Every section is shown, in the order the server sent them. */
export function visibleSections(why: DecisionWhy | null): DecisionSection[] {
  return why?.sections || []
}

export function isRenderable(why: DecisionWhy | null): boolean {
  return Boolean(why && why.available)
}
