/** Phone layout breakpoint — matches the hamburger shell in styles.css. */
export const PHONE_MAX_WIDTH_PX = 820

export function isPhoneLayout(width = defaultViewportWidth()): boolean {
  return width <= PHONE_MAX_WIDTH_PX
}

export function chartHeightForWidth(width: number): number {
  return isPhoneLayout(width) ? 200 : 360
}

export function thesisSheetClassName(hasClose: boolean): string {
  return ['reco-sheet', 'thesis-sheet', hasClose ? 'has-close' : ''].filter(Boolean).join(' ')
}

function defaultViewportWidth(): number {
  if (typeof window === 'undefined') return 1280
  return window.innerWidth
}
