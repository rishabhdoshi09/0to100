import { useEffect, useState } from 'react'

/** Phone layout breakpoint — matches the hamburger shell in styles.css. */
export const PHONE_MAX_WIDTH_PX = 820

export function isPhoneLayout(width?: number): boolean {
  if (width != null) return width <= PHONE_MAX_WIDTH_PX
  if (typeof window === 'undefined') return false
  return window.matchMedia(`(max-width: ${PHONE_MAX_WIDTH_PX}px)`).matches
}

export function usePhoneLayout(): boolean {
  const [phone, setPhone] = useState(() => isPhoneLayout())
  useEffect(() => {
    const media = window.matchMedia(`(max-width: ${PHONE_MAX_WIDTH_PX}px)`)
    const sync = () => setPhone(media.matches)
    sync()
    media.addEventListener('change', sync)
    return () => media.removeEventListener('change', sync)
  }, [])
  return phone
}

export function chartHeightForWidth(width: number): number {
  return isPhoneLayout(width) ? 200 : 360
}

export function thesisSheetClassName(hasClose: boolean): string {
  return ['reco-sheet', 'thesis-sheet', hasClose ? 'has-close' : ''].filter(Boolean).join(' ')
}

export function shouldPortalThesis(hasClose: boolean, phone: boolean): boolean {
  return hasClose && phone
}
