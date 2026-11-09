/** Display truth for the US virtual-paper lane on the Home card.
 *
 * The backend (`execution/us_autopilot.py` via `product/us_market_status.py`)
 * is the only authority. This helper only words it: an unrestored or
 * unreadable paper state is a BLOCKER the operator must act on, never a
 * routine "paper auto OFF".
 */
export type UsPaperState = {
  armed?: boolean
  disarmed_reason?: string | null
  state_recovery_required?: boolean
  state_recovery_error?: string | null
  accounting_reconciliation_required?: boolean
  new_entries_paused?: boolean
  entries_paused_reason?: string | null
}

export function usPaperModeLabel(paper: UsPaperState | undefined): string {
  if (!paper) return ' · paper status unknown'
  if (paper.state_recovery_required) return ' · paper BLOCKED: saved state must be restored'
  if (paper.new_entries_paused) return ' · paper entries paused (India-only); open positions still supervised'
  if (paper.armed === true) return ' · paper auto ON'
  const reason = String(paper.disarmed_reason || '').trim()
  return reason && reason !== 'user' ? ` · paper auto OFF (${reason})` : ' · paper auto OFF'
}

export function usPaperBlockers(paper: UsPaperState | undefined): string[] {
  if (!paper) return []
  const rows: string[] = []
  if (paper.state_recovery_required) {
    rows.push(
      'US paper is blocked: its saved state is missing or unreadable while trade history exists. '
      + 'No entries, exits or P&L settlement run until the saved state is restored and the stack restarted.'
      + (paper.state_recovery_error ? ` (${paper.state_recovery_error})` : ''),
    )
  } else if (paper.accounting_reconciliation_required) {
    rows.push(
      'US paper realized P&L was carried over from a legacy watermark and may omit trades that closed out of order. '
      + 'Reconcile it against the paper journal before relying on the pool figure.',
    )
  }
  return rows
}
