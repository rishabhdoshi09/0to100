import type { AccessMode } from './api'

/** Fail-closed placeholder used only while access cannot be proven.
 *
 * Mutations stay disabled (public_read_only=true), but the desk must not claim
 * that this deployment IS public read-only: that is a verified fact only when
 * the backend says so. One slow /api/access request at startup used to set
 * "PUBLIC READ ONLY" for the whole browser session on a private host.
 */
export const UNVERIFIED_ACCESS: AccessMode = {
  public_read_only: true,
  mutation_policy: 'UNVERIFIED',
  operator_token_configured: false,
  unsafe_methods: ['POST', 'PUT', 'PATCH', 'DELETE'],
  live_money_unlocked: false,
}

/** Accept a backend access projection only when it is a verified answer. */
export function verifiedAccess(raw: unknown): AccessMode | null {
  if (!raw || typeof raw !== 'object') return null
  const row = raw as Partial<AccessMode>
  const policy = String(row.mutation_policy || '').toUpperCase()
  if (typeof row.public_read_only !== 'boolean' || !policy || policy === 'UNVERIFIED') return null
  return {
    public_read_only: row.public_read_only,
    mutation_policy: String(row.mutation_policy),
    operator_token_configured: Boolean(row.operator_token_configured),
    unsafe_methods: Array.isArray(row.unsafe_methods) ? row.unsafe_methods.map(String) : UNVERIFIED_ACCESS.unsafe_methods,
    live_money_unlocked: row.live_money_unlocked === true,
  }
}

export type AccessPill = { label: string; title: string }

export function accessPill(mode: AccessMode | null): AccessPill | null {
  if (!mode) return null
  if (String(mode.mutation_policy || '').toUpperCase() === 'UNVERIFIED') {
    return {
      label: 'ACCESS UNVERIFIED',
      title: 'Operator access could not be verified yet. Actions stay disabled; QuantTerm re-checks with every health probe.',
    }
  }
  if (mode.public_read_only) {
    return {
      label: 'PUBLIC READ ONLY',
      title: 'Public visitor mode: market/research reads are available, state-changing operator actions are disabled.',
    }
  }
  return null
}
