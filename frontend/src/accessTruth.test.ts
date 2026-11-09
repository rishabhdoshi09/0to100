import { describe, expect, it } from 'vitest'

import { UNVERIFIED_ACCESS, accessPill, verifiedAccess } from './accessTruth'

describe('operator access truth (9 Nov P0: PUBLIC READ ONLY on a private host)', () => {
  const privateHost = {
    public_read_only: false, mutation_policy: 'PRIVATE_OPERATOR', operator_token_configured: false,
    unsafe_methods: ['POST'], live_money_unlocked: false,
  }
  it('a failed access check is unverified and fail-closed, not "public read only"', () => {
    expect(UNVERIFIED_ACCESS.public_read_only).toBe(true)                 // actions stay disabled
    expect(accessPill(UNVERIFIED_ACCESS)?.label).toBe('ACCESS UNVERIFIED')
  })
  it('re-verification from a health probe replaces the unverified placeholder', () => {
    expect(verifiedAccess(privateHost)?.public_read_only).toBe(false)
    expect(accessPill(verifiedAccess(privateHost))).toBeNull()
  })
  it('only a verified backend answer may change access', () => {
    expect(verifiedAccess(null)).toBeNull()
    expect(verifiedAccess({ public_read_only: null, mutation_policy: 'UNVERIFIED' })).toBeNull()
    expect(verifiedAccess({ public_read_only: false })).toBeNull()
    expect(verifiedAccess({ ...privateHost, live_money_unlocked: 'yes' })?.live_money_unlocked).toBe(false)
  })
  it('a verified public deployment is still labelled PUBLIC READ ONLY', () => {
    const pub = verifiedAccess({ ...privateHost, public_read_only: true, mutation_policy: 'READ_ONLY' })
    expect(accessPill(pub)?.label).toBe('PUBLIC READ ONLY')
  })
})
