import { describe, expect, it } from 'vitest'
import { deskDeliveryCopy } from './deskDelivery'

describe('desk delivery copy', () => {
  it('never requires Telegram', () => {
    expect(deskDeliveryCopy({})).toContain('no Telegram required')
    expect(deskDeliveryCopy({ configured: true, listener_running: true, last_send_ok: true }))
      .toContain('optional extra')
  })

  it('says the desk still works when the bot is unreachable', () => {
    const line = deskDeliveryCopy({
      configured: true,
      listener_running: true,
      bot_reachable: false,
    })
    expect(line).toContain('unreachable')
    expect(line).toContain('still works')
  })

  it('says the desk still works when Telegram send failed', () => {
    const line = deskDeliveryCopy({
      configured: true,
      listener_running: true,
      last_send_ok: false,
      last_error: 'unauthorized — token or chat id rejected',
    })
    expect(line).toContain('still works')
    expect(line).toContain('unauthorized')
  })
})
