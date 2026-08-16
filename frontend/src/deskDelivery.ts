export type TelegramDelivery = {
  configured?: boolean
  listener_running?: boolean
  last_send_ok?: boolean | null
  last_error?: string
  note?: string
}

export function deskDeliveryCopy(telegram?: TelegramDelivery | null): string {
  const tg = telegram || {}
  if (tg.configured && tg.last_send_ok === false) {
    return `Telegram last send failed (${tg.last_error || 'unknown'}). The desk still works in this browser.`
  }
  if (tg.configured && !tg.listener_running) {
    return 'Telegram is set but incoming chats are silent until the API listener starts. The desk still works in this browser.'
  }
  if (!tg.configured) {
    return 'Telegram is not set. Open this URL on any phone browser — no Telegram required.'
  }
  return 'This page is the desk. Telegram is optional extra — not everyone has it.'
}
