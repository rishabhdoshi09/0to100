import { describe, expect, it } from 'vitest'
import { API_DOWN_MESSAGE, readJson } from './http'

describe('readJson', () => {
  it('maps an empty Vite proxy 500 to a start-the-stack message', async () => {
    await expect(readJson(new Response('', { status: 500 }))).rejects.toThrow(API_DOWN_MESSAGE)
  })

  it('keeps a FastAPI error body', async () => {
    await expect(readJson(new Response('{"detail":"boom"}', { status: 500 }))).rejects.toThrow('{"detail":"boom"}')
  })

  it('returns JSON on 200', async () => {
    await expect(readJson<{ ok: boolean }>(new Response('{"ok":true}', { status: 200 }))).resolves.toEqual({ ok: true })
  })
})
