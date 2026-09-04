import { describe, expect, it, vi } from 'vitest'
import { API_DOWN_MESSAGE, REQUEST_TIMEOUT_MESSAGE, fetchJson, readJson, withTimeout } from './http'

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

describe('withTimeout', () => {
  it('aborts after the requested bound', async () => {
    const signal = withTimeout(15)
    await new Promise((resolve) => setTimeout(resolve, 30))
    expect(signal.aborted).toBe(true)
  })
})

describe('fetchJson', () => {
  it('maps AbortError to a bounded timeout message', async () => {
    vi.stubGlobal('fetch', () => Promise.reject(Object.assign(new Error('aborted'), { name: 'AbortError' })))
    await expect(fetchJson('/api/health', { timeoutMs: 5 })).rejects.toThrow(REQUEST_TIMEOUT_MESSAGE)
    vi.unstubAllGlobals()
  })

  it('passes a timeout signal to fetch', async () => {
    const seen: AbortSignal[] = []
    vi.stubGlobal('fetch', (_input: RequestInfo, init?: RequestInit) => {
      if (init?.signal) seen.push(init.signal)
      return Promise.resolve(new Response('{"ok":true}', { status: 200 }))
    })
    await expect(fetchJson<{ ok: boolean }>('/api/health', { timeoutMs: 50 })).resolves.toEqual({ ok: true })
    expect(seen).toHaveLength(1)
    expect(seen[0].aborted).toBe(false)
    vi.unstubAllGlobals()
  })

  it('releases the timeout as soon as a successful request finishes', async () => {
    const clearSpy = vi.spyOn(globalThis, 'clearTimeout')
    vi.stubGlobal('fetch', () => Promise.resolve(new Response('{"ok":true}', { status: 200 })))

    await expect(fetchJson<{ ok: boolean }>('/api/health', { timeoutMs: 30_000, dedupe: false })).resolves.toEqual({ ok: true })

    expect(clearSpy).toHaveBeenCalled()
    clearSpy.mockRestore()
    vi.unstubAllGlobals()
  })
})
