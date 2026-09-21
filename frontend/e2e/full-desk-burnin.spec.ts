import { expect, test, type Locator } from '@playwright/test'

const WORKSPACES = [
  ['Today', 'Today'],
  ['Opportunities', 'Opportunities'],
  ['Research', 'Research'],
  ['Portfolio', 'Portfolio'],
  ['System', 'System'],
] as const

const TOOLS = [
  ['Scanner', 'Market Scanner'],
  ['Company', 'Company Intelligence'],
  ['Reports', 'Market Reports'],
  ['Watchlist', 'Watchlist'],
  ['Compare', 'Compare'],
  ['Decision', 'Why This Decision'],
  ['Evidence', 'Forward Evidence'],
  ['Strategies', 'Strategies'],
  ['Backtest', 'Backtest'],
  ['Data', 'Research Data'],
  ['Coverage', 'Coverage'],
] as const

const SECONDARY_VIEWS = [
  ['Market Overview', 'Market Overview'],
  ['News & Events', 'News & Events'],
  ['Education', 'Education'],
  ['F&O Desk', 'F&O Desk'],
  ['Long-Term Picks', 'Long-Term Picks'],
  ['Stock Investigator', 'Company Intelligence'],
] as const

const CRITICAL_READS = [
  '/api/health',
  '/api/dashboard',
  '/api/radar-home',
  '/api/decision-simulation-gate',
  '/api/paper-autopilot',
  '/api/research-status',
  '/api/forward-evidence',
  '/api/product-contract',
] as const

async function clickPrimaryNavButton(nav: Locator, name: string) {
  const button = nav.getByRole('button', { name, exact: true })
  await expect(button).toBeVisible()
  // The real sidebar is independently scrollable. Chromium can repeatedly
  // report a deeply clipped button as "outside of the viewport" even after
  // scrollIntoView(). Trigger the actual DOM button click after proving the
  // control is rendered, visible and enabled; this exercises the same React
  // onClick handler without weakening any workspace assertion below.
  await expect(button).toBeEnabled()
  await button.evaluate((element: HTMLButtonElement) => {
    element.scrollIntoView({ block: 'center', inline: 'nearest' })
    element.click()
  })
}

test('10-hour accelerated full-desk burn-in keeps every visible tab and backend surface alive', async ({ page, request }) => {
  const rounds = Math.max(1, Number(process.env.QT_BURNIN_ROUNDS || '10'))
  const pageErrors: string[] = []
  const serverErrors: string[] = []

  page.on('pageerror', (error) => pageErrors.push(error.message))
  page.on('response', (response) => {
    if (response.url().includes('/api/') && response.status() >= 500) {
      serverErrors.push(`${response.status()} ${response.request().method()} ${response.url()}`)
    }
  })

  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Today', level: 1, exact: true })).toBeVisible()
  const nav = page.getByRole('navigation', { name: 'Primary navigation' })

  for (let hour = 1; hour <= rounds; hour += 1) {
    for (const endpoint of CRITICAL_READS) {
      const response = await request.get(endpoint, { timeout: 30_000 })
      expect(response.status(), `virtual hour ${hour}: ${endpoint}`).toBeLessThan(500)
      expect(response.ok(), `virtual hour ${hour}: ${endpoint}`).toBeTruthy()
    }

    for (const [button, title] of WORKSPACES) {
      await clickPrimaryNavButton(nav, button)
      await expect(page.getByRole('heading', { name: title, level: 1, exact: true })).toBeVisible()
      await expect(page.locator('.workspace')).toBeVisible()
      await page.waitForTimeout(100)
    }

    const tools = page.locator('details.nav-advanced')
    if (!(await tools.evaluate((element) => (element as HTMLDetailsElement).open))) {
      await tools.locator('summary').click()
    }
    for (const [button, title] of TOOLS) {
      await clickPrimaryNavButton(nav, button)
      await expect(page.getByRole('heading', { name: title, level: 1, exact: true })).toBeVisible()
      await expect(page.locator('.workspace')).toBeVisible()
      await page.waitForTimeout(100)
    }

    // These views still exist in App.tsx even though the simplified daily
    // sidebar no longer exposes them as peer tabs. Force-load each route from
    // the same persisted navigation state App itself consumes.
    for (const [route, title] of SECONDARY_VIEWS) {
      await page.evaluate((nextRoute) => {
        const current = JSON.parse(window.sessionStorage.getItem('quantterm-nav') || '{}')
        window.sessionStorage.setItem('quantterm-nav', JSON.stringify({
          active: nextRoute,
          selected: current.selected || '',
          compare: Array.isArray(current.compare) ? current.compare : [],
        }))
      }, route)
      await page.reload()
      await expect(page.getByRole('heading', { name: title, level: 1, exact: true })).toBeVisible()
      await expect(page.locator('.workspace')).toBeVisible()
      await page.waitForTimeout(100)
    }

    // Prove a real frontend click makes the backend answer, not just that a
    // component can render cached state.
    const dashboardResponse = page.waitForResponse(
      (response) => response.url().includes('/api/dashboard') && response.request().method() === 'GET',
      { timeout: 30_000 },
    )
    await page.getByRole('button', { name: 'Refresh dashboard' }).click()
    expect((await dashboardResponse).status()).toBeLessThan(500)

    await clickPrimaryNavButton(nav, 'Today')
    await expect(page.getByRole('heading', { name: 'Today', level: 1, exact: true })).toBeVisible()

    const cards = page.locator('.home-os-best-trades > div')
    expect(await cards.count()).toBeLessThanOrEqual(5)
  }

  expect(pageErrors, 'uncaught browser exceptions').toEqual([])
  expect(serverErrors, 'backend 5xx responses observed by the real frontend').toEqual([])
})
