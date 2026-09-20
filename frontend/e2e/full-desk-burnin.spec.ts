import { expect, test } from '@playwright/test'

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
  await expect(page.locator('h1')).toHaveText('Today')

  for (let hour = 1; hour <= rounds; hour += 1) {
    for (const endpoint of CRITICAL_READS) {
      const response = await request.get(endpoint, { timeout: 30_000 })
      expect(response.status(), `virtual hour ${hour}: ${endpoint}`).toBeLessThan(500)
      expect(response.ok(), `virtual hour ${hour}: ${endpoint}`).toBeTruthy()
    }

    for (const [button, title] of WORKSPACES) {
      await page.getByRole('button', { name: button, exact: true }).click()
      await expect(page.locator('h1')).toHaveText(title)
      await expect(page.locator('.workspace')).toBeVisible()
      await page.waitForTimeout(100)
    }

    const tools = page.locator('details.nav-advanced')
    if (!(await tools.getAttribute('open'))) {
      await tools.locator('summary').click()
    }
    for (const [button, title] of TOOLS) {
      await page.getByRole('button', { name: button, exact: true }).click()
      await expect(page.locator('h1')).toHaveText(title)
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

    await page.getByRole('button', { name: 'Today', exact: true }).click()
    await expect(page.locator('h1')).toHaveText('Today')

    const cards = page.locator('.home-os-best-trades > div')
    expect(await cards.count()).toBeLessThanOrEqual(5)
  }

  expect(pageErrors, 'uncaught browser exceptions').toEqual([])
  expect(serverErrors, 'backend 5xx responses observed by the real frontend').toEqual([])
})
