import { expect, test } from '@playwright/test'

// Regression coverage for two rendering bugs found during a live Home-page
// audit (screenshot-verified before/after):
//
// 1. ".top-status button { width: 34px; height: 34px }" (sized for the
//    small square icon-only refresh button) also matched the depth-toggle's
//    "Simple"/"Professional" buttons and the "What is this?" trigger, since
//    both are <button> descendants of .top-status. Their text rendered
//    clipped to a few letters ("Simp", "Profes", "What i") with no wrap or
//    ellipsis, at an ordinary 1440px desktop width.
//
// 2. ".home-os-past" cards (the "No eligible trade" / LEARNING IMPACT / US
//    PAPER MARKET blocks on Home) never gave their <strong>/<small>
//    children a block display, so adjacent inline elements' text ran
//    together with no line break or space, e.g. "No eligible tradeThe
//    production thesis did not qualify..." and "paper auto ONidle · All".
//
// These are layout bugs a status-code/console-error check cannot catch, so
// this spec asserts the actual rendered geometry and text instead.

test('depth-toggle and help-trigger buttons are not squeezed to icon-button width', async ({ page }) => {
  await page.goto('/')
  const toggle = page.locator('.display-depth-toggle')
  await expect(toggle).toBeVisible()

  const simple = toggle.getByRole('button', { name: 'Simple' })
  const professional = toggle.getByRole('button', { name: 'Professional' })
  await expect(simple).toBeVisible()
  await expect(professional).toBeVisible()

  // A 34px-wide box (the icon-button size that was leaking onto these text
  // buttons) cannot fit "Professional" at any readable font size. Assert a
  // floor comfortably above that regression value, well below what the text
  // actually needs (~70-90px), so this fails loudly if the bug returns
  // without being so tight it flakes on minor padding/font tweaks.
  const simpleBox = await simple.boundingBox()
  const professionalBox = await professional.boundingBox()
  expect(simpleBox?.width ?? 0).toBeGreaterThan(45)
  expect(professionalBox?.width ?? 0).toBeGreaterThan(45)

  const helpTrigger = page.locator('.experience-help-trigger')
  await expect(helpTrigger).toBeVisible()
  await expect(helpTrigger).toHaveText('What is this?')
  const helpBox = await helpTrigger.boundingBox()
  expect(helpBox?.width ?? 0).toBeGreaterThan(45)
})

test('home-os-past cards give strong/small children a line break, not run-on text', async ({ page }) => {
  await page.goto('/')
  const card = page.locator('.home-os-card')
  await expect(card).toBeVisible({ timeout: 30_000 })

  const pastBlocks = card.locator('.home-os-past')
  const count = await pastBlocks.count()
  expect(count).toBeGreaterThan(0)

  for (let i = 0; i < count; i += 1) {
    const block = pastBlocks.nth(i)
    const strong = block.locator('strong').first()
    const small = block.locator('small').first()
    if ((await strong.count()) === 0 || (await small.count()) === 0) continue
    const [strongDisplay, smallDisplay] = await Promise.all([
      strong.evaluate((el) => getComputedStyle(el).display),
      small.evaluate((el) => getComputedStyle(el).display),
    ])
    expect(strongDisplay).toBe('block')
    expect(smallDisplay).toBe('block')
  }
})
