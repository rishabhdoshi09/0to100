/** Keep mounted pages stateful, but stop their hidden polling. */
export function shouldPollPage(activePage: string | undefined, pages: readonly string[]): boolean {
  return activePage == null || pages.includes(activePage)
}
