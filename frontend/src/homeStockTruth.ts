/**
 * An old cached Home panel must not present its BUY picks as current after a
 * Kite snapshot rotation, incomplete data refresh or stale scan. False here
 * means "refreshing/unverified", NOT "the market has no eligible trades".
 */
export function verifiedHomeStockScan(
  scan: { available: boolean; scanned_at?: string; provenance?: { data_current?: boolean } },
  dataReady: boolean,
  homeScannedAt: string | undefined,
): boolean {
  return Boolean(
    dataReady && scan.available && scan.scanned_at &&
    scan.provenance?.data_current !== false &&
    homeScannedAt && homeScannedAt === scan.scanned_at,
  )
}
