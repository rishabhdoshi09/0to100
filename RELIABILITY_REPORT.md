# QuantTerm operational reliability and disaster recovery — initial review
Date: 2026-11-07 | Status: PARTIAL / HOST NOT ACCESSED.

## Storage incident recorded from operator brief
Original runtime: /Volumes/QuantTermStorage/QuantTerm/runtime.
Operator reports APFS sparsebundle mount failure and superblock errors on external ExFAT drive, with checksum-verified backup at /Volumes/Expansion/QuantTermStorage_BACKUP_20261107.sparsebundle and read-only recovered mount /Volumes/QuantTermRecovery. These are operator-provided incident claims, not freshly revalidated by this audit. Preserve originals, backup, and recovery image; never run repair/overwrite while auditing.

## Controls visible in code
core/runtime_paths.py implements QT_RUNTIME_ROOT_REQUIRE_EXISTING, refusing an absent configured root in strict installed-host mode. SnapshotStore activates with an atomic ACTIVE swap, commits immutable content-addressed CSVs, and has verify_snapshot for manifest/data hashes. scan/bulk_fetcher.py verifies snapshot before adopting broker bars. The complete launcher delegates installed-host lifecycle to launchd rather than intentionally starting a second owner.
These protections cannot prove the Mac's currently loaded plist, environment, mount, actual service SHA, active job count or WAL state.

## Verified defect and isolated correction
SnapshotStore.get_active_snapshot() collapses malformed ACTIVE pointers and missing snapshot targets into None. The original scan_source_status() treated probe errors with empty used ID as safe. Audit branch now rejects an existing-but-invalid pointer, probe errors, and a Kite label with no active snapshot. Regression tests are synthetic and production storage is untouched.

## Additional risks requiring bounded investigation
1. Cross-file generation consistency: auxiliary recommendation/discovery files can be written before final scan publication. Inject rotation in a disposable runtime.
2. A single CSV/hash check on every HTTP read would hurt performance; establish a verified snapshot metadata cache with explicit invalidation, subject to security review.
3. Validate Atomic SQLite WAL backups, checkpoint handling, in-flight paper positions and restart idempotency on a disposable copy.
4. Test abrupt mount loss, full disk, power failure, stale symlink/mountpoint, invalid ACTIVE pointer and invalid token; no silent empty-runtime fallback.
5. Measure file descriptors, RSS/swap, scan I/O, queue growth and endpoint latency under realistic host load.

## Disaster recovery protocol (not executed)
- Freeze and preserve source images and checksum manifest.
- Verify backup via read-only filesystem checks and cryptographic hash comparison.
- Copy only into a disposable destination; never modify recovered or original disk.
- With all trading services stopped in the sandbox, inspect SQLite integrity and WAL consistency, and reconcile positions/evidence by generation.
- Start one isolated PAPER-only observer with redirected runtime; prove no unexpected orders and no duplicate jobs on restart.
- Document recovery-point objective, recovery-time objective, evidence gaps and rollback procedure before any host migration.

## Security
Source review alone does not establish authn/authz, secret scanning, dependency patch status or last-mile broker interlocks. No secrets should be printed, included in reports, or committed. Do not authorize live orders.

## Verdict
Disaster recovery and installed-host reliability remain NOT VERIFIED. No production writes/restarts/deploys were performed.
