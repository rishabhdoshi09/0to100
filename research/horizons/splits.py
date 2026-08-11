"""Purged / embargoed splits bound to a TargetSpec (reuses research.harness)."""
from __future__ import annotations

import numpy as np

from research.harness import purged_kfold_indices
from research.horizons.spec import OverlapPolicy, TargetSpec


def purged_splits_for_target(
    n: int,
    target: TargetSpec,
    *,
    k: int = 5,
    embargo: int | float | None = None,
) -> list[tuple]:
    """Return ``(train_idx, test_idx)`` folds honouring the target's horizon.

    Delegates to ``research.harness.purged_kfold_indices`` so the research harness
    remains the single statistical implementation. Adding a horizon never requires
    editing the harness — only a new ``TargetSpec``.
    """
    if n <= 0:
        return []
    if target.overlap_policy == OverlapPolicy.ALLOW_WITH_WARNING:
        # Still compute purged indices for transparency, but callers must not
        # treat ALLOW_WITH_WARNING results as promotion-grade evidence.
        pass
    emb = target.effective_embargo_bars if embargo is None else embargo
    return purged_kfold_indices(
        n,
        k=k,
        embargo=emb,
        label_horizon=target.effective_purge_bars,
    )


def assert_no_label_leakage(
    train_idx,
    test_idx,
    *,
    label_horizon: int,
    mode: str = "strict",
) -> None:
    """Raise ``AssertionError`` if train labels overlap the test block.

    mode='strict':
        Any train sample whose exit bar (i + H) falls inside the test block, or
        whose feature index is inside the test block, is leakage.
    mode='harness':
        Matches ``research.harness.purged_kfold_indices`` purge width (H-1 bars
        immediately before the test block plus the test block itself).
    """
    train = np.asarray(train_idx, dtype=int)
    test = np.asarray(test_idx, dtype=int)
    if train.size == 0 or test.size == 0:
        return
    t0, t1 = int(test.min()), int(test.max())
    h = max(0, int(label_horizon))
    leaked = []
    if mode == "harness":
        lab = max(0, h - 1)
        purge_lo = max(0, t0 - lab)
        forbidden = set(range(purge_lo, t1 + 1))
        for i in train:
            if int(i) in forbidden:
                leaked.append(int(i))
    else:
        for i in train:
            i = int(i)
            if t0 <= i <= t1:
                leaked.append(i)
                continue
            if h > 0 and i < t0 and (i + h) >= t0:
                leaked.append(i)
    if leaked:
        raise AssertionError(
            f"label leakage ({mode}): {len(leaked)} train indices overlap test "
            f"window [{t0}, {t1}] at horizon={h} (e.g. {leaked[:5]})"
        )


def train_val_test_embargo_slices(
    n: int,
    target: TargetSpec,
    *,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> dict:
    """Single walk-forward contiguous split with embargo gaps between segments.

    Returns index arrays for train / val / test after dropping embargo buffers
    that would otherwise let overlapping labels cross segment boundaries.
    """
    if n <= 0:
        return {"train": np.array([], dtype=int), "val": np.array([], dtype=int),
                "test": np.array([], dtype=int)}
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("train_frac/val_frac must be positive and sum to < 1")

    emb = int(target.effective_embargo_bars)
    purge = int(target.effective_purge_bars)

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    # Embargo + purge gaps: after train before val, after val before test.
    gap = max(emb, purge)
    train_idx = np.arange(0, max(0, train_end - gap))
    val_start = min(n, train_end + gap)
    val_idx = np.arange(val_start, max(val_start, val_end - gap))
    test_start = min(n, val_end + gap)
    test_idx = np.arange(test_start, n)

    # Drop train/val samples whose label exit would fall into a later segment.
    def _filter(idx, forbidden_lo, forbidden_hi):
        if idx.size == 0:
            return idx
        keep = []
        for i in idx:
            exit_i = int(i) + purge
            if forbidden_lo <= exit_i <= forbidden_hi or forbidden_lo <= i <= forbidden_hi:
                continue
            keep.append(int(i))
        return np.asarray(keep, dtype=int)

    if val_idx.size:
        train_idx = _filter(train_idx, int(val_idx.min()), int(val_idx.max()))
    if test_idx.size:
        train_idx = _filter(train_idx, int(test_idx.min()), int(test_idx.max()))
        val_idx = _filter(val_idx, int(test_idx.min()), int(test_idx.max()))

    return {"train": train_idx, "val": val_idx, "test": test_idx}
