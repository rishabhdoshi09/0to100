"""launchd's StandardErrorPath grew to 181 MB on the owner's Mac (no rotation)."""
from __future__ import annotations

import os

from product.host_entrypoint import trim_inherited_log


def test_append_mode_log_is_trimmed_to_its_newest_lines(tmp_path):
    path = tmp_path / "launchd.err.log"
    path.write_bytes(b"".join(b"line %06d\n" % i for i in range(20000)))
    fd = os.open(path, os.O_RDWR | os.O_APPEND)
    try:
        assert trim_inherited_log(fd, max_bytes=1000, keep_bytes=200) is True
        os.write(fd, b"after\n")
    finally:
        os.close(fd)
    data = path.read_bytes()
    assert len(data) < 400
    assert data.startswith(b"[HOST LOG] trimmed")
    assert b"line 019999\n" in data and data.endswith(b"after\n")
    assert b"\x00" not in data


def test_small_or_non_append_logs_are_left_alone(tmp_path):
    small = tmp_path / "small.log"
    small.write_bytes(b"x" * 100)
    fd = os.open(small, os.O_RDWR | os.O_APPEND)
    try:
        assert trim_inherited_log(fd, max_bytes=1000, keep_bytes=10) is False
    finally:
        os.close(fd)
    big = tmp_path / "big.log"
    big.write_bytes(b"y\n" * 5000)
    fd = os.open(big, os.O_RDWR)          # no O_APPEND: trimming would leave a hole
    try:
        assert trim_inherited_log(fd, max_bytes=1000, keep_bytes=10) is False
    finally:
        os.close(fd)
    assert big.stat().st_size == 10000
