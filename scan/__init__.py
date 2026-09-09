"""Canonical scan package initialization."""

from .long_term_consistency import install as _install_long_term_consistency
from .nonblocking_scan_consistency import install as _install_nonblocking_scan_consistency

_install_long_term_consistency()
_install_nonblocking_scan_consistency()

__all__ = []
