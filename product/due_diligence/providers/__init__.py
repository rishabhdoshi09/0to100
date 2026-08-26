"""Multi-source adapters for Investigate. GET never calls fetch()."""
from product.due_diligence.providers.base import (
    FetchResult,
    ProviderError,
    ProviderPolicy,
    SourceAdapter,
    archive_bytes,
    empty_normalized,
    merge_normalized,
)

__all__ = [
    "FetchResult",
    "ProviderError",
    "ProviderPolicy",
    "SourceAdapter",
    "archive_bytes",
    "empty_normalized",
    "merge_normalized",
]
