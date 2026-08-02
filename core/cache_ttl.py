"""Process-local TTL cache — Streamlit-free alternative to @st.cache_data for data modules."""
from __future__ import annotations

import time
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def ttl_cache(ttl_seconds: float = 3600) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        store: dict[tuple[Any, ...], Any] = {}
        times: dict[tuple[Any, ...], float] = {}

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()
            if key in store and now - times.get(key, 0.0) < ttl_seconds:
                return store[key]
            value = func(*args, **kwargs)
            store[key] = value
            times[key] = now
            return value

        def clear() -> None:
            store.clear()
            times.clear()

        wrapper.clear = clear  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator
