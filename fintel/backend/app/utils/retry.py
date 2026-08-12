from __future__ import annotations
import functools
from tenacity import retry, stop_after_attempt, wait_exponential


def with_retry(max_attempts: int = 3, min_wait: float = 1.0, max_wait: float = 10.0):
    def decorator(func):
        @functools.wraps(func)
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            reraise=True,
        )
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        return wrapper
    return decorator
