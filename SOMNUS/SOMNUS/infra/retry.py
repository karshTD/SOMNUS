"""CockroachDB serializable-isolation retry.

CockroachDB uses SERIALIZABLE isolation and *expects* clients to retry on
error code 40001 (serialization_failure). Without this, any concurrent
consolidation transaction can simply fail. This is the single most commonly
missed CockroachDB client requirement.
"""

from __future__ import annotations

import functools
import logging
import random
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

SERIALIZATION_FAILURE = "40001"
MAX_ATTEMPTS = 5
BASE_BACKOFF = 0.05


def _is_retryable(exc: BaseException) -> bool:
    code = getattr(exc, "pgcode", None)
    if code == SERIALIZATION_FAILURE:
        return True
    # psycopg2 wraps the code on the .pgerror/.args in some paths
    return SERIALIZATION_FAILURE in str(getattr(exc, "pgerror", "") or "")


def with_retry(fn: Callable[..., T]) -> Callable[..., T]:
    """Retry a DB operation on 40001 with exponential backoff and jitter."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        last: BaseException | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - re-raised below
                if not _is_retryable(exc):
                    raise
                last = exc
                backoff = BASE_BACKOFF * (2**attempt) + random.uniform(0, 0.05)
                logger.warning(
                    "CRDB serialization failure (attempt %d/%d), retrying in %.3fs",
                    attempt + 1,
                    MAX_ATTEMPTS,
                    backoff,
                )
                time.sleep(backoff)
        raise RuntimeError(f"Exhausted {MAX_ATTEMPTS} retries") from last

    return wrapper
