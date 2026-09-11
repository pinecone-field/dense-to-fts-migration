"""Bounded retry for the write paths that run long enough to meet a transient failure.

A load or a replay can run for hours, so a single timed-out request should not end it.
Retries are safe here because every write this toolkit makes is idempotent: document
upserts replace the whole document, and deletes of an absent id are no-ops.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypeVar

from pinecone.errors import (
    PineconeConnectionError,
    PineconeTimeoutError,
    RateLimitError,
    ServiceError,
)

T = TypeVar("T")

RETRYABLE = (PineconeTimeoutError, PineconeConnectionError, RateLimitError, ServiceError)
DEFAULT_ATTEMPTS = 4
DEFAULT_BACKOFF_SECONDS = 2.0


def with_retry(
    operation: Callable[[], T],
    attempts: int = DEFAULT_ATTEMPTS,
    backoff: float = DEFAULT_BACKOFF_SECONDS,
    on_retry: Callable[[int, BaseException], Any] | None = None,
) -> T:
    """Run `operation`, retrying transient failures with exponential backoff.

    Only transport-level and service-level failures are retried. A 4xx means the
    request itself is wrong and will be wrong again, so it is raised immediately.
    """
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except RETRYABLE as exc:
            if attempt == attempts:
                raise
            if on_retry is not None:
                on_retry(attempt, exc)
            time.sleep(backoff * (2 ** (attempt - 1)))
    raise AssertionError("unreachable")
