"""Redis-backed deduplication store for article fingerprints.

Spec section 2.4: SHA256(title + source_id) with 7-day TTL.
"""

from __future__ import annotations

import hashlib
from typing import Optional

try:
    import redis as _redis
except ImportError:
    _redis = None  # type: ignore[assignment]


class RedisDedup:
    """Thin wrapper over a Redis set for article deduplication.

    Parameters
    ----------
    redis_url : str, optional
        Redis connection string. Falls back to ``redis://localhost:6379/0``.
    ttl_seconds : int
        Key expiration in seconds (default 604 800 = 7 days).
    """

    def __init__(
        self,
        redis_url: str | None = None,
        ttl_seconds: int = 604_800,
    ) -> None:
        self.ttl = ttl_seconds
        if _redis is None:
            self._client = None
        else:
            self._client = _redis.from_url(
                redis_url or "redis://localhost:6379/0",
                decode_responses=True,
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def exists(self, key: str) -> bool:
        """Return ``True`` if the dedup key already exists."""
        if self._client is None:
            return False
        return bool(self._client.exists(key))

    def set(self, key: str) -> None:
        """Persist a dedup key with the configured TTL."""
        if self._client is not None:
            self._client.setex(key, self.ttl, "1")

    def flush(self) -> None:
        """Remove all dedup keys from Redis."""
        if self._client is not None:
            self._client.flushdb()

    def set_fuzzy_threshold(self, ratio: float = 0.85) -> None:
        """Configure the fuzzy dedup threshold (spec 2.4).

        Note: fuzzy matching is a placeholder — a full implementation would
        compare ``difflib.SequenceMatcher`` ratios against this threshold
        across source pairs.
        """
        self._fuzzy_ratio = ratio