"""Feature store with Redis (real-time) and Parquet (historical).

Spec section 6: Redis for sub-100 ms API reads, Parquet for ML training.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from rag_geopolitik.config import DATA_DIR, get_settings

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]

try:
    import redis as _redis
except ImportError:
    _redis = None  # type: ignore[assignment]


_REDIS_PREFIX = "feature:"


def _redis_key(ticker: str, dt: date) -> str:
    return f"{_REDIS_PREFIX}{ticker}:{dt.isoformat()}"


def _market_key(dt: date) -> str:
    return f"{_REDIS_PREFIX}market:{dt.isoformat()}"


def _flow_key(dt: date) -> str:
    return f"{_REDIS_PREFIX}flow:{dt.isoformat()}"


class FeatureStore:
    """Dual-layer feature store backed by Redis (online) and Parquet (offline).

    Parameters
    ----------
    redis_url : str, optional
    parquet_dir : Path, optional
        Directory where historical parquet files live.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        parquet_dir: Path | None = None,
    ) -> None:
        settings = get_settings()
        self.parquet_dir = parquet_dir or DATA_DIR / "features"

        self._redis = (
            _redis.from_url(redis_url or settings.redis_url, decode_responses=True)
            if _redis is not None
            else None
        )
        self.parquet_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Online store (Redis)
    # ------------------------------------------------------------------ #
    def put_ticker_features(
        self, ticker: str, dt: date, features: dict[str, Any]
    ) -> None:
        """Write per-ticker feature dict for a given date to Redis."""
        if self._redis is None:
            return
        key = _redis_key(ticker, dt)
        self._redis.setex(key, 86_400, json.dumps(features, default=str))

    def get_ticker_features(self, ticker: str, dt: date) -> dict[str, Any] | None:
        """Read per-ticker feature dict from Redis."""
        if self._redis is None:
            return None
        raw = self._redis.get(_redis_key(ticker, dt))
        return json.loads(raw) if raw else None

    def put_market_features(self, dt: date, features: dict[str, Any]) -> None:
        """Write market-wide features to Redis."""
        if self._redis is None:
            return
        self._redis.setex(_market_key(dt), 86_400, json.dumps(features, default=str))

    def get_market_features(self, dt: date) -> dict[str, Any] | None:
        """Read market-wide features from Redis."""
        if self._redis is None:
            return None
        raw = self._redis.get(_market_key(dt))
        return json.loads(raw) if raw else None

    def put_flow_features(self, dt: date, features: dict[str, Any]) -> None:
        """Write foreign-flow proxy features to Redis."""
        if self._redis is None:
            return
        self._redis.setex(_flow_key(dt), 86_400, json.dumps(features, default=str))

    def get_flow_features(self, dt: date) -> dict[str, Any] | None:
        """Read foreign-flow proxy features from Redis."""
        if self._redis is None:
            return None
        raw = self._redis.get(_flow_key(dt))
        return json.loads(raw) if raw else None

    # ------------------------------------------------------------------ #
    # Offline store (Parquet)
    # ------------------------------------------------------------------ #
    def append_training_row(self, row: dict[str, Any]) -> None:
        """Append a single training row to the parquet dataset.

        The row is expected to contain at least ``date``, ``ticker``, and all
        feature columns listed in spec 6.3 plus target labels.
        """
        if pd is None:
            return
        df = pd.DataFrame([row])
        path = self.parquet_dir / "training_data.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_parquet(path, index=False)

    def load_training_data(self) -> pd.DataFrame:
        """Load the full training dataset from parquet."""
        path = self.parquet_dir / "training_data.parquet"
        if pd is None or not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def load_features_since(
        self, since: date, ticker: str | None = None
    ) -> pd.DataFrame:
        """Load historical features filtered by date and optionally ticker."""
        df = self.load_training_data()
        if df.empty:
            return df
        df = df[df["date"] >= since.isoformat()]
        if ticker:
            df = df[df["ticker"] == ticker]
        return df