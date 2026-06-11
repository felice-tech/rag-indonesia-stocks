"""Builds the four feature buckets described in spec section 5.

Each bucket is a standalone method so that callers can compute only what they
need.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import Any

from rag_geopolitik.schemas import ExtractedEvent


# --------------------------------------------------------------------------- #
# Bucket 1: Macro / Geopolitical  (spec 5.2)
# --------------------------------------------------------------------------- #
def build_macro_geo_features(
    events: list[ExtractedEvent],
) -> dict[str, Any]:
    """Aggregate macro/geopolitical features from a list of recent events.

    Parameters
    ----------
    events
        Extracted events (normally from the last 24 hours).

    Returns
    -------
    dict
        Keys: ``sentiment_score``, ``source_weighted_sentiment``,
        ``event_velocity``, ``novelty_score``, ``geopolitical_risk_index``.
    """
    if not events:
        return {
            "sentiment_score": 0.0,
            "source_weighted_sentiment": 0.0,
            "event_velocity": 0,
            "novelty_score": 0.0,
            "geopolitical_risk_index": 0.0,
        }

    n = len(events)
    raw_sum = sum(e.sentiment_score for e in events)
    weighted_sum = sum(e.sentiment_score * e.source_credibility for e in events)
    novelty = sum(e.novelty_score for e in events) / n

    # Geopolitical risk: weighted count of geopolitical events weighted by
    # magnitude and sentiment intensity
    geo_events = [e for e in events if e.event_type == "geopolitical"]
    risk = sum(
        abs(e.sentiment_score) * e.source_credibility * _magnitude_weight(e.magnitude.value)
        for e in geo_events
    )
    risk = min(risk / max(n, 1), 1.0)

    return {
        "sentiment_score": round(raw_sum / n, 4),
        "source_weighted_sentiment": round(weighted_sum / n, 4),
        "event_velocity": n,
        "novelty_score": round(novelty, 4),
        "geopolitical_risk_index": round(risk, 4),
    }


# --------------------------------------------------------------------------- #
# Bucket 2: Sector / Commodity (spec 5.3) — data-loading stubs
# --------------------------------------------------------------------------- #
def build_commodity_features() -> dict[str, Any]:
    """Fetch current commodity price deltas from market data sources.

    Returns stub zero values.  Replace with real ``yfinance`` calls in
    production.
    """
    return {
        "nickel_price_delta_1d": 0.0,
        "nickel_price_delta_5d": 0.0,
        "cpo_price_delta_1d": 0.0,
        "coal_price_delta_1d": 0.0,
        "gold_price_delta_1d": 0.0,
        "oil_price_delta_1d": 0.0,
        "commodity_sentiment": 0.0,
        "sector_exposure_score": 0.0,
    }


# --------------------------------------------------------------------------- #
# Bucket 3: Market Microstructure (spec 5.4) — data-loading stubs
# --------------------------------------------------------------------------- #
def build_market_features() -> dict[str, Any]:
    """Fetch IHSG / ticker returns and volume data.

    Returns stub zero values.  Replace with real ``yfinance`` / IDX data.
    """
    return {
        "ihsg_return_1d": 0.0,
        "ihsg_return_3d": 0.0,
        "ihsg_return_5d": 0.0,
        "ihsg_volatility_10d": 0.0,
        "lq45_vs_ihsg_rs": 0.0,
        "ticker_return_1d": 0.0,
        "ticker_return_5d": 0.0,
        "ticker_volume_ratio": 0.0,
        "earnings_season": False,
        "quarter": _current_quarter(),
    }


# --------------------------------------------------------------------------- #
# Bucket 4: Foreign Flow Proxy (spec 5.5) — data-loading stubs
# --------------------------------------------------------------------------- #
def build_foreign_flow_features() -> dict[str, Any]:
    """Fetch DXY, US10Y, VIX, EM ETF, and IDX foreign flow data.

    Returns stub zero values.  Replace with real market-data scraping.
    """
    return {
        "dxy_delta_1d": 0.0,
        "dxy_delta_5d": 0.0,
        "us10y_yield_delta": 0.0,
        "vix_level": 15.0,
        "vix_delta_1d": 0.0,
        "em_etf_flow": 0.0,
        "jisdor_delta": 0.0,
        "idx_foreign_net_buy": 0.0,
        "idx_foreign_net_buy_5d": 0.0,
        "broker_flow_estimate": 0.0,
    }


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class FeatureBuilder:
    """Aggregates all four feature buckets into a single flat dictionary."""

    def build_all(
        self,
        events: list[ExtractedEvent],
    ) -> dict[str, Any]:
        """Return a single dict containing every feature from all four buckets."""
        features: dict[str, Any] = {}
        features.update(build_macro_geo_features(events))
        features.update(build_commodity_features())
        features.update(build_market_features())
        features.update(build_foreign_flow_features())
        return features

    def build_for_ticker(
        self,
        ticker: str,
        events: list[ExtractedEvent],
    ) -> dict[str, Any]:
        """Build features scoped to a single ticker where possible.

        The macro/flow buckets are market-level; commodity features remain
        generic here (override with ticker-specific weights in production).
        """
        features = self.build_all(events)
        features["ticker"] = ticker
        return features


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _magnitude_weight(mag: str) -> float:
    return {"high": 1.0, "medium": 0.5, "low": 0.2}.get(mag, 0.3)


_MONTH_TO_QUARTER = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4}


def _current_quarter() -> int:
    return _MONTH_TO_QUARTER.get(date.today().month, 1)