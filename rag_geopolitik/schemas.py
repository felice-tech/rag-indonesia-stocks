"""Shared data models used across the pipeline.

These dataclasses mirror the schemas described in the technical specification
(sections 2.3, 4.4, and 11).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class SentimentDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class Magnitude(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FlowDirection(str, Enum):
    NET_BUY = "net_buy"
    NET_SELL = "net_sell"
    NEUTRAL = "neutral"

    @classmethod
    def from_label(cls, label: int) -> "FlowDirection":
        """Map a model label (-1, 0, 1) to a flow direction."""
        return {1: cls.NET_BUY, -1: cls.NET_SELL, 0: cls.NEUTRAL}[label]


# --------------------------------------------------------------------------- #
# News / article schemas (spec 2.3)
# --------------------------------------------------------------------------- #
@dataclass
class RawArticle:
    """A normalised news article as produced by the collector."""

    id: str
    source_id: str
    credibility_score: float
    title: str
    body: str
    url: str
    published_at: datetime  # always UTC
    language: str  # "id" or "en"
    category: str  # macro | commodity | earnings | geopolitical
    raw_html: str | None = None


@dataclass
class Chunk:
    """A retrieved text chunk with its blended relevance score."""

    chunk_text: str
    article_id: str
    source_id: str
    credibility_score: float
    published_at: datetime
    category: str
    score: float = 0.0  # raw cosine similarity
    final_score: float = 0.0  # blended similarity + credibility


# --------------------------------------------------------------------------- #
# Event schemas (spec 4.4)
# --------------------------------------------------------------------------- #
@dataclass
class ExtractedEvent:
    """A structured event extracted from an article."""

    event_id: str
    article_id: str
    event_type: str
    entities: list[str]
    tickers_affected: list[str]
    sentiment_score: float  # -1.0 (bearish) .. +1.0 (bullish)
    sentiment_direction: SentimentDirection
    magnitude: Magnitude
    novelty_score: float  # 0..1
    published_at: datetime
    source_credibility: float
    raw_text_snippet: str


@dataclass
class HistoricalEvent:
    """A past event with its observed market impact."""

    event_id: str
    event_type: str
    entities: list[str]
    published_at: datetime
    similarity_score: float
    affected_returns_5d: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Prediction / recommendation outputs (spec 11)
# --------------------------------------------------------------------------- #
@dataclass
class FlowPrediction:
    """Market-level foreign-flow prediction (Model A output)."""

    direction: FlowDirection
    confidence: float  # 0..1
    estimated_value: str | None = None
    driving_factors: list[str] = field(default_factory=list)


@dataclass
class StockOpportunity:
    """A per-ticker opportunity with explanation (Model B output)."""

    ticker: str
    outperform_probability: float  # 0..1
    confidence: float  # 0..1
    horizon_days: int = 5
    summary: str = ""
    explanation_bullets: list[str] = field(default_factory=list)
    top_shap_features: dict[str, float] = field(default_factory=dict)
    historical_reference: dict | None = None
    risk_flags: list[str] = field(default_factory=list)
    novelty_score: float = 0.0
