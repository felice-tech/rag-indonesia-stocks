"""Event extraction: NER, classification, sentiment, entity-to-ticker linking.

Spec section 4 — mirrors the pipeline described in 4.1–4.4.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from rag_geopolitik.constants import EVENT_TYPES, ALL_EVENT_SUBTYPES, map_entities_to_tickers
from rag_geopolitik.schemas import (
    ExtractedEvent,
    Magnitude,
    RawArticle,
    SentimentDirection,
)

try:
    import spacy
except ImportError:
    spacy = None  # type: ignore[assignment]

try:
    from langdetect import detect as _langdetect
except ImportError:
    _langdetect = None  # type: ignore[assignment]


# A simple rule-based sentiment lexicon for Indonesian and English.
_SENTIMENT_LEXICON: dict[str, float] = {
    # English
    "positive": 0.6,
    "negative": -0.6,
    "increase": 0.5,
    "decrease": -0.5,
    "growth": 0.7,
    "decline": -0.7,
    "bullish": 0.8,
    "bearish": -0.8,
    "rally": 0.7,
    "crash": -0.9,
    "profit": 0.6,
    "loss": -0.6,
    "surge": 0.7,
    "plunge": -0.8,
    # Indonesian
    "naik": 0.5,
    "turun": -0.5,
    "positif": 0.6,
    "negatif": -0.6,
    "meningkat": 0.6,
    "menurun": -0.6,
    "optimis": 0.7,
    "pesimis": -0.7,
    "keuntungan": 0.6,
    "kerugian": -0.6,
    "kenaikan": 0.6,
    "penurunan": -0.6,
    "stabil": 0.1,
    "tidak pasti": -0.4,
    "resiko": -0.5,
    "peluang": 0.5,
}


def _rule_based_sentiment(text: str) -> tuple[float, SentimentDirection]:
    """Compute a simple lexicon-based sentiment score (-1 .. +1)."""
    lower = text.lower()
    hits = 0.0
    count = 0
    for word, score in _SENTIMENT_LEXICON.items():
        if word in lower:
            hits += score
            count += 1
    avg = hits / count if count else 0.0
    avg = max(-1.0, min(1.0, avg))

    if avg > 0.1:
        return avg, SentimentDirection.BULLISH
    if avg < -0.1:
        return avg, SentimentDirection.BEARISH
    return avg, SentimentDirection.NEUTRAL


def _detect_category(text: str) -> str:
    """Rule-based event-category detection."""
    lower = text.lower()
    macro_kw = ["gdp", "inflation", "rate", "trade balance", "pdb", "inflasi", "suku bunga"]
    geopolitical_kw = [
        "sanctions", "trade war", "conflict", "election", "war",
        "sanksi", "perang dagang", "konflik", "pemilu", "ketegangan",
    ]
    commodity_kw = ["nickel", "cpo", "coal", "oil", "gold", "nikel", "batu bara", "minyak", "emas"]
    corporate_kw = [
        "earnings", "dividend", "merger", "acquisition", "rights issue",
        "laba", "dividen", "akuisisi", "right issue",
    ]
    capital_kw = [
        "foreign inflow", "foreign outflow", "capital flow", "risk on", "risk off",
        "capital inflow", "capital outflow",
    ]

    if any(k in lower for k in capital_kw):
        return "capital_flow"
    if any(k in lower for k in geopolitical_kw):
        return "geopolitical"
    if any(k in lower for k in commodity_kw):
        return "commodity"
    if any(k in lower for k in macro_kw):
        return "macro"
    if any(k in lower for k in corporate_kw):
        return "corporate"
    return "macro"  # default


def _estimate_magnitude(sentiment_abs: float) -> Magnitude:
    if sentiment_abs > 0.6:
        return Magnitude.HIGH
    if sentiment_abs > 0.3:
        return Magnitude.MEDIUM
    return Magnitude.LOW


class EventExtractor:
    """Pipeline that converts a ``RawArticle`` into one or more ``ExtractedEvent``.

    Parameters
    ----------
    spacy_model : str
        spaCy model to load (default ``"xx_ent_wiki_sm"`` for multilingual NER).
        If the model is not installed, extraction falls back to rule-based.
    """

    def __init__(self, spacy_model: str = "xx_ent_wiki_sm") -> None:
        self._nlp = None
        if spacy is not None:
            try:
                self._nlp = spacy.load(spacy_model)
            except OSError:
                # Model not installed — fall back to rule-based extraction
                pass

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def extract(self, article: RawArticle) -> ExtractedEvent | None:
        """Run the full extraction pipeline on a single article.

        Returns ``None`` if no meaningful event could be extracted.
        """
        text = f"{article.title}. {article.body}"

        # 1. Language detection (optional)
        lang = self._detect_lang(text) or article.language

        # 2. NER
        entities = self._extract_entities(text)

        # 3. Event-type classification
        event_type = _detect_category(text)

        # 4. Sentiment
        sentiment_score, sentiment_dir = _rule_based_sentiment(text)

        # 5. Entity-to-ticker linking
        tickers = map_entities_to_tickers(entities)

        if not tickers and sentiment_score == 0.0:
            return None

        # 6. Novelty (placeholder — requires historical comparison)
        novelty = 0.5

        snippet = self._extract_snippet(text)

        return ExtractedEvent(
            event_id=str(uuid.uuid4()),
            article_id=article.id,
            event_type=event_type,
            entities=entities,
            tickers_affected=tickers,
            sentiment_score=sentiment_score,
            sentiment_direction=sentiment_dir,
            magnitude=_estimate_magnitude(abs(sentiment_score)),
            novelty_score=novelty,
            published_at=article.published_at,
            source_credibility=article.credibility_score,
            raw_text_snippet=snippet,
        )

    def extract_batch(
        self, articles: list[RawArticle]
    ) -> list[ExtractedEvent]:
        """Extract events from a batch of articles."""
        return [e for a in articles if (e := self.extract(a)) is not None]

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _detect_lang(text: str) -> str | None:
        if _langdetect is None:
            return None
        try:
            return _langdetect(text)
        except Exception:
            return None

    def _extract_entities(self, text: str) -> list[str]:
        """Extract named entities via spaCy (if available); else fall back."""
        if self._nlp is None:
            return []
        doc = self._nlp(text[:100_000])  # limit text length
        seen: set[str] = set()
        entities: list[str] = []
        for ent in doc.ents:
            label = ent.label_
            # Keep only relevant entity types
            if label in {"ORG", "PERSON", "GPE", "LOC", "PRODUCT", "EVENT"}:
                key = ent.text.strip().lower()
                if key not in seen and len(key) > 1:
                    seen.add(key)
                    entities.append(ent.text.strip())
        return entities

    @staticmethod
    def _extract_snippet(text: str, max_chars: int = 300) -> str:
        """Return the first meaningful sentence(s) as the extraction snippet."""
        clean = text.strip()
        if len(clean) <= max_chars:
            return clean
        # Try to break at a sentence boundary within max_chars
        for sep in (". ", ".\n", "! ", "? "):
            idx = clean.rfind(sep, 0, max_chars)
            if idx != -1:
                return clean[: idx + 1]
        return clean[:max_chars] + "..."