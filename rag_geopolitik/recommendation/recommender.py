"""Recommendation engine: confidence adjuster, historical matching, and explanation generation.

Spec section 8 — full pipeline from model output to final recommendation.
Uses Google Gemini Flash Lite (free) for LLM explanation.
"""

from __future__ import annotations

from typing import Any

from rag_geopolitik.config import get_settings
from rag_geopolitik.constants import RISK_FLAGS
from rag_geopolitik.engine.retriever import WeightedRetriever
from rag_geopolitik.models.stock_model import StockModel
from rag_geopolitik.schemas import (
    ExtractedEvent,
    FlowPrediction,
    HistoricalEvent,
    StockOpportunity,
)

try:
    from google.genai import Client as GeminiClient
except ImportError:
    GeminiClient = None  # type: ignore[assignment]


class Recommender:
    """Generates final stock recommendations with confidence and explanation.

    Parameters
    ----------
    stock_model : StockModel
    flow_prediction : FlowPrediction, optional
    retriever : WeightedRetriever, optional
    gemini_api_key : str, optional
    """

    def __init__(
        self,
        stock_model: StockModel | None = None,
        flow_prediction: FlowPrediction | None = None,
        retriever: WeightedRetriever | None = None,
        gemini_api_key: str | None = None,
    ) -> None:
        self.stock_model = stock_model
        self.flow_prediction = flow_prediction
        self.retriever = retriever
        settings = get_settings()

        self._client = None
        if GeminiClient is not None:
            api_key = gemini_api_key or settings.gemini_api_key
            if api_key:
                self._client = GeminiClient(api_key=api_key)
        self._llm_model = settings.llm_model

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def recommend(
        self,
        ticker: str,
        ticker_features: dict[str, Any],
        events: list[ExtractedEvent],
        shap_features: dict[str, float] | None = None,
    ) -> StockOpportunity:
        """Run the full recommendation pipeline for a single ticker."""
        # 1. Model prediction
        opp = (
            self.stock_model.predict(ticker, ticker_features)
            if self.stock_model is not None
            else StockOpportunity(
                ticker=ticker,
                outperform_probability=0.0,
                confidence=0.0,
            )
        )

        # 2. Adjust confidence
        ticker_events = [e for e in events if ticker in e.tickers_affected]
        avg_cred = (
            sum(e.source_credibility for e in ticker_events) / max(len(ticker_events), 1)
        )
        n_sources = len({e.article_id for e in ticker_events})
        avg_novelty = (
            sum(e.novelty_score for e in ticker_events) / max(len(ticker_events), 1)
        )

        opp.confidence = self._adjust_confidence(
            model_confidence=opp.confidence,
            novelty_score=avg_novelty,
            source_credibility_avg=avg_cred,
            n_sources_confirming=n_sources,
        )

        # 3. Risk flags
        opp.risk_flags = self._compute_risk_flags(
            ticker_features, avg_novelty, n_sources
        )

        # 4. Top SHAP features
        if shap_features:
            opp.top_shap_features = shap_features

        # 5. Historical matching
        if self.retriever is not None and ticker_events:
            opp.historical_reference = self._match_historical(
                ticker_events[0], ticker
            )

        # 6. Explanation
        opp.summary, opp.explanation_bullets = self._generate_explanation(
            ticker, opp, shap_features or {}
        )

        opp.novelty_score = avg_novelty
        return opp

    def recommend_batch(
        self,
        ticker_features: dict[str, dict[str, Any]],
        events: list[ExtractedEvent],
        shap_features: dict[str, dict[str, float]] | None = None,
    ) -> list[StockOpportunity]:
        """Run recommendation for multiple tickers."""
        return [
            self.recommend(
                ticker, feats, events,
                shap_features=(shap_features or {}).get(ticker),
            )
            for ticker, feats in ticker_features.items()
        ]

    # ------------------------------------------------------------------ #
    # Internal: confidence adjuster (spec 8.3)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _adjust_confidence(
        model_confidence: float,
        novelty_score: float,
        source_credibility_avg: float,
        n_sources_confirming: int,
    ) -> float:
        novelty_penalty = (1 - novelty_score) * 0.2
        source_boost = (source_credibility_avg - 0.5) * 0.1
        confirmation_boost = min(n_sources_confirming / 5, 1) * 0.1
        adjusted = model_confidence - novelty_penalty + source_boost + confirmation_boost
        return round(min(max(adjusted, 0.1), 0.99), 2)

    # ------------------------------------------------------------------ #
    # Internal: risk flags (spec 11.3)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_risk_flags(
        features: dict[str, Any],
        novelty: float,
        n_sources: int,
    ) -> list[str]:
        flags: list[str] = []
        if novelty < 0.4:
            flags.append("low_novelty")
        if n_sources <= 1:
            flags.append("single_source")
        if features.get("vix_level", 0) > 25:
            flags.append("high_vix")
        if features.get("ticker_volume_ratio", 1) < 0.5:
            flags.append("low_liquidity")
        if features.get("earnings_season", False):
            flags.append("earnings_window")
        return flags

    # ------------------------------------------------------------------ #
    # Internal: historical matching (spec 8.2)
    # ------------------------------------------------------------------ #
    def _match_historical(
        self, event: ExtractedEvent, ticker: str
    ) -> dict[str, Any] | None:
        if self.retriever is None:
            return None
        query = f"{event.event_type} {' '.join(event.entities)}"
        chunks = self.retriever.retrieve(query, top_k=3)
        if not chunks:
            return None
        return {
            "event": event.event_type,
            "similar_chunks": [c.chunk_text[:200] for c in chunks],
            "avg_similarity": round(
                sum(c.final_score for c in chunks) / len(chunks), 2
            ),
        }

    # ------------------------------------------------------------------ #
    # Internal: explanation generator — Google Gemini Flash Lite (free)
    # ------------------------------------------------------------------ #
    def _generate_explanation(
        self,
        ticker: str,
        opp: StockOpportunity,
        shap_features: dict[str, float],
    ) -> tuple[str, list[str]]:
        """Generate explanation via Google Gemini Flash Lite or template fallback."""
        if self._client is None:
            return self._template_explanation(ticker, opp, shap_features)

        context = ""
        if self.retriever is not None:
            chunks = self.retriever.retrieve(f"{ticker} outlook", top_k=3)
            context = "\n".join(c.chunk_text[:300] for c in chunks)

        flow_dir = (
            self.flow_prediction.direction.value
            if self.flow_prediction else "unknown"
        )

        prompt = f"""
Anda adalah analis pasar modal Indonesia. Jelaskan mengapa {ticker} memiliki potensi {'outperform' if opp.outperform_probability > 0.5 else 'netral'} dalam 5 hari ke depan.

Probabilitas outperform: {opp.outperform_probability:.0%}
Prediksi foreign flow: {flow_dir}

Konteks berita terkini:
{context}

Berikan penjelasan dalam:
1. Summary (1 kalimat singkat)
2. 3-5 bullet point alasan (bahasa Indonesia)
"""
        try:
            resp = self._client.models.generate_content(
                model=self._llm_model,
                contents=prompt,
            )
            text = resp.text or ""
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            summary = lines[0] if lines else f"Analisis untuk {ticker}"
            bullets = [l for l in lines[1:] if l.startswith("-") or l.startswith("*")]
            return summary, bullets[:5]
        except Exception:
            return self._template_explanation(ticker, opp, shap_features)

    @staticmethod
    def _template_explanation(
        ticker: str,
        opp: StockOpportunity,
        shap_features: dict[str, float],
    ) -> tuple[str, list[str]]:
        """Fallback template when no LLM is available."""
        summary = (
            f"{ticker} memiliki probabilitas outperform "
            f"{opp.outperform_probability:.0%} dengan confidence "
            f"{opp.confidence:.0%}"
        )
        bullets = [
            f"Probabilitas outperform: {opp.outperform_probability:.0%}",
            f"Confidence: {opp.confidence:.0%}",
        ]
        for feat, val in list(shap_features.items())[:3]:
            direction = "positif" if val > 0 else "negatif"
            bullets.append(f"Feature '{feat}' kontribusi {direction}: {val:.3f}")
        return summary, bullets