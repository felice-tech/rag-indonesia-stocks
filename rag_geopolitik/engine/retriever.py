"""Weighted RAG retrieval blending semantic similarity with source credibility.

Spec section 3.3. Uses Google Gemini Embedding (free) instead of OpenAI.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

from rag_geopolitik.config import get_settings
from rag_geopolitik.schemas import Chunk

try:
    from google.genai import Client as GeminiClient
except ImportError:
    GeminiClient = None  # type: ignore[assignment]

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Filter, FieldCondition, Range
except ImportError:
    QdrantClient = None  # type: ignore[assignment]
    Filter = None  # type: ignore[assignment]
    FieldCondition = None  # type: ignore[assignment]
    Range = None  # type: ignore[assignment]


_COLLECTION_NAME = "news_kb"


class WeightedRetriever:
    """Retrieve chunks with a score that blends similarity and source credibility.

    Parameters
    ----------
    qdrant_url : str, optional
    gemini_api_key : str, optional
    """

    def __init__(
        self,
        qdrant_url: str | None = None,
        gemini_api_key: str | None = None,
    ) -> None:
        settings = get_settings()
        self.sim_weight = settings.retrieval_similarity_weight
        self.cred_weight = settings.retrieval_credibility_weight
        self.max_age_days = settings.retrieval_max_age_days
        self.embedding_model = settings.embedding_model

        self._client = None
        if GeminiClient is not None:
            api_key = gemini_api_key or settings.gemini_api_key
            if api_key:
                self._client = GeminiClient(api_key=api_key)

        self._qdrant = (
            QdrantClient(url=qdrant_url or settings.qdrant_url)
            if QdrantClient is not None
            else None
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: int = 10) -> list[Chunk]:
        """Search, re-rank via credibility blend, and return the top-k chunks."""
        raw = self._search(query, top_k * 2)
        ranked = self._re_rank(raw)
        return ranked[:top_k]

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _embed(self, text: str) -> list[float]:
        """Embed a single query string via Google Gemini; returns zero vector on failure."""
        if self._client is None:
            return [0.0] * 768
        try:
            result = self._client.models.embed_content(
                model=self.embedding_model,
                contents=text,
                config={"task_type": "RETRIEVAL_QUERY"},
            )
            return list(result.embeddings[0].values)
        except Exception:
            return [0.0] * 768

    def _search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Run a Qdrant search with a freshness filter."""
        if self._qdrant is None:
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=self.max_age_days)
        date_filter = Filter(
            must=[
                FieldCondition(
                    key="published_at",
                    range=Range(gte=cutoff.isoformat()),
                )
            ]
        )

        results = self._qdrant.search(
            collection_name=_COLLECTION_NAME,
            query_vector=self._embed(query),
            limit=limit,
            query_filter=date_filter,
        )
        return [
            {
                "score": r.score,
                "payload": r.payload,
            }
            for r in results
        ]

    def _re_rank(self, raw: list[dict[str, Any]]) -> list[Chunk]:
        """Blend cosine similarity with source credibility score."""
        chunks: list[Chunk] = []
        for r in raw:
            p = r["payload"]
            cred = float(p.get("credibility_score", 0.5))
            chunks.append(
                Chunk(
                    chunk_text=p.get("chunk_text", ""),
                    article_id=p.get("article_id", ""),
                    source_id=p.get("source_id", ""),
                    credibility_score=cred,
                    published_at=datetime.fromisoformat(
                        p.get("published_at", datetime.now(timezone.utc).isoformat())
                    ),
                    category=p.get("category", ""),
                    score=r["score"],
                    final_score=(
                        self.sim_weight * r["score"]
                        + self.cred_weight * cred
                    ),
                )
            )
        chunks.sort(key=lambda c: c.final_score, reverse=True)
        return chunks