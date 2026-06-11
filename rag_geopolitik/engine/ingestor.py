"""Article ingestion pipeline: chunk, embed, and store into Qdrant.

Spec section 3.2. Uses Google Gemini Embedding (free) instead of OpenAI.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from rag_geopolitik.config import get_settings
from rag_geopolitik.schemas import RawArticle

try:
    from google.genai import Client as GeminiClient
except ImportError:
    GeminiClient = None  # type: ignore[assignment]

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import PointStruct
except ImportError:
    QdrantClient = None  # type: ignore[assignment]
    PointStruct = None  # type: ignore[assignment]


_COLLECTION_NAME = "news_kb"
_EMBEDDING_DIM = 768  # Google text-embedding-004 outputs 768-dim vectors


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split *text* into overlapping chunks of token-approximate size."""
    approx_tokens = len(text) // 4
    if approx_tokens <= chunk_size:
        return [text]

    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < approx_tokens:
        char_start = start * 4
        char_end = (start + chunk_size) * 4
        chunks.append(text[char_start:char_end])
        start += step
    return chunks or [text]


class Ingestor:
    """Embeds article chunks (Google Gemini) and persists them into Qdrant.

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
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
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
        self._ensure_collection()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def ingest(self, article: RawArticle) -> int:
        """Chunk, embed, and upload *article* into the vector store.

        Returns
        -------
        int
            Number of chunks stored.
        """
        chunks = _chunk_text(article.body, self.chunk_size, self.chunk_overlap)
        if not chunks:
            return 0

        embeddings = self._embed(chunks)
        if embeddings is None:
            return 0

        self._upload(article, chunks, embeddings)
        return len(chunks)

    def ingest_batch(self, articles: list[RawArticle]) -> int:
        """Ingest multiple articles; returns total chunks stored."""
        return sum(self.ingest(a) for a in articles)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _embed(self, texts: list[str]) -> list[list[float]] | None:
        """Call Google Gemini embedding API; return list of vectors or None."""
        if self._client is None:
            return None
        try:
            result = self._client.models.embed_content(
                model=self.embedding_model,
                contents=texts,
                config={"task_type": "RETRIEVAL_DOCUMENT"},
            )
            return [list(v.values) for v in result.embeddings]
        except Exception:
            return None

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not already exist."""
        if self._qdrant is None:
            return
        collections = self._qdrant.get_collections().collections
        if not any(c.name == _COLLECTION_NAME for c in collections):
            self._qdrant.create_collection(
                collection_name=_COLLECTION_NAME,
                vectors_config={"size": _EMBEDDING_DIM, "distance": "Cosine"},
            )

    def _upload(
        self,
        article: RawArticle,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """Upsert chunk points into Qdrant."""
        if self._qdrant is None:
            return

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={
                    "article_id": article.id,
                    "source_id": article.source_id,
                    "credibility_score": article.credibility_score,
                    "published_at": article.published_at.isoformat(),
                    "category": article.category,
                    "chunk_text": chunk,
                },
            )
            for chunk, emb in zip(chunks, embeddings)
        ]

        self._qdrant.upsert(
            collection_name=_COLLECTION_NAME,
            points=points,
        )