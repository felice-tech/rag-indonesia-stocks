"""News crawler: fetch, parse, normalise, and attach source metadata.

Mirrors the ``NewsCrawler`` design from spec section 2.2.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Generator

from rag_geopolitik.collector.dedup import RedisDedup
from rag_geopolitik.config import Source, load_source_registry
from rag_geopolitik.schemas import RawArticle


def _sha256_hash(title: str, source_id: str) -> str:
    """Return a deterministic deduplication hash."""
    return hashlib.sha256(
        f"{title}|{source_id}".encode("utf-8")
    ).hexdigest()


def _parse_timestamp(raw: str) -> datetime:
    """Best-effort parsing of an ISO-8601 timestamp; falls back to now."""
    try:
        return datetime.fromisoformat(raw).astimezone(timezone.utc)
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


class NewsCrawler:
    """Manages crawling, parsing, and deduplication for all configured sources.

    Parameters
    ----------
    dedup_ttl_seconds : int
        How long a dedup key lives in Redis (default 7 days).
    """

    def __init__(self, dedup_ttl_seconds: int = 604_800) -> None:
        self.source_registry: dict[str, Source] = load_source_registry()
        self.dedup_store = RedisDedup(ttl_seconds=dedup_ttl_seconds)
        self.dedup_ttl = dedup_ttl_seconds

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def crawl(self, source_id: str) -> list[RawArticle]:
        """Crawl a single source and return new (non-deduped) articles.

        Parameters
        ----------
        source_id : str
            Key into the source registry (e.g. ``"reuters"``).

        Returns
        -------
        list[RawArticle]
            Articles that passed deduplication.
        """
        source = self.source_registry[source_id]
        raw_articles = self._fetch_source(source)
        return self._deduplicate(source, raw_articles)

    def crawl_all(self) -> list[RawArticle]:
        """Crawl every configured source and return all new articles."""
        articles: list[RawArticle] = []
        for source_id in self.source_registry:
            articles.extend(self.crawl(source_id))
        return articles

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _fetch_source(self, source: Source) -> list[RawArticle]:
        """Placeholder: implement source-specific fetching logic below.

        Subclass or monkey-patch this method to add real HTTP / Playwright
        calls for each source.
        """
        # TODO: implement per-source fetchers (RSS, Scrapy, Playwright)
        return []

    def _deduplicate(
        self, source: Source, articles: list[RawArticle]
    ) -> list[RawArticle]:
        """Remove articles already seen in the dedup store."""
        fresh: list[RawArticle] = []
        for article in articles:
            key = _sha256_hash(article.title, source.id)
            if not self.dedup_store.exists(key):
                self.dedup_store.set(key)
                fresh.append(article)
        return fresh

    # ------------------------------------------------------------------ #
    # Builder helper (spec 2.3)
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_article(
        source: Source,
        title: str,
        body: str,
        url: str,
        published_at: str | datetime,
        language: str,
        category: str,
        raw_html: str | None = None,
    ) -> RawArticle:
        """Convenience builder that enriches a ``RawArticle`` with source metadata."""
        if isinstance(published_at, str):
            published_at = _parse_timestamp(published_at)

        return RawArticle(
            id=str(uuid.uuid4()),
            source_id=source.id,
            credibility_score=source.credibility_score,
            title=title.strip(),
            body=body.strip(),
            url=url,
            published_at=published_at,
            language=language,
            category=category,
            raw_html=raw_html,
        )