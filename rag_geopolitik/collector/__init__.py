"""News collection layer: crawling, normalisation, and deduplication."""

from rag_geopolitik.collector.crawler import NewsCrawler
from rag_geopolitik.collector.dedup import RedisDedup

__all__ = ["NewsCrawler", "RedisDedup"]
