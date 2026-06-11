"""RAG Geopolitik & Investasi.

A retrieval-augmented pipeline that turns geopolitical / macro / commodity news
into foreign-flow predictions and LQ45 stock opportunities.

See ``rag_geopolitik_investasi_spec.md`` for the full technical specification.

Exposed sub-packages
--------------------
- ``collector``      — News crawling, normalisation, and deduplication
- ``engine``         — RAG knowledge base: chunking, embedding, Qdrant, weighted retrieval
- ``extraction``     — Event extraction: NER, classification, sentiment, ticker linking
- ``features``       — Feature engineering (4 buckets) and feature store (Redis + Parquet)
- ``models``         — ML prediction: Model A (XGBoost flow) and Model B (LightGBM stock)
- ``recommendation`` — Confidence adjuster, historical matching, explanation generation
- ``api``            — FastAPI REST endpoints
- ``dashboard``      — Streamlit interactive dashboard
- ``alerter``        — Telegram alert bot
- ``scheduler``      — APScheduler periodic pipeline
"""

from __future__ import annotations

from rag_geopolitik import config, constants, schemas
from rag_geopolitik.collector import NewsCrawler, RedisDedup
from rag_geopolitik.engine import Ingestor, WeightedRetriever
from rag_geopolitik.extraction import EventExtractor
from rag_geopolitik.features import FeatureBuilder, FeatureStore
from rag_geopolitik.models import FlowModel, StockModel
from rag_geopolitik.recommendation import Recommender
from rag_geopolitik.api import create_app
from rag_geopolitik.dashboard import run_dashboard
from rag_geopolitik.alerter import AlertBot
from rag_geopolitik.scheduler import PipelineScheduler

__version__ = "0.1.0"

__all__ = [
    # Config / domain
    "config",
    "constants",
    "schemas",
    # Collector
    "NewsCrawler",
    "RedisDedup",
    # RAG
    "Ingestor",
    "WeightedRetriever",
    # Extraction
    "EventExtractor",
    # Features
    "FeatureBuilder",
    "FeatureStore",
    # Models
    "FlowModel",
    "StockModel",
    # Recommendation
    "Recommender",
    # API / Dashboard / Alert
    "create_app",
    "run_dashboard",
    "AlertBot",
    # Pipeline
    "PipelineScheduler",
]