"""Centralised configuration and source-registry loading.

All runtime configuration is sourced from environment variables (see
``.env.example``) so that secrets never live in the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
SOURCES_REGISTRY_PATH = CONFIG_DIR / "sources.yaml"


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Google Gemini (free tier — uses Flash Lite)
    gemini_api_key: str = ""

    # Infrastructure
    qdrant_url: str = "http://localhost:6333"
    redis_url: str = "redis://localhost:6379/0"

    # Models (Gemini Flash Lite — completely free)
    embedding_model: str = "models/text-embedding-004"      # Free Google embedding
    llm_model: str = "gemini-2.0-flash-lite"                # Google Flash Lite (free)

    # RAG retrieval weights
    retrieval_similarity_weight: float = 0.7
    retrieval_credibility_weight: float = 0.3
    retrieval_max_age_days: int = 7

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Alert thresholds
    alert_min_outperform_prob: float = 0.75
    alert_min_confidence: float = 0.70


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance."""
    return Settings()


# --------------------------------------------------------------------------- #
# Source registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Source:
    """A single news source and its credibility weighting."""

    id: str
    name: str
    type: str
    language: str
    credibility_score: float
    notes: str = ""


@lru_cache(maxsize=1)
def load_source_registry(path: Path | None = None) -> dict[str, Source]:
    """Load the YAML source registry keyed by source id."""
    registry_path = path or SOURCES_REGISTRY_PATH
    with registry_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    return {
        entry["id"]: Source(
            id=entry["id"],
            name=entry["name"],
            type=entry["type"],
            language=entry["language"],
            credibility_score=float(entry["credibility_score"]),
            notes=entry.get("notes", ""),
        )
        for entry in raw["sources"]
    }