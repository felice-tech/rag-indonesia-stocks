"""FastAPI application with endpoints for predictions, features, and explanations.

Spec section 9.1 — endpoints mirroring the API spec.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_geopolitik.features.store import FeatureStore
from rag_geopolitik.models.flow_model import FlowModel
from rag_geopolitik.models.stock_model import StockModel
from rag_geopolitik.recommendation.recommender import Recommender
from rag_geopolitik.schemas import (
    FlowPrediction,
    RawArticle,
    StockOpportunity,
)


class HealthResponse(BaseModel):
    status: str = "ok"
    models_trained: bool = False


class DailyPredictionResponse(BaseModel):
    date: str
    flow_prediction: FlowPrediction | None = None
    opportunities: list[StockOpportunity] = []


def create_app(
    flow_model: FlowModel | None = None,
    stock_model: StockModel | None = None,
    recommender: Recommender | None = None,
    feature_store: FeatureStore | None = None,
) -> FastAPI:
    """Factory that wires together the FastAPI application.

    Parameters
    ----------
    flow_model : FlowModel, optional
    stock_model : StockModel, optional
    recommender : Recommender, optional
    feature_store : FeatureStore, optional
    """
    app = FastAPI(
        title="RAG Geopolitik & Investasi API",
        description="LQ45 foreign-flow prediction and stock opportunity API",
        version="0.1.0",
    )

    # Store dependencies in app state so endpoints can access them.
    app.state.flow_model = flow_model
    app.state.stock_model = stock_model
    app.state.recommender = recommender
    app.state.feature_store = feature_store

    # ------------------------------------------------------------------ #
    # Endpoints
    # ------------------------------------------------------------------ #
    @app.get("/health", response_model=HealthResponse, tags=["System"])
    def health():
        models_trained = (
            (flow_model is not None and flow_model.is_trained)
            or (stock_model is not None and stock_model.is_trained)
        )
        return HealthResponse(models_trained=models_trained)

    @app.get(
        "/api/v1/prediction/daily",
        response_model=DailyPredictionResponse,
        tags=["Prediction"],
    )
    def daily_prediction():
        """Return all predictions for today."""
        today_str = date.today().isoformat()
        flow = flow_model.predict({}) if flow_model is not None else None
        return DailyPredictionResponse(
            date=today_str,
            flow_prediction=flow,
        )

    @app.get(
        "/api/v1/prediction/ticker/{ticker}",
        response_model=StockOpportunity,
        tags=["Prediction"],
    )
    def ticker_prediction(ticker: str):
        """Detail prediction for a single ticker."""
        ticker_upper = ticker.upper()
        if recommender is None:
            raise HTTPException(status_code=503, detail="Recommender not available")

        try:
            opp = recommender.recommend(
                ticker=ticker_upper,
                ticker_features={},
                events=[],
            )
            return opp
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get(
        "/api/v1/foreign-flow/prediction",
        response_model=FlowPrediction,
        tags=["Foreign Flow"],
    )
    def foreign_flow_prediction():
        """Market-level foreign-flow prediction."""
        if flow_model is None:
            raise HTTPException(status_code=503, detail="Flow model not available")
        return flow_model.predict({})

    @app.post(
        "/api/v1/news/ingest",
        status_code=202,
        tags=["News"],
    )
    def ingest_article(article: RawArticle):
        """Manual ingest endpoint for testing."""
        # In a real setup this would queue the article for ingestion.
        return {"ingested": article.id}

    @app.get(
        "/api/v1/features/{ticker}/{date_str}",
        tags=["Features"],
    )
    def get_features(ticker: str, date_str: str):
        """View feature vector for a ticker on a given date."""
        if feature_store is None:
            raise HTTPException(status_code=503, detail="Feature store not available")
        try:
            dt = date.fromisoformat(date_str)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format")

        feats = feature_store.get_ticker_features(ticker.upper(), dt)
        if feats is None:
            raise HTTPException(status_code=404, detail="Features not found")
        return feats

    @app.get(
        "/api/v1/explanation/{ticker}",
        response_model=StockOpportunity,
        tags=["Explanation"],
    )
    def get_explanation(ticker: str):
        """Full explanation for a ticker."""
        return ticker_prediction(ticker)

    return app