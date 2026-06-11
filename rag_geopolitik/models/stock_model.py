"""Model B — Stock Outperform Probability (LightGBM).

Spec section 7.2: predicts per-ticker probability of outperforming IHSG
over a 5-day horizon.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rag_geopolitik.config import DATA_DIR
from rag_geopolitik.schemas import StockOpportunity

try:
    import lightgbm as lgb
    import joblib
except ImportError:
    lgb = None  # type: ignore[assignment]
    joblib = None  # type: ignore[assignment]


# Feature columns expected by the model (spec 7.2).
STOCK_FEATURES = [
    "predicted_flow_direction",
    "predicted_flow_confidence",
    "ihsg_return_1d",
    "ihsg_return_5d",
    "ticker_return_1d",
    "ticker_return_5d",
    "ticker_volume_ratio",
    "sector_exposure_score",
    "relevant_commodity_delta",
    "ticker_sentiment_score",
    "ticker_event_velocity",
    "novelty_score",
    "dxy_delta_5d",
    "vix_level",
]

_MODEL_DIR = DATA_DIR / "models"


class StockModel:
    """Per-ticker outperform classifier (Model B, LightGBM).

    Parameters
    ----------
    model_path : Path, optional
        Path to a pre-trained ``LGBMClassifier`` pickle.
    """

    def __init__(self, model_path: Path | str | None = None) -> None:
        self._model: Any = None
        if isinstance(model_path, str):
            model_path = Path(model_path)
        self._model_path = model_path or _MODEL_DIR / "stock_model.pkl"

        if lgb is not None:
            self._model = lgb.LGBMClassifier(
                n_estimators=500,
                num_leaves=31,
                learning_rate=0.03,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                min_child_samples=20,
                random_state=42,
            )

        if model_path and model_path.exists() and joblib is not None:
            self._model = joblib.load(model_path)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def is_trained(self) -> bool:
        """Return ``True`` if the model has been fitted."""
        if self._model is None:
            return False
        return hasattr(self._model, "classes_")

    def predict(
        self,
        ticker: str,
        features: dict[str, Any],
    ) -> StockOpportunity:
        """Predict the probability that *ticker* will outperform IHSG in 5 days.

        Parameters
        ----------
        ticker : str
            LQ45 ticker symbol.
        features : dict
            Must contain all keys in ``STOCK_FEATURES`` (or sensible defaults).

        Returns
        -------
        StockOpportunity
        """
        if self._model is None or not self.is_trained:
            return StockOpportunity(
                ticker=ticker,
                outperform_probability=0.0,
                confidence=0.0,
            )

        X = np.array([[features.get(f, 0.0) for f in STOCK_FEATURES]])
        proba = float(self._model.predict_proba(X)[0, 1])
        confidence = max(proba, 1 - proba)

        return StockOpportunity(
            ticker=ticker,
            outperform_probability=round(proba, 2),
            confidence=round(confidence, 2),
        )

    def predict_batch(
        self,
        ticker_features: dict[str, dict[str, Any]],
    ) -> list[StockOpportunity]:
        """Run prediction for multiple tickers.

        Parameters
        ----------
        ticker_features
            Mapping ``{ticker: feature_dict}``.

        Returns
        -------
        list[StockOpportunity]
        """
        return [
            self.predict(ticker, feats)
            for ticker, feats in ticker_features.items()
        ]

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        """Fit the LightGBM classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,) with binary labels 0 / 1
        eval_set
            Optional validation set for early stopping.
        """
        if self._model is None:
            return

        kwargs: dict[str, Any] = {"X": X, "y": y}
        if eval_set is not None and lgb is not None:
            kwargs["eval_set"] = eval_set
            kwargs["callbacks"] = [lgb.early_stopping(10)]

        self._model.fit(**kwargs)

    def save(self, path: Path | None = None) -> None:
        """Persist the trained model to disk."""
        dest = path or self._model_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if joblib is not None and self._model is not None:
            joblib.dump(self._model, dest)