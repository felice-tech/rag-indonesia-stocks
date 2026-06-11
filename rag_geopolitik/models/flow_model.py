"""Model A — Foreign Flow Direction Classifier (XGBoost).

Spec section 7.1: predicts net foreign buy/sell direction at market level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rag_geopolitik.config import DATA_DIR
from rag_geopolitik.schemas import FlowDirection, FlowPrediction

try:
    from xgboost import XGBClassifier
    import joblib
except ImportError:
    XGBClassifier = None  # type: ignore[assignment,misc]
    joblib = None  # type: ignore[assignment]


# Feature columns expected by the model (spec 7.1).
FLOW_FEATURES = [
    "dxy_delta_1d",
    "dxy_delta_5d",
    "us10y_yield_delta",
    "vix_level",
    "vix_delta_1d",
    "em_etf_flow",
    "jisdor_delta",
    "ihsg_return_3d",
    "source_weighted_sentiment",
    "geopolitical_risk_index",
    "idx_foreign_net_buy_5d",
]

_MODEL_DIR = DATA_DIR / "models"


class FlowModel:
    """Foreign-flow direction classifier (Model A).

    Parameters
    ----------
    model_path : Path, optional
        Path to a pre-trained ``XGBClassifier`` pickle.  If ``None`` a fresh
        untrained model is used.
    """

    def __init__(self, model_path: Path | str | None = None) -> None:
        self._model: Any = None
        # Accept both Path and str
        if isinstance(model_path, str):
            model_path = Path(model_path)
        self._model_path = model_path or _MODEL_DIR / "flow_model.pkl"

        if XGBClassifier is not None:
            self._model = XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="mlogloss",
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

    def predict(self, features: dict[str, Any]) -> FlowPrediction:
        """Predict foreign-flow direction from a feature dictionary.

        Parameters
        ----------
        features
            Must contain all keys in ``FLOW_FEATURES``.

        Returns
        -------
        FlowPrediction
        """
        if self._model is None or not self.is_trained:
            return FlowPrediction(
                direction=FlowDirection.NEUTRAL,
                confidence=0.0,
                estimated_value=None,
                driving_factors=[],
            )

        X = np.array([[features.get(f, 0.0) for f in FLOW_FEATURES]])
        pred_mapped = int(self._model.predict(X)[0])
        # Remap back: 0 -> -1, 1 -> 0, 2 -> 1
        _REMAP = {0: -1, 1: 0, 2: 1}
        pred = _REMAP[pred_mapped]
        proba = self._model.predict_proba(X)[0].tolist()
        confidence = max(proba)

        direction = FlowDirection.from_label(pred)

        # Simple driving factor identification: feature with largest absolute
        # contribution (approximated by SHAP later; here we use raw correlation)
        driving = self._top_driving(features)

        return FlowPrediction(
            direction=direction,
            confidence=round(confidence, 2),
            estimated_value=self._estimate_value(direction),
            driving_factors=driving,
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        """Fit the XGBoost classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,) with labels -1, 0, 1
        eval_set
            Optional validation set for early stopping.
        """
        if self._model is None:
            return
        # XGBoost requires labels starting at 0; remap -1 -> 0, 0 -> 1, 1 -> 2
        y_mapped = y.copy()
        y_mapped[y == -1] = 0
        y_mapped[y == 0] = 1
        y_mapped[y == 1] = 2

        eval_set_mapped = None
        if eval_set:
            eval_set_mapped = [
                (X_val, (y_val.copy() if y_val is not None else None))
                for X_val, y_val in eval_set
            ]
            for i, (_, y_val) in enumerate(eval_set):
                if y_val is not None:
                    yv = eval_set_mapped[i][1].copy()
                    yv[y_val == -1] = 0
                    yv[y_val == 0] = 1
                    yv[y_val == 1] = 2
                    eval_set_mapped[i] = (X_val, yv)

        self._model.fit(X, y_mapped, eval_set=eval_set_mapped)

    def save(self, path: Path | None = None) -> None:
        """Persist the trained model to disk."""
        dest = path or self._model_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if joblib is not None and self._model is not None:
            joblib.dump(self._model, dest)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _top_driving(features: dict[str, Any], top_n: int = 3) -> list[str]:
        """Return feature names with the largest absolute value as a heuristic."""
        sorted_feats = sorted(
            features.items(), key=lambda kv: abs(kv[1] if isinstance(kv[1], (int, float)) else 0), reverse=True
        )
        return [f"{k}: {v:+.2f}" for k, v in sorted_feats[:top_n]]

    @staticmethod
    def _estimate_value(direction: FlowDirection) -> str | None:
        mapping = {
            FlowDirection.NET_BUY: "+1.2T",
            FlowDirection.NET_SELL: "-800B",
            FlowDirection.NEUTRAL: "0",
        }
        return mapping.get(direction)