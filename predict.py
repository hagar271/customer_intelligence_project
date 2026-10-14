"""Thin wrappers around the SAVED models. No training and no re-implemented preprocessing:
the imputer / log / scaler live INSIDE the saved churn pipeline."""
import numpy as np
import pandas as pd

import config


def risk_level(prob: float, low_max: float, high_min: float) -> str:
    """Low: p < low_max | Medium: low_max <= p < high_min | High: p >= high_min."""
    if prob < low_max:
        return "Low"
    if prob < high_min:
        return "Medium"
    return "High"


def predict_churn(model, features: pd.DataFrame, feature_columns: list) -> np.ndarray:
    """Churn probability. Columns are re-ordered to EXACTLY the training order."""
    missing = [c for c in feature_columns if c not in features.columns]
    if missing:
        raise ValueError(f"Missing feature column(s) for the churn model: {missing}")
    X = features[feature_columns].astype(float)          # exact order used in training
    return model.predict_proba(X)[:, 1]


def predict_cluster(kmeans, scaler, recency: float, frequency: float, monetary: float) -> int:
    """Same steps as the K-Means notebook: log1p -> scaler -> kmeans.predict."""
    X = pd.DataFrame([[recency, frequency, monetary]], columns=config.RFM_FEATURES)
    X_scaled = scaler.transform(np.log1p(X))
    return int(kmeans.predict(X_scaled)[0])


def add_risk_column(df: pd.DataFrame, low_max: float, high_min: float, prob_col: str = "ChurnProbability") -> pd.DataFrame:
    """Adds a 'Risk' column (Low / Medium / High) using the same thresholds as risk_level()."""
    out = df.copy()
    out["Risk"] = pd.cut(out[prob_col], bins=[-np.inf, low_max, high_min, np.inf],
                         labels=["Low", "Medium", "High"], right=False).astype(str)
    return out


def predict_clusters(kmeans, scaler, features: pd.DataFrame) -> np.ndarray:
    """Segment (cluster id) for many customers at once: log1p -> scaler -> kmeans.predict."""
    X = features[config.RFM_FEATURES].astype(float)
    return kmeans.predict(scaler.transform(np.log1p(X)))
