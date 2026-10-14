"""Central place for file names, constants and colours.

If a teammate renames a file, change it HERE only (nothing else in the app hard-codes file names).
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _pick_artifacts_dir() -> Path:
    """Where the result files live: CIS_ARTIFACTS_DIR, else the 'artifacts' folder, else next to app.py."""
    env = os.environ.get("CIS_ARTIFACTS_DIR")
    if env:
        return Path(env)
    for candidate in (BASE_DIR / "artifacts", BASE_DIR):
        if (candidate / "churn_model_meta.json").exists() or (candidate / "sales_kpis.json").exists():
            return candidate
    return BASE_DIR


ARTIFACTS_DIR = _pick_artifacts_dir()

# key -> (file name, who produces it, where it is used, required?)
ARTIFACTS = {
    "sales_summary":    ("sales_kpis.json",                     "Pipeline (small)",     "Overview, Sales Analysis (+3 sales_*.csv)", False),
    "transactions":     ("clean_online_retail.csv",             "Data Cleaning (big)",  "Only if sales_summary is missing",     False),
    "rfm":              ("rfm_kmeans_customers.csv",            "RFM + K-Means",        "Segmentation, Overview",               True),
    "kmeans_model":     ("kmeans_model.joblib",                 "K-Means",              "Customer Prediction (segment)",        False),
    "kmeans_scaler":    ("kmeans_scaler.joblib",                "K-Means",              "Customer Prediction (segment)",        False),
    "churn_model":      ("churn_model_pipeline.joblib",         "Churn model",          "Customer Prediction",                  True),
    "churn_meta":       ("churn_model_meta.json",               "Churn model",          "Feature order, thresholds, metrics",   True),
    "churn_scores":     ("churn_scores_current_customers.csv",  "Churn model",          "Customer Prediction, Overview",        True),
    "churn_comparison": ("churn_model_comparison.csv",          "Model evaluation",     "Churn Prediction",                     False),
    "churn_test_pred":  ("churn_test_predictions.csv",          "Model evaluation",     "Confusion matrix, ROC",                False),
    "churn_labels":     ("churn_label_summary.csv",             "Churn label",          "Churn rate / distribution",            False),
    "churn_importance": ("churn_feature_importance.csv",        "Model evaluation",     "Feature importance",                   False),
    "segment_names":    ("segment_names.json",                  "RFM + K-Means",        "Segment names on every page",          False),
}
SEGMENT_OVERRIDE_FILE = "segment_names.json"   # optional: {"0": "Champions", "1": "Lost", ...}

RFM_FEATURES = ["Recency", "Frequency", "Monetary"]

# Used only if churn_model_meta.json does not contain them (the notebook writes both).
DEFAULT_FEATURE_COLUMNS = [
    "Recency", "Frequency", "Monetary", "AvgOrderValue", "TotalItems", "UniqueProducts",
    "TenureDays", "AvgDaysBetweenPurchases", "Orders_Last90d", "IsOneTimeBuyer",
]
DEFAULT_RISK_THRESHOLDS = {"low_max": 0.4, "high_min": 0.7}   # Low < 0.4 <= Medium < 0.7 <= High
DEFAULT_DECISION_THRESHOLD = 0.5

RISK_LABELS = {"Low": "Low Risk", "Medium": "Medium Risk", "High": "High Risk"}
RISK_COLORS = {"Low": "#2E9E6B", "Medium": "#F2A541", "High": "#D64545"}
PRIMARY = "#2E86AB"
ACCENT = "#A23B72"
