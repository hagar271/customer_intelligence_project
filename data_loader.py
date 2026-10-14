"""Loading of the files produced by the team. Nothing is trained or recomputed here,
except simple aggregations (sums / counts) of the cleaned transactions for the charts."""
import json
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

import aggregates
import config


class ArtifactError(Exception):
    """A required file is missing or does not have the expected columns."""


# ---------------------------------------------------------------- paths
def path_of(key: str):
    return config.ARTIFACTS_DIR / config.ARTIFACTS[key][0]


def exists(key: str) -> bool:
    return path_of(key).exists()


def _mtime(key: str) -> float:
    # Part of the cache key -> if a teammate replaces a file, the app reloads it automatically.
    return path_of(key).stat().st_mtime if exists(key) else 0.0


def require(key: str):
    if not exists(key):
        raise ArtifactError(
            f"Missing file **{config.ARTIFACTS[key][0]}** "
            f"(produced by: {config.ARTIFACTS[key][1]}). "
            f"Copy it into `{config.ARTIFACTS_DIR}`."
        )
    return path_of(key)


def validate_columns(df: pd.DataFrame, columns: list, file_label: str):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ArtifactError(f"`{file_label}` is missing the column(s): {missing}. Found: {list(df.columns)}")


def artifact_status() -> pd.DataFrame:
    rows = []
    for key, (fname, owner, used_in, required) in config.ARTIFACTS.items():
        rows.append({
            "File": fname, "From": owner, "Used in": used_in,
            "Required": "yes" if required else "optional",
            "Found": "✅" if exists(key) else ("❌" if required else "➖"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- generic cached readers
@st.cache_data(show_spinner=False)
def _read_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _read_json(path: str, mtime: float) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def _read_joblib(path: str, mtime: float):
    return joblib.load(path)


def load_csv(key: str) -> pd.DataFrame:
    return _read_csv(str(require(key)), _mtime(key))


def load_optional_csv(key: str):
    return load_csv(key) if exists(key) else None


# ---------------------------------------------------------------- transactions -> small summary tables
@st.cache_data(show_spinner="Loading and summarising the transactions (first run only)…")
def _sales_aggregates(path: str, mtime: float) -> dict:
    df = pd.read_csv(path, usecols=lambda c: c in aggregates.SALES_COLUMNS, parse_dates=["InvoiceDate"],
                     dtype={"Invoice": str, "StockCode": str})
    return aggregates.compute_sales_aggregates(df)


@st.cache_data(show_spinner=False)
def _read_sales_summary(folder: str, mtime: float) -> dict:
    return aggregates.load_sales_summary(folder)


def load_sales_aggregates() -> dict:
    """Prefers the small sales_*.csv / sales_kpis.json files (made by the pipeline or notebook). If only the big
    cleaned CSV exists, it is summarised once and the small files are saved next to it for next time."""
    folder = config.ARTIFACTS_DIR
    if aggregates.summary_exists(folder):
        return _read_sales_summary(str(folder), _mtime("sales_summary"))
    if not exists("transactions"):
        raise ArtifactError(
            "No sales data found. Put the 4 small files (sales_kpis.json, sales_monthly.csv, sales_products.csv, "
            f"sales_countries.csv) or clean_online_retail.csv in `{folder}`, "
            "or build them on the **⚙️ Run Pipeline** page.")
    agg = _sales_aggregates(str(path_of("transactions")), _mtime("transactions"))
    try:
        aggregates.save_sales_summary(agg, folder)
    except OSError:
        pass                                       # read-only folder (e.g. a deployed app): just use it in memory
    return agg


def missing_required() -> list:
    """Names of the required files that are not there yet (sales data = either of the two sales files)."""
    missing = [v[0] for k, v in config.ARTIFACTS.items() if v[3] and not exists(k)]
    if not (aggregates.summary_exists(config.ARTIFACTS_DIR) or exists("transactions")):
        missing.append("sales_kpis.json + sales_*.csv (or clean_online_retail.csv)")
    return missing


# ---------------------------------------------------------------- customers / segments
def load_rfm() -> pd.DataFrame:
    df = load_csv("rfm")
    validate_columns(df, ["Customer ID"] + config.RFM_FEATURES + ["Cluster"], config.ARTIFACTS["rfm"][0])
    df = df.copy()
    df["Customer ID"] = df["Customer ID"].astype(int)
    df["Cluster"] = df["Cluster"].astype(int)
    return df


def load_scores() -> pd.DataFrame:
    df = load_csv("churn_scores")
    validate_columns(df, ["Customer ID", "ChurnProbability"], config.ARTIFACTS["churn_scores"][0])
    df = df.copy()
    df["Customer ID"] = df["Customer ID"].astype(int)
    return df


# ---------------------------------------------------------------- churn artifacts
def load_meta() -> dict:
    meta = dict(_read_json(str(require("churn_meta")), _mtime("churn_meta")))
    meta.setdefault("feature_columns", config.DEFAULT_FEATURE_COLUMNS)
    meta.setdefault("risk_thresholds", config.DEFAULT_RISK_THRESHOLDS)
    meta.setdefault("decision_threshold", config.DEFAULT_DECISION_THRESHOLD)
    meta.setdefault("horizon_days", 90)
    return meta


def load_churn_model():
    return _read_joblib(str(require("churn_model")), _mtime("churn_model"))


def load_kmeans():
    """Returns (kmeans, scaler) or (None, None) if the optional files are missing."""
    if exists("kmeans_model") and exists("kmeans_scaler"):
        return (_read_joblib(str(path_of("kmeans_model")), _mtime("kmeans_model")),
                _read_joblib(str(path_of("kmeans_scaler")), _mtime("kmeans_scaler")))
    return None, None


def load_segment_overrides() -> dict:
    p = config.ARTIFACTS_DIR / config.SEGMENT_OVERRIDE_FILE
    if p.exists():
        try:
            return {int(k): str(v) for k, v in _read_json(str(p), p.stat().st_mtime).items()}
        except (ValueError, json.JSONDecodeError):
            return {}
    return {}
