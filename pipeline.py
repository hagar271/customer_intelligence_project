"""One-click pipeline: raw Online Retail II  ->  every file the dashboard needs.

This is the SAME logic as the notebook (Cleaning -> RFM -> K-Means -> Churn label -> models),
packaged as functions so it can be launched from the 'Run Pipeline' page.
Only this module trains models. The dashboard pages just load what it saves.
"""
import json
import math
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

import aggregates
import config

RANDOM_STATE = 42
FEATURE_COLS = config.DEFAULT_FEATURE_COLUMNS
NON_PRODUCT_CODES = ["POST", "DOT", "M", "BANK CHARGES", "AMAZONFEE", "CRUK", "C2",
                     "ADJUST", "ADJUST2", "TEST001", "TEST002", "SAMPLES", "B", "D", "S"]
BASELINE = "Baseline (Recency only)"
RISK_LOW_MAX, RISK_HIGH_MIN = 0.4, 0.7
THRESHOLD = 0.5

REQUIRED_RAW_COLUMNS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]


def _noop(fraction: float, message: str):
    pass


# =====================================================================  1) read + clean
def read_raw(file) -> pd.DataFrame:
    """Reads the uploaded .xlsx (all sheets are combined) or .csv."""
    name = getattr(file, "name", str(file)).lower()
    if name.endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(file, sheet_name=None)
        return pd.concat(sheets.values(), ignore_index=True)
    return pd.read_csv(file)


def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns={"CustomerID": "Customer ID", "UnitPrice": "Price", "InvoiceNo": "Invoice"})
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"The file is missing the column(s) {missing}. Found: {list(df.columns)}")

    df = df.drop_duplicates()
    df = df.dropna(subset=["Description", "Customer ID"]).copy()
    df["Customer ID"] = df["Customer ID"].astype(int)

    df = df[~df["Invoice"].astype(str).str.startswith("C")]          # cancelled invoices
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)].copy()

    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    non_product = df["StockCode"].isin(NON_PRODUCT_CODES) | df["StockCode"].str.startswith("GIFT_")
    df = df[~non_product].copy()

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Invoice"] = df["Invoice"].astype(str)
    df["TotalPrice"] = df["Quantity"] * df["Price"]
    df["Country"] = df["Country"].str.strip().replace({"Unspecified": "Unknown", "EIRE": "Ireland", "RSA": "South Africa"})
    return df.reset_index(drop=True)


# =====================================================================  2) RFM + K-Means
def _quartile_score(series: pd.Series, reverse: bool = False) -> pd.Series:
    pct = series.rank(pct=True, method="average")
    if reverse:
        pct = 1 - pct + (1 / len(series))
    return pd.cut(pct, bins=[0, 0.25, 0.5, 0.75, 1.0], labels=[1, 2, 3, 4], include_lowest=True).astype(int)


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("Customer ID").agg(
        LastPurchase=("InvoiceDate", "max"), FirstPurchase=("InvoiceDate", "min"),
        Frequency=("Invoice", "nunique"), Monetary=("TotalPrice", "sum")).reset_index()
    rfm["Recency"] = (snapshot - rfm["LastPurchase"]).dt.days
    rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"]
    rfm["TenureDays"] = (rfm["LastPurchase"] - rfm["FirstPurchase"]).dt.days
    rfm = rfm[["Customer ID", "Recency", "Frequency", "Monetary", "AvgOrderValue", "TenureDays"]]
    rfm["R_Score"] = _quartile_score(rfm["Recency"], reverse=True)
    rfm["F_Score"] = _quartile_score(rfm["Frequency"])
    rfm["M_Score"] = _quartile_score(rfm["Monetary"])
    rfm["RFM_Total"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]
    rfm["RFM_Segment"] = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)
    return rfm


def _name_segment(row) -> str:
    recent = row["Recency"] < 0
    good = int(row["Frequency"] > 0) + int(row["Monetary"] > 0)
    if recent and good == 2:
        return "Champions"
    if recent and good == 1:
        return "Potential Loyalists"
    if recent:
        return "New / Promising"
    if good >= 1:
        return "At Risk"
    return "Lost / Hibernating"


def run_kmeans(rfm: pd.DataFrame):
    X_log = np.log1p(rfm[config.RFM_FEATURES])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    results = []
    for k in range(2, 9):
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(X_scaled)
        results.append((k, silhouette_score(X_scaled, labels)))
    best_k = max((r for r in results if r[0] >= 3), key=lambda r: r[1])[0]

    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    out = rfm.copy()
    out["Cluster"] = kmeans.fit_predict(X_scaled)

    z = (X_log.groupby(out["Cluster"]).mean() - X_log.mean()) / X_log.std()
    names = {int(c): _name_segment(r) for c, r in z.iterrows()}
    counts = Counter(names.values())
    names = {c: (f"{n} (Cluster {c})" if counts[n] > 1 else n) for c, n in names.items()}
    out["Segment"] = out["Cluster"].map(names)
    return out, kmeans, scaler, names, best_k


# =====================================================================  3) churn: features, labels, models
def build_features(tx: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    hist = tx[tx["InvoiceDate"] < cutoff]
    feats = hist.groupby("Customer ID").agg(
        LastPurchase=("InvoiceDate", "max"), FirstPurchase=("InvoiceDate", "min"),
        Frequency=("Invoice", "nunique"), Monetary=("TotalPrice", "sum"),
        TotalItems=("Quantity", "sum"), UniqueProducts=("StockCode", "nunique"))
    feats["Recency"] = (cutoff - feats["LastPurchase"]).dt.days
    feats["TenureDays"] = (cutoff - feats["FirstPurchase"]).dt.days
    feats["AvgOrderValue"] = feats["Monetary"] / feats["Frequency"]

    days = hist[["Customer ID", "InvoiceDate"]].copy()
    days["Date"] = days["InvoiceDate"].dt.normalize()
    days = days.drop_duplicates(["Customer ID", "Date"]).sort_values(["Customer ID", "Date"])
    days["gap"] = days.groupby("Customer ID")["Date"].diff().dt.days
    feats["AvgDaysBetweenPurchases"] = days.groupby("Customer ID")["gap"].mean()
    feats["IsOneTimeBuyer"] = feats["AvgDaysBetweenPurchases"].isna().astype(int)

    recent = hist[hist["InvoiceDate"] >= cutoff - pd.Timedelta(days=90)]
    feats["Orders_Last90d"] = recent.groupby("Customer ID")["Invoice"].nunique()
    feats["Orders_Last90d"] = feats["Orders_Last90d"].fillna(0)
    return feats[FEATURE_COLS]


def build_label(tx: pd.DataFrame, customer_ids, cutoff: pd.Timestamp, horizon_days: int) -> pd.Series:
    end = cutoff + pd.Timedelta(days=horizon_days)
    future = tx[(tx["InvoiceDate"] >= cutoff) & (tx["InvoiceDate"] < end)]
    returned = set(future["Customer ID"].unique())
    return pd.Series((~pd.Index(customer_ids).isin(returned)).astype(int), index=customer_ids, name="Churn")


def _make_pipe(model, scale=False):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps += [("log1p", FunctionTransformer(np.log1p)), ("scaler", StandardScaler())]
    return Pipeline(steps + [("model", model)])


def _candidates(fast: bool, use_xgb: bool) -> dict:
    grids = {
        "lr": {"model__C": [0.1, 1]} if fast else {"model__C": [0.01, 0.1, 1, 10]},
        "dt": ({"model__max_depth": [3, 5, 7], "model__min_samples_leaf": [30]} if fast
               else {"model__max_depth": [3, 5, 7, 10], "model__min_samples_leaf": [10, 30, 50]}),
        "rf": ({"model__n_estimators": [200], "model__max_depth": [8, None], "model__min_samples_leaf": [10]} if fast
               else {"model__n_estimators": [200, 400], "model__max_depth": [6, 10, None], "model__min_samples_leaf": [5, 20]}),
        "xgb": ({"model__n_estimators": [200], "model__max_depth": [3, 5], "model__learning_rate": [0.1]} if fast
                else {"model__n_estimators": [200, 400], "model__max_depth": [3, 5], "model__learning_rate": [0.05, 0.1]}),
    }
    cands = {
        "Logistic Regression": (_make_pipe(LogisticRegression(max_iter=2000, class_weight="balanced",
                                                              random_state=RANDOM_STATE), scale=True), grids["lr"]),
        "Decision Tree": (_make_pipe(DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)), grids["dt"]),
        "Random Forest": (_make_pipe(RandomForestClassifier(class_weight="balanced_subsample",
                                                            random_state=RANDOM_STATE, n_jobs=1)), grids["rf"]),
    }
    if use_xgb:
        try:
            from xgboost import XGBClassifier
            cands["XGBoost"] = (_make_pipe(XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1)), grids["xgb"])
        except ImportError:
            pass
    return cands


# =====================================================================  4) the whole run
def run_pipeline(source, out_dir, horizon_days: int = 90, n_train_snapshots: int = 4, snapshot_step_days: int = 30,
                 fast: bool = True, use_xgb: bool = True, progress=_noop, save_clean_csv: bool = False) -> dict:
    """Runs everything and writes all dashboard files into `out_dir`. Returns a small summary dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    progress(0.02, "Reading the uploaded file (this can take a minute for the Excel file)…")
    raw = source if isinstance(source, pd.DataFrame) else read_raw(source)

    progress(0.15, "Cleaning the transactions…")
    df = clean_transactions(raw)
    del raw
    if df.empty:
        raise ValueError("No rows left after cleaning – check that this is the Online Retail II file.")
    if save_clean_csv:                                                      # big file, not needed by the dashboard
        df.to_csv(out / config.ARTIFACTS["transactions"][0], index=False)
    aggregates.save_sales_summary(aggregates.compute_sales_aggregates(df), out)

    progress(0.25, "Building RFM table and K-Means segments…")
    rfm = build_rfm(df)
    rfm_km, kmeans, km_scaler, seg_names, best_k = run_kmeans(rfm)
    rfm_km.to_csv(out / config.ARTIFACTS["rfm"][0], index=False)
    joblib.dump(kmeans, out / config.ARTIFACTS["kmeans_model"][0], compress=3)
    joblib.dump(km_scaler, out / config.ARTIFACTS["kmeans_scaler"][0])
    with open(out / config.ARTIFACTS["segment_names"][0], "w") as f:
        json.dump({str(k): v for k, v in seg_names.items()}, f, indent=2)

    progress(0.40, "Building churn labels (temporal split, no leakage)…")
    data_start = df["InvoiceDate"].min()
    data_end = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    test_cutoff = data_end - pd.Timedelta(days=horizon_days)
    latest_train = test_cutoff - pd.Timedelta(days=horizon_days)
    train_cutoffs = sorted(latest_train - pd.Timedelta(days=snapshot_step_days * i) for i in range(n_train_snapshots))
    if (train_cutoffs[0] - data_start).days < 120:
        raise ValueError("Not enough history for this setting. Use fewer training snapshots or a shorter horizon.")

    X_parts, y_parts, id_parts = [], [], []
    for c in train_cutoffs:
        Xc = build_features(df, c)
        X_parts.append(Xc)
        y_parts.append(build_label(df, Xc.index, c, horizon_days))
        id_parts.append(Xc.index.to_numpy())
    X_train = pd.concat(X_parts, ignore_index=True)
    y_train = pd.concat(y_parts, ignore_index=True)
    groups = np.concatenate(id_parts)
    X_test = build_features(df, test_cutoff)
    y_test = build_label(df, X_test.index, test_cutoff, horizon_days)

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cands = _candidates(fast, use_xgb)
    fitted = {}
    for i, (name, (pipe, grid)) in enumerate(cands.items()):
        progress(0.45 + 0.35 * i / len(cands), f"Training and tuning: {name}…")
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1)
        gs.fit(X_train, y_train, groups=groups)
        fitted[name] = gs.best_estimator_

    baseline = Pipeline([
        ("select", ColumnTransformer([("recency", "passthrough", ["Recency"])])),
        ("log1p", FunctionTransformer(np.log1p)), ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000))]).fit(X_train, y_train)
    all_models = {BASELINE: baseline, **fitted}

    progress(0.82, "Evaluating on the test period…")
    rows, probas = [], {}
    for name, model in all_models.items():
        p_te = model.predict_proba(X_test)[:, 1]
        pred_te = (p_te >= THRESHOLD).astype(int)
        pred_tr = (model.predict_proba(X_train)[:, 1] >= THRESHOLD).astype(int)
        probas[name] = p_te
        rows.append({"Model": name, "Accuracy": accuracy_score(y_test, pred_te),
                     "Precision": precision_score(y_test, pred_te, zero_division=0),
                     "Recall": recall_score(y_test, pred_te), "F1": f1_score(y_test, pred_te),
                     "ROC-AUC": roc_auc_score(y_test, p_te), "Train F1": f1_score(y_train, pred_tr)})
    comparison = pd.DataFrame(rows).set_index("Model")
    comparison["F1 gap (train-test)"] = comparison["Train F1"] - comparison["F1"]

    ranking = comparison.drop(index=BASELINE).sort_values(["F1", "ROC-AUC"], ascending=False)
    best_name = ranking.index[0]
    best_model = fitted[best_name]
    est = best_model.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        importance, imp_type = pd.Series(est.feature_importances_, index=FEATURE_COLS), "feature_importance"
    else:
        importance, imp_type = pd.Series(np.abs(est.coef_[0]), index=FEATURE_COLS), "abs_coefficient"

    progress(0.90, "Saving models and result files…")
    joblib.dump(best_model, out / config.ARTIFACTS["churn_model"][0], compress=3)
    metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    meta = {
        "best_model": best_name, "feature_columns": FEATURE_COLS,
        "target": "Churn (1 = no purchase in the next HORIZON_DAYS days)",
        "horizon_days": horizon_days, "decision_threshold": THRESHOLD,
        "train_cutoff": str(train_cutoffs[-1].date()), "train_cutoffs": [str(c.date()) for c in train_cutoffs],
        "test_cutoff": str(test_cutoff.date()),
        "risk_thresholds": {"low_max": RISK_LOW_MAX, "high_min": RISK_HIGH_MIN},
        "sklearn_version": sklearn.__version__,
        "test_metrics": comparison.loc[best_name, metric_cols].round(4).to_dict(),
        "baseline_metrics": comparison.loc[BASELINE, metric_cols].round(4).to_dict(),
        "brier_score": round(float(brier_score_loss(y_test, probas[best_name])), 4),
    }
    with open(out / config.ARTIFACTS["churn_meta"][0], "w") as f:
        json.dump(meta, f, indent=2)
    comparison.round(4).to_csv(out / config.ARTIFACTS["churn_comparison"][0])

    test_pred = pd.DataFrame({"Customer ID": X_test.index, "Churn_True": y_test.values})
    for name, p in probas.items():
        test_pred[f"proba__{name}"] = p
    test_pred.to_csv(out / config.ARTIFACTS["churn_test_pred"][0], index=False)

    label_summary = pd.DataFrame({
        "Set": ["Train", "Test"],
        "Cutoff": [f"{train_cutoffs[0].date()} to {train_cutoffs[-1].date()} ({len(train_cutoffs)} snapshots)", str(test_cutoff.date())],
        "Customers": [len(y_train), len(y_test)], "Churned": [int(y_train.sum()), int(y_test.sum())]})
    label_summary["ChurnRate"] = label_summary["Churned"] / label_summary["Customers"]
    label_summary.to_csv(out / config.ARTIFACTS["churn_labels"][0], index=False)
    (importance.rename("Importance").rename_axis("Feature").reset_index().assign(Type=imp_type)
        .to_csv(out / config.ARTIFACTS["churn_importance"][0], index=False))

    progress(0.95, "Scoring all current customers…")
    X_now = build_features(df, data_end)
    scores = pd.DataFrame({"Customer ID": X_now.index, "ChurnProbability": best_model.predict_proba(X_now)[:, 1]})
    scores["RiskLevel"] = pd.cut(scores["ChurnProbability"], bins=[0, RISK_LOW_MAX, RISK_HIGH_MIN, 1.0001],
                                 labels=["Low", "Medium", "High"], right=False)
    scores = scores.merge(rfm_km[["Customer ID", "Cluster", "Segment"]], on="Customer ID", how="left")
    scores = scores.merge(X_now.reset_index(), on="Customer ID", how="left")
    scores.to_csv(out / config.ARTIFACTS["churn_scores"][0], index=False)

    progress(1.0, "Done ✅")
    return {"rows_clean": len(df), "customers": int(rfm.shape[0]), "n_clusters": int(best_k),
            "best_model": best_name, "test_metrics": meta["test_metrics"], "baseline_metrics": meta["baseline_metrics"],
            "train_churn_rate": float(y_train.mean()), "test_churn_rate": float(y_test.mean())}
