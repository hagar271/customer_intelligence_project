"""Page 5 - Predict churn from data YOU provide: type one customer, upload a file, or pick an existing customer.
No ML logic lives here: we only build the input rows and call the SAVED model."""
import io

import numpy as np
import pandas as pd
import streamlit as st

import charts, data_loader, pipeline, predict, segments
from utils import fmt_int, fmt_money, page_header, parse_customer_id, risk_badge, show_chart

MODES = ["✍️ Enter customer data", "📂 Upload a file (many customers)", "🔎 Existing customer"]

ACTIONS = {
    "Low": "Keep the customer engaged: regular newsletter, loyalty rewards.",
    "Medium": "Monitor closely: send a reminder or a personalised offer before the window ends.",
    "High": "Act now: win-back campaign (discount or personal outreach).",
}

# Ready-made examples so you can see the result immediately and then edit any value.
PRESETS = {
    "Loyal, active customer": dict(Recency=12, Frequency=14, Monetary=5200.0, TotalItems=3800, UniqueProducts=160,
                                   TenureDays=600, AvgDays=38.0, Orders90=3, OneDay=False),
    "Slipping away": dict(Recency=130, Frequency=5, Monetary=1100.0, TotalItems=650, UniqueProducts=70,
                          TenureDays=520, AvgDays=85.0, Orders90=0, OneDay=False),
    "New one-time buyer": dict(Recency=25, Frequency=1, Monetary=180.0, TotalItems=90, UniqueProducts=25,
                               TenureDays=25, AvgDays=0.0, Orders90=1, OneDay=True),
}


def render():
    page_header("🎯 Customer Prediction", "Enter your own data and get the churn risk instantly")
    meta = data_loader.load_meta()
    model = data_loader.load_churn_model()
    feature_cols = meta["feature_columns"]
    low, high = meta["risk_thresholds"]["low_max"], meta["risk_thresholds"]["high_min"]
    horizon = meta["horizon_days"]
    scores = data_loader.load_scores()
    rfm = data_loader.load_rfm()
    seg_map = segments.build_segment_map(rfm)

    st.caption(f"Churn probability = model score for the customer **not** buying in the next {horizon} days. "
               f"Risk levels: **Low** < {low:.0%} · **Medium** {low:.0%}–{high:.0%} · **High** ≥ {high:.0%}.")

    mode = st.radio("How do you want to provide the data?", MODES, horizontal=True)
    if mode == MODES[0]:
        _enter(scores, model, feature_cols, seg_map, low, high, horizon)
    elif mode == MODES[1]:
        _upload(model, feature_cols, seg_map, low, high, horizon)
    else:
        _existing(scores, model, feature_cols, seg_map, low, high, horizon)


# ------------------------------------------------------------------ 1) type one customer
def _enter(scores, model, feature_cols, seg_map, low, high, horizon):
    kmeans, scaler = data_loader.load_kmeans()
    med = scores[feature_cols].median(numeric_only=True)

    def typical(name, fallback):
        v = med.get(name, fallback)
        return fallback if pd.isna(v) else v

    presets = {"Typical customer (median of your data)": dict(
        Recency=int(typical("Recency", 60)), Frequency=max(1, int(typical("Frequency", 3))),
        Monetary=float(round(typical("Monetary", 800.0), 2)), TotalItems=max(1, int(typical("TotalItems", 300))),
        UniqueProducts=max(1, int(typical("UniqueProducts", 40))), TenureDays=int(typical("TenureDays", 250)),
        AvgDays=float(round(typical("AvgDaysBetweenPurchases", 60.0), 1)), Orders90=int(typical("Orders_Last90d", 0)),
        OneDay=False), **PRESETS}

    choice = st.selectbox("Start from an example (then change any value)", list(presets))
    v = presets[choice]
    key = lambda name: f"{name}__{choice}"      # new keys per example -> the form shows the example's values

    st.write("Describe the customer **as of today**. Average Order Value and the one-time-buyer flag are calculated for you.")
    with st.form("custom_customer"):
        c1, c2, c3 = st.columns(3)
        recency = c1.number_input("Recency (days since last purchase)", 0, 1000, v["Recency"], key=key("rec"),
                                  help="How many days ago the customer last bought something.")
        frequency = c2.number_input("Frequency (number of orders)", 1, 5000, v["Frequency"], key=key("freq"),
                                    help="How many separate orders (invoices) the customer has placed.")
        monetary = c3.number_input("Monetary (total spent, £)", 0.01, 10_000_000.0, v["Monetary"], key=key("mon"),
                                   help="Total money the customer has spent with you.")
        c4, c5, c6 = st.columns(3)
        items = c4.number_input("Total items purchased", 1, 10_000_000, v["TotalItems"], key=key("items"))
        unique_products = c5.number_input("Distinct products purchased", 1, 10_000, v["UniqueProducts"], key=key("uniq"))
        tenure = c6.number_input("Tenure (days since first purchase)", 0, 1000, v["TenureDays"], key=key("ten"),
                                 help="How long the customer has been buying from you.")
        c7, c8, c9 = st.columns(3)
        orders90 = c7.number_input("Orders in the last 90 days", 0, 5000, v["Orders90"], key=key("o90"))
        avg_days = c8.number_input("Average days between purchases", 0.0, 1000.0, v["AvgDays"], key=key("avg"))
        one_day = c9.checkbox("Bought on a single day only", value=v["OneDay"], key=key("one"),
                              help="Tick this if there are no repeat purchase days (then the average above is ignored).")
        submitted = st.form_submit_button("Predict churn", type="primary")

    if not submitted:
        st.info("Fill the form (or pick an example) and press **Predict churn**.")
        return

    one_time = bool(one_day or frequency == 1)
    errors = []
    if tenure < recency:
        errors.append("Tenure cannot be smaller than Recency (the first purchase cannot be after the last one).")
    if orders90 > frequency:
        errors.append("Orders in the last 90 days cannot exceed the total number of orders.")
    if recency >= 90 and orders90 > 0:
        errors.append("Recency is 90+ days, so there can be no orders in the last 90 days.")
    if items < unique_products:
        errors.append("Total items cannot be smaller than the number of distinct products.")
    if not one_time and avg_days > tenure:
        errors.append("Average days between purchases cannot be larger than the customer's tenure.")
    if errors:
        for e in errors:
            st.error(e)
        return

    values = {
        "Recency": recency, "Frequency": frequency, "Monetary": monetary,
        "AvgOrderValue": monetary / frequency, "TotalItems": items, "UniqueProducts": unique_products,
        "TenureDays": tenure, "AvgDaysBetweenPurchases": np.nan if one_time else avg_days,
        "Orders_Last90d": orders90, "IsOneTimeBuyer": int(one_time),
    }
    features = pd.DataFrame([values])
    try:
        prob = float(predict.predict_churn(model, features, feature_cols)[0])
    except ValueError as e:
        st.error(f"Cannot score this customer: {e}")
        return

    segment = None
    if kmeans is not None and scaler is not None:
        segment = seg_map.get(predict.predict_cluster(kmeans, scaler, recency, frequency, monetary))
    _show_result(prob, low, high, horizon, pd.Series(values), segment, features[feature_cols])


# ------------------------------------------------------------------ 2) upload a file with many customers
@st.cache_data(show_spinner="Cleaning the file and scoring the customers…")
def _score_file(name, data, _model, _kmeans, _scaler, feature_cols, low, high, seg_items):
    f = io.BytesIO(data)
    f.name = name
    df = pipeline.clean_transactions(pipeline.read_raw(f))
    if df.empty:
        raise ValueError("No valid purchases are left after cleaning. Check that this is a transactions file.")
    cutoff = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)       # "today" = day after the last purchase
    feats = pipeline.build_features(df, cutoff)                                # same function used in training
    out = feats.reset_index()
    out["ChurnProbability"] = predict.predict_churn(_model, feats, list(feature_cols))
    out = predict.add_risk_column(out, low, high)
    if _kmeans is not None and _scaler is not None:
        out["Segment"] = pd.Series(predict.predict_clusters(_kmeans, _scaler, feats)).map(dict(seg_items)).to_numpy()
    first = ["Customer ID", "ChurnProbability", "Risk"] + (["Segment"] if "Segment" in out.columns else [])
    return out[first + [c for c in out.columns if c not in first]].sort_values("ChurnProbability", ascending=False), cutoff


def _upload(model, feature_cols, seg_map, low, high, horizon):
    st.write("Upload a **transactions file** (CSV or Excel) for the customers you care about. "
             "The app builds each customer's features from their purchases, then scores them with the saved model.")
    with st.expander("What must the file contain?"):
        st.markdown(
            "- Columns like Online Retail II: `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, "
            "`Price` (or `UnitPrice`), `Customer ID` (or `CustomerID`), `Country`.\n"
            "- Include each customer's **full purchase history** — features such as tenure and average days between "
            "purchases need it.\n"
            "- \"Today\" is taken as the day after the last purchase date in the file.\n"
            "- Cancelled invoices (starting with `C`) and invalid rows are removed automatically."
        )
    file = st.file_uploader("Transactions file", type=["csv", "xlsx", "xls"], key="batch_file")
    if file is None:
        return

    kmeans, scaler = data_loader.load_kmeans()
    try:
        result, cutoff = _score_file(file.name, file.getvalue(), model, kmeans, scaler, tuple(feature_cols),
                                     low, high, tuple(seg_map.items()))
    except Exception as e:
        st.error(f"Could not score this file: {e}")
        return

    st.success(f"Scored {len(result):,} customers (reference date: {cutoff:%d %b %Y}).")
    m1, m2, m3 = st.columns(3)
    m1.metric("Customers", fmt_int(len(result)))
    m2.metric("Average churn score", f"{result['ChurnProbability'].mean():.1%}")
    m3.metric("High-risk customers", f"{(result['Risk'] == 'High').mean():.1%}")

    a, b = st.columns([2, 3])
    with a:
        show_chart(charts.risk_distribution(result["Risk"].value_counts()))
    with b:
        st.dataframe(result.round(3), hide_index=True)
    st.download_button("⬇️ Download the predictions (CSV)", result.round(4).to_csv(index=False).encode("utf-8"),
                       file_name="churn_predictions.csv", mime="text/csv")


# ------------------------------------------------------------------ 3) an existing customer
def _existing(scores, model, feature_cols, seg_map, low, high, horizon):
    examples = ", ".join(str(x) for x in scores["Customer ID"].sample(3, random_state=2))
    text = st.text_input("Customer ID", placeholder="e.g. 12347", key="pred_customer_id")
    if not text.strip():
        st.info(f"Enter a Customer ID. Examples: {examples}")
        return
    cid, error = parse_customer_id(text)
    if error:
        st.error(error)
        return
    row = scores[scores["Customer ID"] == cid]
    if row.empty:
        st.error(f"Customer **{cid}** was not found. Try one of: {examples}")
        return
    row = row.iloc[[0]]

    try:
        prob = float(predict.predict_churn(model, row, feature_cols)[0])
    except ValueError as e:
        st.error(f"Cannot score this customer: {e}")
        return
    stored = float(row["ChurnProbability"].iloc[0])
    if abs(prob - stored) > 1e-3:
        st.warning("The live prediction differs from the stored score file. The model file and the scores file "
                   "may come from different training runs – please re-export both from the same run.")

    cluster = row["Cluster"].iloc[0] if "Cluster" in row.columns else np.nan
    segment = seg_map.get(int(cluster)) if pd.notna(cluster) else None
    _show_result(prob, low, high, horizon, row.iloc[0], segment, row[feature_cols])


# ------------------------------------------------------------------ shared result card
def _show_result(prob, low, high, horizon, info, segment, features_df):
    level = predict.risk_level(prob, low, high)
    st.divider()
    left, right = st.columns([2, 3])
    with left:
        show_chart(charts.probability_gauge(prob, low, high))
    with right:
        st.markdown(f"### {prob:.1%} churn probability")
        st.markdown(risk_badge(level), unsafe_allow_html=True)
        st.write("")
        st.write(f"**Segment:** {segment if segment else 'n/a (K-Means files not available)'}")
        st.write(f"**Recommended action:** {ACTIONS[level]}")

    r1, r2, r3 = st.columns(3)
    r1.metric("Recency (days)", fmt_int(info["Recency"]))
    r2.metric("Frequency (orders)", fmt_int(info["Frequency"]))
    r3.metric("Monetary", fmt_money(info["Monetary"]))

    with st.expander("Exact model input (same columns and order as training)"):
        st.dataframe(features_df.T.rename(columns=lambda c: "Value"))
