"""Page 1 - Overview: business KPIs + a snapshot of the whole intelligence pipeline."""
import pandas as pd
import streamlit as st

import charts, config, data_loader, predict, segments
from utils import fmt_int, fmt_money, fmt_pct, page_header, pipeline_strip, show_chart


def render():
    page_header("📊 Customer Intelligence System",
                "Online Retail II · from raw transactions to segments and churn risk")
    pipeline_strip("Dashboard")

    agg = data_loader.load_sales_aggregates()
    k, monthly, countries = agg["kpis"], agg["monthly"], agg["countries"]

    # ---------------- business KPIs
    st.subheader("Business overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", fmt_money(k["revenue"]))
    c2.metric("Total Orders", fmt_int(k["orders"]))
    c3.metric("Total Customers", fmt_int(k["customers"]))
    c4.metric("Average Order Value", f"£{k['aov']:,.2f}")
    st.caption(f"Cleaned data from {k['date_min']:%d %b %Y} to {k['date_max']:%d %b %Y}.")

    left, right = st.columns([3, 2])
    with left:
        show_chart(charts.revenue_line(monthly))
        if k["partial_last_month"]:
            st.caption(f"⚠️ {k['last_month']:%b %Y} is an incomplete month (data ends {k['date_max']:%d %b}) — the drop at the end is expected.")
    with right:
        top5 = countries.head(5)[["Country", "Revenue"]].copy()
        other = countries["Revenue"].iloc[5:].sum()
        if other > 0:
            top5 = pd.concat([top5, pd.DataFrame([{"Country": "Other", "Revenue": other}])], ignore_index=True)
        show_chart(charts.donut(top5, "Country", "Revenue", "Revenue share by country"))

    # ---------------- intelligence snapshot (segments + churn)
    st.subheader("Customer intelligence snapshot")
    try:
        rfm = data_loader.load_rfm()
        seg_map = segments.build_segment_map(rfm)
        meta = data_loader.load_meta()
        scores = data_loader.load_scores()
    except data_loader.ArtifactError as e:
        st.info(f"Segments / churn results are not available yet. {e}")
        scores = None

    if scores is not None:
        low, high = meta["risk_thresholds"]["low_max"], meta["risk_thresholds"]["high_min"]
        scores = predict.add_risk_column(scores, low, high)
        high_mask = scores["Risk"] == "High"
        metrics = meta.get("test_metrics", {})

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Customer segments", fmt_int(rfm["Cluster"].nunique()))
        s2.metric("High-risk customers", fmt_pct(high_mask.mean()),
                  help=f"Churn probability ≥ {high:.0%} for the next {meta['horizon_days']} days.")
        if "Monetary" in scores.columns and scores["Monetary"].sum() > 0:
            s3.metric("Past revenue from high-risk customers",
                      fmt_pct(scores.loc[high_mask, "Monetary"].sum() / scores["Monetary"].sum()))
        else:
            s3.metric("Past revenue from high-risk customers", "n/a")
        s4.metric("Churn model", meta["best_model"].replace(" (optional)", ""),
                  delta=f"ROC-AUC {metrics['ROC-AUC']:.2f}" if "ROC-AUC" in metrics else None, delta_color="off")

        a, b = st.columns(2)
        with a:
            seg_counts = rfm["Cluster"].map(seg_map).value_counts().reset_index()
            seg_counts.columns = ["Segment", "Customers"]
            show_chart(charts.donut(seg_counts, "Segment", "Customers", "Customers by segment"))
        with b:
            show_chart(charts.risk_distribution(scores["Risk"].value_counts()))

    # ---------------- key findings (computed, not hard-coded)
    st.subheader("Key findings")
    peak = monthly.loc[monthly["Revenue"].idxmax()]
    top_country = countries.iloc[0]
    st.markdown(
        f"- **Peak month:** {peak['Month']:%B %Y} with {fmt_money(peak['Revenue'])} in revenue.\n"
        f"- **Market concentration:** {top_country['Country']} generates {fmt_pct(top_country['Revenue'] / k['revenue'])} of revenue.\n"
        f"- **Retention challenge:** {fmt_pct(k['one_time_buyers_pct'])} of customers placed only one order."
    )
    st.caption("Use the sidebar to explore sales, segments, the churn model and single-customer predictions.")
