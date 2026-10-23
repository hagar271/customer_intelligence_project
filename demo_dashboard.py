"""Demo dashboard shown when generated artifacts are absent.

All numbers in this page are illustrative placeholders for presentation only.
They must never be interpreted as measured results from the Online Retail II dataset.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def _style(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=55, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=-0.28))
    return fig


def render():
    st.title("📊 Customer Intelligence System")
    st.caption("Online Retail II · Sales Analytics · Customer Segmentation · Churn Intelligence")
    st.info(
        "DEMO PREVIEW — These are illustrative sample values to demonstrate the dashboard layout. "
        "They are not calculated from the original dataset, and the prediction visuals are not live model outputs."
    )

    # Illustrative sample data, generated in code; no uploaded dataset or artifacts are needed.
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", "£1.24M", help="Illustrative demo value")
    c2.metric("Total Orders", "18,420", help="Illustrative demo value")
    c3.metric("Total Customers", "4,317", help="Illustrative demo value")
    c4.metric("Average Order Value", "£67.32", help="Illustrative demo value")

    months = pd.date_range("2010-12-01", periods=13, freq="MS")
    revenue = [52, 48, 71, 65, 88, 79, 102, 96, 118, 111, 139, 154, 132]
    orders = [820, 770, 1080, 990, 1310, 1200, 1490, 1430, 1680, 1590, 1900, 2130, 1810]
    monthly = pd.DataFrame({"Month": months, "Revenue (£000)": revenue, "Orders": orders})

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Sales performance")
        fig = px.line(monthly, x="Month", y="Revenue (£000)", markers=True,
                      title="Monthly revenue trend (illustrative)")
        st.plotly_chart(_style(fig), use_container_width=True)
    with right:
        st.subheader("Revenue by market")
        countries = pd.DataFrame({
            "Country": ["United Kingdom", "Germany", "France", "EIRE", "Other"],
            "Revenue (£000)": [780, 125, 104, 82, 149],
        })
        fig = px.pie(countries, names="Country", values="Revenue (£000)", hole=0.55,
                     title="Illustrative revenue share")
        st.plotly_chart(_style(fig), use_container_width=True)

    st.divider()
    st.header("🧩 Customer Segmentation")
    seg1, seg2 = st.columns([2, 3])
    segment_counts = pd.DataFrame({
        "Segment": ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Hibernating"],
        "Customers": [610, 930, 785, 1020, 972],
    })
    with seg1:
        st.metric("Example segments", "5")
        st.dataframe(segment_counts, hide_index=True, use_container_width=True)
    with seg2:
        fig = px.bar(segment_counts, x="Segment", y="Customers", color="Segment",
                     title="Illustrative customer distribution")
        fig.update_layout(showlegend=False)
        st.plotly_chart(_style(fig), use_container_width=True)

    st.divider()
    st.header("🔮 Churn Risk & Model Evaluation")
    k1, k2, k3 = st.columns(3)
    k1.metric("Illustrative churn rate", "24.8%")
    k2.metric("Example model F1", "0.78")
    k3.metric("Example ROC-AUC", "0.84")
    risk = pd.DataFrame({"Risk level": ["Low", "Medium", "High"], "Customers": [2100, 1240, 977]})
    models = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
        "F1 score": [0.70, 0.73, 0.78],
        "ROC-AUC": [0.77, 0.79, 0.84],
    })
    left, right = st.columns(2)
    with left:
        fig = px.bar(risk, x="Risk level", y="Customers", color="Risk level",
                     title="Illustrative churn-risk distribution")
        st.plotly_chart(_style(fig), use_container_width=True)
    with right:
        fig = px.bar(models, x="Model", y=["F1 score", "ROC-AUC"], barmode="group",
                     title="Illustrative model comparison")
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(_style(fig), use_container_width=True)

    st.divider()
    st.header("⚙️ Machine Learning Workflow")
    steps = [
        ("01 · Data Cleaning", "Handle missing values, duplicates, cancellations and returns."),
        ("02 · EDA & Sales KPIs", "Explore revenue, orders, products, markets and monthly trends."),
        ("03 · RFM Analysis", "Calculate Recency, Frequency and Monetary value per customer."),
        ("04 · K-Means Clustering", "Group customers into behavior-based segments."),
        ("05 · Churn Prediction", "Create time-aware churn labels and train classification models."),
        ("06 · Evaluation", "Compare models using F1, Recall, ROC-AUC and confusion matrices."),
    ]
    for title, detail in steps:
        with st.expander(title):
            st.write(detail)

    st.caption(
        "To show the project's actual computed results and run real predictions, the generated summary files "
        "and trained model artifacts must be made available to the app. The original Excel file itself is not "
        "needed in the deployed repository for this preview."
    )
