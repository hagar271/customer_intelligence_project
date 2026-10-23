"""Artifact-free presentation dashboard.

All values are clearly labelled illustrative examples, not results computed from the dataset.
"""
import pandas as pd
import plotly.express as px
import streamlit as st


def _style(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=55, b=10))
    return fig


def render(page="📊 Overview"):
    st.title(page)
    st.caption("Customer Intelligence System · Online Retail II")
    st.info("DEMO PREVIEW — charts and numbers below are illustrative placeholders, not computed dataset results or live predictions.")

    if page == "📊 Overview":
        st.subheader("Business overview")
        cols = st.columns(4)
        for col, label, value in zip(cols, ["Total Revenue", "Total Orders", "Total Customers", "Average Order Value"],
                                     ["£1.24M", "18,420", "4,317", "£67.32"]):
            col.metric(label, value)
        monthly = pd.DataFrame({"Month": pd.date_range("2010-12-01", periods=12, freq="MS"),
                                "Revenue (£000)": [52, 48, 71, 65, 88, 79, 102, 96, 118, 111, 139, 154]})
        countries = pd.DataFrame({"Country": ["United Kingdom", "Germany", "France", "EIRE", "Other"],
                                  "Revenue (£000)": [780, 125, 104, 82, 149]})
        a, b = st.columns(2)
        a.plotly_chart(_style(px.line(monthly, x="Month", y="Revenue (£000)", markers=True,
                                      title="Monthly revenue trend (demo)")), use_container_width=True)
        b.plotly_chart(_style(px.pie(countries, names="Country", values="Revenue (£000)", hole=.5,
                                     title="Revenue by market (demo)")), use_container_width=True)
        st.subheader("Project workflow")
        st.markdown("Data Cleaning → EDA → RFM Analysis → K-Means Segmentation → Churn Prediction → Model Evaluation")

    elif page == "🛒 Sales Analysis":
        st.subheader("Sales performance")
        months = pd.date_range("2010-12-01", periods=12, freq="MS")
        sales = pd.DataFrame({"Month": months, "Revenue (£)": [52000,48000,71000,65000,88000,79000,102000,96000,118000,111000,139000,154000],
                              "Orders": [820,770,1080,990,1310,1200,1490,1430,1680,1590,1900,2130]})
        a,b = st.columns(2)
        a.plotly_chart(_style(px.line(sales, x="Month", y="Revenue (£)", markers=True, title="Monthly revenue (demo)")), use_container_width=True)
        b.plotly_chart(_style(px.bar(sales, x="Month", y="Orders", title="Monthly orders (demo)")), use_container_width=True)
        countries = pd.DataFrame({"Country":["United Kingdom","Germany","France","EIRE","Other"],"Revenue (£)":[780000,125000,104000,82000,149000]})
        st.plotly_chart(_style(px.bar(countries, x="Country", y="Revenue (£)", title="Revenue by country (demo)")), use_container_width=True)

    elif page == "🧩 Customer Segmentation":
        st.subheader("RFM-based customer groups")
        segments = pd.DataFrame({"Segment":["Champions","Loyal Customers","Potential Loyalists","At Risk","Hibernating"],
                                 "Customers":[610,930,785,1020,972], "Avg Recency (days)":[12,35,48,110,190],
                                 "Avg Frequency":[18,10,5,4,1], "Avg Monetary (£)":[1250,720,340,280,65]})
        a,b = st.columns([2,3])
        a.dataframe(segments, hide_index=True, use_container_width=True)
        b.plotly_chart(_style(px.bar(segments, x="Segment", y="Customers", color="Segment",
                                     title="Customers by segment (demo)")), use_container_width=True)
        st.plotly_chart(_style(px.scatter(segments, x="Avg Recency (days)", y="Avg Monetary (£)",
                                          size="Customers", color="Segment", title="Recency vs monetary (demo)")), use_container_width=True)

    elif page == "🔮 Churn Prediction":
        st.subheader("Churn model evaluation")
        cols=st.columns(3)
        cols[0].metric("Example churn rate","24.8%")
        cols[1].metric("Example F1 score","0.78")
        cols[2].metric("Example ROC-AUC","0.84")
        models = pd.DataFrame({"Model":["Logistic Regression","Decision Tree","Random Forest"],
                               "F1":[.70,.73,.78],"ROC-AUC":[.77,.79,.84],"Recall":[.72,.75,.81]})
        st.dataframe(models, hide_index=True, use_container_width=True)
        st.plotly_chart(_style(px.bar(models, x="Model", y=["F1","ROC-AUC","Recall"], barmode="group",
                                      title="Model comparison (demo)")), use_container_width=True)
        cm = pd.DataFrame({"Predicted Active":[120,22],"Predicted Churn":[18,65]}, index=["Actual Active","Actual Churn"])
        st.subheader("Confusion matrix (demo)")
        st.dataframe(cm, use_container_width=True)

    elif page == "🎯 Customer Prediction":
        st.subheader("Customer churn risk preview")
        st.write("This interactive example demonstrates the intended prediction interface. It does not call a trained model.")
        with st.form("demo_customer_form"):
            recency = st.slider("Recency (days since last order)", 0, 365, 90)
            frequency = st.slider("Number of orders", 1, 100, 5)
            monetary = st.number_input("Total spend (£)", min_value=0.0, value=500.0, step=50.0)
            submitted = st.form_submit_button("Estimate example risk")
        if submitted:
            score = min(.95, max(.05, .15 + recency/500 - frequency/250 - monetary/10000))
            cols=st.columns(2)
            cols[0].metric("Illustrative churn risk", f"{score:.0%}")
            cols[1].metric("Risk category", "High" if score >= .65 else "Medium" if score >= .4 else "Low")
            st.caption("Illustrative calculation only — not a trained machine-learning prediction.")

    st.caption("To show authentic project results, the precomputed outputs and trained model must be available. The original Excel file does not need to be deployed.")
