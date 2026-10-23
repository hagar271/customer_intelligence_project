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
        st.subheader("Predict customer churn risk")
        st.write("Enter customer behavior details below. The current artifact-free preview demonstrates the old input form, but does not use a trained model.")
        preset = st.selectbox("Start from an example", ["Typical customer", "Loyal, active customer", "Slipping away", "New one-time buyer"])
        examples = {
            "Typical customer": {"recency": 60, "frequency": 3, "monetary": 800.0, "items": 300, "unique": 40, "tenure": 250, "avg_days": 60.0, "orders90": 0, "one_day": False},
            "Loyal, active customer": {"recency": 12, "frequency": 14, "monetary": 5200.0, "items": 3800, "unique": 160, "tenure": 600, "avg_days": 38.0, "orders90": 3, "one_day": False},
            "Slipping away": {"recency": 130, "frequency": 5, "monetary": 1100.0, "items": 650, "unique": 70, "tenure": 520, "avg_days": 85.0, "orders90": 0, "one_day": False},
            "New one-time buyer": {"recency": 25, "frequency": 1, "monetary": 180.0, "items": 90, "unique": 25, "tenure": 25, "avg_days": 0.0, "orders90": 1, "one_day": True},
        }
        v = examples[preset]
        with st.form("demo_customer_form"):
            a,b,c = st.columns(3)
            recency = a.number_input("Recency (days since last purchase)", 0, 1000, v["recency"])
            frequency = b.number_input("Frequency (number of orders)", 1, 5000, v["frequency"])
            monetary = c.number_input("Monetary (total spent, £)", min_value=0.01, max_value=10000000.0, value=v["monetary"])
            d,e,f = st.columns(3)
            items = d.number_input("Total items purchased", 1, 10000000, v["items"])
            unique = e.number_input("Distinct products purchased", 1, 10000, v["unique"])
            tenure = f.number_input("Tenure (days since first purchase)", 0, 1000, v["tenure"])
            g,h,i = st.columns(3)
            orders90 = g.number_input("Orders in the last 90 days", 0, 5000, v["orders90"])
            avg_days = h.number_input("Average days between purchases", 0.0, 1000.0, v["avg_days"])
            one_day = i.checkbox("Bought on a single day only", value=v["one_day"])
            submitted = st.form_submit_button("Predict churn", type="primary")
        if submitted:
            errors = []
            if tenure < recency:
                errors.append("Tenure cannot be smaller than Recency.")
            if orders90 > frequency:
                errors.append("Orders in the last 90 days cannot exceed total orders.")
            if recency >= 90 and orders90 > 0:
                errors.append("Recency is 90+ days, so orders in the last 90 days must be zero.")
            if items < unique:
                errors.append("Total items cannot be smaller than distinct products.")
            one_time = bool(one_day or frequency == 1)
            if not one_time and avg_days > tenure:
                errors.append("Average days between purchases cannot exceed tenure.")
            if errors:
                for error in errors:
                    st.error(error)
            else:
                score = min(.95, max(.05, .15 + recency/500 - frequency/250 - monetary/10000))
                cols = st.columns(3)
                cols[0].metric("Illustrative churn risk", f"{score:.0%}")
                cols[1].metric("Risk category", "High" if score >= .65 else "Medium" if score >= .4 else "Low")
                cols[2].metric("Average order value", f"£{monetary/frequency:,.2f}")
                st.caption("This is an illustrative calculation only, not a trained machine-learning prediction.")
                st.write("**Submitted features**")
                st.dataframe(pd.DataFrame([{
                    "Recency": recency, "Frequency": frequency, "Monetary": monetary,
                    "Average order value": monetary/frequency, "Total items": items,
                    "Distinct products": unique, "Tenure days": tenure,
                    "Average days between purchases": avg_days, "Orders last 90 days": orders90,
                    "One-time buyer": one_time,
                }]), hide_index=True, use_container_width=True)

    st.caption("To show authentic project results, the precomputed outputs and trained model must be available. The original Excel file does not need to be deployed.")
