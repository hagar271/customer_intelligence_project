"""Page 2 - Sales analysis."""
import streamlit as st

import charts, data_loader
from utils import fmt_int, fmt_money, page_header, show_chart


def render():
    page_header("🛒 Sales Analysis", "Revenue, orders, products and countries")
    agg = data_loader.load_sales_aggregates()
    monthly, products, countries, k = agg["monthly"], agg["products"], agg["countries"], agg["kpis"]

    # ---------------- time analysis with a month-range filter
    st.subheader("Revenue and orders over time")
    labels = monthly["Month"].dt.strftime("%Y-%m").tolist()
    start, end = st.select_slider("Month range", options=labels, value=(labels[0], labels[-1]))
    sel = monthly[(monthly["Month"].dt.strftime("%Y-%m") >= start) & (monthly["Month"].dt.strftime("%Y-%m") <= end)]

    m1, m2, m3 = st.columns(3)
    m1.metric("Revenue in range", fmt_money(sel["Revenue"].sum()))
    m2.metric("Orders in range", fmt_int(sel["Orders"].sum()))
    best = sel.loc[sel["Revenue"].idxmax()]
    m3.metric("Best month", f"{best['Month']:%b %Y}", delta=fmt_money(best["Revenue"]), delta_color="off")

    a, b = st.columns(2)
    with a:
        show_chart(charts.revenue_line(sel))
    with b:
        show_chart(charts.orders_bar(sel))
    if k["partial_last_month"] and sel["Month"].max() == k["last_month"]:
        st.caption(f"⚠️ {k['last_month']:%b %Y} is incomplete (data ends {k['date_max']:%d %b %Y}).")

    # ---------------- products
    st.subheader("Top products")
    st.caption("All-time figures (not affected by the month filter). Grouped by product code.")
    p1, p2 = st.columns([1, 2])
    metric = p1.radio("Rank products by", ["Revenue", "Quantity"], horizontal=True)
    n_products = p2.slider("Number of products", 5, 25, 10)
    top = products.nlargest(n_products, metric)
    show_chart(charts.top_bar(top, "Description", metric,
                              f"Top {n_products} products by {metric.lower()}",
                              hover=["StockCode", "Orders"]))

    # ---------------- countries
    st.subheader("Top countries")
    c1, c2 = st.columns([1, 2])
    exclude_uk = c1.checkbox("Exclude United Kingdom", value=False,
                             help="The UK dominates sales; excluding it makes the other markets visible.")
    n_countries = c2.slider("Number of countries", 5, 15, 10)
    d = countries[countries["Country"] != "United Kingdom"] if exclude_uk else countries
    show_chart(charts.top_bar(d.head(n_countries), "Country", "Revenue",
                              f"Top {n_countries} countries by revenue (£)", color="#3B8686",
                              hover=["Orders", "Customers"]))

    with st.expander("Show the data tables"):
        st.dataframe(monthly.assign(Month=monthly["Month"].dt.strftime("%Y-%m")).round(2), hide_index=True)
        st.dataframe(countries.round(2), hide_index=True)
