"""Page 3 - Customer segmentation (K-Means on RFM) + single-customer lookup."""
import streamlit as st

import charts, data_loader, segments
from utils import fmt_int, fmt_money, page_header, parse_customer_id, show_chart


def render():
    page_header("🧩 Customer Segmentation", "K-Means clusters built from Recency, Frequency and Monetary (RFM)")
    rfm = data_loader.load_rfm()
    seg_map = segments.build_segment_map(rfm)
    rfm = rfm.assign(Segment=rfm["Cluster"].map(seg_map))
    summary = segments.cluster_summary(rfm, seg_map)

    with st.expander("What are R, F and M?"):
        st.markdown(
            "- **Recency** – days since the customer's last purchase (lower = more recent).\n"
            "- **Frequency** – number of distinct orders.\n"
            "- **Monetary** – total money spent (£).\n\n"
            "K-Means groups customers with a similar R/F/M profile. The segment names "
            "(Champions, At Risk, …) are assigned by a simple rule that compares each cluster with the average customer."
        )

    # ---------------- clusters overview
    st.subheader("Customers per cluster")
    a, b = st.columns([3, 2])
    with a:
        show_chart(charts.cluster_bar(summary))
    with b:
        show_chart(charts.donut(summary.reset_index().assign(Label=lambda d: d["Cluster"].astype(str) + " – " + d["Segment"]),
                                "Label", "Customers", "Share of customers"))

    st.subheader("Cluster summary")
    st.dataframe(summary.drop(columns=["Total Monetary (£)"]).round(1), column_config={
        "Customers": st.column_config.NumberColumn(format="%d")})

    # ---------------- segment information
    st.subheader("Segment information")
    cols = st.columns(2)
    for i, (cluster, row) in enumerate(summary.iterrows()):
        desc, action = segments.describe(row["Segment"])
        with cols[i % 2].container(border=True):
            st.markdown(f"**Cluster {cluster} · {row['Segment']}**")
            st.caption(f"{fmt_int(row['Customers'])} customers · {row['% of customers']:.1f}% of customers · "
                       f"{row['% of revenue']:.1f}% of revenue")
            st.write(desc)
            st.markdown(f"**Suggested action:** {action}")

    # ---------------- distributions
    st.subheader("RFM distributions")
    d1, d2 = st.columns([1, 3])
    feature = d1.selectbox("Feature", ["Recency", "Frequency", "Monetary"])
    use_log = d1.checkbox("Log scale (recommended for Frequency / Monetary)", value=feature != "Recency")
    with d2:
        show_chart(charts.rfm_hist(rfm, feature, use_log))
    show_chart(charts.rfm_scatter(rfm))

    # ---------------- customer lookup
    st.subheader("🔎 Customer lookup")
    examples = ", ".join(str(x) for x in rfm["Customer ID"].sample(3, random_state=1))
    text = st.text_input("Customer ID", placeholder="e.g. 12347", help=f"Examples from the data: {examples}",
                         key="seg_customer_id")
    if not text.strip():
        st.info(f"Enter a Customer ID to see its RFM values and segment. Examples: {examples}")
        return
    cid, error = parse_customer_id(text)
    if error:
        st.error(error)
        return
    match = rfm[rfm["Customer ID"] == cid]
    if match.empty:
        st.error(f"Customer **{cid}** was not found in the segmentation data. Try one of: {examples}")
        return

    row = match.iloc[0]
    avg = rfm[rfm["Cluster"] == row["Cluster"]][["Recency", "Frequency", "Monetary"]].mean()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Recency (days)", fmt_int(row["Recency"]), delta=f"cluster avg {avg['Recency']:.0f}", delta_color="off")
    m2.metric("Frequency (orders)", fmt_int(row["Frequency"]), delta=f"cluster avg {avg['Frequency']:.1f}", delta_color="off")
    m3.metric("Monetary", fmt_money(row["Monetary"]), delta=f"cluster avg {fmt_money(avg['Monetary'])}", delta_color="off")
    m4.metric("Cluster", int(row["Cluster"]))
    m5.metric("Segment", row["Segment"])
    desc, action = segments.describe(row["Segment"])
    st.success(f"**{row['Segment']}** — {desc}  \n**Suggested action:** {action}")
