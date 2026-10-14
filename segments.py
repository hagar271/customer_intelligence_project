"""Gives each K-Means cluster a business-friendly segment name.

The rule is transparent (easy to explain): the K-Means was trained on log(1+R), log(1+F), log(1+M),
so we compare each cluster's average (in that log space) with the average of ALL customers:
  * Recency  below average  -> bought recently ("good")
  * Frequency above average -> buys often      ("good")
  * Monetary above average  -> spends a lot    ("good")
The team can override the names with artifacts/segment_names.json.
"""
from collections import Counter

import numpy as np
import pandas as pd

import config
import data_loader

SEGMENT_INFO = {
    "Champions": (
        "Bought recently, buy often and spend the most.",
        "Reward them: loyalty perks, early access to new products, ask for reviews."),
    "Potential Loyalists": (
        "Bought recently and are strong on either frequency or spending.",
        "Nurture them: cross-sell related products, invite them to a loyalty programme."),
    "New / Promising": (
        "Bought recently but with low frequency and low spending so far.",
        "Onboard them: welcome offers and a nudge to make the second purchase."),
    "At Risk": (
        "Used to be valuable customers but have not bought for a long time.",
        "Win them back: personal offer or a 'we miss you' message before they are lost."),
    "Lost / Hibernating": (
        "Long time since the last purchase, with low frequency and low spending.",
        "Low-cost reactivation only (e-mail / discount); do not over-invest."),
}


def base_name(segment: str) -> str:
    return segment.split(" (Cluster")[0]


def describe(segment: str):
    return SEGMENT_INFO.get(base_name(segment), ("Custom segment defined by the team.", "—"))


def build_segment_map(rfm: pd.DataFrame) -> dict:
    """cluster id -> segment name."""
    logs = np.log1p(rfm[config.RFM_FEATURES])
    z = (logs.groupby(rfm["Cluster"]).mean() - logs.mean()) / logs.std().replace(0, 1)

    names = {}
    for cluster, row in z.iterrows():
        recent = row["Recency"] < 0
        good = int(row["Frequency"] > 0) + int(row["Monetary"] > 0)
        if recent and good == 2:
            name = "Champions"
        elif recent and good == 1:
            name = "Potential Loyalists"
        elif recent:
            name = "New / Promising"
        elif good >= 1:
            name = "At Risk"
        else:
            name = "Lost / Hibernating"
        names[int(cluster)] = name

    counts = Counter(names.values())              # two clusters with the same name -> add the id
    names = {c: (f"{n} (Cluster {c})" if counts[n] > 1 else n) for c, n in names.items()}
    names.update(data_loader.load_segment_overrides())
    return names


def cluster_summary(rfm: pd.DataFrame, segment_map: dict) -> pd.DataFrame:
    g = rfm.groupby("Cluster")
    out = pd.DataFrame({
        "Customers": g.size(),
        "Avg Recency (days)": g["Recency"].mean(),
        "Avg Frequency (orders)": g["Frequency"].mean(),
        "Avg Monetary (£)": g["Monetary"].mean(),
        "Total Monetary (£)": g["Monetary"].sum(),
    })
    out["% of customers"] = out["Customers"] / out["Customers"].sum() * 100
    out["% of revenue"] = out["Total Monetary (£)"] / out["Total Monetary (£)"].sum() * 100
    out.insert(0, "Segment", out.index.map(segment_map))
    return out.sort_values("Avg Monetary (£)", ascending=False)
