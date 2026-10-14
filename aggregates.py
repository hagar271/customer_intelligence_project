"""Summary tables for the Overview / Sales pages (pure pandas, no Streamlit)."""
import json
from pathlib import Path

import pandas as pd

SALES_COLUMNS = {"Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price",
                 "Customer ID", "Country", "TotalPrice"}


def compute_sales_aggregates(df: pd.DataFrame) -> dict:
    """KPIs + monthly / product / country tables from the cleaned transactions."""
    if "TotalPrice" not in df.columns:
        df = df.assign(TotalPrice=df["Quantity"] * df["Price"])
    revenue = float(df["TotalPrice"].sum())
    orders = int(df["Invoice"].nunique())
    orders_per_customer = df.groupby("Customer ID")["Invoice"].nunique()
    date_min, date_max = df["InvoiceDate"].min(), df["InvoiceDate"].max()

    month = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    monthly = (df.groupby(month)
                 .agg(Revenue=("TotalPrice", "sum"), Orders=("Invoice", "nunique"),
                      Customers=("Customer ID", "nunique"))
                 .reset_index())
    monthly = monthly.rename(columns={monthly.columns[0]: "Month"})

    last_month = monthly["Month"].max()
    last_month_end = last_month + pd.offsets.MonthEnd(0)
    partial_last_month = bool(date_max.normalize() < last_month_end.normalize())

    desc = df.groupby("StockCode")["Description"].agg(lambda s: s.mode().iloc[0])
    products = (df.groupby("StockCode")
                  .agg(Revenue=("TotalPrice", "sum"), Quantity=("Quantity", "sum"), Orders=("Invoice", "nunique"))
                  .join(desc).reset_index())

    countries = (df.groupby("Country")
                   .agg(Revenue=("TotalPrice", "sum"), Orders=("Invoice", "nunique"),
                        Customers=("Customer ID", "nunique"))
                   .reset_index().sort_values("Revenue", ascending=False))

    kpis = {
        "revenue": revenue, "orders": orders, "customers": int(orders_per_customer.shape[0]),
        "aov": revenue / orders if orders else 0.0,
        "one_time_buyers_pct": float((orders_per_customer == 1).mean()),
        "date_min": date_min, "date_max": date_max,
        "partial_last_month": partial_last_month, "last_month": last_month,
    }
    return {"kpis": kpis, "monthly": monthly, "products": products, "countries": countries}




# ---- saving / loading the summary as plain CSV + JSON (works across pandas / Python versions) ----
SUMMARY_FILES = ["sales_kpis.json", "sales_monthly.csv", "sales_products.csv", "sales_countries.csv"]


def save_sales_summary(agg: dict, folder) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    kpis = {k: (v.isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in agg["kpis"].items()}
    with open(folder / "sales_kpis.json", "w") as f:
        json.dump(kpis, f, indent=2)
    agg["monthly"].to_csv(folder / "sales_monthly.csv", index=False)
    agg["products"].to_csv(folder / "sales_products.csv", index=False)
    agg["countries"].to_csv(folder / "sales_countries.csv", index=False)


def summary_exists(folder) -> bool:
    return all((Path(folder) / name).exists() for name in SUMMARY_FILES)


def load_sales_summary(folder) -> dict:
    folder = Path(folder)
    with open(folder / "sales_kpis.json") as f:
        kpis = json.load(f)
    for key in ("date_min", "date_max", "last_month"):
        kpis[key] = pd.Timestamp(kpis[key])
    return {
        "kpis": kpis,
        "monthly": pd.read_csv(folder / "sales_monthly.csv", parse_dates=["Month"]),
        "products": pd.read_csv(folder / "sales_products.csv", dtype={"StockCode": str}),
        "countries": pd.read_csv(folder / "sales_countries.csv"),
    }
