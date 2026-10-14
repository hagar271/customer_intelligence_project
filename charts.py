"""All Plotly figures in one place, so the pages stay short and readable."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config


def _style(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), title_font_size=16,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.3))
    return fig


# ------------------------------------------------------------ sales
def revenue_line(monthly: pd.DataFrame):
    fig = px.line(monthly, x="Month", y="Revenue", markers=True, title="Monthly Revenue (£)")
    fig.update_traces(line_color=config.PRIMARY)
    fig.update_yaxes(tickprefix="£", tickformat=",.0f", title=None)
    fig.update_xaxes(title=None)
    return _style(fig)


def orders_bar(monthly: pd.DataFrame):
    fig = px.bar(monthly, x="Month", y="Orders", title="Monthly Orders (unique invoices)")
    fig.update_traces(marker_color=config.ACCENT)
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None)
    return _style(fig)


def top_bar(df: pd.DataFrame, label_col: str, value_col: str, title: str, color: str = config.PRIMARY, hover=None):
    d = df.sort_values(value_col, ascending=False)
    fig = px.bar(d, x=value_col, y=label_col, orientation="h", title=title, hover_data=hover)
    fig.update_traces(marker_color=color)
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(title=None, tickformat=",.0f")
    return _style(fig, height=max(340, 32 * len(d) + 110))


def donut(df: pd.DataFrame, names: str, values: str, title: str):
    fig = px.pie(df, names=names, values=values, hole=0.55, title=title)
    fig.update_traces(textposition="inside", textinfo="percent")
    return _style(fig)


# ------------------------------------------------------------ segmentation
def cluster_bar(summary: pd.DataFrame):
    d = summary.reset_index()
    d["Label"] = d["Cluster"].astype(str) + " – " + d["Segment"]
    fig = px.bar(d, x="Label", y="Customers", color="Label", title="Customers per cluster",
                 text="Customers")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title=None)
    return _style(fig)


def rfm_hist(df: pd.DataFrame, column: str, use_log: bool):
    d = df.copy()
    x = column
    if use_log:
        x = f"log(1 + {column})"
        d[x] = np.log1p(d[column])
    fig = px.histogram(d, x=x, color="Segment", nbins=50, barmode="overlay", opacity=0.65,
                       title=f"{column} distribution by segment")
    return _style(fig)


def rfm_scatter(df: pd.DataFrame):
    sample = df.sample(min(len(df), 3000), random_state=42)
    fig = px.scatter(sample, x="Recency", y="Monetary", color="Segment", log_y=True, opacity=0.6,
                     hover_data=["Customer ID", "Frequency"],
                     title="Recency vs Monetary (log scale, sample of customers)")
    return _style(fig, height=430)


# ------------------------------------------------------------ churn
def churn_split_bar(label_summary: pd.DataFrame):
    d = label_summary.melt(id_vars="Set", value_vars=["Customers", "Churned"], var_name="Group", value_name="Count")
    d["Group"] = d["Group"].replace({"Customers": "All customers", "Churned": "Churned (1)"})
    fig = px.bar(d, x="Set", y="Count", color="Group", barmode="group",
                 title="Churn label: train vs test", color_discrete_sequence=[config.PRIMARY, config.RISK_COLORS["High"]])
    return _style(fig)


def risk_distribution(counts: pd.Series):
    d = counts.reindex(["Low", "Medium", "High"]).fillna(0).reset_index()
    d.columns = ["Risk", "Customers"]
    d["Label"] = d["Risk"].map(config.RISK_LABELS)
    fig = px.bar(d, x="Label", y="Customers", color="Risk", color_discrete_map=config.RISK_COLORS,
                 title="Current customers by churn-risk level", text="Customers")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title=None)
    return _style(fig)


def confusion_heatmap(cm: np.ndarray, title: str):
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", aspect="auto",
                    x=["Predicted: Active", "Predicted: Churn"], y=["Actual: Active", "Actual: Churn"], title=title)
    fig.update_coloraxes(showscale=False)
    return _style(fig, height=360)


def roc_figure(curves: list):
    fig = go.Figure()
    for name, fpr, tpr, auc in curves:
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                             line=dict(dash="dash", color="gray")))
    fig.update_layout(title="ROC curves (test set)", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return _style(fig, height=420)


def metrics_bar(comparison: pd.DataFrame):
    metrics = [m for m in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"] if m in comparison.columns]
    d = comparison.reset_index().melt(id_vars=comparison.index.name or "index", value_vars=metrics,
                                      var_name="Metric", value_name="Score")
    d = d.rename(columns={comparison.index.name or "index": "Model"})
    fig = px.bar(d, x="Model", y="Score", color="Metric", barmode="group", title="Model comparison (test set)")
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(title=None)
    return _style(fig, height=420)


def importance_bar(df: pd.DataFrame, title: str):
    d = df.sort_values("Importance")
    fig = px.bar(d, x="Importance", y="Feature", orientation="h", title=title)
    fig.update_traces(marker_color=config.PRIMARY)
    fig.update_yaxes(title=None)
    return _style(fig)


def probability_gauge(prob: float, low_max: float, high_min: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": "%", "valueformat": ".1f"},
        title={"text": "Churn probability"},
        gauge={
            "axis": {"range": [0, 100]}, "bar": {"color": "#1F2937", "thickness": 0.25},
            "steps": [
                {"range": [0, low_max * 100], "color": config.RISK_COLORS["Low"]},
                {"range": [low_max * 100, high_min * 100], "color": config.RISK_COLORS["Medium"]},
                {"range": [high_min * 100, 100], "color": config.RISK_COLORS["High"]},
            ],
        }))
    fig.update_layout(height=290, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def pr_figure(curves: list):
    fig = go.Figure()
    for name, precision, recall, ap in curves:
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"{name} (AP={ap:.3f})"))
    fig.update_layout(title="Precision-Recall curves (test set)", xaxis_title="Recall", yaxis_title="Precision")
    fig.update_yaxes(range=[0, 1.02])
    return _style(fig, height=420)


def calibration_figure(mean_pred, frac_pos, brier: float, model_name: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode="lines+markers", name=model_name))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfectly calibrated",
                             line=dict(dash="dash", color="gray")))
    fig.update_layout(title=f"Calibration (Brier score = {brier:.3f})",
                      xaxis_title="Mean predicted churn probability", yaxis_title="Actual churn rate")
    return _style(fig, height=420)


def risk_by_segment(df: pd.DataFrame):
    """Share of Low / Medium / High risk customers inside each segment (needs columns Segment, Risk)."""
    share = pd.crosstab(df["Segment"], df["Risk"], normalize="index").mul(100)
    share = share.reindex(columns=["Low", "Medium", "High"], fill_value=0)
    d = share.reset_index().melt(id_vars="Segment", var_name="Risk", value_name="Share")
    fig = px.bar(d, x="Segment", y="Share", color="Risk", color_discrete_map=config.RISK_COLORS,
                 category_orders={"Risk": ["Low", "Medium", "High"]}, title="Churn-risk mix inside each segment (%)")
    fig.update_yaxes(title=None, range=[0, 100])
    fig.update_xaxes(title=None)
    return _style(fig)
