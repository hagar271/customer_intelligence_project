"""Small UI helpers shared by all pages."""
import inspect

import streamlit as st

import config

# Newer Streamlit versions use width="stretch"; older ones use use_container_width=True.
_NEW_WIDTH_API = "width" in inspect.signature(st.plotly_chart).parameters


def show_chart(fig, key=None):
    if _NEW_WIDTH_API:
        st.plotly_chart(fig, width="stretch", key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


def inject_css():
    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] {
            background-color: rgba(128,128,128,0.08);
            border: 1px solid rgba(128,128,128,0.25);
            border-radius: 12px; padding: 14px 16px;
        }
        .pipeline {display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin: 4px 0 18px 0;}
        .pipeline .step {background:rgba(46,134,171,0.12); border:1px solid rgba(46,134,171,0.4);
                         padding:4px 10px; border-radius:14px; font-size:0.85rem;}
        .pipeline .here {background:#2E86AB; color:white; border-color:#2E86AB;}
        .pipeline .arrow {opacity:0.5;}
        .badge {color:white; padding:4px 14px; border-radius:14px; font-weight:600; font-size:1.05rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = ""):
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def pipeline_strip(current: str = "Dashboard"):
    steps = ["Data Cleaning", "EDA", "RFM", "K-Means Segments", "Churn Label", "Churn Model", "Dashboard"]
    html = []
    for i, s in enumerate(steps):
        cls = "step here" if s == current else "step"
        html.append(f'<span class="{cls}">{s}</span>')
        if i < len(steps) - 1:
            html.append('<span class="arrow">→</span>')
    st.markdown(f'<div class="pipeline">{"".join(html)}</div>', unsafe_allow_html=True)


def fmt_money(x: float) -> str:
    return f"£{x:,.0f}"


def fmt_int(x) -> str:
    return f"{int(x):,}"


def fmt_pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def risk_badge(level: str) -> str:
    color = config.RISK_COLORS[level]
    return f'<span class="badge" style="background:{color}">{config.RISK_LABELS[level]}</span>'


def parse_customer_id(text: str):
    """Returns (customer_id, error_message). Exactly one of them is None."""
    text = (text or "").strip()
    if not text:
        return None, "Please enter a Customer ID."
    try:
        value = float(text)
    except ValueError:
        return None, "Customer ID must be a number, for example 12347."
    if not value.is_integer() or value <= 0:
        return None, "Customer ID must be a positive whole number."
    return int(value), None
