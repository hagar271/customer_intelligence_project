"""Page 4 - Churn prediction: label definition, model performance, comparison and an action list.
Everything is READ from files saved by the modelling notebook - nothing is retrained here."""
import pandas as pd
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)

import charts, data_loader, predict, segments
from utils import fmt_money, fmt_pct, page_header, show_chart

PROBA_PREFIX = "proba__"
BASELINE_PREFIX = "Baseline"


def _delta(metrics: dict, baseline: dict, key: str):
    """'+0.03 vs baseline' text for a KPI card (None if the baseline is not available)."""
    if key in metrics and key in baseline:
        return f"{metrics[key] - baseline[key]:+.2f} vs baseline"
    return None


def render():
    page_header("🔮 Churn Prediction", "Which customers are likely to stop buying?")
    meta = data_loader.load_meta()
    horizon, best = meta["horizon_days"], meta["best_model"]
    low, high = meta["risk_thresholds"]["low_max"], meta["risk_thresholds"]["high_min"]

    train_cutoffs = meta.get("train_cutoffs") or [meta.get("train_cutoff", "?")]
    with st.expander("How is churn defined? (temporal label, no leakage)", expanded=False):
        st.markdown(
            f"- Pick a **cutoff date**. Features (RFM and others) are computed **only from purchases before it**.\n"
            f"- Look at the next **{horizon} days**: if the customer buys again → **Churn = 0**, otherwise → **Churn = 1**.\n"
            f"- **Temporal split:** the model learns from {len(train_cutoffs)} earlier cutoff(s) "
            f"(`{train_cutoffs[0]}` … `{train_cutoffs[-1]}`) and is tested on a later one (`{meta.get('test_cutoff', '?')}`): "
            f"learn from the past, predict the future.\n"
            f"- RFM describes past behaviour (feature generation); the churn label is future behaviour."
        )

    labels = data_loader.load_optional_csv("churn_labels")
    comparison = data_loader.load_optional_csv("churn_comparison")
    test_pred = data_loader.load_optional_csv("churn_test_pred")
    importance = data_loader.load_optional_csv("churn_importance")
    scores = predict.add_risk_column(data_loader.load_scores(), low, high)

    # ---------------- headline numbers
    st.subheader("Overall churn")
    m, base = meta.get("test_metrics", {}), meta.get("baseline_metrics", {})
    c1, c2, c3, c4, c5 = st.columns(5)
    if labels is not None:
        test_row = labels[labels["Set"] == "Test"].iloc[0]
        train_row = labels[labels["Set"] == "Train"].iloc[0]
        c1.metric(f"Churn rate (test, {horizon}d)", fmt_pct(test_row["ChurnRate"]),
                  delta=f"train {fmt_pct(train_row['ChurnRate'])}", delta_color="off")
    else:
        c1.metric(f"Churn rate (test, {horizon}d)", "n/a")
    c2.metric("Best model", best.replace(" (optional)", ""))
    c3.metric("Recall", f"{m['Recall']:.2f}" if "Recall" in m else "n/a", delta=_delta(m, base, "Recall"))
    c4.metric("F1-score", f"{m['F1']:.2f}" if "F1" in m else "n/a", delta=_delta(m, base, "F1"))
    c5.metric("ROC-AUC", f"{m['ROC-AUC']:.2f}" if "ROC-AUC" in m else "n/a", delta=_delta(m, base, "ROC-AUC"))
    if base:
        st.caption("The small grey/green/red note under each metric compares the selected model with a simple "
                   "baseline that only looks at Recency.")

    a, b = st.columns(2)
    with a:
        if labels is not None:
            show_chart(charts.churn_split_bar(labels))
        else:
            st.info("`churn_label_summary.csv` not found.")
    with b:
        show_chart(charts.risk_distribution(scores["Risk"].value_counts()))
        st.caption(f"Risk levels: Low < {low:.0%} ≤ Medium < {high:.0%} ≤ High "
                   f"(model score for NOT buying in the next {horizon} days).")

    # ---------------- model performance
    st.subheader("Model performance (test set)")
    if test_pred is None:
        st.info("`churn_test_predictions.csv` not found – confusion matrix and ROC curves need it.")
    else:
        models = [c[len(PROBA_PREFIX):] for c in test_pred.columns if c.startswith(PROBA_PREFIX)]
        y_true = test_pred["Churn_True"].to_numpy()

        p1, p2 = st.columns(2)
        selected = p1.selectbox("Model", models, index=models.index(best) if best in models else 0)
        default_t = float(meta["decision_threshold"])
        threshold = p2.slider("Decision threshold", 0.05, 0.95, default_t, 0.05,
                              help="Predict 'churn' when the score is at least this value.")
        proba = test_pred[PROBA_PREFIX + selected].to_numpy()
        pred = (proba >= threshold).astype(int)
        if abs(threshold - default_t) > 1e-9:
            st.caption("What-if threshold: metrics are recomputed from the saved test scores (no retraining).")

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Accuracy", f"{accuracy_score(y_true, pred):.3f}")
        k2.metric("Precision", f"{precision_score(y_true, pred, zero_division=0):.3f}")
        k3.metric("Recall", f"{recall_score(y_true, pred, zero_division=0):.3f}")
        k4.metric("F1-score", f"{f1_score(y_true, pred, zero_division=0):.3f}")
        k5.metric("ROC-AUC", f"{roc_auc_score(y_true, proba):.3f}")

        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        left, right = st.columns(2)
        with left:
            show_chart(charts.confusion_heatmap(cm, f"Confusion matrix – {selected}"))
            st.caption(f"Missed churners (false negatives): **{fn}** · false alarms (false positives): **{fp}**")
        with right:
            roc_curves, pr_curves = [], []
            for name in models:
                p_name = test_pred[PROBA_PREFIX + name]
                fpr, tpr, _ = roc_curve(y_true, p_name)
                roc_curves.append((name, fpr, tpr, roc_auc_score(y_true, p_name)))
                prec, rec, _ = precision_recall_curve(y_true, p_name)
                pr_curves.append((name, prec, rec, average_precision_score(y_true, p_name)))
            show_chart(charts.roc_figure(roc_curves))

        left2, right2 = st.columns(2)
        with left2:
            show_chart(charts.pr_figure(pr_curves))
        with right2:
            frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
            show_chart(charts.calibration_figure(mean_pred, frac_pos, brier_score_loss(y_true, proba), selected))
        st.caption("Calibration shows whether a score of 70% really means ~70% of such customers churn. "
                   "Scores are best read as a **risk ranking**; the Low/Medium/High bands use fixed thresholds.")

    with st.expander("Why are Recall and F1 more important than Accuracy here?"):
        st.markdown(
            "- **Recall**: of all customers who really churn, how many do we catch? A missed churner is lost revenue we never tried to save.\n"
            "- **Precision**: of the customers we flag, how many really churn? Low precision wastes retention budget.\n"
            "- **F1-score**: balances Precision and Recall in one number.\n"
            "- **Accuracy** can be misleading when one class is much bigger than the other.\n"
            "- **ROC-AUC**: how well the model *ranks* customers by risk, independent of the threshold."
        )

    # ---------------- comparison
    st.subheader("Model comparison")
    if comparison is None:
        st.info("`churn_model_comparison.csv` not found.")
    else:
        comparison = comparison.set_index(comparison.columns[0])
        comparison.index.name = "Model"
        show = comparison.sort_values([c for c in ["F1", "ROC-AUC"] if c in comparison.columns], ascending=False)
        st.dataframe(show.round(3))
        st.caption(f"**Selected model: {best}** — chosen by F1-score, then ROC-AUC (not by Accuracy alone). "
                   "The Baseline (Recency only) is a reference and is not eligible.")
        show_chart(charts.metrics_bar(comparison))

    if importance is not None and {"Feature", "Importance"} <= set(importance.columns):
        st.subheader("What drives the prediction?")
        is_importance = importance["Type"].iloc[0] == "feature_importance" if "Type" in importance.columns else True
        kind = "Feature importance" if is_importance else "|Coefficient| on scaled features"
        show_chart(charts.importance_bar(importance, f"{kind} – {best}"))

    # ---------------- segments x risk + action list (connects K-Means with the churn model)
    st.subheader("Churn risk by customer segment")
    try:
        rfm = data_loader.load_rfm()
        seg_map = segments.build_segment_map(rfm)
        scores["Segment"] = scores["Cluster"].map(seg_map) if "Cluster" in scores.columns else None
    except data_loader.ArtifactError:
        scores["Segment"] = None

    if scores["Segment"].notna().any():
        show_chart(charts.risk_by_segment(scores.dropna(subset=["Segment"])))
    else:
        st.info("Segment information is not available in the scores file.")

    st.subheader("🎯 Who should we contact first?")
    st.caption("Ranked by **revenue at risk** = past spending × churn score. Download the list for the retention team.")
    if "Monetary" in scores.columns:
        pool = scores[scores["Risk"].isin(["High", "Medium"])].copy()
        f1_, f2_ = st.columns([2, 1])
        seg_options = sorted(pool["Segment"].dropna().unique())
        chosen = f1_.multiselect("Filter by segment (optional)", seg_options)
        top_n = f2_.slider("Rows to show", 10, 100, 20, 10)
        if chosen:
            pool = pool[pool["Segment"].isin(chosen)]
        pool["Revenue at risk (£)"] = pool["Monetary"] * pool["ChurnProbability"]
        pool = pool.sort_values("Revenue at risk (£)", ascending=False)
        cols = [c for c in ["Customer ID", "Segment", "Risk", "ChurnProbability", "Recency", "Frequency",
                            "Monetary", "Revenue at risk (£)"] if c in pool.columns]
        st.dataframe(pool[cols].head(top_n).round(2), hide_index=True)
        st.caption(f"{len(pool):,} customers match · total revenue at risk: {fmt_money(pool['Revenue at risk (£)'].sum())}")
        st.download_button("⬇️ Download the full list (CSV)", pool[cols].round(4).to_csv(index=False).encode("utf-8"),
                           file_name="retention_priority_list.csv", mime="text/csv")
    else:
        st.info("The scores file has no 'Monetary' column, so the priority list cannot be built.")
