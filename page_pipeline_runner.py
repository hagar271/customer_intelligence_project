"""Page 6 - Run Pipeline: upload the raw dataset and build every file the dashboard needs.
This is the ONLY place where models are trained. All other pages just load the saved files."""
import streamlit as st

import config, data_loader, pipeline
from utils import fmt_int, fmt_pct, page_header


def render():
    page_header("⚙️ Run Pipeline", "Upload the raw dataset → one click → the whole dashboard is ready")

    missing = data_loader.missing_required()
    if missing:
        st.info("**First time here?** Upload `online_retail_II.xlsx` below and press **Run**. "
                "The app will clean the data, build RFM, K-Means segments and the churn model, "
                "and save everything. When it finishes, open any other page from the sidebar.")
    else:
        st.success("All required files are already available. You only need this page to rebuild them from a new dataset.")

    uploaded = st.file_uploader("Online Retail II file (.xlsx or .csv)", type=["xlsx", "xls", "csv"],
                                help="The original Excel file with both sheets, or a CSV with the same columns.")

    with st.expander("Advanced settings"):
        c1, c2 = st.columns(2)
        horizon = c1.select_slider("Churn horizon (days)", options=[60, 90, 120], value=90,
                                   help="A customer who does not buy within this many days after the cutoff is labelled churn.")
        snapshots = c2.slider("Training snapshots", 1, 6, 4,
                              help="How many past cutoff dates are used to create training examples (1 = simplest).")
        fast = st.checkbox("Fast mode (smaller tuning grids) – recommended", value=True,
                           help="Untick for the full grid search used in the notebook (slower, a few minutes more).")
        use_xgb = st.checkbox("Include XGBoost (optional model)", value=True)

    if not missing:
        st.warning("Running the pipeline will **overwrite** the existing result files.")

    run = st.button("🚀 Run the full pipeline", type="primary", disabled=uploaded is None)
    if uploaded is None:
        st.caption("Select a file to enable the button.")

    if run:
        st.caption("Please keep this page open and do not click anything until it finishes.")
        bar = st.progress(0.0, text="Starting…")
        with st.status("Running the pipeline…", expanded=True) as status:
            def on_progress(fraction: float, message: str):
                bar.progress(min(max(fraction, 0.0), 1.0), text=message)
                status.write(message)

            try:
                summary = pipeline.run_pipeline(uploaded, config.ARTIFACTS_DIR, horizon_days=horizon,
                                                n_train_snapshots=snapshots, fast=fast, use_xgb=use_xgb,
                                                progress=on_progress)
            except Exception as e:           # show the reason instead of a blank page
                status.update(label="Pipeline failed", state="error")
                st.error(f"Could not finish: {e}")
                return
            status.update(label="Pipeline finished", state="complete")

        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Everything is ready. Open **Overview** (or any page) from the sidebar.")

        m = summary["test_metrics"]
        b = summary["baseline_metrics"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Clean rows", fmt_int(summary["rows_clean"]))
        k2.metric("Customers", fmt_int(summary["customers"]))
        k3.metric("Segments", summary["n_clusters"])
        k4.metric("Best churn model", summary["best_model"])
        k5, k6, k7 = st.columns(3)
        k5.metric("F1 (test)", f"{m['F1']:.3f}", delta=f"{m['F1'] - b['F1']:+.3f} vs baseline")
        k6.metric("Recall (test)", f"{m['Recall']:.3f}", delta=f"{m['Recall'] - b['Recall']:+.3f} vs baseline")
        k7.metric("Churn rate train → test",
                  f"{fmt_pct(summary['train_churn_rate'])} → {fmt_pct(summary['test_churn_rate'])}")

    with st.expander("Files status"):
        st.dataframe(data_loader.artifact_status(), hide_index=True)
