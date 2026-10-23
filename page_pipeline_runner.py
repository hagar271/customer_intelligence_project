"""Page 6 - Prepare result artifacts for the dashboard.

The deployed app does not need the original Excel dataset if the precomputed
result files and saved model are committed under the artifacts/ directory.
This page still allows an upload-based rebuild as a fallback.
"""
import streamlit as st

import config, data_loader, pipeline
from utils import fmt_int, fmt_pct, page_header


def render():
    page_header("⚙️ Pipeline & Deployment", "Use precomputed artifacts online; rebuild locally when possible")

    missing = data_loader.missing_required()
    if missing:
        st.warning(
            "The dashboard's required result/model files are not present in this GitHub revision. "
            "The Python source code alone cannot reproduce the real charts or predictions. "
            "To deploy without uploading the original dataset, run the pipeline on your own computer, "
            "then commit the required generated files listed below into the repository's artifacts/ folder."
        )
        st.markdown("**Required deployment files**")
        st.code(
            "artifacts/rfm_kmeans_customers.csv\n"
            "artifacts/churn_model_pipeline.joblib\n"
            "artifacts/churn_model_meta.json\n"
            "artifacts/churn_scores_current_customers.csv\n"
            "artifacts/sales_kpis.json\n"
            "artifacts/sales_monthly.csv\n"
            "artifacts/sales_products.csv\n"
            "artifacts/sales_countries.csv",
            language="text",
        )
        st.caption(
            "Do not upload online_retail_II.xlsx or clean_online_retail.csv to a public repository. "
            "The files above should contain only the precomputed outputs needed by the dashboard."
        )
        st.write("Files currently missing:")
        st.code("\n".join(f"- {name}" for name in missing), language="text")
    else:
        st.success("All required result/model files are available. The dashboard can run without the original dataset.")

    with st.expander("Optional: rebuild from a dataset upload", expanded=False):
        st.info(
            "This fallback requires the original Online Retail II file and may take time or exceed the resources "
            "available on Streamlit Community Cloud. Prefer building artifacts locally and committing only the "
            "required result/model files."
        )
        uploaded = st.file_uploader(
            "Online Retail II file (.xlsx or .csv)",
            type=["xlsx", "xls", "csv"],
            help="Use this only if you intentionally want to rebuild the pipeline in the running app.",
        )

        with st.expander("Advanced settings"):
            c1, c2 = st.columns(2)
            horizon = c1.select_slider("Churn horizon (days)", options=[60, 90, 120], value=90)
            snapshots = c2.slider("Training snapshots", 1, 6, 4)
            fast = st.checkbox("Fast mode (smaller tuning grids)", value=True)
            use_xgb = st.checkbox("Include XGBoost (optional model)", value=True)

        run = st.button("🚀 Run the full pipeline", type="primary", disabled=uploaded is None)
        if uploaded is None:
            st.caption("Select a file to enable the button.")

        if run:
            bar = st.progress(0.0, text="Starting…")
            with st.status("Running the pipeline…", expanded=True) as status:
                def on_progress(fraction: float, message: str):
                    bar.progress(min(max(fraction, 0.0), 1.0), text=message)
                    status.write(message)

                try:
                    summary = pipeline.run_pipeline(
                        uploaded, config.ARTIFACTS_DIR, horizon_days=horizon,
                        n_train_snapshots=snapshots, fast=fast, use_xgb=use_xgb,
                        progress=on_progress,
                    )
                except Exception as e:
                    status.update(label="Pipeline failed", state="error")
                    st.error(f"Could not finish: {e}")
                    return
                status.update(label="Pipeline finished", state="complete")

            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Pipeline finished. Refresh the app to load the newly generated files.")
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
