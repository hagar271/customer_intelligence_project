# Customer Intelligence System — Streamlit dashboard

This repository contains the dashboard source code. **The original Online Retail II dataset is not required by the deployed app when the precomputed result files and saved models are included.** The source-code-only repository does not contain those result/model files yet, so the dashboard will show a deployment checklist until they are added.

## Recommended deployment (keep the original dataset off GitHub)

1. On your own computer, place the original `online_retail_II.xlsx` beside `build_results.bat`.
2. Run `build_results.bat` and wait for the pipeline to finish. This generates the result tables and trained model files locally.
3. Run `run_app.bat` and confirm the dashboard works locally.
4. Upload the generated **result/model files only** to this repository, either in the project root or in an `artifacts/` folder. Do **not** upload `online_retail_II.xlsx` or `clean_online_retail.csv`.
5. In Streamlit Community Cloud, deploy this repository using branch `main` and main file `app.py`. If it is already deployed, restart/reboot the app after committing the generated files.

## Required generated files

The app needs the following files to show the full dashboard without rebuilding the pipeline:

- `rfm_kmeans_customers.csv`
- `churn_model_pipeline.joblib`
- `churn_model_meta.json`
- `churn_scores_current_customers.csv`
- `sales_kpis.json`
- `sales_monthly.csv`
- `sales_products.csv`
- `sales_countries.csv`

Optional files enable extra charts and features:

- `kmeans_model.joblib`, `kmeans_scaler.joblib`, `segment_names.json`
- `churn_model_comparison.csv`, `churn_test_predictions.csv`, `churn_label_summary.csv`, `churn_feature_importance.csv`

The saved model's scikit-learn version must match the version installed in `requirements.txt`. `build_results.bat` / `build_results.py` pins the version after a successful local build.

## Run locally

- Windows: double-click `run_app.bat` to launch the dashboard.
- Or run `streamlit run app.py` in the project environment.

## Project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard entry point |
| `page_*.py` | Overview, Sales, Segmentation, Churn, Customer Prediction, Pipeline/Deployment |
| `pipeline.py` | Data cleaning, RFM, K-Means, churn labels and model training |
| `build_results.py` / `.bat` | Builds results locally from the original dataset |
| `config.py`, `data_loader.py`, `aggregates.py`, `segments.py`, `predict.py`, `charts.py`, `utils.py` | Supporting modules |

**Privacy note:** inspect generated customer-level CSV files before committing them to a public repository. The dashboard uses customer IDs in some files for lookup/prediction. Never publish the original dataset or raw transaction-level exports.
