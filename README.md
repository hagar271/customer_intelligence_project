# Customer Intelligence System — Streamlit dashboard

This repository contains the Customer Intelligence System dashboard source code.

## Demo preview without uploading the dataset

The app now opens a **Dashboard Preview** automatically when generated result/model files are absent. This preview shows the intended layout for sales analytics, customer segmentation, churn-risk visuals, model comparison, and the ML workflow without requiring `online_retail_II.xlsx`.

**Important:** values shown in preview mode are illustrative sample values, not real results computed from the Online Retail II dataset. Model metrics and churn visuals in preview mode are not predictions from a trained model.

## Show the actual project results (optional)

If you want the dashboard to show the real computed results and perform real customer predictions, the generated summary files and saved model must be made available to the app. The original Excel file itself does not need to be uploaded to GitHub or Streamlit Cloud.

1. On your own computer, place `online_retail_II.xlsx` beside `build_results.bat`.
2. Run `build_results.bat` and wait for the pipeline to finish.
3. Run `run_app.bat` and confirm the dashboard works locally.
4. Upload only the generated result/model files to this repository; do **not** upload `online_retail_II.xlsx` or `clean_online_retail.csv`.
5. Deploy this repository in Streamlit Community Cloud with branch `main` and main file `app.py`. If it is already deployed, reboot the app after committing changes.

## Generated result files

Core files used for real results include:

- `rfm_kmeans_customers.csv`
- `churn_model_pipeline.joblib`
- `churn_model_meta.json`
- `churn_scores_current_customers.csv`
- `sales_kpis.json`
- `sales_monthly.csv`
- `sales_products.csv`
- `sales_countries.csv`

Optional files enable additional charts and segmentation/prediction features:

- `kmeans_model.joblib`, `kmeans_scaler.joblib`, `segment_names.json`
- `churn_model_comparison.csv`, `churn_test_predictions.csv`, `churn_label_summary.csv`, `churn_feature_importance.csv`

## Run locally

- Windows: double-click `run_app.bat`.
- Or run `streamlit run app.py` in the project environment.

## Main files

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard entry point and missing-artifact fallback |
| `demo_dashboard.py` | Presentation preview shown when generated artifacts are absent |
| `page_*.py` | Overview, Sales, Segmentation, Churn, Customer Prediction, Pipeline/Deployment |
| `pipeline.py` | Data cleaning, RFM, K-Means, churn labels and model training |
| `build_results.py` / `.bat` | Builds results locally from the original dataset |
| `config.py`, `data_loader.py`, `aggregates.py`, `segments.py`, `predict.py`, `charts.py`, `utils.py` | Supporting modules |

**Privacy note:** never publish the original dataset or raw transaction-level exports. Inspect customer-level CSV files for identifiers before committing generated artifacts to a public repository.
