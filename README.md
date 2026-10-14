# Customer Intelligence System — Streamlit dashboard

One flat folder (no sub-folders), so it uploads to GitHub in one go.

## 3 steps
1. **Build the results (once).** Double-click `build_results.bat`, choose `online_retail_II.xlsx` when the window asks.
   It installs the libraries, builds every result file next to `app.py`, and pins the scikit-learn version in `requirements.txt`.
2. **Check locally (optional).** Double-click `run_app.bat` → the dashboard opens in your browser.
3. **Publish.** Double-click `publish_to_github.bat` (needs Git) and paste the address of a NEW EMPTY GitHub repository.
   Everything is uploaded in one go. The raw dataset and any huge file are skipped automatically (`.gitignore`).

Then go to https://share.streamlit.io → **Create app** → pick the repository, branch `main`, main file `app.py` → **Deploy**.

> No Git? Upload every file in this folder to GitHub with **Add file → Upload files** (select all files, drag them in).
> Keep the dataset (`.xlsx`) out of that upload.

## What is inside
| File | Purpose |
|---|---|
| `app.py` | Dashboard entry point (the "Main file" on Streamlit Cloud) |
| `page_*.py` | The pages: Overview, Sales, Segmentation, Churn, Customer Prediction, Run Pipeline |
| `pipeline.py` | The whole ML pipeline (cleaning → RFM → K-Means → churn label → models) |
| `build_results.py` / `.bat` | Builds the result files from the dataset, no browser needed |
| `publish_to_github.bat` | One-click upload to GitHub |
| `config.py`, `data_loader.py`, `aggregates.py`, `segments.py`, `predict.py`, `charts.py`, `utils.py` | Helpers |

## Result files created by step 1 (all small)
`sales_kpis.json`, `sales_monthly.csv`, `sales_products.csv`, `sales_countries.csv`, `rfm_kmeans_customers.csv`,
`kmeans_model.joblib`, `kmeans_scaler.joblib`, `segment_names.json`, `churn_model_pipeline.joblib`, `churn_model_meta.json`,
`churn_scores_current_customers.csv`, `churn_model_comparison.csv`, `churn_test_predictions.csv`,
`churn_label_summary.csv`, `churn_feature_importance.csv`.

## The pages
- **Customer Prediction:** type one customer's data (with ready-made examples), upload a transactions file for many customers, or pick an existing customer.
- **Churn Prediction:** churn rate, confusion matrix, ROC / PR / calibration, model comparison vs a baseline, risk by segment, priority list (CSV).
