# Reproducibility Audit

This audit responds to the main review gap: the project already contains the data outputs and analytical evidence, but the evidence needs to be easy to verify.

## What Is Reproducible From The Repository

| Area | Status | Evidence |
|---|---|---|
| Processed customer-level data | Available | `data/processed/final_customer_analysis_train.csv.gz`, `final_customer_analysis_test.csv.gz`, `final_customer_analysis_table.csv.gz` |
| Feature count and target baseline | Available | `outputs/tables/step08_ml/16_full_271_feature_inventory.csv` |
| SQL aggregation logic | Available | `sql/01_create_base_application.sql` to `sql/07_descriptive_statistics_and_segments.sql` |
| Python pipeline logic | Available | `src/`, `run_pipeline.py`, `requirements.txt` |
| Descriptive/statistical evidence | Available | `outputs/tables/step04_descriptive/`, `outputs/figures/step04_descriptive/` |
| Diagnostic Logistic Regression | Available | `outputs/tables/step07_diagnostic/`, `outputs/tables/step08_ml/12_*`, `13_*` |
| ML benchmark metrics | Available | `outputs/tables/step08_ml/extracted_tables/` |
| Dashboard deliverable | Available | `dashboard/dashboard.pbix`, `dashboard/dashboard.pdf`, `dashboard/screenshots/` |
| Website deliverable | Available | `site/` |
| Metric traceability tests | Added | `tests/test_repository_contract.py` |

## Known Limits

| Limit | Reason | Impact |
|---|---|---|
| Raw Home Credit CSV files are not committed | Size and licensing practicality | Reviewer must download raw data separately to rebuild from zero |
| Notebook output cells are not embedded | The final evidence was extracted into committed CSV/PNG outputs | Reviewer should audit `outputs/` and `docs/METRICS_SOURCE_OF_TRUTH.csv` instead of relying on notebook rendering |
| Binary trained model object is not committed | The project is positioned as analytics and decision support, not production serving | Model-card artifacts document champion selection, metrics, feature signals, and governance checks |
| Python tests are lightweight | Avoids requiring large LFS data in CI | CI checks repository contracts and metric traceability, not full model retraining |

## Positioning

Use this project as:

- Credit risk analytics portfolio
- Data analyst / risk analyst case study
- SQL and Python data engineering evidence
- Power BI dashboard and business communication evidence
- ML scoring support and model governance evidence

Do not present it as:

- A production underwriting system
- A fully regulated Basel/IFRS9 PD model
- A live model-serving API with monitoring and retraining

## Suggested Interview Framing

This repository is reproducible at the evidence layer: the processed customer-level data, SQL/Python pipeline scripts, dashboard, output tables, figures, and model metrics are committed. Raw Kaggle files are excluded, but the pipeline documents where they should be placed. The ML model is presented as a ranking and prioritization layer, while Logistic Regression and SHAP are used to explain risk drivers.
