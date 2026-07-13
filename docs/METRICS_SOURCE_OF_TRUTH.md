# Metrics Source of Truth

This file documents where each headline portfolio metric comes from. The goal is to make the README, website, CV bullets, and presentation traceable back to committed output tables.

## How To Audit

1. Open `docs/METRICS_SOURCE_OF_TRUTH.csv`.
2. For each row, use `source_file`, optional `selector_column` and `selector_value`, then read `source_column`.
3. The repository test `tests/test_repository_contract.py` validates that the committed source files still match the stated values.

## Metric Groups

| Group | Source files |
|---|---|
| Data volume and feature count | `outputs/tables/step08_ml/16_full_271_feature_inventory.csv` |
| Step 8 data split | `outputs/tables/step08_ml/extracted_tables/v3_advanced_cell06_out02_table01.csv` |
| ML benchmark | `outputs/tables/step08_ml/extracted_tables/v3_advanced_cell09_out04_table01.csv`, `v3_advanced_cell16_out01_table01.csv` |
| Cross-validation | `outputs/tables/step08_ml/extracted_tables/v3_advanced_cell11_out14_table01.csv` |
| Decision bands | `outputs/tables/step08_ml/extracted_tables/v3_advanced_cell12_out01_table01.csv` |
| Diagnostic Logistic Regression | `outputs/tables/step08_ml/12_core_interpretable_logit_model_metrics.csv`, `13_core_interpretable_logit_odds_ratio_with_ci_pvalue.csv` |
| SHAP explainability | `outputs/tables/step08_ml/extracted_tables/v3_advanced_cell16_out09_table01.csv` |

## Interpretation Guardrails

- The LightGBM and ensemble metrics are validation metrics for risk ranking and prioritization, not an automated rejection policy.
- The Logistic Regression metrics and odds ratios are used for diagnostic interpretation, not as the strongest predictive model.
- Raw Kaggle CSV files are intentionally not committed. The processed train/test/master files are committed with Git LFS for portfolio reproducibility.
- The repository does not claim to be a production credit decisioning service. It is an end-to-end analytics and ML support project.
