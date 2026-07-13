# Metrics Source of Truth

This file documents where each headline portfolio metric comes from. The goal is to make the README, website, CV bullets, and presentation traceable back to committed output tables.

## How To Audit

1. Open `docs/METRICS_SOURCE_OF_TRUTH.csv`.
2. For each row, use `source_file`, optional `selector_column` and `selector_value`, then read `source_column`.
3. The repository test `tests/test_repository_contract.py` validates that the committed source files still match the stated values.

## Metric Groups

| Group | Source files |
|---|---|
| Data volume and feature count | `outputs/final/portfolio_baseline.csv`, `outputs/final/feature_inventory.csv` |
| Step 8 data split | `outputs/final/data_split_report.csv` |
| Champion model | `outputs/final/champion_validation_metrics.csv` |
| Challenger benchmark | `outputs/final/model_comparison.csv` |
| Cross-validation | `outputs/final/cross_validation_metrics.csv` |
| Review-priority bands | `outputs/final/decision_bands.csv` |
| Dashboard rule-based segments | `outputs/final/dashboard_rule_segments.csv` |
| Diagnostic Logistic Regression | `outputs/final/diagnostic_model_metrics.csv`, `outputs/final/diagnostic_odds_ratios.csv` |
| SHAP explainability | `outputs/final/shap_importance.csv` |

## Interpretation Guardrails

- The LightGBM and ensemble metrics are validation metrics for risk ranking and prioritization, not an automated rejection policy.
- The champion single model is LightGBM v3. The weighted ensemble is retained as a calibrated challenger/benchmark.
- The Logistic Regression metrics and odds ratios are used for diagnostic interpretation, not as the strongest predictive model.
- Dashboard default-rate/lift segments and Logistic Regression odds ratios are different metric families and should not be substituted for one another.
- Raw Kaggle CSV files are intentionally not committed. The processed train/test/master files are committed with Git LFS for portfolio reproducibility.
- The repository does not claim to be a production credit decisioning service. It is an end-to-end analytics and ML support project.
