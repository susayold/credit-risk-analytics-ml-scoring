# SQL ETL Version - Credit Risk Project

## Purpose

This folder adds a SQL-first ETL reference layer for the Credit Risk Analytics project.

The original project was executed mainly with Python/pandas for data preparation and Python ML libraries for modeling. This SQL package documents how the data engineering parts can be implemented with SQL:

- data cleaning flags
- missing-value flags
- application feature engineering
- historical table aggregation
- customer-level master table build
- descriptive statistics and risk segmentation queries

## Scope

This SQL package supports the CV wording:

> Used SQL joins and aggregations for data cleaning, missing-value flagging, feature engineering, and consolidating multi-table historical credit data at the customer level.

## What SQL Covers

| Project area | SQL coverage |
|---|---|
| Step 3 Data Cleaning | Missing flags, special-value flags, cleaned date/employment fields |
| Step 4 Descriptive Analytics | Descriptive statistics and quantile/bin default-rate queries |
| Step 5 Master Table Aggregation | Full customer-level aggregation and left joins by `SK_ID_CURR` |
| Step 6 Dashboard Data | SQL master table can feed dashboard datasets |

## What SQL Does Not Replace

SQL does not replace the ML model training layer:

- Logistic Regression
- LightGBM
- PR-AUC / ROC-AUC evaluation
- SHAP explainability
- fairness/sensitivity model tests

Those are still handled by Python because ML libraries are more appropriate for model training and explainability.

## Recommended Interview Explanation

Use this wording:

> The project was executed mainly in Python, but I added a SQL ETL version for the data engineering layer. SQL handles cleaning flags, missing-value flags, feature engineering, multi-table aggregation, and the customer-level master table. Python is used for model training and ML explainability.

