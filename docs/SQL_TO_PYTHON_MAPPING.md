# SQL to Python Mapping

## Summary

The SQL scripts map to the Python pipeline as follows.

| SQL file | Replaces / documents Python logic | Main SQL concepts |
|---|---|---|
| `01_create_base_application.sql` | Load labeled `application_train` as base dataset | `CREATE VIEW`, `SELECT` |
| `02_cleaning_missing_flags.sql` | Step 3 cleaning flags and missing flags | `CASE WHEN`, `IS NULL`, `COALESCE` |
| `03_feature_engineering_application.sql` | Step 3 application feature engineering | ratio formulas, `NULLIF`, derived columns |
| `04_aggregate_bureau_and_bureau_balance.sql` | Step 5 bureau and bureau_balance aggregation | `JOIN`, `GROUP BY`, `COUNT`, `SUM`, `AVG`, `MAX`, `MIN` |
| `05_aggregate_previous_pos_installments_credit_card.sql` | Step 5 previous/POS/installment/credit-card aggregation | `GROUP BY`, flags, rates |
| `06_build_customer_master_table.sql` | Step 5 final customer-level master table | `LEFT JOIN`, history coverage flags |
| `07_descriptive_statistics_and_segments.sql` | Step 4 descriptive statistics and segmentation | `AVG`, `STDDEV_SAMP`, `NTILE`, default rate |

## Interview-Safe Explanation

Do not say:

> The whole project was done only in SQL.

Say:

> I implemented the analytics and ML pipeline in Python, then added a SQL ETL version for the data engineering layer. SQL covers joins, cleaning flags, missing flags, feature engineering, aggregation, and master-table construction. Python remains the modeling layer.

