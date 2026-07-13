/*
07_descriptive_statistics_and_segments.sql

Creates descriptive statistics and default-rate segmentation queries.
This maps to Step 4 descriptive analytics and quantile/bin logic.
*/

/* Portfolio baseline */
CREATE OR REPLACE VIEW v07_portfolio_baseline AS
SELECT
    COUNT(*) AS n_applications,
    AVG(TARGET * 1.0) AS baseline_default_rate
FROM v06_customer_master_table
WHERE TARGET IS NOT NULL;

/* Selected descriptive statistics for business-facing variables */
CREATE OR REPLACE VIEW v07_selected_descriptive_stats AS
SELECT
    'APP_CREDIT_INCOME_RATIO' AS variable_name,
    COUNT(APP_CREDIT_INCOME_RATIO) AS n_non_missing,
    AVG(APP_CREDIT_INCOME_RATIO) AS mean_value,
    STDDEV_SAMP(APP_CREDIT_INCOME_RATIO) AS std_value,
    MIN(APP_CREDIT_INCOME_RATIO) AS min_value,
    MAX(APP_CREDIT_INCOME_RATIO) AS max_value
FROM v06_customer_master_table
UNION ALL
SELECT
    'APP_ANNUITY_INCOME_RATIO',
    COUNT(APP_ANNUITY_INCOME_RATIO),
    AVG(APP_ANNUITY_INCOME_RATIO),
    STDDEV_SAMP(APP_ANNUITY_INCOME_RATIO),
    MIN(APP_ANNUITY_INCOME_RATIO),
    MAX(APP_ANNUITY_INCOME_RATIO)
FROM v06_customer_master_table
UNION ALL
SELECT
    'CC_UTILIZATION_MEAN',
    COUNT(CC_UTILIZATION_MEAN),
    AVG(CC_UTILIZATION_MEAN),
    STDDEV_SAMP(CC_UTILIZATION_MEAN),
    MIN(CC_UTILIZATION_MEAN),
    MAX(CC_UTILIZATION_MEAN)
FROM v06_customer_master_table
UNION ALL
SELECT
    'INST_LATE_PAYMENT_RATE',
    COUNT(INST_LATE_PAYMENT_RATE),
    AVG(INST_LATE_PAYMENT_RATE),
    STDDEV_SAMP(INST_LATE_PAYMENT_RATE),
    MIN(INST_LATE_PAYMENT_RATE),
    MAX(INST_LATE_PAYMENT_RATE)
FROM v06_customer_master_table;

/* Quantile bin example: credit/income ratio default-rate curve */
CREATE OR REPLACE VIEW v07_credit_income_quantile_default_rate AS
WITH binned AS (
    SELECT
        SK_ID_CURR,
        TARGET,
        APP_CREDIT_INCOME_RATIO,
        NTILE(5) OVER (ORDER BY APP_CREDIT_INCOME_RATIO) AS credit_income_bin
    FROM v06_customer_master_table
    WHERE TARGET IS NOT NULL
      AND APP_CREDIT_INCOME_RATIO IS NOT NULL
)
SELECT
    credit_income_bin,
    COUNT(*) AS n_customers,
    AVG(TARGET * 1.0) AS default_rate,
    MIN(APP_CREDIT_INCOME_RATIO) AS bin_min,
    MAX(APP_CREDIT_INCOME_RATIO) AS bin_max
FROM binned
GROUP BY credit_income_bin;

/* High-risk segment example used for CV/business explanation */
CREATE OR REPLACE VIEW v07_credit_card_over_100_segment AS
SELECT
    CASE
        WHEN CC_UTILIZATION_MAX > 1 THEN 'CC utilization above 100%'
        ELSE 'CC utilization at or below 100%'
    END AS segment_name,
    COUNT(*) AS n_customers,
    AVG(TARGET * 1.0) AS default_rate
FROM v06_customer_master_table
WHERE TARGET IS NOT NULL
  AND HAS_CREDIT_CARD_HISTORY = 1
GROUP BY
    CASE
        WHEN CC_UTILIZATION_MAX > 1 THEN 'CC utilization above 100%'
        ELSE 'CC utilization at or below 100%'
    END;

