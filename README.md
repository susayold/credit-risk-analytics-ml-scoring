# Credit Risk Analytics & ML Scoring Pipeline

End-to-end credit risk analytics project on Home Credit loan applications.

The project builds a customer-level risk analytics table, creates SQL ETL references, produces dashboard-ready outputs, and supports ML-based risk prioritization.

## Highlights

- Built on 307K+ labeled loan applications.
- Created a 271-feature customer-level master table.
- Identified high-risk credit card utilization segment with 25.5% default rate, 3.16x above the 8.07% portfolio baseline.
- ML review prioritization reached ROC-AUC 0.7907, PR-AUC 0.3127, and Lift@10 3.66x.
- Reviewing the top 29.6% highest-risk applications captured about 67.9% of defaults, improving review efficiency by 2.27x versus random review.

## Repository Structure

```text
credit-risk-analytics-ml-scoring/
  src/                 Python pipeline and analytics scripts
  sql/                 SQL ETL version for cleaning, feature engineering, joins, and aggregation
  notebooks/           ML notebook for model benchmarking and governance
  data/
    raw/               Place Kaggle raw CSV files here
    processed/         Final processed master outputs
  outputs/
    tables/            Result tables by project step
    figures/           Result figures by project step
  dashboard/           Power BI dashboard
  reports/             Final report and presentation
  docs/                SQL mapping and project documentation
```

## Data Setup

Download the Home Credit Default Risk dataset from Kaggle and place these files in `data/raw/`:

```text
application_train.csv
application_test.csv
bureau.csv
bureau_balance.csv
previous_application.csv
POS_CASH_balance.csv
installments_payments.csv
credit_card_balance.csv
sample_submission.csv
HomeCredit_columns_description.csv
```

Raw CSV files are intentionally ignored by Git because they are large. The processed master outputs are included and tracked with Git LFS.

## Run the Pipeline

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main pipeline:

```bash
python run_pipeline.py
```

If raw data exists in `data/raw`, the script runs Step 5 master-table aggregation first. If raw data is not available, it uses the included processed master output to run descriptive analytics.

To run only the descriptive analytics from the processed file:

```bash
python run_pipeline.py --skip-master
```

## SQL ETL Version

The `sql/` folder documents the SQL-first data engineering layer:

- cleaning flags
- missing-value flags
- feature engineering
- historical table aggregation
- customer-level master table build
- descriptive statistics and segment queries

Python remains the modeling layer for Logistic Regression, LightGBM, SHAP, and model evaluation.

## Dashboard

Power BI dashboard:

```text
dashboard/credit_risk_dashboard.pbix
```

This file is larger than normal GitHub file limits and should be tracked using Git LFS.

## Git LFS

Before pushing to GitHub:

```bash
git lfs install
git lfs track "*.pbix" "*.csv.gz" "*.pptx" "*.docx" "*.pdf" "*.xlsx"
git add .gitattributes
```

Then commit and push normally.

