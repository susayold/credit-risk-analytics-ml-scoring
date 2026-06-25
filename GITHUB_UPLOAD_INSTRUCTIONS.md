# GitHub Upload Instructions

## 1. Initialize Git

```bash
cd D:/Code/DA/credit-risk-analytics-ml-scoring
git init
```

## 2. Enable Git LFS

This repository includes files larger than 100 MB:

- `data/processed/final_customer_analysis_table.csv.gz`
- `data/processed/final_customer_analysis_train.csv.gz`
- `dashboard/credit_risk_dashboard.pbix`

Use Git LFS before committing:

```bash
git lfs install
git lfs track "*.pbix" "*.csv.gz" "*.pptx" "*.docx" "*.pdf" "*.xlsx"
git add .gitattributes
```

## 3. Commit

```bash
git add .
git commit -m "Add credit risk analytics and ML scoring pipeline"
```

## 4. Connect Remote Repository

```bash
git remote add origin https://github.com/<your-username>/<your-repo>.git
git branch -M main
git push -u origin main
```

## Important

Raw Kaggle CSV files are not committed. They should be downloaded into `data/raw/` before running the full ETL pipeline.

The processed outputs and dashboard are included with Git LFS so the repository still shows complete results.

