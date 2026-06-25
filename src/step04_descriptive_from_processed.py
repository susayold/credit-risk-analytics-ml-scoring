from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RUN_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step04_descriptive_run"
TABLE_DIR = RUN_OUTPUT_DIR / "tables"
FIG_DIR = RUN_OUTPUT_DIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def find_train_master() -> Path:
    candidates = [
        PROJECT_ROOT / "outputs" / "step05_master_table_run" / "data" / "final_customer_analysis_train.csv.gz",
        PROCESSED_DIR / "final_customer_analysis_train.csv.gz",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find final_customer_analysis_train.csv.gz")


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    rows = []
    for col in numeric_cols:
        s = df[col]
        rows.append(
            {
                "feature": col,
                "count": int(s.count()),
                "missing_count": int(s.isna().sum()),
                "missing_pct": float(s.isna().mean() * 100),
                "mean": s.mean(),
                "std": s.std(),
                "min": s.min(),
                "p25": s.quantile(0.25),
                "median_p50": s.quantile(0.50),
                "p75": s.quantile(0.75),
                "p95": s.quantile(0.95),
                "max": s.max(),
                "skewness": s.skew(),
                "kurtosis": s.kurtosis(),
                "sum": s.sum(),
            }
        )
    return pd.DataFrame(rows).sort_values("feature")


def quantile_default_rate(df: pd.DataFrame, feature: str, n_bins: int = 5) -> pd.DataFrame:
    work = df[["TARGET", feature]].dropna().copy()
    work["quantile_bin"] = pd.qcut(work[feature], q=n_bins, labels=False, duplicates="drop") + 1
    return (
        work.groupby("quantile_bin")
        .agg(
            n_customers=("TARGET", "size"),
            default_rate=("TARGET", "mean"),
            feature_min=(feature, "min"),
            feature_max=(feature, "max"),
        )
        .reset_index()
    )


def main() -> None:
    input_path = find_train_master()
    print(f"Loading {input_path}")
    df = pd.read_csv(input_path)

    desc = descriptive_stats(df)
    desc.to_csv(TABLE_DIR / "descriptive_all_numeric_features.csv", index=False)

    selected = [
        "APP_CREDIT_INCOME_RATIO",
        "APP_ANNUITY_INCOME_RATIO",
        "APP_AGE_YEARS",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "BUREAU_DEBT_TO_CREDIT_SUM_RATIO",
        "INST_LATE_PAYMENT_RATE",
        "INST_UNDERPAYMENT_RATE",
        "CC_UTILIZATION_MEAN",
        "CC_UTILIZATION_MAX",
    ]
    available = [c for c in selected if c in df.columns]
    if "TARGET" in df.columns and available:
        corr = df[["TARGET"] + available].corr(method="spearman")
        corr.to_csv(TABLE_DIR / "selected_spearman_correlation.csv")

        quantile_outputs = []
        for feature in available[:6]:
            q = quantile_default_rate(df, feature)
            q.insert(0, "feature", feature)
            quantile_outputs.append(q)
        pd.concat(quantile_outputs, ignore_index=True).to_csv(TABLE_DIR / "quantile_default_rates.csv", index=False)

    if "TARGET" in df.columns:
        baseline = pd.DataFrame(
            [
                {
                    "n_applications": len(df),
                    "n_default": int(df["TARGET"].sum()),
                    "baseline_default_rate": float(df["TARGET"].mean()),
                }
            ]
        )
        baseline.to_csv(TABLE_DIR / "portfolio_baseline_default_rate.csv", index=False)

    print(f"Saved descriptive outputs to {TABLE_DIR}")


if __name__ == "__main__":
    main()

