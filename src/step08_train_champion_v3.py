"""
Step 8 - Champion ML model and governance evidence.

This script is the official entry point for the Step 8 model layer.

Two modes are available:

1. evidence (default)
   Rebuilds canonical outputs/final/*.csv from committed notebook evidence.
   This is fast and is used by CI and repository contract tests.

2. train
   Retrains a LightGBM v3-style champion from the processed customer-level
   table, then exports validation, calibration, score-distribution shift, SHAP,
   fairness and decision-band tables. Exact metrics can vary with library
   versions and hardware; the committed evidence tables remain the published
   source of truth.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(os.environ.get("CREDIT_RISK_PROJECT_DIR", Path(__file__).resolve().parents[1]))
OUTPUT_FINAL = PROJECT_ROOT / "outputs" / "final"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

RANDOM_STATE = 42
TARGET = "TARGET"
ID_COL = "SK_ID_CURR"
VALIDATION_ROWS = 61_504
CALIBRATION_ROWS = 49_202

EVIDENCE = {
    "feature_inventory": PROJECT_ROOT / "outputs/tables/step08_ml/16_full_271_feature_inventory.csv",
    "data_split": PROJECT_ROOT / "outputs/tables/step08_ml/extracted_tables/v3_advanced_cell06_out02_table01.csv",
    "holdout_models": PROJECT_ROOT / "outputs/tables/step08_ml/extracted_tables/v3_advanced_cell09_out04_table01.csv",
    "cv_models": PROJECT_ROOT / "outputs/tables/step08_ml/extracted_tables/v3_advanced_cell11_out14_table01.csv",
    "decision_bands": PROJECT_ROOT / "outputs/tables/step08_ml/extracted_tables/v3_advanced_cell12_out01_table01.csv",
    "psi_summary": PROJECT_ROOT / "outputs/tables/step08_ml/extracted_tables/v3_advanced_cell13_out01_table01.csv",
    "psi_features": PROJECT_ROOT / "outputs/tables/step08_ml/extracted_tables/v3_advanced_cell13_out02_table01.csv",
    "fairness": PROJECT_ROOT / "outputs/tables/step08_ml/extracted_tables/v3_advanced_cell15_out03_table01.csv",
    "final_model_comparison": PROJECT_ROOT / "outputs/tables/step08_ml/extracted_tables/v3_advanced_cell16_out01_table01.csv",
    "shap": PROJECT_ROOT / "outputs/tables/step08_ml/extracted_tables/v3_advanced_cell16_out09_table01.csv",
    "diagnostic_metrics": PROJECT_ROOT / "outputs/tables/step08_ml/12_core_interpretable_logit_model_metrics.csv",
    "diagnostic_or": PROJECT_ROOT / "outputs/tables/step08_ml/13_core_interpretable_logit_odds_ratio_with_ci_pvalue.csv",
    "dashboard_segments": PROJECT_ROOT / "outputs/tables/step07_diagnostic/segment_risk_index_all_variables.csv",
}


def display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        if rows:
            fieldnames = list(rows[0].keys())
        else:
            raise ValueError(f"Cannot infer CSV fieldnames for empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logging.info("Wrote %s", display_path(path))


def copy_csv(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    logging.info("Wrote %s", display_path(dst))


def select_one(rows: list[dict[str, str]], column: str, value: str) -> dict[str, str]:
    selected = [row for row in rows if row.get(column) == value]
    if len(selected) != 1:
        raise ValueError(f"Expected one row where {column}={value!r}, found {len(selected)}")
    return selected[0]


def rename_scorecard_label(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    renamed = []
    for row in rows:
        out = dict(row)
        if out.get("model_name") == "Logistic scorecard-like v3":
            out["model_name"] = "WOE Logistic Regression Benchmark"
        renamed.append(out)
    return renamed


def export_portfolio_baseline(output_dir: Path = OUTPUT_FINAL) -> None:
    row = read_rows(EVIDENCE["feature_inventory"])[0]
    rows = [
        {
            "metric_id": "labeled_rows",
            "value_numeric": row["total_rows"],
            "value_display": f"{int(row['total_rows']):,}",
            "source_note": "application_train labeled population",
        },
        {
            "metric_id": "final_master_columns",
            "value_numeric": row["total_columns_in_train"],
            "value_display": row["total_columns_in_train"],
            "source_note": "customer-level master table width",
        },
        {
            "metric_id": "portfolio_default_rate",
            "value_numeric": row["target_default_rate"],
            "value_display": f"{float(row['target_default_rate']) * 100:.2f}%",
            "source_note": "TARGET=1 baseline in labeled population",
        },
    ]
    write_rows(output_dir / "portfolio_baseline.csv", rows)


def export_dashboard_rule_segments(output_dir: Path = OUTPUT_FINAL) -> None:
    source_rows = read_rows(EVIDENCE["dashboard_segments"])
    keys = {
        ("DIAG_CC_UTILIZATION_GROUP", ">100%"),
        ("DIAG_BUREAU_OVERDUE_GROUP", ">=2 overdue loans"),
        ("DIAG_BUREAU_DEBT_CREDIT_GROUP", ">100%"),
    }
    rows = [
        row
        for row in source_rows
        if (row.get("variable"), row.get("segment")) in keys
    ]
    write_rows(output_dir / "dashboard_rule_segments.csv", rows)


def export_decision_bands(output_dir: Path = OUTPUT_FINAL) -> None:
    label_map = {
        "AUTO_APPROVE": "LOW_PRIORITY_REVIEW",
        "MANUAL_REVIEW": "MANUAL_REVIEW",
        "STRICT_REVIEW_OR_REJECT": "ENHANCED_REVIEW",
    }
    rows = []
    for row in read_rows(EVIDENCE["decision_bands"]):
        out = dict(row)
        out["presentation_label"] = label_map.get(out["decision"], out["decision"])
        out["decision_scope"] = "Illustrative review-priority bands, not production approval rules."
        rows.append(out)
    write_rows(output_dir / "decision_bands.csv", rows)


def export_fairness_and_sensitivity(output_dir: Path = OUTPUT_FINAL) -> None:
    copy_csv(EVIDENCE["fairness"], output_dir / "fairness_feature_gaps_original.csv")
    sensitivity_rows = [
        {"version": "Original", "roc_auc": "0.7838", "delta_auc": "0"},
        {"version": "Drop occupation/organization", "roc_auc": "0.7835", "delta_auc": "-0.0003"},
        {"version": "Drop occupation/organization/gender", "roc_auc": "0.7824", "delta_auc": "-0.0014"},
    ]
    fairness_summary_rows = [
        {"group": "Occupation", "original_gap": "0.268", "drop_occupation_organization": "0.238", "drop_occupation_organization_gender": "0.229"},
        {"group": "Organization", "original_gap": "0.185", "drop_occupation_organization": "0.197", "drop_occupation_organization_gender": "0.180"},
        {"group": "Gender", "original_gap": "0.048", "drop_occupation_organization": "0.049", "drop_occupation_organization_gender": "0.030"},
    ]
    write_rows(output_dir / "sensitivity_performance.csv", sensitivity_rows)
    write_rows(output_dir / "fairness_summary.csv", fairness_summary_rows)


def export_evidence_outputs(output_dir: Path = OUTPUT_FINAL) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    export_portfolio_baseline(output_dir)
    copy_csv(EVIDENCE["feature_inventory"], output_dir / "feature_inventory.csv")
    copy_csv(EVIDENCE["data_split"], output_dir / "data_split_report.csv")

    holdout_rows = read_rows(EVIDENCE["holdout_models"])
    champion = select_one(holdout_rows, "model_name", "LightGBM v3")
    write_rows(output_dir / "champion_validation_metrics.csv", [champion])

    comparison_rows = rename_scorecard_label(read_rows(EVIDENCE["final_model_comparison"]))
    write_rows(output_dir / "model_comparison.csv", comparison_rows)
    copy_csv(EVIDENCE["cv_models"], output_dir / "cross_validation_metrics.csv")
    export_decision_bands(output_dir)
    copy_csv(EVIDENCE["psi_summary"], output_dir / "psi_summary.csv")
    copy_csv(EVIDENCE["psi_features"], output_dir / "psi_feature_summary.csv")
    export_fairness_and_sensitivity(output_dir)
    copy_csv(EVIDENCE["shap"], output_dir / "shap_importance.csv")
    copy_csv(EVIDENCE["diagnostic_metrics"], output_dir / "diagnostic_model_metrics.csv")
    copy_csv(EVIDENCE["diagnostic_or"], output_dir / "diagnostic_odds_ratios.csv")
    export_dashboard_rule_segments(output_dir)
    write_final_manifest(output_dir)


def write_final_manifest(output_dir: Path = OUTPUT_FINAL) -> None:
    rows = []
    for path in sorted(output_dir.glob("*.csv")):
        if path.name == "final_outputs_manifest.csv":
            continue
        rows.append(
            {
                "relative_path": display_path(path),
                "size_bytes": path.stat().st_size,
                "status": "final",
                "artifact_type": "csv_evidence",
            }
        )
    write_rows(output_dir / "final_outputs_manifest.csv", rows)


def require_train_dependencies() -> None:
    missing: list[str] = []
    for module in ["numpy", "pandas", "sklearn", "lightgbm", "scipy"]:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise RuntimeError(
            "Missing training dependencies: "
            + ", ".join(missing)
            + ". Install with: pip install -r requirements-lock.txt"
        )


def load_published_split_frames(df, split_dir: Path = SPLIT_DIR):
    import pandas as pd

    split_files = {
        "fit_train": split_dir / "fit_train_ids.csv",
        "calibration": split_dir / "calibration_ids.csv",
        "final_validation": split_dir / "final_validation_ids.csv",
    }
    missing = [str(path) for path in split_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing committed Step 8 split ID files: "
            + ", ".join(missing)
            + ". The train mode requires fixed split IDs to reproduce the published split counts."
        )

    keyed = df.set_index(ID_COL, drop=False)
    frames = {}
    seen_ids: set[int] = set()
    for split_name, path in split_files.items():
        split_ids = pd.read_csv(path)
        required_cols = {ID_COL, TARGET}
        if not required_cols.issubset(split_ids.columns):
            raise ValueError(f"{path} must contain {sorted(required_cols)}")
        ids = split_ids[ID_COL].astype(int).tolist()
        if len(ids) != len(set(ids)):
            raise ValueError(f"{path} contains duplicate {ID_COL} values")
        overlap = seen_ids.intersection(ids)
        if overlap:
            raise ValueError(f"Split ID overlap detected in {split_name}: {len(overlap)} duplicate IDs")
        seen_ids.update(ids)
        missing_ids = sorted(set(ids).difference(keyed.index.astype(int)))
        if missing_ids:
            raise ValueError(f"{path} contains IDs not found in processed train data; first missing ID: {missing_ids[0]}")

        part = keyed.loc[ids].reset_index(drop=True)
        expected_target = split_ids[TARGET].astype(int).reset_index(drop=True)
        actual_target = part[TARGET].astype(int).reset_index(drop=True)
        if not actual_target.equals(expected_target):
            raise ValueError(f"{path} TARGET values do not match processed train data")
        frames[split_name] = part

    if len(seen_ids) != len(df):
        raise ValueError(f"Split files cover {len(seen_ids):,} IDs, but processed train has {len(df):,} rows")
    return frames


def split_xy(frame):
    y = frame[TARGET].astype(int)
    X = frame.drop(columns=[TARGET])
    if ID_COL in X.columns:
        X = X.drop(columns=[ID_COL])
    return X, y


def run_train_mode(train_file: Path, test_file: Path, output_dir: Path) -> None:
    require_train_dependencies()

    import pandas as pd
    from lightgbm import LGBMClassifier
    from scipy.stats import ks_2samp
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    logging.info("Loading processed train file: %s", train_file)
    df = pd.read_csv(train_file, compression="gzip")
    if TARGET not in df.columns:
        raise ValueError(f"{TARGET} column is required in {train_file}")

    df = add_v3_features(df)
    split_frames = load_published_split_frames(df, SPLIT_DIR)
    X_fit, y_fit = split_xy(split_frames["fit_train"])
    X_cal, y_cal = split_xy(split_frames["calibration"])
    X_val, y_val = split_xy(split_frames["final_validation"])

    data_split_rows = [
        {"split": "fit_train", "rows": len(X_fit), "default_rate": float(y_fit.mean())},
        {"split": "calibration", "rows": len(X_cal), "default_rate": float(y_cal.mean())},
        {"split": "final_validation", "rows": len(X_val), "default_rate": float(y_val.mean())},
    ]
    if test_file.exists():
        opener = gzip.open if test_file.suffix == ".gz" else open
        with opener(test_file, "rt", encoding="utf-8", errors="ignore") as handle:
            test_rows = sum(1 for _ in handle) - 1
        data_split_rows.append({"split": "test", "rows": test_rows, "default_rate": ""})

    X_fit = prepare_lgbm_frame(X_fit)
    X_cal = prepare_lgbm_frame(X_cal)
    X_val = prepare_lgbm_frame(X_val)

    model = LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        colsample_bytree=0.7,
        subsample=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=80,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )
    model.fit(X_fit, y_fit, eval_set=[(X_cal, y_cal)], eval_metric="auc")

    calibrator = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    calibrator.fit(X_cal, y_cal)
    raw_score = model.predict_proba(X_val)[:, 1]
    cal_score = calibrator.predict_proba(X_val)[:, 1]

    champion_row = evaluate_scores("LightGBM v3 retrain", y_val.to_numpy(), cal_score, time.time() - started, ks_2samp)
    write_rows(output_dir / "data_split_report.csv", data_split_rows)
    write_rows(output_dir / "champion_validation_metrics.csv", [champion_row])
    write_rows(output_dir / "decision_bands.csv", build_decision_bands(y_val.to_numpy(), cal_score))
    write_rows(output_dir / "calibration_score_distribution_shift.csv", build_score_distribution_shift(raw_score, cal_score))
    write_rows(output_dir / "fairness_summary.csv", build_fairness_rows(X_val, y_val.to_numpy(), cal_score))
    write_rows(output_dir / "shap_importance.csv", build_shap_rows(model, X_val))
    write_final_manifest(output_dir)


def add_v3_features(df):
    import numpy as np

    out = df.copy()
    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in out.columns]
    if ext_cols:
        out["V3_EXT_SOURCE_MEAN"] = out[ext_cols].mean(axis=1)
        out["V3_EXT_SOURCE_MIN"] = out[ext_cols].min(axis=1)
        out["V3_EXT_SOURCE_MAX"] = out[ext_cols].max(axis=1)
    if {"EXT_SOURCE_2", "EXT_SOURCE_3"}.issubset(out.columns):
        out["V3_EXT_SOURCE_2_x_EXT_SOURCE_3"] = out["EXT_SOURCE_2"] * out["EXT_SOURCE_3"]
    if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(out.columns):
        out["V3_ANNUITY_CREDIT_RATIO"] = out["AMT_ANNUITY"] / out["AMT_CREDIT"].replace(0, np.nan)
    if {"AMT_GOODS_PRICE", "AMT_CREDIT"}.issubset(out.columns):
        out["V3_GOODS_CREDIT_RATIO"] = out["AMT_GOODS_PRICE"] / out["AMT_CREDIT"].replace(0, np.nan)
    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(out.columns):
        out["V3_CREDIT_INCOME_RATIO"] = out["AMT_CREDIT"] / out["AMT_INCOME_TOTAL"].replace(0, np.nan)
    if {"V3_EXT_SOURCE_MEAN", "V3_ANNUITY_CREDIT_RATIO"}.issubset(out.columns):
        out["V3_EXT_MEAN_X_CREDIT_BURDEN"] = out["V3_EXT_SOURCE_MEAN"] * out["V3_ANNUITY_CREDIT_RATIO"]
    return out


def prepare_lgbm_frame(df):
    import numpy as np
    import pandas as pd

    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].fillna("Missing").astype("category")
        elif pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].replace([np.inf, -np.inf], np.nan)
            out[col] = out[col].fillna(out[col].median())
    return out


def evaluate_scores(model_name: str, y_true, score, runtime_seconds: float, ks_2samp_func) -> dict[str, object]:
    import numpy as np
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

    y_true = np.asarray(y_true).astype(int)
    score = np.clip(np.asarray(score, dtype=float), 1e-7, 1 - 1e-7)
    row = {
        "model_name": model_name,
        "runtime_seconds": round(runtime_seconds, 6),
        "n": int(len(y_true)),
        "event_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "gini": float(2 * roc_auc_score(y_true, score) - 1),
        "pr_auc": float(average_precision_score(y_true, score)),
        "log_loss": float(log_loss(y_true, score, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, score)),
        "ks": float(ks_2samp_func(score[y_true == 1], score[y_true == 0]).statistic),
    }
    for pct in [0.05, 0.10, 0.20, 0.30]:
        row.update(top_percentile_metrics(y_true, score, pct))
    return row


def top_percentile_metrics(y_true, score, pct: float) -> dict[str, object]:
    import numpy as np

    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    k = max(1, int(math.ceil(len(y_true) * pct)))
    idx = np.argsort(-score)[:k]
    default_top = float(y_true[idx].sum())
    precision_top = default_top / k
    recall_top = default_top / max(float(y_true.sum()), 1.0)
    lift_top = precision_top / float(y_true.mean())
    label = int(pct * 100)
    return {
        f"precision_at_top_{label}pct": precision_top,
        f"recall_at_top_{label}pct": recall_top,
        f"lift_at_top_{label}pct": lift_top,
    }


def build_decision_bands(y_true, score) -> list[dict[str, object]]:
    import numpy as np
    import pandas as pd

    df = pd.DataFrame({"target": y_true, "score": score})
    q70 = df["score"].quantile(0.704409)
    q90 = df["score"].quantile(0.899909)
    df["decision"] = np.where(
        df["score"] >= q90,
        "ENHANCED_REVIEW",
        np.where(df["score"] >= q70, "MANUAL_REVIEW", "LOW_PRIORITY_REVIEW"),
    )
    rows = []
    for decision, group in df.groupby("decision"):
        rows.append(
            {
                "decision": decision,
                "customers": int(len(group)),
                "customer_share": float(len(group) / len(df)),
                "default_rate": float(group["target"].mean()),
                "avg_score": float(group["score"].mean()),
                "defaults": int(group["target"].sum()),
                "decision_scope": "Illustrative review-priority bands, not production approval rules.",
            }
        )
    return rows


def build_score_distribution_shift(reference_score, comparison_score, bins: int = 10) -> list[dict[str, object]]:
    import numpy as np
    import pandas as pd

    reference_score = np.asarray(reference_score, dtype=float)
    comparison_score = np.asarray(comparison_score, dtype=float)
    edges = np.unique(np.quantile(reference_score, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return [{"shift_type": "raw_vs_calibrated_score_shift", "score_distribution_shift_index": 0.0}]
    ref_bins = pd.cut(reference_score, edges, include_lowest=True, duplicates="drop")
    cmp_bins = pd.cut(comparison_score, edges, include_lowest=True, duplicates="drop")
    ref_pct = ref_bins.value_counts(normalize=True, sort=False).replace(0, 1e-6)
    cmp_pct = cmp_bins.value_counts(normalize=True, sort=False).replace(0, 1e-6)
    shift_index = float(((cmp_pct - ref_pct) * np.log(cmp_pct / ref_pct)).sum())
    return [{"shift_type": "raw_vs_calibrated_score_shift", "score_distribution_shift_index": shift_index}]


def build_fairness_rows(X, y_true, score) -> list[dict[str, object]]:
    import numpy as np
    import pandas as pd

    rows = []
    high_risk = score >= np.quantile(score, 0.70)
    for col in ["OCCUPATION_TYPE", "ORGANIZATION_TYPE", "CODE_GENDER"]:
        if col not in X.columns:
            continue
        tmp = pd.DataFrame({"group": X[col].astype(str), "target": y_true, "high_risk": high_risk})
        grouped = tmp.groupby("group").agg(rows=("target", "size"), high_risk_rate=("high_risk", "mean")).reset_index()
        grouped = grouped[grouped["rows"] >= 200]
        gap = float(grouped["high_risk_rate"].max() - grouped["high_risk_rate"].min()) if not grouped.empty else 0.0
        rows.append({"group": col, "original_gap": gap, "note": "High-risk-rate gap by observed group."})
    return rows or [{"group": "not_available", "original_gap": "", "note": "Fairness columns were not present."}]


def build_shap_rows(model, X_val) -> list[dict[str, object]]:
    try:
        import numpy as np
        import shap
    except ImportError:
        return [{"feature": "shap_not_available", "mean_abs_shap": "", "note": "Install shap to export SHAP values."}]

    sample = X_val.sample(min(2000, len(X_val)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    if isinstance(values, list):
        values = values[1]
    mean_abs = np.abs(values).mean(axis=0)
    rows = [
        {"feature": feature, "mean_abs_shap": float(value)}
        for feature, value in zip(sample.columns, mean_abs)
    ]
    return sorted(rows, key=lambda row: row["mean_abs_shap"], reverse=True)[:40]


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Train or rebuild Step 8 champion ML outputs.")
    parser.add_argument("--mode", choices=["evidence", "train"], default="evidence")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=PROJECT_ROOT / "data/processed/final_customer_analysis_train.csv.gz",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=PROJECT_ROOT / "data/processed/final_customer_analysis_test.csv.gz",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_FINAL)
    args = parser.parse_args()

    if args.mode == "evidence":
        export_evidence_outputs(args.output_dir)
    else:
        run_train_mode(args.train_file, args.test_file, args.output_dir)


if __name__ == "__main__":
    main()
