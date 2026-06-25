"""
Step 6 â€” Diagnostic Analytics: Validating Dashboard Insights

Purpose
-------
This script validates the main patterns found in the Power BI dashboard before
moving to Machine Learning Support.

It does NOT try to find the best Kaggle prediction model. Instead, it compares
several diagnostic logistic specifications and produces business-oriented
tables:

1. Diagnostic feature engineering
2. Risk index / lift by segment
3. Contribution analysis
4. Statistical tests and effect sizes
5. Interaction diagnostics
6. Diagnostic logistic model comparison
7. Odds ratio interpretation table

Input
-----
Preferred:
    D:/Code/DA/1/step4_outputs/data/final_customer_analysis_train.csv.gz

Fallback:
    D:/Code/DA/1/powerbi_inputs/final_customer_analysis_train.csv

Kaggle:
    Put final_customer_analysis_train.csv or .csv.gz in a Kaggle input dataset,
    then update INPUT_PATH manually below if auto-detection cannot find it.

Outputs
-------
    step6_outputs/
        step6_diagnostic_analytics_report.html
        step6_key_results_for_chat.md
        data/diagnostic_customer_dataset.csv.gz
        tables/*.csv
        tables/research_logistic_model_fit.csv
        tables/research_logistic_coefficients.csv
        tables/step6_diagnostic_summary_tables.xlsx
        charts/*.png
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# 0. Configuration
# =============================================================================

LOCAL_PROJECT_DIR = Path("D:/Code/DA/1")
PROJECT_DIR = LOCAL_PROJECT_DIR if LOCAL_PROJECT_DIR.exists() else Path.cwd()

# Optional heavy ML packages can be installed locally to the project folder,
# which is useful when the user site-packages drive has limited free space.
LOCAL_PACKAGE_DIR = PROJECT_DIR / ".python_packages"
if LOCAL_PACKAGE_DIR.exists():
    sys.path.insert(0, str(LOCAL_PACKAGE_DIR))

# If you run on Kaggle and auto-detection fails, set this manually, e.g.
# INPUT_PATH = Path("/kaggle/input/your-dataset/final_customer_analysis_train.csv.gz")
INPUT_PATH: Path | None = None

OUTPUT_DIR = PROJECT_DIR / "step6_outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
CHART_DIR = OUTPUT_DIR / "charts"
DATA_DIR = OUTPUT_DIR / "data"
PROGRESS_LOG_PATH = OUTPUT_DIR / "step6_progress.log"
PROGRESS_STATE_PATH = OUTPUT_DIR / "step6_progress_state.json"

TARGET = "TARGET"
RANDOM_STATE = 42
SCRIPT_START_TIME = time.time()

# For statsmodels research-style logistic regression. None means use all rows.
# This is slower, but gives the full-data inferential result.
INFERENTIAL_MAX_ROWS: int | None = None

# For sklearn model comparison. None means use all rows for maximum-quality
# model comparison results. This can be slower, but avoids sampling noise.
MODEL_COMPARISON_MAX_ROWS: int | None = None
MODEL_TEST_SIZE = 0.25
RUN_OPTIONAL_BOOSTING_MODELS = True

# Diagnostic add-ons. None means use all available rows.
MARGINAL_EFFECT_MAX_ROWS: int | None = None
VIF_MAX_ROWS: int | None = None

# Minimum group size flag for diagnostic caution.
SMALL_SAMPLE_THRESHOLD = 500


# =============================================================================
# 1. Utility
# =============================================================================

def ensure_dirs() -> None:
    for p in [OUTPUT_DIR, TABLE_DIR, CHART_DIR, DATA_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def format_duration(seconds: float | int | None) -> str:
    if seconds is None or pd.isna(seconds):
        return "unknown"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def reset_progress_log() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    header = [
        "Step 6 Diagnostic Analytics progress log",
        f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "-" * 90,
    ]
    PROGRESS_LOG_PATH.write_text("\n".join(header) + "\n", encoding="utf-8")
    if PROGRESS_STATE_PATH.exists():
        PROGRESS_STATE_PATH.unlink()


def write_progress_state(stage: str, done: int, total: int, current: str, status: str, start_time: float) -> None:
    total = max(total, 1)
    elapsed = time.time() - start_time
    pct_done = done / total
    eta = (elapsed / done * (total - done)) if done > 0 else None
    state = {
        "stage": stage,
        "status": status,
        "current": current,
        "done": int(done),
        "total": int(total),
        "percent": round(pct_done * 100, 2),
        "elapsed_seconds": round(elapsed, 2),
        "elapsed": format_duration(elapsed),
        "eta_seconds": round(eta, 2) if eta is not None else None,
        "eta": format_duration(eta),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        PROGRESS_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception:
        pass


def log(msg: str) -> None:
    elapsed = format_duration(time.time() - SCRIPT_START_TIME)
    timestamp = time.strftime("%H:%M:%S")
    line = f"[STEP 6][{timestamp}][elapsed {elapsed}] {msg}"
    print(line, flush=True)
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with PROGRESS_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


class ProgressTracker:
    def __init__(self, stage: str, total: int):
        self.stage = stage
        self.total = max(total, 1)
        self.done = 0
        self.start_time = time.time()
        log(f"[{self.stage}] START | total tasks={self.total}")
        write_progress_state(self.stage, self.done, self.total, "Starting", "running", self.start_time)

    def start(self, item: str) -> None:
        next_step = min(self.done + 1, self.total)
        pct_now = (self.done / self.total) * 100
        log(f"[{self.stage}] RUNNING {next_step}/{self.total} ({pct_now:.1f}%) | {item}")
        write_progress_state(self.stage, self.done, self.total, item, "running", self.start_time)

    def finish(self, item: str, status: str = "done") -> None:
        self.done = min(self.done + 1, self.total)
        elapsed = time.time() - self.start_time
        eta = (elapsed / self.done * (self.total - self.done)) if self.done else None
        pct_done = (self.done / self.total) * 100
        log(
            f"[{self.stage}] {status.upper()} {self.done}/{self.total} "
            f"({pct_done:.1f}%) | {item} | elapsed={format_duration(elapsed)} | ETA={format_duration(eta)}"
        )
        write_progress_state(self.stage, self.done, self.total, item, status, self.start_time)

    def skip(self, item: str, reason: str) -> None:
        self.done = min(self.done + 1, self.total)
        elapsed = time.time() - self.start_time
        eta = (elapsed / self.done * (self.total - self.done)) if self.done else None
        pct_done = (self.done / self.total) * 100
        log(
            f"[{self.stage}] SKIP {self.done}/{self.total} "
            f"({pct_done:.1f}%) | {item} | reason={reason} | ETA={format_duration(eta)}"
        )
        write_progress_state(self.stage, self.done, self.total, item, "skipped", self.start_time)

    def done_all(self) -> None:
        elapsed = time.time() - self.start_time
        log(f"[{self.stage}] COMPLETE | elapsed={format_duration(elapsed)}")
        write_progress_state(self.stage, self.total, self.total, "Complete", "complete", self.start_time)


def find_input_path() -> Path:
    if INPUT_PATH is not None:
        return Path(INPUT_PATH)

    candidates = [
        PROJECT_DIR / "step4_outputs/data/final_customer_analysis_train.csv.gz",
        PROJECT_DIR / "powerbi_inputs/final_customer_analysis_train.csv",
        PROJECT_DIR / "final_customer_analysis_train.csv.gz",
        PROJECT_DIR / "final_customer_analysis_train.csv",
    ]

    for c in candidates:
        if c.exists():
            return c

    # Kaggle auto-discovery
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        matches = list(kaggle_input.rglob("final_customer_analysis_train.csv.gz"))
        matches += list(kaggle_input.rglob("final_customer_analysis_train.csv"))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        "Cannot find final_customer_analysis_train.csv(.gz). "
        "Please set INPUT_PATH at the top of this script."
    )


def read_final_dataset(path: Path) -> pd.DataFrame:
    log(f"Reading final analytical dataset: {path}")
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Input data must contain TARGET column. Columns found: {len(df.columns)}")
    df = df[df[TARGET].notna()].copy()
    df[TARGET] = df[TARGET].astype(int)
    log(f"Loaded shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    return df


def has_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def pct(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.2%}"


def pp(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x * 100:.2f} pp"


def fmt_float(x: float, n: int = 4) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.{n}f}"


def ordered_cat(series: pd.Series, categories: list[str]) -> pd.Categorical:
    return pd.Categorical(series, categories=categories, ordered=True)


def save_df(df: pd.DataFrame, name: str) -> Path:
    path = TABLE_DIR / name
    if df is None or (df.empty and len(df.columns) == 0):
        df = pd.DataFrame({"note": ["No data produced for this table."]})
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# =============================================================================
# 2. Diagnostic Feature Engineering
# =============================================================================

def make_group_features(df: pd.DataFrame) -> pd.DataFrame:
    log("Creating diagnostic grouped features...")
    out = df.copy()

    # -------------------------
    # Customer profile
    # -------------------------
    if has_col(out, "APP_AGE_YEARS"):
        age = safe_num(out["APP_AGE_YEARS"])
    elif has_col(out, "DAYS_BIRTH"):
        age = (-safe_num(out["DAYS_BIRTH"]) / 365.25)
    else:
        age = pd.Series(np.nan, index=out.index)

    age_bins = [-np.inf, 30, 40, 50, 60, np.inf]
    age_labels = ["1. <=30", "2. 31-40", "3. 41-50", "4. 51-60", "5. 61+"]
    out["DIAG_AGE_GROUP"] = pd.cut(age, bins=age_bins, labels=age_labels).astype("object")
    out["DIAG_AGE_GROUP"] = out["DIAG_AGE_GROUP"].fillna("Unknown")

    for c in [
        "NAME_CONTRACT_TYPE",
        "CODE_GENDER",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
    ]:
        if has_col(out, c):
            out[f"DIAG_{c}"] = out[c].astype("object").fillna("Unknown")

    if has_col(out, "OCCUPATION_TYPE"):
        occ = out["OCCUPATION_TYPE"].astype("object").fillna("Unknown")
        top_occ = occ.value_counts(dropna=False).head(10).index
        out["DIAG_OCCUPATION_GROUP"] = np.where(occ.isin(top_occ), occ, "Other")
    else:
        out["DIAG_OCCUPATION_GROUP"] = "Unknown"

    # -------------------------
    # Affordability
    # -------------------------
    credit_income = safe_num(out.get("APP_CREDIT_INCOME_RATIO", pd.Series(np.nan, index=out.index)))
    credit_income_labels = ["1. <=1x", "2. 1x-2x", "3. 2x-4x", "4. 4x-6x", "5. >6x"]
    out["DIAG_CREDIT_INCOME_GROUP"] = pd.cut(
        credit_income,
        bins=[-np.inf, 1, 2, 4, 6, np.inf],
        labels=credit_income_labels,
    ).astype("object").fillna("Unknown")

    annuity_income = safe_num(out.get("APP_ANNUITY_INCOME_RATIO", pd.Series(np.nan, index=out.index)))
    annuity_income_labels = ["1. <=10%", "2. 10-20%", "3. 20-30%", "4. 30-40%", "5. 40-60%", "6. 60%+"]
    out["DIAG_ANNUITY_INCOME_GROUP"] = pd.cut(
        annuity_income,
        bins=[-np.inf, 0.10, 0.20, 0.30, 0.40, 0.60, np.inf],
        labels=annuity_income_labels,
    ).astype("object").fillna("Unknown")

    # -------------------------
    # Credit history
    # -------------------------
    has_bureau = safe_num(out.get("HAS_BUREAU_HISTORY", pd.Series(0, index=out.index))).fillna(0)
    active = safe_num(out.get("BUREAU_ACTIVE_COUNT", pd.Series(np.nan, index=out.index))).fillna(0)
    out["DIAG_BUREAU_ACTIVE_LOAN_GROUP"] = np.select(
        [
            has_bureau.eq(0),
            active.eq(0),
            active.eq(1),
            active.between(2, 3, inclusive="both"),
            active.gt(3),
        ],
        ["No bureau history", "0 active", "1 active", "2-3 active", ">3 active"],
        default="Unknown",
    )

    overdue = safe_num(out.get("BUREAU_OVERDUE_LOAN_COUNT", pd.Series(np.nan, index=out.index))).fillna(0)
    out["DIAG_BUREAU_OVERDUE_GROUP"] = np.select(
        [
            has_bureau.eq(0),
            overdue.eq(0),
            overdue.eq(1),
            overdue.ge(2),
        ],
        ["No bureau history", "No overdue", "1 overdue loan", ">=2 overdue loans"],
        default="Unknown",
    )

    debt_credit = safe_num(out.get("BUREAU_DEBT_TO_CREDIT_SUM_RATIO", pd.Series(np.nan, index=out.index)))
    out["DIAG_BUREAU_DEBT_CREDIT_GROUP"] = np.where(
        has_bureau.eq(0),
        "No bureau history",
        pd.cut(
            debt_credit,
            bins=[-np.inf, 0, 0.25, 0.50, 1.00, np.inf],
            labels=["No bureau debt", "<=25%", "25%-50%", "50%-100%", ">100%"],
        ).astype("object").fillna("Unknown"),
    )

    has_prev = safe_num(out.get("HAS_PREVIOUS_APPLICATION", pd.Series(0, index=out.index))).fillna(0)
    prev_refusal = safe_num(out.get("PREV_REFUSAL_RATE", pd.Series(np.nan, index=out.index))).fillna(0)
    out["DIAG_PREVIOUS_REFUSAL_GROUP"] = np.select(
        [
            has_prev.eq(0),
            prev_refusal.eq(0),
            prev_refusal.gt(0) & prev_refusal.le(0.25),
            prev_refusal.gt(0.25) & prev_refusal.le(0.50),
            prev_refusal.gt(0.50),
        ],
        ["No previous application", "0% refusal", "<=25% refusal", "25%-50% refusal", ">50% refusal"],
        default="Unknown",
    )

    # -------------------------
    # Payment behavior
    # -------------------------
    has_inst = safe_num(out.get("HAS_INSTALLMENT_HISTORY", pd.Series(0, index=out.index))).fillna(0)
    late = safe_num(out.get("INST_LATE_PAYMENT_RATE", pd.Series(np.nan, index=out.index))).fillna(0)
    out["DIAG_LATE_PAYMENT_GROUP"] = np.select(
        [
            has_inst.eq(0),
            late.eq(0),
            late.gt(0) & late.le(0.10),
            late.gt(0.10) & late.le(0.30),
            late.gt(0.30),
        ],
        ["No installment history", "No late payment", "Late <=10%", "Late 10%-30%", "Late >30%"],
        default="Unknown",
    )

    underpay = safe_num(out.get("INST_UNDERPAYMENT_RATE", pd.Series(np.nan, index=out.index))).fillna(0)
    out["DIAG_UNDERPAYMENT_GROUP"] = np.select(
        [
            has_inst.eq(0),
            underpay.eq(0),
            underpay.gt(0) & underpay.le(0.10),
            underpay.gt(0.10) & underpay.le(0.30),
            underpay.gt(0.30),
        ],
        ["No installment history", "No underpayment", "Underpay <=10%", "Underpay 10%-30%", "Underpay >30%"],
        default="Unknown",
    )

    pay_ratio = safe_num(out.get("INST_PAYMENT_RATIO_MEAN", pd.Series(np.nan, index=out.index)))
    out["DIAG_PAYMENT_RATIO_GROUP"] = np.where(
        has_inst.eq(0),
        "No installment history",
        pd.cut(
            pay_ratio,
            bins=[-np.inf, 0.80, 1.00, 1.05, np.inf],
            labels=["<80% paid", "80%-100% paid", "Full/near full paid", ">105% paid"],
        ).astype("object").fillna("Unknown"),
    )

    has_cc = safe_num(out.get("HAS_CREDIT_CARD_HISTORY", pd.Series(0, index=out.index))).fillna(0)
    cc_util = safe_num(out.get("CC_UTILIZATION_MEAN", pd.Series(np.nan, index=out.index)))
    out["DIAG_CC_UTILIZATION_GROUP"] = np.where(
        has_cc.eq(0),
        "No credit card history",
        np.where(
            cc_util.le(0).fillna(False),
            "Zero/no utilization",
            pd.cut(
                cc_util,
                bins=[-np.inf, 0.30, 0.70, 1.00, np.inf],
                labels=["<=30%", "30%-70%", "70%-100%", ">100%"],
            ).astype("object").fillna("Unknown"),
        ),
    )

    has_pos = safe_num(out.get("HAS_POS_CASH_HISTORY", pd.Series(0, index=out.index))).fillna(0)
    pos_dpd = safe_num(out.get("POS_DPD_MONTH_COUNT", pd.Series(np.nan, index=out.index))).fillna(0)
    out["DIAG_POS_DPD_GROUP"] = np.select(
        [
            has_pos.eq(0),
            pos_dpd.eq(0),
            pos_dpd.gt(0),
        ],
        ["No POS/CASH history", "No POS DPD", "Has POS DPD"],
        default="Unknown",
    )

    # -------------------------
    # Rebuild Page 6 style rule-based risk scores
    # -------------------------
    affordability_score = (
        ((credit_income > 2) & (credit_income <= 6)).astype(int)
        + (annuity_income > 0.20).astype(int)
    )

    credit_history_score = (
        (has_bureau.eq(0)).astype(int)
        + (active > 3).astype(int)
        + (overdue > 0).astype(int)
        + (debt_credit > 0.50).fillna(False).astype(int)
        + (prev_refusal > 0.25).astype(int)
    )

    payment_behavior_score = (
        (late > 0.10).astype(int)
        + (underpay > 0.10).astype(int)
        + (pay_ratio < 1.00).fillna(False).astype(int)
        + (cc_util > 0.70).fillna(False).astype(int)
        + (pos_dpd > 0).astype(int)
    )

    out["DIAG_AFFORDABILITY_SIGNAL_SCORE"] = affordability_score
    out["DIAG_CREDIT_HISTORY_SIGNAL_SCORE"] = credit_history_score
    out["DIAG_PAYMENT_BEHAVIOR_SIGNAL_SCORE"] = payment_behavior_score
    out["DIAG_RISK_SIGNAL_SCORE"] = affordability_score + credit_history_score + payment_behavior_score

    risk_score = out["DIAG_RISK_SIGNAL_SCORE"]
    out["DIAG_RISK_SIGNAL_GROUP"] = np.select(
        [
            risk_score <= 1,
            risk_score.between(2, 3, inclusive="both"),
            risk_score.between(4, 5, inclusive="both"),
            risk_score >= 6,
        ],
        ["1. Low signal", "2. Medium signal", "3. High signal", "4. Very high signal"],
        default="Unknown",
    )

    # Interaction variables for the diagnostic model comparison.
    out["DIAG_INT_AFFORDABILITY_LATE"] = (
        out["DIAG_CREDIT_INCOME_GROUP"].astype(str) + " | " + out["DIAG_LATE_PAYMENT_GROUP"].astype(str)
    )
    out["DIAG_INT_REFUSAL_OVERDUE"] = (
        out["DIAG_PREVIOUS_REFUSAL_GROUP"].astype(str) + " | " + out["DIAG_BUREAU_OVERDUE_GROUP"].astype(str)
    )
    out["DIAG_INT_PAYMENT_CC"] = (
        out["DIAG_PAYMENT_RATIO_GROUP"].astype(str) + " | " + out["DIAG_CC_UTILIZATION_GROUP"].astype(str)
    )

    diag_cols = [c for c in out.columns if c.startswith("DIAG_")]
    log(f"Created {len(diag_cols)} diagnostic columns")
    return out


# =============================================================================
# 3. Risk Index / Contribution Analysis
# =============================================================================

DIAGNOSTIC_GROUP_VARS = [
    "DIAG_AGE_GROUP",
    "DIAG_NAME_CONTRACT_TYPE",
    "DIAG_NAME_INCOME_TYPE",
    "DIAG_NAME_EDUCATION_TYPE",
    "DIAG_OCCUPATION_GROUP",
    "DIAG_CREDIT_INCOME_GROUP",
    "DIAG_ANNUITY_INCOME_GROUP",
    "DIAG_BUREAU_ACTIVE_LOAN_GROUP",
    "DIAG_BUREAU_OVERDUE_GROUP",
    "DIAG_BUREAU_DEBT_CREDIT_GROUP",
    "DIAG_PREVIOUS_REFUSAL_GROUP",
    "DIAG_LATE_PAYMENT_GROUP",
    "DIAG_UNDERPAYMENT_GROUP",
    "DIAG_PAYMENT_RATIO_GROUP",
    "DIAG_CC_UTILIZATION_GROUP",
    "DIAG_POS_DPD_GROUP",
    "DIAG_RISK_SIGNAL_GROUP",
]


def segment_risk_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    base_rate = df[TARGET].mean()
    total_customers = len(df)
    total_defaults = df[TARGET].sum()

    g = (
        df.groupby(group_col, dropna=False)[TARGET]
        .agg(customer_count="count", default_customers="sum", default_rate="mean")
        .reset_index()
        .rename(columns={group_col: "segment"})
    )
    g.insert(0, "variable", group_col)
    g["non_default_customers"] = g["customer_count"] - g["default_customers"]
    g["customer_share"] = g["customer_count"] / total_customers
    g["default_contribution_share"] = np.where(total_defaults > 0, g["default_customers"] / total_defaults, np.nan)
    g["risk_index"] = g["default_rate"] / base_rate
    g["risk_difference"] = g["default_rate"] - base_rate
    g["small_sample_flag"] = np.where(g["customer_count"] < SMALL_SAMPLE_THRESHOLD, 1, 0)
    g["business_priority_score"] = g["risk_difference"].clip(lower=0) * g["default_customers"]
    return g.sort_values(["variable", "risk_index"], ascending=[True, False])


def make_all_segment_tables(df: pd.DataFrame) -> pd.DataFrame:
    log("Building segment risk index and contribution tables...")
    frames = []
    for col in DIAGNOSTIC_GROUP_VARS:
        if col in df.columns:
            frames.append(segment_risk_table(df, col))
    out = pd.concat(frames, ignore_index=True)
    save_df(out, "segment_risk_index_all_variables.csv")

    top_lift = (
        out[out["customer_count"] >= SMALL_SAMPLE_THRESHOLD]
        .sort_values("risk_index", ascending=False)
        .head(30)
        .copy()
    )
    save_df(top_lift, "top_30_highest_risk_index_segments.csv")

    top_impact = (
        out[out["risk_difference"] > 0]
        .sort_values("business_priority_score", ascending=False)
        .head(30)
        .copy()
    )
    save_df(top_impact, "top_30_business_priority_segments.csv")
    return out


# =============================================================================
# 4. Statistical Tests / Effect Size
# =============================================================================

def cramers_v_from_chi2(chi2: float, n: int, r: int, k: int) -> float:
    if n <= 0:
        return np.nan
    denom = n * (min(k - 1, r - 1))
    if denom <= 0:
        return np.nan
    return math.sqrt(chi2 / denom)


def categorical_tests(df: pd.DataFrame, group_vars: list[str]) -> pd.DataFrame:
    log("Running categorical chi-square tests and CramÃ©r's V...")
    try:
        from scipy.stats import chi2_contingency
        scipy_available = True
    except Exception:
        chi2_contingency = None
        scipy_available = False
        log("  scipy is not available. Chi-square p-values will be left blank, but CramÃ©r's V will still be calculated.")

    rows = []
    for col in group_vars:
        if col not in df.columns:
            continue
        temp = df[[col, TARGET]].dropna().copy()
        if temp[col].nunique(dropna=False) < 2:
            continue
        ct = pd.crosstab(temp[col], temp[TARGET])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        if scipy_available:
            chi2, p_value, dof, expected = chi2_contingency(ct)
        else:
            observed = ct.values.astype(float)
            row_sums = observed.sum(axis=1, keepdims=True)
            col_sums = observed.sum(axis=0, keepdims=True)
            total = observed.sum()
            expected = row_sums @ col_sums / total if total > 0 else np.zeros_like(observed)
            with np.errstate(divide="ignore", invalid="ignore"):
                chi2 = np.nansum((observed - expected) ** 2 / expected)
            dof = (ct.shape[0] - 1) * (ct.shape[1] - 1)
            p_value = np.nan
        v = cramers_v_from_chi2(chi2, ct.values.sum(), ct.shape[0], ct.shape[1])
        rows.append(
            {
                "variable": col,
                "n": int(ct.values.sum()),
                "levels": int(ct.shape[0]),
                "chi2": chi2,
                "dof": dof,
                "p_value": p_value,
                "cramers_v": v,
                "effect_size_label": effect_size_label_cramers(v),
            }
        )
    out = pd.DataFrame(rows).sort_values("cramers_v", ascending=False)
    save_df(out, "categorical_statistical_tests.csv")
    return out


def effect_size_label_cramers(v: float) -> str:
    if pd.isna(v):
        return "Unknown"
    if v < 0.05:
        return "Very weak"
    if v < 0.10:
        return "Weak"
    if v < 0.20:
        return "Moderate"
    return "Strong"


NUMERIC_TEST_VARS = [
    "APP_CREDIT_INCOME_RATIO",
    "APP_ANNUITY_INCOME_RATIO",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "APP_AGE_YEARS",
    "BUREAU_ACTIVE_COUNT",
    "BUREAU_OVERDUE_LOAN_COUNT",
    "BUREAU_DEBT_TO_CREDIT_SUM_RATIO",
    "PREV_REFUSAL_RATE",
    "INST_LATE_PAYMENT_RATE",
    "INST_UNDERPAYMENT_RATE",
    "INST_PAYMENT_RATIO_MEAN",
    "CC_UTILIZATION_MEAN",
    "POS_DPD_MONTH_RATE",
    "DIAG_RISK_SIGNAL_SCORE",
]


def numeric_tests(df: pd.DataFrame, numeric_vars: list[str]) -> pd.DataFrame:
    log("Running numerical Mann-Whitney / KS tests and effect sizes...")
    try:
        from scipy.stats import mannwhitneyu, ks_2samp
        scipy_available = True
    except Exception:
        mannwhitneyu = None
        ks_2samp = None
        scipy_available = False
        log("  scipy is not available. Numeric test p-values will be left blank, but rank effect size and KS statistic will still be calculated.")

    rows = []
    y = df[TARGET].astype(int)
    for col in numeric_vars:
        if col not in df.columns:
            continue
        x = safe_num(df[col])
        non_default = x[(y == 0) & x.notna()]
        default = x[(y == 1) & x.notna()]
        if len(non_default) < 30 or len(default) < 30:
            continue

        try:
            if not scipy_available:
                raise RuntimeError("scipy not available")
            u_stat, mw_p = mannwhitneyu(default, non_default, alternative="two-sided")
            auc_rank = u_stat / (len(default) * len(non_default))
            cliffs_delta = 2 * auc_rank - 1
        except Exception:
            combined = pd.concat(
                [
                    pd.DataFrame({"x": default.values, "g": 1}),
                    pd.DataFrame({"x": non_default.values, "g": 0}),
                ],
                ignore_index=True,
            )
            ranks = combined["x"].rank(method="average")
            n1 = len(default)
            n0 = len(non_default)
            rank_sum_default = ranks[combined["g"].eq(1)].sum()
            u_stat = rank_sum_default - n1 * (n1 + 1) / 2
            auc_rank = u_stat / (n1 * n0) if n1 > 0 and n0 > 0 else np.nan
            cliffs_delta = 2 * auc_rank - 1 if pd.notna(auc_rank) else np.nan
            mw_p = np.nan

        try:
            if not scipy_available:
                raise RuntimeError("scipy not available")
            ks_stat, ks_p = ks_2samp(default, non_default)
        except Exception:
            ks_stat = manual_ks_stat(default.values, non_default.values)
            ks_p = np.nan

        rows.append(
            {
                "variable": col,
                "n_non_default": int(len(non_default)),
                "n_default": int(len(default)),
                "mean_non_default": non_default.mean(),
                "mean_default": default.mean(),
                "median_non_default": non_default.median(),
                "median_default": default.median(),
                "mean_difference_default_minus_non_default": default.mean() - non_default.mean(),
                "mannwhitney_p_value": mw_p,
                "rank_auc_default_vs_non_default": auc_rank,
                "cliffs_delta": cliffs_delta,
                "abs_cliffs_delta": abs(cliffs_delta) if pd.notna(cliffs_delta) else np.nan,
                "ks_statistic": ks_stat,
                "ks_p_value": ks_p,
                "effect_size_label": effect_size_label_cliffs(abs(cliffs_delta) if pd.notna(cliffs_delta) else np.nan),
            }
        )

    out = pd.DataFrame(rows).sort_values(["abs_cliffs_delta", "ks_statistic"], ascending=False)
    save_df(out, "numeric_statistical_tests.csv")
    return out


def manual_ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    """Compute two-sample KS statistic without scipy."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    if len(a) == 0 or len(b) == 0:
        return np.nan
    values = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, values, side="right") / len(a)
    cdf_b = np.searchsorted(b, values, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def effect_size_label_cliffs(abs_delta: float) -> str:
    if pd.isna(abs_delta):
        return "Unknown"
    if abs_delta < 0.147:
        return "Very small"
    if abs_delta < 0.33:
        return "Small"
    if abs_delta < 0.474:
        return "Medium"
    return "Large"


# =============================================================================
# 5. Interaction Diagnostics
# =============================================================================

INTERACTION_PAIRS = [
    ("DIAG_CREDIT_INCOME_GROUP", "DIAG_ANNUITY_INCOME_GROUP"),
    ("DIAG_CREDIT_INCOME_GROUP", "DIAG_LATE_PAYMENT_GROUP"),
    ("DIAG_LATE_PAYMENT_GROUP", "DIAG_UNDERPAYMENT_GROUP"),
    ("DIAG_BUREAU_ACTIVE_LOAN_GROUP", "DIAG_PREVIOUS_REFUSAL_GROUP"),
    ("DIAG_BUREAU_DEBT_CREDIT_GROUP", "DIAG_PREVIOUS_REFUSAL_GROUP"),
    ("DIAG_CREDIT_HISTORY_SIGNAL_SCORE", "DIAG_PAYMENT_BEHAVIOR_SIGNAL_SCORE"),
]


def interaction_table(df: pd.DataFrame, row_col: str, col_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    temp = df[[row_col, col_col, TARGET]].copy()
    temp[row_col] = temp[row_col].astype(str)
    temp[col_col] = temp[col_col].astype(str)
    rate = pd.pivot_table(temp, values=TARGET, index=row_col, columns=col_col, aggfunc="mean")
    count = pd.pivot_table(temp, values=TARGET, index=row_col, columns=col_col, aggfunc="count")
    return rate, count


def make_interaction_outputs(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    log("Building interaction diagnostic matrices...")
    outputs = {}
    for row_col, col_col in INTERACTION_PAIRS:
        if row_col not in df.columns or col_col not in df.columns:
            continue
        rate, count = interaction_table(df, row_col, col_col)
        safe_name = f"{row_col}__x__{col_col}".replace("DIAG_", "").lower()
        rate_path = TABLE_DIR / f"interaction_default_rate_{safe_name}.csv"
        count_path = TABLE_DIR / f"interaction_customer_count_{safe_name}.csv"
        rate.to_csv(rate_path, encoding="utf-8-sig")
        count.to_csv(count_path, encoding="utf-8-sig")
        chart_path = plot_heatmap(rate, f"Default Rate: {row_col} x {col_col}", f"heatmap_{safe_name}.png")
        outputs[safe_name] = {
            "rate_csv": str(rate_path),
            "count_csv": str(count_path),
            "chart": str(chart_path),
        }
    return outputs


def plot_heatmap(rate_table: pd.DataFrame, title: str, filename: str) -> Path:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(11, 5.8))
    ax = sns.heatmap(
        rate_table,
        annot=True,
        fmt=".2%",
        cmap="coolwarm",
        linewidths=0.4,
        linecolor="#F2ECFF",
        cbar_kws={"format": "%.0f%%"},
    )
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    path = CHART_DIR / filename
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


# =============================================================================
# 6. Diagnostic Model Comparison
# =============================================================================

PROFILE_FEATURES = [
    "DIAG_AGE_GROUP",
    "DIAG_CODE_GENDER",
    "DIAG_NAME_CONTRACT_TYPE",
    "DIAG_NAME_INCOME_TYPE",
    "DIAG_NAME_EDUCATION_TYPE",
    "DIAG_OCCUPATION_GROUP",
]

AFFORDABILITY_FEATURES = [
    "DIAG_CREDIT_INCOME_GROUP",
    "DIAG_ANNUITY_INCOME_GROUP",
]

CREDIT_HISTORY_FEATURES = [
    "DIAG_BUREAU_ACTIVE_LOAN_GROUP",
    "DIAG_BUREAU_OVERDUE_GROUP",
    "DIAG_BUREAU_DEBT_CREDIT_GROUP",
    "DIAG_PREVIOUS_REFUSAL_GROUP",
]

PAYMENT_BEHAVIOR_FEATURES = [
    "DIAG_LATE_PAYMENT_GROUP",
    "DIAG_UNDERPAYMENT_GROUP",
    "DIAG_PAYMENT_RATIO_GROUP",
    "DIAG_CC_UTILIZATION_GROUP",
    "DIAG_POS_DPD_GROUP",
]

INTERACTION_FEATURES = [
    "DIAG_INT_AFFORDABILITY_LATE",
    "DIAG_INT_REFUSAL_OVERDUE",
    "DIAG_INT_PAYMENT_CC",
]

# A smaller, stable set for research-style inferential logistic regression.
# This is intentionally narrower than the sklearn model-comparison stage.
# It avoids repeated no-history categories and strongly overlapping payment
# variables that can produce non-convergence or meaningless coefficient tables.
INFERENTIAL_PROFILE_FEATURES = [
    "DIAG_AGE_GROUP",
    "DIAG_NAME_CONTRACT_TYPE",
    "DIAG_NAME_INCOME_TYPE",
    "DIAG_NAME_EDUCATION_TYPE",
]

INFERENTIAL_AFFORDABILITY_FEATURES = [
    "DIAG_CREDIT_INCOME_GROUP",
    "DIAG_ANNUITY_INCOME_GROUP",
]

INFERENTIAL_CREDIT_HISTORY_FEATURES = [
    "DIAG_BUREAU_DEBT_CREDIT_GROUP",
    "DIAG_PREVIOUS_REFUSAL_GROUP",
]

INFERENTIAL_PAYMENT_FEATURES = [
    "DIAG_LATE_PAYMENT_GROUP",
    "DIAG_CC_UTILIZATION_GROUP",
]

INFERENTIAL_PAYMENT_SENSITIVITY_FEATURES = [
    "DIAG_UNDERPAYMENT_GROUP",
]

REFERENCE_GROUPS = {
    "DIAG_AGE_GROUP": "3. 41-50",
    "DIAG_NAME_CONTRACT_TYPE": "Cash loans",
    "DIAG_NAME_INCOME_TYPE": "Working",
    "DIAG_NAME_EDUCATION_TYPE": "Secondary / secondary special",
    "DIAG_CREDIT_INCOME_GROUP": "1. <=1x",
    "DIAG_ANNUITY_INCOME_GROUP": "1. <=10%",
    "DIAG_BUREAU_DEBT_CREDIT_GROUP": "No bureau debt",
    "DIAG_PREVIOUS_REFUSAL_GROUP": "0% refusal",
    "DIAG_LATE_PAYMENT_GROUP": "No late payment",
    "DIAG_CC_UTILIZATION_GROUP": "No credit card history",
    "DIAG_UNDERPAYMENT_GROUP": "No underpayment",
}


def existing_features(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def make_model_specs(df: pd.DataFrame) -> dict[str, dict]:
    profile = existing_features(df, PROFILE_FEATURES)
    affordability = existing_features(df, AFFORDABILITY_FEATURES)
    credit = existing_features(df, CREDIT_HISTORY_FEATURES)
    payment = existing_features(df, PAYMENT_BEHAVIOR_FEATURES)
    full = profile + affordability + credit + payment
    interactions = existing_features(df, INTERACTION_FEATURES)
    return {
        "00_null_baseline": {"features": [], "class_weight": None, "description": "Null model using base default rate only"},
        "01_profile_only": {"features": profile, "class_weight": None, "description": "Customer profile variables only"},
        "02_affordability_only": {"features": affordability, "class_weight": None, "description": "Loan affordability burden variables only"},
        "03_credit_history_only": {"features": credit, "class_weight": None, "description": "External credit and previous application history only"},
        "04_payment_behavior_only": {"features": payment, "class_weight": None, "description": "Installment, credit-card and POS/CASH payment behavior only"},
        "05_full_controlled_main_effects": {"features": full, "class_weight": None, "description": "All selected diagnostic feature groups, main effects only"},
        "06_full_plus_selected_interactions": {"features": full + interactions, "class_weight": None, "description": "Full model plus selected business interactions"},
        "07_full_balanced_sensitivity": {"features": full, "class_weight": "balanced", "description": "Sensitivity model with class_weight='balanced'"},
    }


def model_comparison(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    log("Running diagnostic logistic model comparison...")
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.dummy import DummyClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            average_precision_score,
            brier_score_loss,
            log_loss,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
    except Exception as e:
        raise ImportError(
            "Step 6 model comparison requires scikit-learn. "
            "Kaggle normally has it preinstalled. If running locally, install it with: "
            "pip install scikit-learn scipy statsmodels seaborn openpyxl "
            "lightgbm xgboost catboost"
        ) from e

    specs = make_model_specs(df)
    y = df[TARGET].astype(int)
    base_rate = y.mean()
    null_logloss = log_loss(y, np.repeat(base_rate, len(y)), labels=[0, 1])

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    rows = []
    fitted_models = {}

    for spec_name, spec in specs.items():
        features = spec["features"]
        log(f"  Fitting {spec_name} with {len(features)} features...")

        if not features:
            clf = DummyClassifier(strategy="prior")
            X_train = pd.DataFrame({"dummy": np.zeros(len(train_idx))})
            X_test = pd.DataFrame({"dummy": np.zeros(len(test_idx))})
            clf.fit(X_train, y.iloc[train_idx])
            pred = clf.predict_proba(X_test)[:, 1]
            n_encoded_features = 0
            fitted_models[spec_name] = clf
        else:
            X = df[features].copy().astype("object").fillna("Unknown")
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            try:
                encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=True)
            except TypeError:
                # Older scikit-learn uses sparse instead of sparse_output.
                encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse=True)
            pre = ColumnTransformer(
                transformers=[("cat", encoder, features)],
                remainder="drop",
            )
            clf = LogisticRegression(
                max_iter=1000,
                solver="saga",
                penalty="l2",
                C=1.0,
                class_weight=spec["class_weight"],
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
            pipe = Pipeline([("preprocess", pre), ("model", clf)])
            pipe.fit(X_train, y.iloc[train_idx])
            pred = pipe.predict_proba(X_test)[:, 1]
            fitted_models[spec_name] = pipe
            try:
                n_encoded_features = pipe.named_steps["preprocess"].get_feature_names_out().shape[0]
            except Exception:
                n_encoded_features = np.nan

        y_test = y.iloc[test_idx]
        pred_label_05 = (pred >= 0.5).astype(int)
        top10_threshold = np.quantile(pred, 0.90)
        pred_top10 = (pred >= top10_threshold).astype(int)
        top20_threshold = np.quantile(pred, 0.80)
        pred_top20 = (pred >= top20_threshold).astype(int)

        ll = log_loss(y_test, pred, labels=[0, 1])
        rows.append(
            {
                "model_spec": spec_name,
                "description": spec["description"],
                "feature_count_before_encoding": len(features),
                "encoded_feature_count": n_encoded_features,
                "class_weight": str(spec["class_weight"]),
                "roc_auc": roc_auc_score(y_test, pred) if len(np.unique(y_test)) == 2 else np.nan,
                "pr_auc": average_precision_score(y_test, pred),
                "log_loss": ll,
                "brier_score": brier_score_loss(y_test, pred),
                "pseudo_r2_vs_null_logloss": 1 - (ll / null_logloss),
                "recall_at_0_5": recall_score(y_test, pred_label_05, zero_division=0),
                "precision_at_0_5": precision_score(y_test, pred_label_05, zero_division=0),
                "recall_at_top_10pct_score": recall_score(y_test, pred_top10, zero_division=0),
                "precision_at_top_10pct_score": precision_score(y_test, pred_top10, zero_division=0),
                "recall_at_top_20pct_score": recall_score(y_test, pred_top20, zero_division=0),
                "precision_at_top_20pct_score": precision_score(y_test, pred_top20, zero_division=0),
            }
        )

    comp = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    save_df(comp, "diagnostic_model_comparison.csv")
    plot_model_comparison(comp)
    return comp, fitted_models


def plot_model_comparison(comp: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    data = comp.sort_values("roc_auc", ascending=True)
    plt.figure(figsize=(11, 5.5))
    ax = sns.barplot(data=data, y="model_spec", x="roc_auc", color="#6C4CFF")
    ax.set_title("Diagnostic Model Comparison by ROC-AUC", fontsize=13, weight="bold")
    ax.set_xlabel("ROC-AUC")
    ax.set_ylabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "diagnostic_model_comparison_auc.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(11, 5.5))
    ax = sns.barplot(data=data, y="model_spec", x="pseudo_r2_vs_null_logloss", color="#FF5C6A")
    ax.set_title("Diagnostic Model Comparison by Pseudo RÂ² vs Null Log Loss", fontsize=13, weight="bold")
    ax.set_xlabel("Pseudo RÂ²")
    ax.set_ylabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "diagnostic_model_comparison_pseudo_r2.png", dpi=180, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# Enhanced model-comparison routine:
# The first model_comparison() above was a specification-only comparison using
# Logistic Regression. This replacement keeps that idea and adds an algorithm
# comparison table. Because this definition appears later in the file, it is the
# one used by main().
# -----------------------------------------------------------------------------

def sample_for_model_comparison(df: pd.DataFrame) -> pd.DataFrame:
    if MODEL_COMPARISON_MAX_ROWS is None or len(df) <= MODEL_COMPARISON_MAX_ROWS:
        return df.copy()
    pieces = []
    for _, group in df.groupby(TARGET, group_keys=False):
        sample_size = min(
            len(group),
            max(1, int(round(MODEL_COMPARISON_MAX_ROWS * len(group) / len(df)))),
        )
        pieces.append(group.sample(n=sample_size, random_state=RANDOM_STATE))
    return pd.concat(pieces).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def top_k_metrics(y_true: np.ndarray, pred_score: np.ndarray, top_pct: float) -> dict[str, float | int]:
    k = max(1, int(math.ceil(len(y_true) * top_pct)))
    top_idx = np.argsort(pred_score)[::-1][:k]
    total_events = int(y_true.sum())
    found_events = int(y_true[top_idx].sum())
    precision = found_events / k if k else np.nan
    recall = found_events / total_events if total_events else np.nan
    baseline = float(y_true.mean())
    lift = precision / baseline if baseline else np.nan
    label = int(top_pct * 100)
    return {
        f"top_{label}pct_count": int(k),
        f"recall_at_top_{label}pct_score": recall,
        f"precision_at_top_{label}pct_score": precision,
        f"lift_at_top_{label}pct_score": lift,
    }


def model_metric_row(
    y_train: pd.Series,
    y_test: pd.Series,
    pred_score: np.ndarray,
    threshold: float = 0.5,
    parameter_count: int | None = None,
) -> dict[str, float | int]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        log_loss,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = y_test.to_numpy(dtype=int)
    pred_score = np.asarray(pred_score, dtype=float)
    null_score = np.repeat(float(y_train.mean()), len(y_true))
    pred_label = (pred_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred_label, labels=[0, 1]).ravel()
    model_ll = log_loss(y_true, pred_score, labels=[0, 1])
    null_ll = log_loss(y_true, null_score, labels=[0, 1])
    n = len(y_true)
    k = int(parameter_count) if parameter_count is not None and not pd.isna(parameter_count) else np.nan
    total_log_likelihood = -model_ll * n
    null_total_log_likelihood = -null_ll * n
    lr_chi2 = 2 * (total_log_likelihood - null_total_log_likelihood)
    lr_df = max(1, int(k) - 1) if pd.notna(k) else np.nan
    try:
        from scipy.stats import chi2
        lr_p = float(chi2.sf(lr_chi2, lr_df)) if pd.notna(lr_df) else np.nan
    except Exception:
        lr_p = np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    fnr = fn / (fn + tp) if (fn + tp) else np.nan
    out: dict[str, float | int] = {
        "train_event_rate": float(y_train.mean()),
        "test_event_rate": float(y_test.mean()),
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, pred_score) if len(np.unique(y_true)) == 2 else np.nan,
        "pr_auc": average_precision_score(y_true, pred_score),
        "log_loss": model_ll,
        "null_log_loss": null_ll,
        "test_log_likelihood": total_log_likelihood,
        "test_null_log_likelihood": null_total_log_likelihood,
        "test_lr_chi2_vs_null": lr_chi2,
        "test_lr_df_approx": lr_df,
        "test_lr_p_value_approx": lr_p,
        "test_aic_approx": (-2 * total_log_likelihood + 2 * k) if pd.notna(k) else np.nan,
        "test_bic_approx": (-2 * total_log_likelihood + np.log(n) * k) if pd.notna(k) else np.nan,
        "brier_score": brier_score_loss(y_true, pred_score),
        "pseudo_r2_vs_null_logloss": 1 - (model_ll / null_ll) if null_ll else np.nan,
        "accuracy_at_0_5": accuracy_score(y_true, pred_label),
        "balanced_accuracy_at_0_5": balanced_accuracy_score(y_true, pred_label),
        "recall_at_0_5": recall_score(y_true, pred_label, zero_division=0),
        "precision_at_0_5": precision_score(y_true, pred_label, zero_division=0),
        "f1_at_0_5": f1_score(y_true, pred_label, zero_division=0),
        "specificity_at_0_5": specificity,
        "sensitivity_at_0_5": sensitivity,
        "npv_at_0_5": npv,
        "fpr_at_0_5": fpr,
        "fnr_at_0_5": fnr,
        "mcc_at_0_5": matthews_corrcoef(y_true, pred_label) if len(np.unique(pred_label)) > 1 else 0.0,
        "predicted_positive_rate_at_0_5": float(pred_label.mean()),
        "tn_at_0_5": int(tn),
        "fp_at_0_5": int(fp),
        "fn_at_0_5": int(fn),
        "tp_at_0_5": int(tp),
        "confusion_matrix_at_0_5": f"TN={int(tn)}, FP={int(fp)}, FN={int(fn)}, TP={int(tp)}",
    }
    out.update(top_k_metrics(y_true, pred_score, 0.10))
    out.update(top_k_metrics(y_true, pred_score, 0.20))
    return out


def make_one_hot_encoder(dense: bool = False):
    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=not dense)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse=not dense)


def fit_sklearn_model(
    estimator,
    features: list[str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    dense: bool = False,
) -> tuple[object, np.ndarray, int | float]:
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    pre = ColumnTransformer(
        transformers=[("cat", make_one_hot_encoder(dense=dense), features)],
        remainder="drop",
    )
    pipe = Pipeline([("preprocess", pre), ("model", estimator)])
    log(f"    Fitting sklearn pipeline: rows={len(X_train):,}, raw_features={len(features)}, dense={dense}")
    pipe.fit(X_train, y_train)
    log("    Predicting test probabilities...")
    if hasattr(pipe, "predict_proba"):
        pred = pipe.predict_proba(X_test)[:, 1]
    elif hasattr(pipe, "decision_function"):
        raw = pipe.decision_function(X_test)
        pred = 1 / (1 + np.exp(-raw))
    else:
        pred = pipe.predict(X_test).astype(float)
    try:
        encoded_features = pipe.named_steps["preprocess"].get_feature_names_out().shape[0]
    except Exception:
        encoded_features = np.nan
    return pipe, pred, encoded_features


def algorithm_specs() -> list[dict]:
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    specs = [
        {
            "algorithm": "logistic_regression_l2",
            "algorithm_family": "Linear / Logistic",
            "estimator": LogisticRegression(max_iter=1000, solver="lbfgs", penalty="l2", C=1.0, n_jobs=-1),
            "dense": False,
            "notes": "Interpretable diagnostic baseline.",
        },
        {
            "algorithm": "logistic_regression_l2_balanced",
            "algorithm_family": "Linear / Logistic",
            "estimator": LogisticRegression(max_iter=1000, solver="lbfgs", penalty="l2", C=1.0, class_weight="balanced", n_jobs=-1),
            "dense": False,
            "notes": "Sensitivity check for class imbalance.",
        },
        {
            "algorithm": "random_forest",
            "algorithm_family": "Tree ensemble",
            "estimator": RandomForestClassifier(
                n_estimators=120,
                max_depth=10,
                min_samples_leaf=80,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "dense": False,
            "notes": "Non-linear benchmark; captures interactions but is less interpretable.",
        },
        {
            "algorithm": "extra_trees",
            "algorithm_family": "Tree ensemble",
            "estimator": ExtraTreesClassifier(
                n_estimators=160,
                max_depth=10,
                min_samples_leaf=80,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "dense": False,
            "notes": "Randomized tree ensemble benchmark.",
        },
        {
            "algorithm": "hist_gradient_boosting",
            "algorithm_family": "Gradient boosting",
            "estimator": HistGradientBoostingClassifier(
                max_iter=180,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=0.05,
                random_state=RANDOM_STATE,
            ),
            "dense": True,
            "notes": "Sklearn gradient boosting benchmark.",
        },
    ]
    if RUN_OPTIONAL_BOOSTING_MODELS:
        try:
            from lightgbm import LGBMClassifier

            specs.append(
                {
                    "algorithm": "lightgbm",
                    "algorithm_family": "Gradient boosting",
                    "estimator": LGBMClassifier(
                        n_estimators=450,
                        learning_rate=0.035,
                        num_leaves=31,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="binary",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                    "dense": False,
                    "notes": "Optional strong tabular-data benchmark.",
                }
            )
        except Exception as e:
            log(f"  Skipping LightGBM: {e}")
        try:
            from xgboost import XGBClassifier

            specs.append(
                {
                    "algorithm": "xgboost",
                    "algorithm_family": "Gradient boosting",
                    "estimator": XGBClassifier(
                        n_estimators=350,
                        learning_rate=0.04,
                        max_depth=4,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss",
                        objective="binary:logistic",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                    "dense": False,
                    "notes": "Optional strong tabular-data benchmark.",
                }
            )
        except Exception as e:
            log(f"  Skipping XGBoost: {e}")
        try:
            from catboost import CatBoostClassifier

            specs.append(
                {
                    "algorithm": "catboost",
                    "algorithm_family": "Gradient boosting",
                    "estimator": CatBoostClassifier(
                        iterations=350,
                        learning_rate=0.04,
                        depth=5,
                        loss_function="Logloss",
                        eval_metric="AUC",
                        random_seed=RANDOM_STATE,
                        verbose=False,
                    ),
                    "dense": False,
                    "notes": "Optional strong categorical tabular benchmark.",
                }
            )
        except Exception as e:
            log(f"  Skipping CatBoost: {e}")
    return specs


def model_comparison(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    log("Running diagnostic model comparison: specification comparison + algorithm comparison...")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
    except Exception as e:
        raise ImportError(
            "Step 6 model comparison requires scikit-learn. "
            "Kaggle normally has it preinstalled. If running locally, install it with: "
            "pip install scikit-learn scipy statsmodels seaborn openpyxl "
            "lightgbm xgboost catboost"
        ) from e

    model_df = sample_for_model_comparison(df)
    if len(model_df) < len(df):
        log(f"  Stratified sample for model comparison: {len(model_df):,} / {len(df):,} rows")

    specs = make_model_specs(model_df)
    all_features = sorted({f for spec in specs.values() for f in spec["features"]})
    y = model_df[TARGET].astype(int)
    train_idx, test_idx = train_test_split(
        np.arange(len(model_df)),
        test_size=MODEL_TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    X_all = model_df[all_features].copy().astype("object").fillna("Unknown")
    X_train_all = X_all.iloc[train_idx].copy()
    X_test_all = X_all.iloc[test_idx].copy()

    fitted_models = {}
    spec_rows = []
    algorithm_rows = []

    log("  Specification comparison: same Logistic Regression, different feature sets...")
    for spec_name, spec in specs.items():
        start = time.time()
        features = spec["features"]
        row = {
            "comparison_type": "Specification comparison",
            "model_spec": spec_name,
            "algorithm": "LogisticRegression" if features else "Null baseline",
            "algorithm_family": "Linear / Logistic" if features else "Baseline",
            "description": spec["description"],
            "feature_set": spec_name,
            "feature_count_before_encoding": len(features),
            "class_weight": str(spec["class_weight"]),
            "preprocessing": "one_hot_min_frequency_50" if features else "none",
            "hyperparameters_summary": "LogisticRegression(max_iter=1000, solver='lbfgs', penalty='l2', C=1.0)",
            "n_rows_available": int(len(df)),
            "n_rows_used": int(len(model_df)),
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "status": "ok",
            "error": "",
        }
        try:
            if not features:
                pred = np.repeat(float(y_train.mean()), len(y_test))
                encoded_features = 0
            else:
                clf = LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    penalty="l2",
                    C=1.0,
                    class_weight=spec["class_weight"],
                    n_jobs=-1,
                )
                pipe, pred, encoded_features = fit_sklearn_model(
                    clf,
                    features,
                    X_train_all[features].copy(),
                    X_test_all[features].copy(),
                    y_train,
                )
                fitted_models[spec_name] = pipe
            row["encoded_feature_count"] = encoded_features
            row.update(model_metric_row(y_train, y_test, pred))
        except Exception as e:
            row["status"] = "failed"
            row["error"] = repr(e)
            row["encoded_feature_count"] = np.nan
        row["runtime_seconds"] = round(time.time() - start, 2)
        spec_rows.append(row)

    log("  Algorithm comparison: same full diagnostic feature set, different algorithms...")
    full_features = specs["06_full_plus_selected_interactions"]["features"]
    algo_specs = [
        {
            "algorithm": "null_baseline",
            "algorithm_family": "Baseline",
            "estimator": None,
            "dense": False,
            "notes": "Predicts train default rate only.",
        }
    ] + algorithm_specs()

    for alg in algo_specs:
        start = time.time()
        row = {
            "comparison_type": "Algorithm comparison",
            "model_spec": f"algo_{alg['algorithm']}",
            "algorithm": alg["algorithm"],
            "algorithm_family": alg["algorithm_family"],
            "description": alg["notes"],
            "feature_set": "06_full_plus_selected_interactions",
            "feature_count_before_encoding": len(full_features),
            "class_weight": "",
            "preprocessing": "one_hot_dense_min_frequency_50" if alg.get("dense") else "one_hot_min_frequency_50",
            "hyperparameters_summary": str(alg["estimator"].get_params())[:1200] if alg["estimator"] is not None else "train default rate",
            "n_rows_available": int(len(df)),
            "n_rows_used": int(len(model_df)),
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "status": "ok",
            "error": "",
        }
        try:
            if alg["estimator"] is None:
                pred = np.repeat(float(y_train.mean()), len(y_test))
                encoded_features = 0
            else:
                pipe, pred, encoded_features = fit_sklearn_model(
                    alg["estimator"],
                    full_features,
                    X_train_all[full_features].copy(),
                    X_test_all[full_features].copy(),
                    y_train,
                    dense=bool(alg.get("dense")),
                )
                fitted_models[alg["algorithm"]] = pipe
            row["encoded_feature_count"] = encoded_features
            row.update(model_metric_row(y_train, y_test, pred))
        except Exception as e:
            row["status"] = "failed"
            row["error"] = repr(e)
            row["encoded_feature_count"] = np.nan
        row["runtime_seconds"] = round(time.time() - start, 2)
        algorithm_rows.append(row)

    spec_comp = pd.DataFrame(spec_rows).sort_values(["roc_auc", "pr_auc"], ascending=False, na_position="last")
    algo_comp = pd.DataFrame(algorithm_rows).sort_values(["roc_auc", "pr_auc"], ascending=False, na_position="last")
    comp = pd.concat([spec_comp, algo_comp], ignore_index=True)
    save_df(spec_comp, "specification_comparison_research_table.csv")
    save_df(algo_comp, "algorithm_comparison_research_table.csv")
    save_df(comp, "diagnostic_model_comparison.csv")
    plot_model_comparison(comp)
    return comp, fitted_models


def plot_model_comparison(comp: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    if "comparison_type" in comp.columns:
        spec_data = comp[comp["comparison_type"].eq("Specification comparison")].copy()
        algo_data = comp[comp["comparison_type"].eq("Algorithm comparison")].copy()
    else:
        spec_data = comp.copy()
        algo_data = pd.DataFrame()

    if spec_data.empty:
        spec_data = comp.copy()
    data = spec_data.sort_values("roc_auc", ascending=True)
    plt.figure(figsize=(11, 5.5))
    ax = sns.barplot(data=data, y="model_spec", x="roc_auc", color="#6C4CFF")
    ax.set_title("Specification Comparison by ROC-AUC", fontsize=13, weight="bold")
    ax.set_xlabel("ROC-AUC")
    ax.set_ylabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "diagnostic_model_comparison_auc.png", dpi=180, bbox_inches="tight")
    plt.savefig(CHART_DIR / "specification_comparison_roc_auc.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(11, 5.5))
    ax = sns.barplot(data=data, y="model_spec", x="pseudo_r2_vs_null_logloss", color="#FF5C6A")
    ax.set_title("Specification Comparison by Pseudo R2 vs Null Log Loss", fontsize=13, weight="bold")
    ax.set_xlabel("Pseudo R2")
    ax.set_ylabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "diagnostic_model_comparison_pseudo_r2.png", dpi=180, bbox_inches="tight")
    plt.savefig(CHART_DIR / "specification_comparison_pseudo_r2.png", dpi=180, bbox_inches="tight")
    plt.close()

    if not algo_data.empty:
        data = algo_data.sort_values("roc_auc", ascending=True)
        plt.figure(figsize=(11, max(4.5, 0.55 * len(data))))
        ax = sns.barplot(data=data, y="algorithm", x="roc_auc", color="#1E8BFF")
        ax.set_title("Algorithm Comparison by ROC-AUC", fontsize=13, weight="bold")
        ax.set_xlabel("ROC-AUC")
        ax.set_ylabel("")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)
        plt.tight_layout()
        plt.savefig(CHART_DIR / "algorithm_comparison_roc_auc.png", dpi=180, bbox_inches="tight")
        plt.close()

        data = algo_data.sort_values("pr_auc", ascending=True)
        plt.figure(figsize=(11, max(4.5, 0.55 * len(data))))
        ax = sns.barplot(data=data, y="algorithm", x="pr_auc", color="#31A873")
        ax.set_title("Algorithm Comparison by PR-AUC", fontsize=13, weight="bold")
        ax.set_xlabel("PR-AUC")
        ax.set_ylabel("")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)
        plt.tight_layout()
        plt.savefig(CHART_DIR / "algorithm_comparison_pr_auc.png", dpi=180, bbox_inches="tight")
        plt.close()


# -----------------------------------------------------------------------------
# Diagnostic-only model-comparison routine:
# This final replacement keeps Step 6 as Diagnostic Analytics, not ML support.
# It runs statistical / econometric binary-outcome models that are applicable to
# the current cross-sectional customer-level dataset and exports an applicability
# matrix for models that are not valid for this data structure.
# -----------------------------------------------------------------------------

def binary_model_catalog() -> list[dict[str, str]]:
    return [
        {"group": "MÃ´ hÃ¬nh cÆ¡ báº£n", "model_name": "Linear Probability Model â€” LPM", "status": "run", "reason": "Applicable: binary TARGET with cross-sectional customer-level features."},
        {"group": "MÃ´ hÃ¬nh cÆ¡ báº£n", "model_name": "Binary Logistic Regression / Logit", "status": "run", "reason": "Applicable: standard diagnostic baseline for binary outcome."},
        {"group": "MÃ´ hÃ¬nh cÆ¡ báº£n", "model_name": "Probit Model", "status": "run", "reason": "Applicable: binary outcome with normal-CDF link."},
        {"group": "MÃ´ hÃ¬nh cÆ¡ báº£n", "model_name": "Complementary Log-Log â€” Cloglog", "status": "run", "reason": "Applicable as rare-event/asymmetric-link diagnostic sensitivity model."},
        {"group": "MÃ´ hÃ¬nh regularization", "model_name": "Lasso Logistic Regression", "status": "run", "reason": "Applicable: regularized Logit with L1 penalty for sparse signal check."},
        {"group": "MÃ´ hÃ¬nh regularization", "model_name": "Ridge Logistic Regression", "status": "run", "reason": "Applicable: regularized Logit with L2 penalty for coefficient stability."},
        {"group": "MÃ´ hÃ¬nh regularization", "model_name": "Elastic Net Logistic Regression", "status": "run", "reason": "Applicable: combines L1 and L2 regularization."},
        {"group": "MÃ´ hÃ¬nh cho dá»¯ liá»‡u hiáº¿m", "model_name": "Rare Events Logistic Regression", "status": "proxy_run", "reason": "Exact King-Zeng rare-events Logit is not natively available; run weighted Logit as a rare-event sensitivity proxy."},
        {"group": "MÃ´ hÃ¬nh cho dá»¯ liá»‡u nhá»", "model_name": "Firth Logistic Regression", "status": "not_run", "reason": "Not suitable for this large dataset; mainly used for small samples or complete separation. Requires specialized package."},
        {"group": "Panel data", "model_name": "Pooled Logit", "status": "not_run", "reason": "Final analytical table is one row per current customer application, not person-period panel data."},
        {"group": "Panel data", "model_name": "Fixed Effects Logit", "status": "not_run", "reason": "Requires repeated observations per entity and within-entity TARGET variation."},
        {"group": "Panel data", "model_name": "Random Effects Logit", "status": "not_run", "reason": "Requires panel structure and random individual effects."},
        {"group": "Panel data", "model_name": "Panel Probit", "status": "not_run", "reason": "Requires panel structure."},
        {"group": "Dynamic panel", "model_name": "Dynamic Panel Logit", "status": "not_run", "reason": "Requires time-indexed outcome Y_t and lagged Y_{t-1}; current target is one current application outcome."},
        {"group": "MÃ´ hÃ¬nh xá»­ lÃ½ ná»™i sinh", "model_name": "IV-Probit", "status": "not_run", "reason": "Requires valid instrumental variables and an explicit endogeneity hypothesis."},
        {"group": "MÃ´ hÃ¬nh xá»­ lÃ½ ná»™i sinh", "model_name": "IV-Logit", "status": "not_run", "reason": "Requires valid instruments/control-function design."},
        {"group": "MÃ´ hÃ¬nh xá»­ lÃ½ ná»™i sinh", "model_name": "Control Function Approach", "status": "not_run", "reason": "Requires a clearly endogenous variable and first-stage model."},
        {"group": "MÃ´ hÃ¬nh hai phÆ°Æ¡ng trÃ¬nh", "model_name": "Bivariate Probit", "status": "not_run", "reason": "Requires two correlated binary outcomes."},
        {"group": "MÃ´ hÃ¬nh hai phÆ°Æ¡ng trÃ¬nh", "model_name": "Recursive Bivariate Probit", "status": "not_run", "reason": "Requires a binary endogenous treatment and a binary outcome with identification strategy."},
        {"group": "Machine Learning", "model_name": "Decision Tree Classifier", "status": "deferred", "reason": "Out of scope for Diagnostic Analytics; move to Machine Learning Support step."},
        {"group": "Machine Learning", "model_name": "Random Forest Classifier", "status": "deferred", "reason": "Out of scope for Diagnostic Analytics; move to Machine Learning Support step."},
        {"group": "Machine Learning", "model_name": "Gradient Boosting Classifier", "status": "deferred", "reason": "Out of scope for Diagnostic Analytics; move to Machine Learning Support step."},
        {"group": "Machine Learning", "model_name": "XGBoost", "status": "deferred", "reason": "Out of scope for Diagnostic Analytics; move to Machine Learning Support step."},
        {"group": "Machine Learning", "model_name": "LightGBM", "status": "deferred", "reason": "Out of scope for Diagnostic Analytics; move to Machine Learning Support step."},
        {"group": "Machine Learning", "model_name": "CatBoost", "status": "deferred", "reason": "Out of scope for Diagnostic Analytics; move to Machine Learning Support step."},
        {"group": "Machine Learning", "model_name": "Support Vector Machine â€” SVM", "status": "deferred", "reason": "Out of scope for Diagnostic Analytics; move to Machine Learning Support step."},
        {"group": "Machine Learning", "model_name": "K-Nearest Neighbors â€” KNN", "status": "deferred", "reason": "Out of scope for Diagnostic Analytics; move to Machine Learning Support step."},
        {"group": "Machine Learning", "model_name": "Naive Bayes", "status": "deferred", "reason": "Out of scope and not a natural fit for this non-text credit-risk table."},
        {"group": "Machine Learning", "model_name": "Neural Network", "status": "deferred", "reason": "Out of scope for Diagnostic Analytics; move to Machine Learning Support step."},
        {"group": "Bayesian", "model_name": "Bayesian Logistic Regression", "status": "not_run", "reason": "Requires Bayesian modeling framework and prior specification; not needed for current diagnostic scope."},
        {"group": "Bayesian / phÃ¢n cáº¥p", "model_name": "Multilevel Logistic Regression", "status": "not_run", "reason": "Requires a defined hierarchy such as branch/region; not available in current final table."},
        {"group": "Survival / event history", "model_name": "Discrete-Time Hazard Model", "status": "not_run", "reason": "Requires person-period data and event timing; current TARGET is a one-time binary outcome."},
        {"group": "Survival analysis", "model_name": "Cox Proportional Hazards Model", "status": "not_run", "reason": "Requires time-to-event and censoring information, not just TARGET 0/1."},
        {"group": "Credit scoring", "model_name": "WOE Logistic Regression", "status": "deferred", "reason": "Valid credit-scoring extension, but requires a separate WOE/binning pipeline to avoid mixing diagnostic grouping with scorecard development."},
        {"group": "Credit scoring", "model_name": "Logistic Scorecard Model", "status": "deferred", "reason": "Valid later scorecard step after WOE/binning; not part of current diagnostic-only Step 6."},
    ]


def save_model_applicability_matrix() -> pd.DataFrame:
    out = pd.DataFrame(binary_model_catalog())
    out["run_in_step6"] = out["status"].isin(["run", "proxy_run"])
    save_df(out, "model_applicability_matrix.csv")
    return out


def safe_probability(pred: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(pred, dtype=float), 1e-6, 1 - 1e-6)


def model_metric_row(
    y_train: pd.Series,
    y_test: pd.Series,
    pred_score: np.ndarray,
    threshold: float = 0.5,
    parameter_count: int | None = None,
) -> dict[str, float | int]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        log_loss,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = y_test.to_numpy(dtype=int)
    pred_score = safe_probability(pred_score)
    null_score = np.repeat(float(y_train.mean()), len(y_true))
    pred_label = (pred_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred_label, labels=[0, 1]).ravel()
    model_ll = log_loss(y_true, pred_score, labels=[0, 1])
    null_ll = log_loss(y_true, null_score, labels=[0, 1])
    n = len(y_true)
    k = int(parameter_count) if parameter_count is not None and not pd.isna(parameter_count) else np.nan
    total_log_likelihood = -model_ll * n
    null_total_log_likelihood = -null_ll * n
    lr_chi2 = 2 * (total_log_likelihood - null_total_log_likelihood)
    lr_df = max(1, int(k) - 1) if pd.notna(k) else np.nan
    try:
        from scipy.stats import chi2
        lr_p = float(chi2.sf(lr_chi2, lr_df)) if pd.notna(lr_df) else np.nan
    except Exception:
        lr_p = np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    fnr = fn / (fn + tp) if (fn + tp) else np.nan
    out: dict[str, float | int] = {
        "train_event_rate": float(y_train.mean()),
        "test_event_rate": float(y_test.mean()),
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, pred_score) if len(np.unique(y_true)) == 2 else np.nan,
        "pr_auc": average_precision_score(y_true, pred_score),
        "log_loss": model_ll,
        "null_log_loss": null_ll,
        "test_log_likelihood": total_log_likelihood,
        "test_null_log_likelihood": null_total_log_likelihood,
        "test_lr_chi2_vs_null": lr_chi2,
        "test_lr_df_approx": lr_df,
        "test_lr_p_value_approx": lr_p,
        "test_aic_approx": (-2 * total_log_likelihood + 2 * k) if pd.notna(k) else np.nan,
        "test_bic_approx": (-2 * total_log_likelihood + np.log(n) * k) if pd.notna(k) else np.nan,
        "brier_score": brier_score_loss(y_true, pred_score),
        "pseudo_r2_vs_null_logloss": 1 - (model_ll / null_ll) if null_ll else np.nan,
        "accuracy_at_0_5": accuracy_score(y_true, pred_label),
        "balanced_accuracy_at_0_5": balanced_accuracy_score(y_true, pred_label),
        "recall_at_0_5": recall_score(y_true, pred_label, zero_division=0),
        "precision_at_0_5": precision_score(y_true, pred_label, zero_division=0),
        "f1_at_0_5": f1_score(y_true, pred_label, zero_division=0),
        "specificity_at_0_5": specificity,
        "sensitivity_at_0_5": sensitivity,
        "npv_at_0_5": npv,
        "fpr_at_0_5": fpr,
        "fnr_at_0_5": fnr,
        "mcc_at_0_5": matthews_corrcoef(y_true, pred_label) if len(np.unique(pred_label)) > 1 else 0.0,
        "predicted_positive_rate_at_0_5": float(pred_label.mean()),
        "tn_at_0_5": int(tn),
        "fp_at_0_5": int(fp),
        "fn_at_0_5": int(fn),
        "tp_at_0_5": int(tp),
        "confusion_matrix_at_0_5": f"TN={int(tn)}, FP={int(fp)}, FN={int(fn)}, TP={int(tp)}",
    }
    out.update(top_k_metrics(y_true, pred_score, 0.10))
    out.update(top_k_metrics(y_true, pred_score, 0.20))
    return out


def diagnostic_model_specs() -> list[dict]:
    from sklearn.linear_model import LinearRegression, LogisticRegression

    return [
        {
            "algorithm": "linear_probability_model_ols",
            "algorithm_family": "Diagnostic / Linear probability",
            "estimator": LinearRegression(),
            "model_name": "Linear Probability Model â€” LPM",
            "dense": False,
            "notes": "OLS on binary TARGET. Predictions are clipped to [0,1] for probability metrics.",
        },
        {
            "algorithm": "binary_logit_approx_unpenalized",
            "algorithm_family": "Diagnostic / Logit",
            "estimator": LogisticRegression(max_iter=1200, solver="lbfgs", penalty="l2", C=1e6, n_jobs=-1),
            "model_name": "Binary Logistic Regression / Logit",
            "dense": False,
            "notes": "Near-unpenalized Logit baseline using very large C.",
        },
        {
            "algorithm": "ridge_logit_l2",
            "algorithm_family": "Diagnostic / Regularized Logit",
            "estimator": LogisticRegression(max_iter=1200, solver="lbfgs", penalty="l2", C=1.0, n_jobs=-1),
            "model_name": "Ridge Logistic Regression",
            "dense": False,
            "notes": "L2-regularized Logit for coefficient stability.",
        },
        {
            "algorithm": "lasso_logit_l1",
            "algorithm_family": "Diagnostic / Regularized Logit",
            "estimator": LogisticRegression(max_iter=1200, solver="saga", penalty="l1", C=1.0, n_jobs=-1, random_state=RANDOM_STATE),
            "model_name": "Lasso Logistic Regression",
            "dense": False,
            "notes": "L1-regularized Logit for sparse variable selection sensitivity.",
        },
        {
            "algorithm": "elastic_net_logit",
            "algorithm_family": "Diagnostic / Regularized Logit",
            "estimator": LogisticRegression(max_iter=1200, solver="saga", penalty="elasticnet", l1_ratio=0.5, C=1.0, n_jobs=-1, random_state=RANDOM_STATE),
            "model_name": "Elastic Net Logistic Regression",
            "dense": False,
            "notes": "Mixed L1/L2 regularized Logit.",
        },
        {
            "algorithm": "weighted_logit_rare_event_proxy",
            "algorithm_family": "Diagnostic / Rare-event sensitivity",
            "estimator": LogisticRegression(max_iter=1200, solver="lbfgs", penalty="l2", C=1.0, class_weight="balanced", n_jobs=-1),
            "model_name": "Rare Events Logistic Regression proxy",
            "dense": False,
            "notes": "Weighted Logit proxy, not exact King-Zeng rare-events Logit.",
        },
    ]


def fit_statsmodels_binary_link(
    link_name: str,
    features: list[str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[object, np.ndarray, int]:
    import statsmodels.api as sm
    from sklearn.compose import ColumnTransformer

    if link_name == "probit":
        link = sm.families.links.Probit()
    elif link_name == "cloglog":
        link = sm.families.links.CLogLog()
    else:
        raise ValueError(f"Unsupported statsmodels link: {link_name}")

    pre = ColumnTransformer(
        transformers=[("cat", make_one_hot_encoder(dense=True), features)],
        remainder="drop",
    )
    log(f"    Encoding features for statsmodels {link_name}: {len(features)} raw features...")
    X_train_enc = pre.fit_transform(X_train)
    X_test_enc = pre.transform(X_test)
    X_train_enc = sm.add_constant(X_train_enc, has_constant="add")
    X_test_enc = sm.add_constant(X_test_enc, has_constant="add")
    log(
        f"    Fitting statsmodels {link_name} GLM: "
        f"train_rows={X_train_enc.shape[0]:,}, encoded_features={X_train_enc.shape[1]:,}"
    )
    res = sm.GLM(
        y_train.astype(float).to_numpy(),
        X_train_enc,
        family=sm.families.Binomial(link=link),
    ).fit(maxiter=120, disp=0)
    log(f"    Finished statsmodels {link_name} GLM.")
    pred = res.predict(X_test_enc)
    return res, pred, int(X_train_enc.shape[1])


def classification_metrics_at_threshold(
    y_true: np.ndarray,
    pred_score: np.ndarray,
    threshold: float,
    model_id: str,
    model_label: str,
    threshold_label: str,
    threshold_type: str,
) -> dict[str, float | int | str]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
    )

    y_true = np.asarray(y_true, dtype=int)
    pred_score = safe_probability(pred_score)
    pred_label = (pred_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred_label, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    fnr = fn / (fn + tp) if (fn + tp) else np.nan
    return {
        "model_id": model_id,
        "model_label": model_label,
        "threshold_type": threshold_type,
        "threshold_label": threshold_label,
        "threshold": float(threshold),
        "customer_count": int(len(y_true)),
        "actual_default_count": int(y_true.sum()),
        "predicted_default_count": int(pred_label.sum()),
        "predicted_positive_rate": float(pred_label.mean()),
        "accuracy": accuracy_score(y_true, pred_label),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred_label),
        "precision": precision_score(y_true, pred_label, zero_division=0),
        "recall_sensitivity": recall_score(y_true, pred_label, zero_division=0),
        "specificity": specificity,
        "npv": npv,
        "f1_score": f1_score(y_true, pred_label, zero_division=0),
        "mcc": matthews_corrcoef(y_true, pred_label) if len(np.unique(pred_label)) > 1 else 0.0,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": f"TN={int(tn)}, FP={int(fp)}, FN={int(fn)}, TP={int(tp)}",
    }


def build_threshold_analysis(y_true: np.ndarray, prediction_store: dict[str, dict]) -> pd.DataFrame:
    probability_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    top_percentiles = [0.05, 0.10, 0.20, 0.30]
    rows = []
    for model_id, meta in prediction_store.items():
        pred = safe_probability(meta["pred_score"])
        model_label = meta["model_label"]
        for t in probability_thresholds:
            rows.append(classification_metrics_at_threshold(y_true, pred, t, model_id, model_label, f"prob >= {t:.2f}", "probability"))
        for pct_value in top_percentiles:
            threshold = float(np.quantile(pred, 1 - pct_value))
            label = f"top {int(pct_value * 100)}%"
            rows.append(classification_metrics_at_threshold(y_true, pred, threshold, model_id, model_label, label, "top_percentile"))
    return pd.DataFrame(rows)


def build_curve_points(y_true: np.ndarray, prediction_store: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import precision_recall_curve, roc_curve

    roc_rows = []
    pr_rows = []
    for model_id, meta in prediction_store.items():
        pred = safe_probability(meta["pred_score"])
        label = meta["model_label"]
        try:
            fpr, tpr, thresholds = roc_curve(y_true, pred)
            for i, (f, t, th) in enumerate(zip(fpr, tpr, thresholds)):
                roc_rows.append(
                    {
                        "model_id": model_id,
                        "model_label": label,
                        "point_index": i,
                        "false_positive_rate": f,
                        "true_positive_rate_recall": t,
                        "threshold": th,
                    }
                )
        except Exception as e:
            log(f"  ROC curve failed for {model_id}: {e}")
        try:
            precision, recall, pr_thresholds = precision_recall_curve(y_true, pred)
            padded_thresholds = np.append(pr_thresholds, np.nan)
            for i, (p, r, th) in enumerate(zip(precision, recall, padded_thresholds)):
                pr_rows.append(
                    {
                        "model_id": model_id,
                        "model_label": label,
                        "point_index": i,
                        "precision": p,
                        "recall": r,
                        "threshold": th,
                    }
                )
        except Exception as e:
            log(f"  PR curve failed for {model_id}: {e}")
    return pd.DataFrame(roc_rows), pd.DataFrame(pr_rows)


def build_calibration_tables(y_true: np.ndarray, prediction_store: dict[str, dict], n_bins: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration_rows = []
    summary_rows = []
    y_true = np.asarray(y_true, dtype=int)
    for model_id, meta in prediction_store.items():
        pred = safe_probability(meta["pred_score"])
        temp = pd.DataFrame({"actual": y_true, "pred": pred})
        try:
            temp["calibration_bin"] = pd.qcut(temp["pred"].rank(method="first"), q=n_bins, labels=False) + 1
        except ValueError:
            temp["calibration_bin"] = pd.cut(temp["pred"], bins=n_bins, labels=False, duplicates="drop") + 1
        grouped = (
            temp.groupby("calibration_bin", dropna=False)
            .agg(
                customer_count=("actual", "count"),
                observed_defaults=("actual", "sum"),
                observed_default_rate=("actual", "mean"),
                expected_defaults=("pred", "sum"),
                mean_predicted_probability=("pred", "mean"),
                min_predicted_probability=("pred", "min"),
                max_predicted_probability=("pred", "max"),
            )
            .reset_index()
        )
        grouped["model_id"] = model_id
        grouped["model_label"] = meta["model_label"]
        grouped["calibration_gap"] = grouped["mean_predicted_probability"] - grouped["observed_default_rate"]
        grouped["abs_calibration_gap"] = grouped["calibration_gap"].abs()
        grouped["weighted_abs_gap"] = grouped["abs_calibration_gap"] * grouped["customer_count"] / grouped["customer_count"].sum()
        calibration_rows.append(grouped)
        summary_rows.append(
            {
                "model_id": model_id,
                "model_label": meta["model_label"],
                "bins": int(grouped["calibration_bin"].nunique()),
                "ece_expected_calibration_error": float(grouped["weighted_abs_gap"].sum()),
                "mce_max_calibration_error": float(grouped["abs_calibration_gap"].max()),
                "mean_calibration_gap": float(grouped["calibration_gap"].mean()),
            }
        )
    calibration = pd.concat(calibration_rows, ignore_index=True) if calibration_rows else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    return calibration, summary


def build_hosmer_lemeshow_tables(calibration: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from scipy.stats import chi2
    except Exception:
        chi2 = None
    if calibration.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    decile = calibration.copy()
    decile["observed_non_defaults"] = decile["customer_count"] - decile["observed_defaults"]
    decile["expected_non_defaults"] = decile["customer_count"] - decile["expected_defaults"]
    eps = 1e-9
    decile["hl_component_default"] = (decile["observed_defaults"] - decile["expected_defaults"]) ** 2 / (decile["expected_defaults"] + eps)
    decile["hl_component_non_default"] = (decile["observed_non_defaults"] - decile["expected_non_defaults"]) ** 2 / (decile["expected_non_defaults"] + eps)
    decile["hl_component_total"] = decile["hl_component_default"] + decile["hl_component_non_default"]
    for (model_id, model_label), g in decile.groupby(["model_id", "model_label"], dropna=False):
        hl_chi2 = float(g["hl_component_total"].sum())
        bins = int(g["calibration_bin"].nunique())
        df_hl = max(1, bins - 2)
        p_value = float(chi2.sf(hl_chi2, df_hl)) if chi2 is not None else np.nan
        rows.append(
            {
                "model_id": model_id,
                "model_label": model_label,
                "bins": bins,
                "hl_chi2": hl_chi2,
                "hl_df": df_hl,
                "hl_p_value": p_value,
                "note": "With very large samples, Hosmer-Lemeshow is highly sensitive; interpret together with calibration plot and Brier Score.",
            }
        )
    return pd.DataFrame(rows), decile


def downsample_curve_points_for_plot(df: pd.DataFrame, max_points_per_model: int = 900) -> pd.DataFrame:
    """Keep full curve CSVs, but plot a stable sample so chart rendering does not hang."""
    if df.empty or "model_id" not in df.columns:
        return df
    sampled_frames = []
    for _, g in df.groupby("model_id", dropna=False, sort=False):
        if len(g) <= max_points_per_model:
            sampled_frames.append(g)
            continue
        idx = np.linspace(0, len(g) - 1, max_points_per_model).round().astype(int)
        sampled_frames.append(g.iloc[np.unique(idx)])
    return pd.concat(sampled_frames, ignore_index=True) if sampled_frames else df.head(0)


def plot_curve_and_calibration_outputs(
    roc_points: pd.DataFrame,
    pr_points: pd.DataFrame,
    calibration: pd.DataFrame,
    threshold_analysis: pd.DataFrame,
) -> None:
    if roc_points.empty and pr_points.empty and calibration.empty and threshold_analysis.empty:
        return
    import matplotlib.pyplot as plt
    import seaborn as sns

    preferred_labels = [
        "Binary Logistic Regression / Logit",
        "Ridge Logistic Regression",
        "Lasso Logistic Regression",
        "Elastic Net Logistic Regression",
        "Linear Probability Model Ã¢â‚¬â€ LPM",
        "06_full_plus_selected_interactions",
    ]

    if not roc_points.empty:
        temp = roc_points[roc_points["model_label"].isin(preferred_labels)].copy()
        if temp.empty:
            temp = roc_points.copy()
        before_rows = len(temp)
        temp = downsample_curve_points_for_plot(temp)
        log(f"  Plotting ROC curve: sampled {len(temp):,} / {before_rows:,} curve points")
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=temp, x="false_positive_rate", y="true_positive_rate_recall", hue="model_label", linewidth=1.8)
        plt.plot([0, 1], [0, 1], linestyle="--", color="#9CA3AF", linewidth=1)
        plt.title("ROC Curve - Main Diagnostic Models", fontsize=13, weight="bold")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate / Recall")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(CHART_DIR / "roc_curve_main_diagnostic_models.png", dpi=180, bbox_inches="tight")
        plt.close()

    if not pr_points.empty:
        temp = pr_points[pr_points["model_label"].isin(preferred_labels)].copy()
        if temp.empty:
            temp = pr_points.copy()
        before_rows = len(temp)
        temp = downsample_curve_points_for_plot(temp)
        log(f"  Plotting Precision-Recall curve: sampled {len(temp):,} / {before_rows:,} curve points")
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=temp, x="recall", y="precision", hue="model_label", linewidth=1.8)
        plt.title("Precision-Recall Curve - Main Diagnostic Models", fontsize=13, weight="bold")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(CHART_DIR / "precision_recall_curve_main_diagnostic_models.png", dpi=180, bbox_inches="tight")
        plt.close()

    if not calibration.empty:
        temp = calibration[calibration["model_label"].isin(["Binary Logistic Regression / Logit", "Ridge Logistic Regression", "06_full_plus_selected_interactions"])].copy()
        if temp.empty:
            temp = calibration.copy()
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=temp, x="mean_predicted_probability", y="observed_default_rate", hue="model_label", marker="o")
        max_axis = max(temp["mean_predicted_probability"].max(), temp["observed_default_rate"].max())
        plt.plot([0, max_axis], [0, max_axis], linestyle="--", color="#9CA3AF", linewidth=1)
        plt.title("Calibration Plot - Predicted Probability vs Actual Default Rate", fontsize=13, weight="bold")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Observed Default Rate")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(CHART_DIR / "calibration_plot_main_diagnostic_models.png", dpi=180, bbox_inches="tight")
        plt.close()

    if not threshold_analysis.empty:
        temp = threshold_analysis[
            threshold_analysis["model_label"].eq("Binary Logistic Regression / Logit")
            & threshold_analysis["threshold_type"].eq("probability")
        ].copy()
        if not temp.empty:
            plot_df = temp.melt(
                id_vars=["threshold_label", "threshold"],
                value_vars=["precision", "recall_sensitivity", "specificity", "balanced_accuracy", "mcc"],
                var_name="metric",
                value_name="value",
            )
            plt.figure(figsize=(9, 5.5))
            sns.lineplot(data=plot_df, x="threshold", y="value", hue="metric", marker="o")
            plt.title("Threshold Analysis - Main Logit Model", fontsize=13, weight="bold")
            plt.xlabel("Probability Threshold")
            plt.ylabel("Metric Value")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(CHART_DIR / "threshold_analysis_main_logit.png", dpi=180, bbox_inches="tight")
            plt.close()


def make_terminal_model_evaluation_table(model_comp: pd.DataFrame) -> pd.DataFrame:
    if model_comp.empty:
        return pd.DataFrame()
    cols = [
        "comparison_type",
        "model_spec",
        "model_name",
        "algorithm",
        "status",
        "roc_auc",
        "pr_auc",
        "log_loss",
        "brier_score",
        "pseudo_r2_vs_null_logloss",
        "accuracy_at_0_5",
        "precision_at_0_5",
        "recall_at_0_5",
        "specificity_at_0_5",
        "balanced_accuracy_at_0_5",
        "mcc_at_0_5",
        "lift_at_top_10pct_score",
        "runtime_seconds",
    ]
    existing = [c for c in cols if c in model_comp.columns]
    out = model_comp[existing].copy()
    if "model_name" in out.columns:
        out["display_model"] = out["model_name"].fillna(out.get("model_spec", ""))
    else:
        out["display_model"] = out.get("model_spec", "")
    first_cols = ["comparison_type", "display_model", "status"]
    remaining = [c for c in out.columns if c not in first_cols + ["model_name", "model_spec", "algorithm"]]
    out = out[[c for c in first_cols if c in out.columns] + remaining]
    return out


def save_and_print_terminal_model_summary(model_comp: pd.DataFrame) -> pd.DataFrame:
    terminal_table = make_terminal_model_evaluation_table(model_comp)
    if terminal_table.empty:
        return terminal_table
    save_df(terminal_table, "terminal_model_evaluation_table.csv")
    terminal_path = TABLE_DIR / "terminal_model_evaluation_table.txt"
    text = terminal_table.to_string(index=False, max_colwidth=34)
    terminal_path.write_text(text, encoding="utf-8")
    log("Full terminal model evaluation table:")
    print("\n" + text + "\n", flush=True)
    try:
        with PROGRESS_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write("\nFULL TERMINAL MODEL EVALUATION TABLE\n")
            f.write(text + "\n")
    except Exception:
        pass
    return terminal_table


def build_full_evaluation_outputs(y_train: pd.Series, y_test: pd.Series, prediction_store: dict[str, dict], model_comp: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if not prediction_store:
        return {}
    log("Building full evaluation outputs: threshold analysis, ROC/PR curves, calibration and Hosmer-Lemeshow...")
    tracker = ProgressTracker("FULL EVALUATION OUTPUTS", 7)
    y_true = y_test.to_numpy(dtype=int)

    tracker.start("threshold analysis for all diagnostic models")
    threshold_analysis = build_threshold_analysis(y_true, prediction_store)
    save_df(threshold_analysis, "threshold_analysis_all_models.csv")
    main_ids = [
        "algorithm::binary_logit_approx_unpenalized",
        "spec::05_full_controlled_main_effects",
        "spec::06_full_plus_selected_interactions",
    ]
    threshold_main = threshold_analysis[threshold_analysis["model_id"].isin(main_ids)].copy()
    save_df(threshold_main, "threshold_analysis_main_logit.csv")
    tracker.finish("threshold analysis saved")

    tracker.start("ROC and Precision-Recall curve point tables")
    roc_points, pr_points = build_curve_points(y_true, prediction_store)
    save_df(roc_points, "roc_curve_points.csv")
    save_df(pr_points, "precision_recall_curve_points.csv")
    tracker.finish(f"curve point tables saved | ROC rows={len(roc_points):,}; PR rows={len(pr_points):,}")

    tracker.start("calibration decile tables")
    calibration, calibration_summary = build_calibration_tables(y_true, prediction_store)
    save_df(calibration, "calibration_by_decile.csv")
    save_df(calibration_summary, "calibration_error_summary.csv")
    tracker.finish("calibration tables saved")

    tracker.start("Hosmer-Lemeshow goodness-of-fit tables")
    hl_summary, hl_deciles = build_hosmer_lemeshow_tables(calibration)
    save_df(hl_summary, "hosmer_lemeshow_test.csv")
    save_df(hl_deciles, "hosmer_lemeshow_decile_table.csv")
    tracker.finish("Hosmer-Lemeshow tables saved")

    tracker.start("full model evaluation summary")
    full_summary = model_comp.copy()
    save_df(full_summary, "full_model_evaluation_summary.csv")
    tracker.finish("full model evaluation summary saved")

    tracker.start("terminal model evaluation table")
    terminal_table = save_and_print_terminal_model_summary(full_summary)
    tracker.finish("terminal model evaluation table saved and printed")

    tracker.start("diagnostic evaluation charts")
    plot_curve_and_calibration_outputs(roc_points, pr_points, calibration, threshold_analysis)
    tracker.finish("diagnostic evaluation charts saved")
    tracker.done_all()
    return {
        "threshold_analysis": threshold_analysis,
        "threshold_main": threshold_main,
        "roc_points": roc_points,
        "pr_points": pr_points,
        "calibration": calibration,
        "calibration_summary": calibration_summary,
        "hosmer_lemeshow": hl_summary,
        "hosmer_lemeshow_deciles": hl_deciles,
        "full_model_evaluation_summary": full_summary,
        "terminal_model_evaluation": terminal_table,
    }


def model_comparison(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    log("Running diagnostic-only model comparison: specification comparison + binary-response model comparison...")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
    except Exception as e:
        raise ImportError(
            "Step 6 model comparison requires scikit-learn and statsmodels. "
            "Install with: pip install scikit-learn scipy statsmodels seaborn openpyxl"
        ) from e

    applicability = save_model_applicability_matrix()
    log(
        "  Applicability matrix saved. "
        f"Run/proxy models: {int(applicability['run_in_step6'].sum())}; "
        f"documented/skipped models: {int((~applicability['run_in_step6']).sum())}"
    )

    model_df = sample_for_model_comparison(df)
    if len(model_df) < len(df):
        log(f"  Stratified sample for model comparison: {len(model_df):,} / {len(df):,} rows")

    specs = make_model_specs(model_df)
    all_features = sorted({f for spec in specs.values() for f in spec["features"]})
    y = model_df[TARGET].astype(int)
    train_idx, test_idx = train_test_split(
        np.arange(len(model_df)),
        test_size=MODEL_TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    X_all = model_df[all_features].copy().astype("object").fillna("Unknown")
    X_train_all = X_all.iloc[train_idx].copy()
    X_test_all = X_all.iloc[test_idx].copy()

    fitted_models = {}
    spec_rows = []
    diagnostic_rows = []
    prediction_store: dict[str, dict] = {}

    log("  Specification comparison: same Logit family, different feature sets...")
    spec_progress = ProgressTracker("SPECIFICATION COMPARISON", len(specs))
    for spec_name, spec in specs.items():
        start = time.time()
        features = spec["features"]
        spec_progress.start(f"{spec_name} | features={len(features)}")
        row = {
            "comparison_type": "Specification comparison",
            "model_spec": spec_name,
            "algorithm": "Binary Logistic Regression / Logit" if features else "Null baseline",
            "algorithm_family": "Diagnostic / Logit" if features else "Baseline",
            "description": spec["description"],
            "feature_set": spec_name,
            "feature_count_before_encoding": len(features),
            "class_weight": str(spec["class_weight"]),
            "preprocessing": "one_hot_min_frequency_50" if features else "none",
            "hyperparameters_summary": "Near-unpenalized LogisticRegression(C=1e6, solver='lbfgs')",
            "n_rows_available": int(len(df)),
            "n_rows_used": int(len(model_df)),
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "status": "ok",
            "error": "",
        }
        try:
            if not features:
                pred = np.repeat(float(y_train.mean()), len(y_test))
                encoded_features = 0
            else:
                estimator = LogisticRegression(max_iter=1200, solver="lbfgs", penalty="l2", C=1e6, class_weight=spec["class_weight"], n_jobs=-1)
                pipe, pred, encoded_features = fit_sklearn_model(
                    estimator,
                    features,
                    X_train_all[features].copy(),
                    X_test_all[features].copy(),
                    y_train,
                )
                fitted_models[spec_name] = pipe
            row["encoded_feature_count"] = encoded_features
            parameter_count = int(encoded_features) + 1 if pd.notna(encoded_features) else None
            row.update(model_metric_row(y_train, y_test, pred, parameter_count=parameter_count))
            prediction_store[f"spec::{spec_name}"] = {
                "model_label": spec_name,
                "comparison_type": "Specification comparison",
                "pred_score": safe_probability(pred),
                "parameter_count": parameter_count,
            }
        except Exception as e:
            row["status"] = "failed"
            row["error"] = repr(e)
            row["encoded_feature_count"] = np.nan
        row["runtime_seconds"] = round(time.time() - start, 2)
        spec_rows.append(row)
        metric_msg = ""
        if pd.notna(row.get("roc_auc", np.nan)):
            metric_msg = f" | ROC-AUC={row['roc_auc']:.4f}, PR-AUC={row['pr_auc']:.4f}"
        spec_progress.finish(f"{spec_name}{metric_msg}", status=row["status"])
    spec_progress.done_all()

    log("  Diagnostic model-family comparison: same full feature set, different binary-response models...")
    full_features = specs["06_full_plus_selected_interactions"]["features"]
    model_family_specs = [
        {
            "algorithm": "null_baseline",
            "algorithm_family": "Baseline",
            "model_name": "Null baseline",
            "estimator": None,
            "dense": False,
            "notes": "Predicts train default rate only.",
        }
    ] + diagnostic_model_specs() + [
        {
            "algorithm": "probit_glm",
            "algorithm_family": "Diagnostic / Probit",
            "model_name": "Probit Model",
            "estimator": "statsmodels_probit",
            "dense": True,
            "notes": "GLM Binomial with Probit link.",
        },
        {
            "algorithm": "cloglog_glm",
            "algorithm_family": "Diagnostic / Cloglog",
            "model_name": "Complementary Log-Log â€” Cloglog",
            "estimator": "statsmodels_cloglog",
            "dense": True,
            "notes": "GLM Binomial with complementary log-log link.",
        },
    ]

    family_progress = ProgressTracker("DIAGNOSTIC MODEL FAMILY", len(model_family_specs))
    for alg in model_family_specs:
        start = time.time()
        family_progress.start(f"{alg['model_name']} | {alg['algorithm']}")
        row = {
            "comparison_type": "Algorithm comparison",
            "model_spec": f"diagnostic_{alg['algorithm']}",
            "algorithm": alg["algorithm"],
            "model_name": alg["model_name"],
            "algorithm_family": alg["algorithm_family"],
            "description": alg["notes"],
            "feature_set": "06_full_plus_selected_interactions",
            "feature_count_before_encoding": len(full_features),
            "class_weight": "",
            "preprocessing": "one_hot_dense_min_frequency_50" if alg.get("dense") else "one_hot_min_frequency_50",
            "hyperparameters_summary": str(alg["estimator"].get_params())[:1200] if hasattr(alg["estimator"], "get_params") else str(alg["estimator"]),
            "n_rows_available": int(len(df)),
            "n_rows_used": int(len(model_df)),
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "status": "ok",
            "error": "",
        }
        try:
            if alg["estimator"] is None:
                pred = np.repeat(float(y_train.mean()), len(y_test))
                encoded_features = 0
            elif alg["estimator"] == "statsmodels_probit":
                model, pred, encoded_features = fit_statsmodels_binary_link(
                    "probit",
                    full_features,
                    X_train_all[full_features].copy(),
                    X_test_all[full_features].copy(),
                    y_train,
                )
                fitted_models[alg["algorithm"]] = model
            elif alg["estimator"] == "statsmodels_cloglog":
                model, pred, encoded_features = fit_statsmodels_binary_link(
                    "cloglog",
                    full_features,
                    X_train_all[full_features].copy(),
                    X_test_all[full_features].copy(),
                    y_train,
                )
                fitted_models[alg["algorithm"]] = model
            else:
                pipe, pred, encoded_features = fit_sklearn_model(
                    alg["estimator"],
                    full_features,
                    X_train_all[full_features].copy(),
                    X_test_all[full_features].copy(),
                    y_train,
                    dense=bool(alg.get("dense")),
                )
                fitted_models[alg["algorithm"]] = pipe
            row["encoded_feature_count"] = encoded_features
            parameter_count = int(encoded_features) + 1 if pd.notna(encoded_features) else None
            row.update(model_metric_row(y_train, y_test, pred, parameter_count=parameter_count))
            prediction_store[f"algorithm::{alg['algorithm']}"] = {
                "model_label": alg["model_name"],
                "comparison_type": "Algorithm comparison",
                "pred_score": safe_probability(pred),
                "parameter_count": parameter_count,
            }
        except Exception as e:
            row["status"] = "failed"
            row["error"] = repr(e)
            row["encoded_feature_count"] = np.nan
        row["runtime_seconds"] = round(time.time() - start, 2)
        diagnostic_rows.append(row)
        metric_msg = ""
        if pd.notna(row.get("roc_auc", np.nan)):
            metric_msg = f" | ROC-AUC={row['roc_auc']:.4f}, PR-AUC={row['pr_auc']:.4f}"
        family_progress.finish(f"{alg['model_name']}{metric_msg}", status=row["status"])
    family_progress.done_all()

    spec_comp = pd.DataFrame(spec_rows).sort_values(["roc_auc", "pr_auc"], ascending=False, na_position="last")
    diagnostic_comp = pd.DataFrame(diagnostic_rows).sort_values(["roc_auc", "pr_auc"], ascending=False, na_position="last")
    comp = pd.concat([spec_comp, diagnostic_comp], ignore_index=True)
    save_df(spec_comp, "specification_comparison_research_table.csv")
    save_df(diagnostic_comp, "algorithm_comparison_research_table.csv")
    save_df(diagnostic_comp, "diagnostic_binary_model_family_comparison.csv")
    save_df(comp, "diagnostic_model_comparison.csv")
    build_full_evaluation_outputs(y_train, y_test, prediction_store, comp)
    plot_model_comparison(comp)
    return comp, fitted_models


# =============================================================================
# 7. Odds Ratio Model
# =============================================================================

def fit_odds_ratio_model(df: pd.DataFrame) -> pd.DataFrame:
    log("Fitting statsmodels full controlled logistic regression for odds ratios...")
    try:
        import statsmodels.api as sm
    except Exception as e:
        log(f"statsmodels is not available. Skipping odds-ratio table. Error: {e}")
        return pd.DataFrame()

    features = (
        existing_features(df, PROFILE_FEATURES)
        + existing_features(df, AFFORDABILITY_FEATURES)
        + existing_features(df, CREDIT_HISTORY_FEATURES)
        + existing_features(df, PAYMENT_BEHAVIOR_FEATURES)
    )

    model_df = df[[TARGET] + features].copy()
    for c in features:
        model_df[c] = model_df[c].astype("object").fillna("Unknown")

    if STATSMODELS_MAX_ROWS is not None and len(model_df) > STATSMODELS_MAX_ROWS:
        log(f"  Sampling {STATSMODELS_MAX_ROWS:,} rows for statsmodels GLM to keep runtime manageable...")
        sampled_parts = []
        for _, group in model_df.groupby(TARGET, group_keys=False):
            sample_size = min(
                len(group),
                max(1, int(STATSMODELS_MAX_ROWS * len(group) / len(model_df))),
            )
            sampled_parts.append(group.sample(n=sample_size, random_state=RANDOM_STATE))
        model_df = (
            pd.concat(sampled_parts, ignore_index=True)
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )

    X = pd.get_dummies(model_df[features], drop_first=True, dtype=float)
    # Remove extremely rare dummy columns to avoid unstable coefficients.
    rare_cols = [c for c in X.columns if X[c].sum() < 30]
    if rare_cols:
        X = X.drop(columns=rare_cols)

    X = sm.add_constant(X, has_constant="add")
    y = model_df[TARGET].astype(float)

    try:
        glm = sm.GLM(y, X, family=sm.families.Binomial())
        res = glm.fit(maxiter=150, disp=0)
    except Exception as e:
        log(f"  GLM failed with error: {e}. Trying regularized fit without p-values...")
        try:
            glm = sm.GLM(y, X, family=sm.families.Binomial())
            res = glm.fit_regularized(alpha=1e-4, maxiter=200)
            coef = res.params
            out = pd.DataFrame(
                {
                    "term": coef.index,
                    "coef": coef.values,
                    "odds_ratio": np.exp(coef.values),
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "p_value": np.nan,
                }
            )
            out = out[out["term"] != "const"].copy()
            out["abs_log_odds"] = out["coef"].abs()
            out = out.sort_values("abs_log_odds", ascending=False)
            save_df(out, "controlled_logistic_odds_ratios.csv")
            plot_odds_ratio(out)
            return out
        except Exception as e2:
            log(f"  Regularized GLM also failed. Skipping odds ratio table. Error: {e2}")
            return pd.DataFrame()

    params = res.params
    conf = res.conf_int()
    out = pd.DataFrame(
        {
            "term": params.index,
            "coef": params.values,
            "odds_ratio": np.exp(params.values),
            "ci_lower": np.exp(conf[0].values),
            "ci_upper": np.exp(conf[1].values),
            "p_value": res.pvalues.values,
        }
    )
    out = out[out["term"] != "const"].copy()
    out["abs_log_odds"] = out["coef"].abs()
    out["direction"] = np.where(out["odds_ratio"] > 1, "Higher odds of default", "Lower odds of default")
    out = out.sort_values("abs_log_odds", ascending=False)
    save_df(out, "controlled_logistic_odds_ratios.csv")

    model_info = pd.DataFrame(
        [
            {
                "n_rows_used": int(len(model_df)),
                "n_features_after_dummies": int(X.shape[1] - 1),
                "aic": getattr(res, "aic", np.nan),
                "bic": getattr(res, "bic", np.nan),
                "llf": getattr(res, "llf", np.nan),
                "converged": str(getattr(res, "converged", "unknown")),
                "note": "Statsmodels GLM uses a sample if STATSMODELS_MAX_ROWS is set.",
            }
        ]
    )
    save_df(model_info, "controlled_logistic_model_info.csv")
    plot_odds_ratio(out)
    return out


def clean_term_label(term: str) -> str:
    term = re.sub(r"^DIAG_", "", term)
    term = term.replace("_", " ")
    return term


def plot_odds_ratio(or_table: pd.DataFrame) -> None:
    if or_table.empty:
        return
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Show strongest effects but avoid infinite/extreme unreadable values.
    temp = or_table.copy()
    temp = temp[np.isfinite(temp["odds_ratio"])]
    temp = temp[(temp["odds_ratio"] > 0) & (temp["odds_ratio"] < 20)]
    top = temp.sort_values("abs_log_odds", ascending=False).head(25).copy()
    top["label"] = top["term"].map(clean_term_label)
    top = top.sort_values("odds_ratio", ascending=True)

    plt.figure(figsize=(10, 9))
    ax = sns.scatterplot(data=top, x="odds_ratio", y="label", hue="direction", s=80)
    if "ci_lower" in top.columns and top["ci_lower"].notna().any():
        for _, row in top.iterrows():
            if pd.notna(row["ci_lower"]) and pd.notna(row["ci_upper"]):
                plt.plot([row["ci_lower"], row["ci_upper"]], [row["label"], row["label"]], color="#8A7CB8", alpha=0.7)
    plt.axvline(1, color="#1F2937", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_title("Controlled Logistic Regression â€” Top Odds Ratios", fontsize=13, weight="bold")
    ax.set_xlabel("Odds Ratio (log scale)")
    ax.set_ylabel("")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "controlled_logistic_top_odds_ratios.png", dpi=180, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# Replacement odds-ratio routine:
# The original wide statsmodels GLM can be unstable because it includes many
# overlapping grouped variables at once. This research-style version fits several
# narrower GLM specifications and exports the classic regression table the user
# expects: beta, standard error, z, p-value, odds ratio, confidence interval and
# pseudo RÂ² metrics.
# -----------------------------------------------------------------------------

def inferential_model_specs(df: pd.DataFrame) -> dict[str, dict]:
    profile = existing_features(df, INFERENTIAL_PROFILE_FEATURES)
    affordability = existing_features(df, INFERENTIAL_AFFORDABILITY_FEATURES)
    credit = existing_features(df, INFERENTIAL_CREDIT_HISTORY_FEATURES)
    payment = existing_features(df, INFERENTIAL_PAYMENT_FEATURES)
    underpay = existing_features(df, INFERENTIAL_PAYMENT_SENSITIVITY_FEATURES)
    core = profile + affordability + credit + payment
    return {
        "01_glm_profile_only": {
            "features": profile,
            "description": "Research GLM: customer profile only",
        },
        "02_glm_affordability_only": {
            "features": affordability,
            "description": "Research GLM: affordability only",
        },
        "03_glm_credit_history_only": {
            "features": credit,
            "description": "Research GLM: credit history only",
        },
        "04_glm_payment_behavior_only": {
            "features": payment,
            "description": "Research GLM: payment behavior only",
        },
        "05_glm_controlled_core": {
            "features": core,
            "description": "Research GLM: controlled core model",
        },
        "06_glm_controlled_plus_underpayment_sensitivity": {
            "features": core + underpay,
            "description": "Research GLM: core plus underpayment sensitivity",
        },
    }


def stratified_model_sample(model_df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or len(model_df) <= max_rows:
        return model_df.copy()
    sampled_parts = []
    for _, group in model_df.groupby(TARGET, group_keys=False):
        sample_size = min(len(group), max(1, int(round(max_rows * len(group) / len(model_df)))))
        sampled_parts.append(group.sample(n=sample_size, random_state=RANDOM_STATE))
    out = pd.concat(sampled_parts, ignore_index=True)
    return out.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def treatment_formula_term(feature: str, data: pd.DataFrame) -> tuple[str, str]:
    values = data[feature].astype("object").fillna("Unknown").astype(str)
    ref = REFERENCE_GROUPS.get(feature)
    if ref not in set(values.dropna().unique()):
        mode = values.mode(dropna=True)
        ref = str(mode.iloc[0]) if len(mode) else "Unknown"
    return f"C({feature}, Treatment(reference={ref!r}))", ref


def parse_glm_term(term: str, reference_map: dict[str, str]) -> tuple[str, str, str]:
    if term == "Intercept":
        return "Intercept", "Intercept", ""
    variable = term
    level = ""
    if term.startswith("C("):
        variable = term.split(",", 1)[0].replace("C(", "")
        if "[T." in term:
            level = term.split("[T.", 1)[1].rstrip("]")
    reference = reference_map.get(variable, "")
    return variable, level, reference


def significance_stars(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    if p_value < 0.10:
        return "."
    return ""


def lr_p_value(lr_stat: float, df_diff: int) -> float:
    if pd.isna(lr_stat) or df_diff <= 0:
        return np.nan
    try:
        from scipy.stats import chi2
        return float(chi2.sf(lr_stat, df_diff))
    except Exception:
        return np.nan


def build_wald_tests_for_glm(res, spec_name: str, description: str, reference_map: dict[str, str]) -> pd.DataFrame:
    try:
        from scipy.stats import chi2
    except Exception:
        chi2 = None
    cov = res.cov_params()
    rows = []
    for variable, reference in reference_map.items():
        terms = [t for t in res.params.index if parse_glm_term(t, reference_map)[0] == variable]
        if not terms:
            continue
        beta = res.params.loc[terms].to_numpy(dtype=float)
        cov_sub = cov.loc[terms, terms].to_numpy(dtype=float)
        try:
            inv_cov = np.linalg.pinv(cov_sub)
            wald_chi2 = float(beta.T @ inv_cov @ beta)
            df_wald = int(len(terms))
            p_value = float(chi2.sf(wald_chi2, df_wald)) if chi2 is not None else np.nan
        except Exception as e:
            wald_chi2 = np.nan
            df_wald = int(len(terms))
            p_value = np.nan
            log(f"    Wald test failed for {spec_name} / {variable}: {e}")
        rows.append(
            {
                "model_spec": spec_name,
                "description": description,
                "variable": variable,
                "reference_group": reference,
                "term_count": len(terms),
                "wald_chi2": wald_chi2,
                "wald_df": df_wald,
                "wald_p_value": p_value,
                "significance": significance_stars(p_value),
            }
        )
    return pd.DataFrame(rows)


def build_marginal_effects_for_glm(
    res,
    model_df: pd.DataFrame,
    features: list[str],
    spec_name: str,
    description: str,
    reference_map: dict[str, str],
) -> pd.DataFrame:
    if not features:
        return pd.DataFrame()
    effect_df = stratified_model_sample(model_df[[TARGET] + features].copy(), MARGINAL_EFFECT_MAX_ROWS)
    rows = []
    log(f"    [{spec_name}] Calculating discrete average marginal effects...")
    for variable in features:
        reference = reference_map.get(variable)
        if reference is None:
            continue
        levels = [str(v) for v in pd.Series(effect_df[variable].astype(str).unique()).dropna().tolist()]
        levels = [v for v in sorted(levels) if v != reference]
        if not levels:
            continue
        base_data = effect_df[features].copy()
        base_data[variable] = reference
        try:
            base_pred = safe_probability(res.predict(base_data))
        except Exception as e:
            log(f"      Marginal-effect baseline failed for {variable}: {e}")
            continue
        for level in levels:
            alt_data = base_data.copy()
            alt_data[variable] = level
            try:
                alt_pred = safe_probability(res.predict(alt_data))
                diff = alt_pred - base_pred
                rows.append(
                    {
                        "model_spec": spec_name,
                        "description": description,
                        "variable": variable,
                        "level": level,
                        "reference_group": reference,
                        "n_rows_used": int(len(effect_df)),
                        "average_marginal_effect": float(np.mean(diff)),
                        "average_marginal_effect_pp": float(np.mean(diff) * 100),
                        "median_marginal_effect": float(np.median(diff)),
                        "min_marginal_effect": float(np.min(diff)),
                        "max_marginal_effect": float(np.max(diff)),
                        "direction": "Higher predicted default probability" if np.mean(diff) > 0 else "Lower predicted default probability",
                    }
                )
            except Exception as e:
                log(f"      Marginal effect failed for {variable}={level}: {e}")
    return pd.DataFrame(rows)


def fit_research_glm_spec(df: pd.DataFrame, spec_name: str, features: list[str], description: str) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    model_df = df[[TARGET] + features].copy()
    for c in features:
        model_df[c] = model_df[c].astype("object").fillna("Unknown").astype(str)
        # Collapse extremely rare categories for inference stability.
        counts = model_df[c].value_counts(dropna=False)
        rare_values = set(counts[counts < 50].index)
        if rare_values:
            model_df[c] = np.where(model_df[c].isin(rare_values), "Other rare", model_df[c])

    model_df = stratified_model_sample(model_df, INFERENTIAL_MAX_ROWS)
    n = len(model_df)
    events = int(model_df[TARGET].sum())

    reference_map = {}
    terms = []
    for f in features:
        term, ref = treatment_formula_term(f, model_df)
        terms.append(term)
        reference_map[f] = ref

    formula = f"{TARGET} ~ " + " + ".join(terms) if terms else f"{TARGET} ~ 1"
    null_formula = f"{TARGET} ~ 1"

    fit_row = {
        "model_spec": spec_name,
        "description": description,
        "n_rows_used": n,
        "event_count": events,
        "event_rate": events / n if n else np.nan,
        "feature_count": len(features),
        "formula": formula,
        "converged": False,
    }

    try:
        log(f"    [{spec_name}] Fitting null GLM baseline...")
        null_res = smf.glm(null_formula, data=model_df, family=sm.families.Binomial()).fit(maxiter=100, disp=0)
        log(f"    [{spec_name}] Fitting full GLM with robust HC3 covariance...")
        res = smf.glm(formula, data=model_df, family=sm.families.Binomial()).fit(
            maxiter=150,
            disp=0,
            cov_type="HC3",
        )
        log(f"    [{spec_name}] Finished GLM fit. Extracting coefficients...")
    except Exception as e:
        fit_row["error"] = str(e)
        return fit_row, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    ll_model = float(res.llf)
    ll_null = float(null_res.llf)
    k = int(len(res.params))
    lr_stat = 2 * (ll_model - ll_null)
    df_diff = max(0, k - 1)
    cox_snell = 1 - np.exp((2 / n) * (ll_null - ll_model)) if n else np.nan
    nagelkerke_denom = 1 - np.exp((2 / n) * ll_null) if n else np.nan
    nagelkerke = cox_snell / nagelkerke_denom if nagelkerke_denom and nagelkerke_denom > 0 else np.nan

    fit_row.update(
        {
            "converged": bool(getattr(res, "converged", True)),
            "log_likelihood": ll_model,
            "null_log_likelihood": ll_null,
            "aic": -2 * ll_model + 2 * k,
            "bic": -2 * ll_model + k * np.log(n),
            "lr_chi2": lr_stat,
            "lr_df": df_diff,
            "lr_p_value": lr_p_value(lr_stat, df_diff),
            "mcfadden_pseudo_r2": 1 - (ll_model / ll_null) if ll_null != 0 else np.nan,
            "cox_snell_pseudo_r2": cox_snell,
            "nagelkerke_pseudo_r2": nagelkerke,
            "parameter_count": k,
            "error": "",
        }
    )

    conf = res.conf_int()
    rows = []
    for term in res.params.index:
        beta = float(res.params[term])
        se = float(res.bse[term])
        z_value = float(res.tvalues[term])
        p_value = float(res.pvalues[term])
        ci_low = float(conf.loc[term, 0])
        ci_high = float(conf.loc[term, 1])
        variable, level, reference = parse_glm_term(term, reference_map)
        rows.append(
            {
                "model_spec": spec_name,
                "term": term,
                "variable": variable,
                "level": level,
                "reference_group": reference,
                "beta": beta,
                "std_error": se,
                "z_value": z_value,
                "p_value": p_value,
                "significance": significance_stars(p_value),
                "odds_ratio": np.exp(beta) if -30 < beta < 30 else (np.inf if beta >= 30 else 0.0),
                "ci_lower": np.exp(ci_low) if -30 < ci_low < 30 else (np.inf if ci_low >= 30 else 0.0),
                "ci_upper": np.exp(ci_high) if -30 < ci_high < 30 else (np.inf if ci_high >= 30 else 0.0),
                "direction": "Higher odds of default" if beta > 0 else "Lower odds of default",
                "abs_log_odds": abs(beta),
            }
        )
    coef_df = pd.DataFrame(rows)
    wald_df = build_wald_tests_for_glm(res, spec_name, description, reference_map)
    marginal_df = build_marginal_effects_for_glm(res, model_df, features, spec_name, description, reference_map)
    return fit_row, coef_df, wald_df, marginal_df


def fit_odds_ratio_model(df: pd.DataFrame) -> pd.DataFrame:
    log("Fitting research-style logistic regression tables: beta, p-value, pseudo RÂ² and odds ratios...")
    try:
        import statsmodels.api  # noqa: F401
        import statsmodels.formula.api  # noqa: F401
    except Exception as e:
        log(f"statsmodels is not available. Skipping research-style logistic tables. Error: {e}")
        return pd.DataFrame()

    fit_rows = []
    coef_frames = []
    wald_frames = []
    marginal_frames = []
    research_specs = inferential_model_specs(df)
    glm_progress = ProgressTracker("RESEARCH GLM / ODDS RATIO", len(research_specs))
    for spec_name, spec in research_specs.items():
        features = spec["features"]
        if not features:
            glm_progress.skip(spec_name, "No features available")
            continue
        glm_progress.start(f"{spec_name} | research_features={len(features)}")
        log(f"  Fitting {spec_name} with {len(features)} research features...")
        fit_row, coef_df, wald_df, marginal_df = fit_research_glm_spec(df, spec_name, features, spec["description"])
        fit_rows.append(fit_row)
        if coef_df is not None and not coef_df.empty:
            coef_frames.append(coef_df)
        if wald_df is not None and not wald_df.empty:
            wald_frames.append(wald_df)
        if marginal_df is not None and not marginal_df.empty:
            marginal_frames.append(marginal_df)
        status = "failed" if fit_row.get("error") else "done"
        metric_msg = ""
        if pd.notna(fit_row.get("mcfadden_pseudo_r2", np.nan)):
            metric_msg = f" | McFadden R2={fit_row['mcfadden_pseudo_r2']:.4f}"
        glm_progress.finish(f"{spec_name}{metric_msg}", status=status)
    glm_progress.done_all()

    fit_table = pd.DataFrame(fit_rows)
    coef_table = pd.concat(coef_frames, ignore_index=True) if coef_frames else pd.DataFrame()
    wald_table = pd.concat(wald_frames, ignore_index=True) if wald_frames else pd.DataFrame()
    marginal_table = pd.concat(marginal_frames, ignore_index=True) if marginal_frames else pd.DataFrame()

    save_df(fit_table, "research_logistic_model_fit.csv")
    save_df(coef_table, "research_logistic_coefficients.csv")
    save_df(wald_table, "research_wald_tests.csv")
    save_df(marginal_table, "research_marginal_effects.csv")

    # Keep old filenames too, so downstream report/key-results still work.
    save_df(fit_table, "controlled_logistic_model_info.csv")
    save_df(coef_table, "controlled_logistic_odds_ratios.csv")

    # Plot the controlled core model if available; otherwise plot all.
    plot_source = coef_table[coef_table["model_spec"].eq("05_glm_controlled_core")].copy()
    if plot_source.empty:
        plot_source = coef_table.copy()
    plot_source = plot_source[plot_source["term"] != "Intercept"].copy()
    if not plot_source.empty:
        plot_odds_ratio(plot_source)

    return coef_table


def compute_vif_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    log("Computing VIF multicollinearity diagnostics for controlled-core research features...")
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except Exception as e:
        log(f"statsmodels VIF is not available. Skipping VIF diagnostics. Error: {e}")
        return pd.DataFrame()

    features = (
        existing_features(df, INFERENTIAL_PROFILE_FEATURES)
        + existing_features(df, INFERENTIAL_AFFORDABILITY_FEATURES)
        + existing_features(df, INFERENTIAL_CREDIT_HISTORY_FEATURES)
        + existing_features(df, INFERENTIAL_PAYMENT_FEATURES)
    )
    if not features:
        return pd.DataFrame()
    model_df = df[features].copy()
    for c in features:
        model_df[c] = model_df[c].astype("object").fillna("Unknown").astype(str)
        counts = model_df[c].value_counts(dropna=False)
        rare_values = set(counts[counts < 50].index)
        if rare_values:
            model_df[c] = np.where(model_df[c].isin(rare_values), "Other rare", model_df[c])
    model_df = stratified_model_sample(pd.concat([df[[TARGET]].reset_index(drop=True), model_df.reset_index(drop=True)], axis=1), VIF_MAX_ROWS)
    X = pd.get_dummies(model_df[features], drop_first=True, dtype=float)
    X = X.loc[:, X.nunique(dropna=False) > 1].copy()
    if X.empty:
        return pd.DataFrame()
    X.insert(0, "const", 1.0)
    rows = []
    vif_progress = ProgressTracker("VIF DIAGNOSTICS", max(X.shape[1] - 1, 1))
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        vif_progress.start(col)
        try:
            vif = float(variance_inflation_factor(X.values, i))
        except Exception as e:
            log(f"  VIF failed for {col}: {e}")
            vif = np.nan
        if pd.isna(vif):
            risk = "Unknown"
            tolerance = np.nan
        else:
            tolerance = 1 / vif if vif not in [0, np.inf] else 0.0
            if vif < 5:
                risk = "Low"
            elif vif < 10:
                risk = "Moderate"
            else:
                risk = "High"
        source_var = col.split("_", 1)[0] if "_" in col else col
        # Recover DIAG source variable more accurately by prefix matching.
        for f in features:
            if col.startswith(f + "_"):
                source_var = f
                break
        rows.append(
            {
                "source_variable": source_var,
                "encoded_feature": col,
                "vif": vif,
                "tolerance": tolerance,
                "vif_risk_level": risk,
                "n_rows_used": int(len(model_df)),
                "note": "VIF is calculated on one-hot encoded controlled-core research features with drop_first=True.",
            }
        )
        vif_progress.finish(f"{col} | VIF={vif:.2f}" if pd.notna(vif) else col)
    vif_progress.done_all()
    out = pd.DataFrame(rows).sort_values("vif", ascending=False, na_position="last")
    save_df(out, "vif_diagnostic_features.csv")
    save_df(out[out["vif"].fillna(0) >= 5].copy(), "high_vif_features.csv")
    return out


# =============================================================================
# 8. Summary / Report
# =============================================================================

def make_charts(segment_table: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    if "DIAG_RISK_SIGNAL_GROUP" in segment_table["variable"].unique():
        temp = segment_table[segment_table["variable"] == "DIAG_RISK_SIGNAL_GROUP"].copy()
        temp = temp.sort_values("segment")
        plt.figure(figsize=(8, 4.8))
        ax = sns.barplot(data=temp, x="segment", y="default_rate", color="#6C4CFF")
        ax.set_title("Default Rate by Diagnostic Risk Signal Group", fontsize=13, weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Default Rate")
        ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
        for container in ax.containers:
            ax.bar_label(container, labels=[f"{v:.2%}" for v in temp["default_rate"]], padding=3, fontsize=8)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(CHART_DIR / "default_rate_by_diagnostic_risk_signal_group.png", dpi=180, bbox_inches="tight")
        plt.close()

    top_impact = (
        segment_table[segment_table["risk_difference"] > 0]
        .sort_values("business_priority_score", ascending=False)
        .head(20)
        .copy()
    )
    top_impact["segment_label"] = top_impact["variable"].str.replace("DIAG_", "", regex=False) + " | " + top_impact["segment"].astype(str)
    top_impact = top_impact.sort_values("business_priority_score", ascending=True)

    plt.figure(figsize=(11, 7))
    ax = sns.barplot(data=top_impact, x="business_priority_score", y="segment_label", color="#FF5C6A")
    ax.set_title("Top Segments by Business Priority Score", fontsize=13, weight="bold")
    ax.set_xlabel("Risk difference x default customers")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "top_business_priority_segments.png", dpi=180, bbox_inches="tight")
    plt.close()


def df_to_html_table(df: pd.DataFrame, max_rows: int | None = 20, percent_cols: list[str] | None = None) -> str:
    if df is None or df.empty:
        return "<p><em>No data.</em></p>"
    temp = df.copy() if max_rows is None else df.head(max_rows).copy()
    percent_cols = percent_cols or []
    for c in percent_cols:
        if c in temp.columns:
            temp[c] = temp[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    for c in temp.select_dtypes(include=[float]).columns:
        if c not in percent_cols:
            temp[c] = temp[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    return temp.to_html(index=False, escape=False, classes="data-table")


def make_excel_summary(tables: dict[str, pd.DataFrame]) -> Path:
    path = TABLE_DIR / "step6_diagnostic_summary_tables.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in tables.items():
            if df is not None and not df.empty:
                sheet = re.sub(r"[^A-Za-z0-9_]", "_", name)[:31]
                df.to_excel(writer, sheet_name=sheet, index=False)
    return path


def make_key_results_markdown(
    df: pd.DataFrame,
    segment_table: pd.DataFrame,
    cat_tests: pd.DataFrame,
    num_tests: pd.DataFrame,
    model_comp: pd.DataFrame,
    odds: pd.DataFrame,
) -> Path:
    base_rate = df[TARGET].mean()
    lines = []
    lines.append("# Step 6 Diagnostic Analytics â€” Key Results for Review")
    lines.append("")
    lines.append(f"- Rows used: {len(df):,}")
    lines.append(f"- Baseline default rate: {base_rate:.2%}")
    lines.append("")

    spec_comp = pd.DataFrame()
    algo_comp = pd.DataFrame()
    if not model_comp.empty and "comparison_type" in model_comp.columns:
        spec_comp = model_comp[model_comp["comparison_type"].eq("Specification comparison")].copy()
        algo_comp = model_comp[model_comp["comparison_type"].eq("Algorithm comparison")].copy()
    else:
        spec_comp = model_comp.copy()

    lines.append("## Top Specification Comparison Results")
    lines.append("")
    lines.append("Same algorithm family: Logistic Regression. Different feature specifications.")
    if not model_comp.empty:
        for _, r in spec_comp.head(5).iterrows():
            lines.append(
                f"- `{r['model_spec']}`: ROC-AUC={r['roc_auc']:.4f}, "
                f"PR-AUC={r['pr_auc']:.4f}, Pseudo R2={r['pseudo_r2_vs_null_logloss']:.4f}, "
                f"Precision@Top10={r['precision_at_top_10pct_score']:.2%}, "
                f"Recall@Top10={r['recall_at_top_10pct_score']:.2%}"
            )
    lines.append("")

    if not algo_comp.empty:
        lines.append("## Top Diagnostic Binary Model-Family Comparison Results")
        lines.append("")
        lines.append("Same full diagnostic feature set. Different statistical/econometric binary-response models.")
        for _, r in algo_comp.head(8).iterrows():
            label = r["model_name"] if "model_name" in algo_comp.columns and pd.notna(r.get("model_name")) else r["algorithm"]
            lines.append(
                f"- `{label}`: ROC-AUC={r['roc_auc']:.4f}, "
                f"PR-AUC={r['pr_auc']:.4f}, Pseudo R2={r['pseudo_r2_vs_null_logloss']:.4f}, "
                f"Precision@Top10={r['precision_at_top_10pct_score']:.2%}, "
                f"Recall@Top10={r['recall_at_top_10pct_score']:.2%}, status={r['status']}"
            )
        lines.append("")

    research_fit_path = TABLE_DIR / "research_logistic_model_fit.csv"
    if research_fit_path.exists():
        research_fit = pd.read_csv(research_fit_path)
        lines.append("## Research-Style Logistic Regression Fit")
        for _, r in research_fit.sort_values("mcfadden_pseudo_r2", ascending=False).head(6).iterrows():
            lines.append(
                f"- `{r['model_spec']}`: McFadden R2={r['mcfadden_pseudo_r2']:.4f}, "
                f"Nagelkerke R2={r['nagelkerke_pseudo_r2']:.4f}, "
                f"AIC={r['aic']:.1f}, BIC={r['bic']:.1f}, converged={r['converged']}"
            )
        lines.append("")

    lines.append("## Top High-Risk Segments")
    top_risk = (
        segment_table[segment_table["customer_count"] >= SMALL_SAMPLE_THRESHOLD]
        .sort_values("risk_index", ascending=False)
        .head(10)
    )
    for _, r in top_risk.iterrows():
        lines.append(
            f"- `{r['variable']} = {r['segment']}`: default={r['default_rate']:.2%}, "
            f"risk_index={r['risk_index']:.2f}, customers={int(r['customer_count']):,}"
        )
    lines.append("")

    lines.append("## Top Business-Priority Segments")
    top_impact = (
        segment_table[segment_table["risk_difference"] > 0]
        .sort_values("business_priority_score", ascending=False)
        .head(10)
    )
    for _, r in top_impact.iterrows():
        lines.append(
            f"- `{r['variable']} = {r['segment']}`: default={r['default_rate']:.2%}, "
            f"default_customers={int(r['default_customers']):,}, "
            f"default_share={r['default_contribution_share']:.2%}"
        )
    lines.append("")

    lines.append("## Strongest Categorical Associations")
    if not cat_tests.empty:
        for _, r in cat_tests.head(10).iterrows():
            lines.append(
                f"- `{r['variable']}`: Cramer's V={r['cramers_v']:.4f}, "
                f"effect={r['effect_size_label']}, p={r['p_value']:.3g}"
            )
    lines.append("")

    lines.append("## Strongest Numeric Effects")
    if not num_tests.empty:
        for _, r in num_tests.head(10).iterrows():
            lines.append(
                f"- `{r['variable']}`: Cliff's delta={r['cliffs_delta']:.4f}, "
                f"KS={r['ks_statistic']:.4f}, effect={r['effect_size_label']}"
            )
    lines.append("")

    lines.append("## Top Controlled Logistic Odds Ratios")
    if odds is not None and not odds.empty:
        temp = odds[np.isfinite(odds["odds_ratio"])].copy()
        if "model_spec" in temp.columns and temp["model_spec"].eq("05_glm_controlled_core").any():
            temp = temp[temp["model_spec"].eq("05_glm_controlled_core")].copy()
        temp = temp[temp["term"] != "Intercept"].copy()
        temp = temp.sort_values("abs_log_odds", ascending=False).head(15)
        for _, r in temp.iterrows():
            ci = ""
            if "ci_lower" in temp.columns and pd.notna(r.get("ci_lower")):
                ci = f", CI=[{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]"
            p = f", p={r['p_value']:.3g}" if pd.notna(r.get("p_value", np.nan)) else ""
            lines.append(f"- `{r['term']}`: OR={r['odds_ratio']:.2f}{ci}{p}")
    lines.append("")

    lines.append("## Files to Send Back")
    lines.append("- `step6_outputs/step6_key_results_for_chat.md`")
    lines.append("- `step6_outputs/step6_diagnostic_analytics_report.html`")
    lines.append("- `step6_outputs/tables/diagnostic_model_comparison.csv`")
    lines.append("- `step6_outputs/tables/full_model_evaluation_summary.csv`")
    lines.append("- `step6_outputs/tables/terminal_model_evaluation_table.txt`")
    lines.append("- `step6_outputs/tables/threshold_analysis_all_models.csv`")
    lines.append("- `step6_outputs/tables/calibration_by_decile.csv`")
    lines.append("- `step6_outputs/tables/hosmer_lemeshow_test.csv`")
    lines.append("- `step6_outputs/tables/specification_comparison_research_table.csv`")
    lines.append("- `step6_outputs/tables/algorithm_comparison_research_table.csv`")
    lines.append("- `step6_outputs/tables/diagnostic_binary_model_family_comparison.csv`")
    lines.append("- `step6_outputs/tables/model_applicability_matrix.csv`")
    lines.append("- `step6_outputs/tables/controlled_logistic_odds_ratios.csv`")
    lines.append("- `step6_outputs/tables/research_wald_tests.csv`")
    lines.append("- `step6_outputs/tables/research_marginal_effects.csv`")
    lines.append("- `step6_outputs/tables/vif_diagnostic_features.csv`")
    lines.append("- `step6_outputs/tables/segment_risk_index_all_variables.csv`")
    lines.append("- Optional: zip the full `step6_outputs` folder.")

    path = OUTPUT_DIR / "step6_key_results_for_chat.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_html_report(
    df: pd.DataFrame,
    segment_table: pd.DataFrame,
    cat_tests: pd.DataFrame,
    num_tests: pd.DataFrame,
    model_comp: pd.DataFrame,
    odds: pd.DataFrame,
    interaction_outputs: dict[str, dict[str, str]],
) -> Path:
    base_rate = df[TARGET].mean()
    top_risk = (
        segment_table[segment_table["customer_count"] >= SMALL_SAMPLE_THRESHOLD]
        .sort_values("risk_index", ascending=False)
        .head(20)
    )
    top_impact = (
        segment_table[segment_table["risk_difference"] > 0]
        .sort_values("business_priority_score", ascending=False)
        .head(20)
    )
    risk_group = segment_table[segment_table["variable"] == "DIAG_RISK_SIGNAL_GROUP"].sort_values("segment")
    applicability_path = TABLE_DIR / "model_applicability_matrix.csv"
    model_applicability = pd.read_csv(applicability_path) if applicability_path.exists() else pd.DataFrame()
    research_fit_path = TABLE_DIR / "research_logistic_model_fit.csv"
    research_fit = pd.read_csv(research_fit_path) if research_fit_path.exists() else pd.DataFrame()
    optional_tables = {
        "terminal_eval": TABLE_DIR / "terminal_model_evaluation_table.csv",
        "full_eval": TABLE_DIR / "full_model_evaluation_summary.csv",
        "threshold_all": TABLE_DIR / "threshold_analysis_all_models.csv",
        "threshold_main": TABLE_DIR / "threshold_analysis_main_logit.csv",
        "calibration_summary": TABLE_DIR / "calibration_error_summary.csv",
        "calibration": TABLE_DIR / "calibration_by_decile.csv",
        "hl_summary": TABLE_DIR / "hosmer_lemeshow_test.csv",
        "hl_deciles": TABLE_DIR / "hosmer_lemeshow_decile_table.csv",
        "wald_tests": TABLE_DIR / "research_wald_tests.csv",
        "marginal_effects": TABLE_DIR / "research_marginal_effects.csv",
        "vif_table": TABLE_DIR / "vif_diagnostic_features.csv",
    }
    loaded_optional = {
        name: pd.read_csv(path) if path.exists() else pd.DataFrame()
        for name, path in optional_tables.items()
    }
    if not model_comp.empty and "comparison_type" in model_comp.columns:
        spec_comp = model_comp[model_comp["comparison_type"].eq("Specification comparison")].copy()
        algo_comp = model_comp[model_comp["comparison_type"].eq("Algorithm comparison")].copy()
    else:
        spec_comp = model_comp.copy()
        algo_comp = pd.DataFrame()
    odds_display = odds.copy() if odds is not None else pd.DataFrame()
    if not odds_display.empty and "model_spec" in odds_display.columns and odds_display["model_spec"].eq("05_glm_controlled_core").any():
        odds_display = odds_display[odds_display["model_spec"].eq("05_glm_controlled_core")].copy()
    if not odds_display.empty and "term" in odds_display.columns:
        odds_display = odds_display[odds_display["term"] != "Intercept"].copy()
        odds_display = odds_display.sort_values("abs_log_odds", ascending=False)

    chart_imgs = []
    for p in [
        CHART_DIR / "default_rate_by_diagnostic_risk_signal_group.png",
        CHART_DIR / "top_business_priority_segments.png",
        CHART_DIR / "diagnostic_model_comparison_auc.png",
        CHART_DIR / "diagnostic_model_comparison_pseudo_r2.png",
        CHART_DIR / "algorithm_comparison_roc_auc.png",
        CHART_DIR / "algorithm_comparison_pr_auc.png",
        CHART_DIR / "controlled_logistic_top_odds_ratios.png",
        CHART_DIR / "roc_curve_main_diagnostic_models.png",
        CHART_DIR / "precision_recall_curve_main_diagnostic_models.png",
        CHART_DIR / "calibration_plot_main_diagnostic_models.png",
        CHART_DIR / "threshold_analysis_main_logit.png",
    ]:
        if p.exists():
            chart_imgs.append(p)
    for info in interaction_outputs.values():
        p = Path(info["chart"])
        if p.exists():
            chart_imgs.append(p)

    chart_html = "\n".join(
        [f'<div class="chart"><img src="{p.relative_to(OUTPUT_DIR).as_posix()}" /></div>' for p in chart_imgs]
    )

    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Step 6 Diagnostic Analytics Report</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 28px;
    color: #1F2937;
    background: #FAF8FF;
}}
h1, h2, h3 {{ color: #4F36E8; }}
.subtitle {{ color: #6B7280; margin-top: -8px; }}
.card {{
    background: #FFFFFF;
    border: 1px solid #E5DDF8;
    border-radius: 12px;
    padding: 18px;
    margin: 18px 0;
    box-shadow: 0 2px 8px rgba(37, 30, 84, 0.08);
}}
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, minmax(150px, 1fr));
    gap: 14px;
}}
.kpi {{
    background: #FFFFFF;
    border-left: 5px solid #6C4CFF;
    border-radius: 10px;
    padding: 14px;
}}
.kpi .label {{ color: #6B7280; font-size: 13px; }}
.kpi .value {{ font-size: 26px; font-weight: 700; margin-top: 6px; }}
table.data-table {{
    border-collapse: collapse;
    min-width: 100%;
    font-size: 13px;
}}
table.data-table th {{
    background: #EEE8FF;
    color: #1F2937;
    padding: 8px;
    border: 1px solid #DDD4F6;
    text-align: left;
}}
table.data-table td {{
    padding: 7px;
    border: 1px solid #EEE8FF;
}}
table.data-table tr:nth-child(even) {{ background: #F7F3FF; }}
.chart img {{
    max-width: 100%;
    background: white;
    border: 1px solid #E5DDF8;
    border-radius: 10px;
    margin: 12px 0;
}}
.note {{
    background: #EEE8FF;
    border-left: 5px solid #6C4CFF;
    padding: 12px;
    border-radius: 8px;
}}
.table-wrap {{
    overflow-x: auto;
    width: 100%;
}}
</style>
</head>
<body>
<h1>Step 6 â€” Diagnostic Analytics</h1>
<p class="subtitle">Validating dashboard insights before Machine Learning Support</p>

<div class="kpi-grid">
  <div class="kpi"><div class="label">Rows</div><div class="value">{len(df):,}</div></div>
  <div class="kpi"><div class="label">Baseline Default Rate</div><div class="value">{base_rate:.2%}</div></div>
  <div class="kpi"><div class="label">Diagnostic Group Variables</div><div class="value">{len(DIAGNOSTIC_GROUP_VARS)}</div></div>
  <div class="kpi"><div class="label">Model Rows Compared</div><div class="value">{len(model_comp)}</div></div>
</div>

<div class="card">
<h2>1. Diagnostic Objective</h2>
<p>
This step does not try to optimize a Kaggle score. It validates whether the dashboard patterns are
statistically and commercially meaningful. The analysis combines risk lift, contribution analysis,
statistical tests, effect sizes, interaction diagnostics and controlled logistic regression.
</p>
</div>

<div class="card">
<h2>2. Risk Signal Group Validation</h2>
{df_to_html_table(risk_group, max_rows=20, percent_cols=["default_rate", "customer_share", "default_contribution_share", "risk_difference"])}
</div>

<div class="card">
<h2>3. Top High-Risk Segments</h2>
{df_to_html_table(top_risk, max_rows=20, percent_cols=["default_rate", "customer_share", "default_contribution_share", "risk_difference"])}
</div>

<div class="card">
<h2>4. Top Business-Priority Segments</h2>
<p class="note">A segment is business-priority when it has both elevated default rate and meaningful default-customer volume.</p>
{df_to_html_table(top_impact, max_rows=20, percent_cols=["default_rate", "customer_share", "default_contribution_share", "risk_difference"])}
</div>

<div class="card">
<h2>5. Statistical Tests â€” Categorical Variables</h2>
{df_to_html_table(cat_tests, max_rows=20)}
</div>

<div class="card">
<h2>6. Statistical Tests â€” Numeric Variables</h2>
{df_to_html_table(num_tests, max_rows=20)}
</div>

<div class="card">
<h2>7A. Model Applicability Matrix</h2>
<p class="note">Not every binary-outcome model is valid for this dataset. This table documents which models are run in Step 6 and why other models are deferred or not applicable.</p>
<div class="table-wrap">{df_to_html_table(model_applicability, max_rows=None)}</div>
</div>

<div class="card">
<h2>7B. Specification Comparison</h2>
<p class="note">Same algorithm family: Logistic Regression. Different feature specifications. This answers which feature group explains default risk better.</p>
<div class="table-wrap">{df_to_html_table(spec_comp, max_rows=None)}</div>
</div>

<div class="card">
<h2>7C. Diagnostic Binary Model-Family Comparison</h2>
<p class="note">Same full diagnostic feature set. Different statistical/econometric binary-response models. Machine Learning models are intentionally excluded from this Step 6 diagnostic-only comparison.</p>
<div class="table-wrap">{df_to_html_table(algo_comp, max_rows=None)}</div>
</div>

<div class="card">
<h2>7D. Combined Model Comparison Table</h2>
<div class="table-wrap">{df_to_html_table(model_comp, max_rows=None)}</div>
</div>

<div class="card">
<h2>7E. Full Model Evaluation Summary</h2>
<p class="note">All model-level fit, classification, ranking and probability-quality metrics in one table.</p>
<div class="table-wrap">{df_to_html_table(loaded_optional["full_eval"], max_rows=None)}</div>
</div>

<div class="card">
<h2>7F. Terminal Model Evaluation Table</h2>
<p class="note">This compact table is also printed in the terminal and saved as <code>terminal_model_evaluation_table.txt</code>.</p>
<div class="table-wrap">{df_to_html_table(loaded_optional["terminal_eval"], max_rows=None)}</div>
</div>

<div class="card">
<h2>7G. Classification Threshold Analysis</h2>
<p class="note">Credit-risk data is imbalanced, so threshold 0.5 is not enough. This section compares probability thresholds and top-risk cutoffs.</p>
<h3>Main Logit / Full Specifications</h3>
<div class="table-wrap">{df_to_html_table(loaded_optional["threshold_main"], max_rows=None)}</div>
<h3>All Models</h3>
<div class="table-wrap">{df_to_html_table(loaded_optional["threshold_all"], max_rows=None)}</div>
</div>

<div class="card">
<h2>7H. Calibration and Hosmer-Lemeshow</h2>
<p class="note">Calibration checks whether predicted probabilities are close to actual default rates. Hosmer-Lemeshow is very sensitive with large samples, so read it with Brier Score and the calibration plot.</p>
<h3>Calibration Error Summary</h3>
<div class="table-wrap">{df_to_html_table(loaded_optional["calibration_summary"], max_rows=None)}</div>
<h3>Calibration by Decile</h3>
<div class="table-wrap">{df_to_html_table(loaded_optional["calibration"], max_rows=None, percent_cols=["observed_default_rate", "mean_predicted_probability", "calibration_gap", "abs_calibration_gap"])}</div>
<h3>Hosmer-Lemeshow Test</h3>
<div class="table-wrap">{df_to_html_table(loaded_optional["hl_summary"], max_rows=None)}</div>
<h3>Hosmer-Lemeshow Decile Table</h3>
<div class="table-wrap">{df_to_html_table(loaded_optional["hl_deciles"], max_rows=None, percent_cols=["observed_default_rate", "mean_predicted_probability", "calibration_gap", "abs_calibration_gap"])}</div>
</div>

<div class="card">
<h2>8. Research-Style Logistic Regression Fit</h2>
<p class="note">For logistic regression, ordinary linear RÂ² is not appropriate. This section reports pseudo RÂ² metrics, AIC/BIC, log-likelihood and likelihood-ratio tests.</p>
<div class="table-wrap">{df_to_html_table(research_fit, max_rows=None, percent_cols=["event_rate"])}</div>
</div>

<div class="card">
<h2>8B. Wald Tests by Variable Group</h2>
<p class="note">Wald tests check whether a categorical variable contributes jointly through all of its dummy levels.</p>
<div class="table-wrap">{df_to_html_table(loaded_optional["wald_tests"], max_rows=None)}</div>
</div>

<div class="card">
<h2>8C. Average Marginal Effects</h2>
<p class="note">Marginal effects translate logit coefficients into percentage-point changes in predicted default probability versus the reference group.</p>
<div class="table-wrap">{df_to_html_table(loaded_optional["marginal_effects"], max_rows=None)}</div>
</div>

<div class="card">
<h2>8D. Multicollinearity / VIF Diagnostics</h2>
<p class="note">VIF checks whether encoded explanatory variables are highly correlated. VIF below 5 is usually low concern; 5-10 moderate; above 10 high.</p>
<div class="table-wrap">{df_to_html_table(loaded_optional["vif_table"], max_rows=None)}</div>
</div>

<div class="card">
<h2>9. Controlled Logistic Regression - Beta, p-value and Odds Ratios</h2>
<p class="note">Each coefficient compares a category against its reference group after controlling for the other variables in the selected model.</p>
<div class="table-wrap">{df_to_html_table(odds_display, max_rows=None)}</div>
</div>

<div class="card">
<h2>10. Charts</h2>
{chart_html}
</div>

<div class="card">
<h2>11. Output Files</h2>
<ul>
<li><code>tables/segment_risk_index_all_variables.csv</code></li>
<li><code>tables/top_30_highest_risk_index_segments.csv</code></li>
<li><code>tables/top_30_business_priority_segments.csv</code></li>
<li><code>tables/categorical_statistical_tests.csv</code></li>
<li><code>tables/numeric_statistical_tests.csv</code></li>
<li><code>tables/diagnostic_model_comparison.csv</code></li>
<li><code>tables/full_model_evaluation_summary.csv</code></li>
<li><code>tables/terminal_model_evaluation_table.csv</code></li>
<li><code>tables/terminal_model_evaluation_table.txt</code></li>
<li><code>tables/threshold_analysis_all_models.csv</code></li>
<li><code>tables/threshold_analysis_main_logit.csv</code></li>
<li><code>tables/roc_curve_points.csv</code></li>
<li><code>tables/precision_recall_curve_points.csv</code></li>
<li><code>tables/calibration_by_decile.csv</code></li>
<li><code>tables/calibration_error_summary.csv</code></li>
<li><code>tables/hosmer_lemeshow_test.csv</code></li>
<li><code>tables/hosmer_lemeshow_decile_table.csv</code></li>
<li><code>tables/specification_comparison_research_table.csv</code></li>
<li><code>tables/algorithm_comparison_research_table.csv</code></li>
<li><code>tables/diagnostic_binary_model_family_comparison.csv</code></li>
<li><code>tables/model_applicability_matrix.csv</code></li>
<li><code>tables/research_logistic_model_fit.csv</code></li>
<li><code>tables/research_logistic_coefficients.csv</code></li>
<li><code>tables/research_wald_tests.csv</code></li>
<li><code>tables/research_marginal_effects.csv</code></li>
<li><code>tables/vif_diagnostic_features.csv</code></li>
<li><code>tables/high_vif_features.csv</code></li>
<li><code>tables/controlled_logistic_odds_ratios.csv</code></li>
<li><code>tables/step6_diagnostic_summary_tables.xlsx</code></li>
<li><code>data/diagnostic_customer_dataset.csv.gz</code></li>
</ul>
</div>
</body>
</html>
"""
    path = OUTPUT_DIR / "step6_diagnostic_analytics_report.html"
    path.write_text(html, encoding="utf-8")
    return path


# =============================================================================
# 9. Main
# =============================================================================

def main() -> None:
    t0 = time.time()
    ensure_dirs()
    reset_progress_log()
    main_progress = ProgressTracker("MAIN PIPELINE", 14)

    main_progress.start("Find and read final analytical dataset")
    input_path = find_input_path()
    df = read_final_dataset(input_path)
    main_progress.finish("Find and read final analytical dataset")

    main_progress.start("Create diagnostic grouped features")
    df_diag = make_group_features(df)
    main_progress.finish("Create diagnostic grouped features")

    main_progress.start("Save diagnostic customer dataset")
    diag_cols = [TARGET, "SK_ID_CURR"] + [c for c in df_diag.columns if c.startswith("DIAG_")]
    diag_cols = [c for c in diag_cols if c in df_diag.columns]
    diag_dataset_path = DATA_DIR / "diagnostic_customer_dataset.csv.gz"
    df_diag[diag_cols].to_csv(diag_dataset_path, index=False, compression="gzip")
    log(f"Saved diagnostic dataset: {diag_dataset_path}")
    main_progress.finish("Save diagnostic customer dataset")

    main_progress.start("Build segment risk index and contribution tables")
    segment_table = make_all_segment_tables(df_diag)
    main_progress.finish("Build segment risk index and contribution tables")

    main_progress.start("Run categorical statistical tests")
    cat_tests = categorical_tests(df_diag, DIAGNOSTIC_GROUP_VARS)
    main_progress.finish("Run categorical statistical tests")

    main_progress.start("Run numeric statistical tests")
    num_tests = numeric_tests(df_diag, NUMERIC_TEST_VARS)
    main_progress.finish("Run numeric statistical tests")

    main_progress.start("Build interaction diagnostic matrices and heatmaps")
    interaction_outputs = make_interaction_outputs(df_diag)
    main_progress.finish("Build interaction diagnostic matrices and heatmaps")

    main_progress.start("Run diagnostic model comparison")
    model_comp, fitted_models = model_comparison(df_diag)
    main_progress.finish("Run diagnostic model comparison")

    main_progress.start("Prepare model comparison subtables")
    if not model_comp.empty and "comparison_type" in model_comp.columns:
        spec_comp = model_comp[model_comp["comparison_type"].eq("Specification comparison")].copy()
        algo_comp = model_comp[model_comp["comparison_type"].eq("Algorithm comparison")].copy()
    else:
        spec_comp = model_comp.copy()
        algo_comp = pd.DataFrame()
    applicability_path = TABLE_DIR / "model_applicability_matrix.csv"
    model_applicability = pd.read_csv(applicability_path) if applicability_path.exists() else pd.DataFrame()
    main_progress.finish("Prepare model comparison subtables")

    main_progress.start("Run research-style GLM odds-ratio tables")
    odds = fit_odds_ratio_model(df_diag)
    main_progress.finish("Run research-style GLM odds-ratio tables")

    main_progress.start("Run VIF multicollinearity diagnostics")
    vif_table = compute_vif_diagnostics(df_diag)
    main_progress.finish("Run VIF multicollinearity diagnostics")

    main_progress.start("Create summary charts")
    make_charts(segment_table)
    main_progress.finish("Create summary charts")

    main_progress.start("Create Excel summary workbook")
    research_fit = pd.read_csv(TABLE_DIR / "research_logistic_model_fit.csv") if (TABLE_DIR / "research_logistic_model_fit.csv").exists() else pd.DataFrame()
    research_coef = pd.read_csv(TABLE_DIR / "research_logistic_coefficients.csv") if (TABLE_DIR / "research_logistic_coefficients.csv").exists() else pd.DataFrame()
    research_wald = pd.read_csv(TABLE_DIR / "research_wald_tests.csv") if (TABLE_DIR / "research_wald_tests.csv").exists() else pd.DataFrame()
    research_marginal = pd.read_csv(TABLE_DIR / "research_marginal_effects.csv") if (TABLE_DIR / "research_marginal_effects.csv").exists() else pd.DataFrame()
    terminal_eval = pd.read_csv(TABLE_DIR / "terminal_model_evaluation_table.csv") if (TABLE_DIR / "terminal_model_evaluation_table.csv").exists() else pd.DataFrame()
    full_eval = pd.read_csv(TABLE_DIR / "full_model_evaluation_summary.csv") if (TABLE_DIR / "full_model_evaluation_summary.csv").exists() else pd.DataFrame()
    threshold_all = pd.read_csv(TABLE_DIR / "threshold_analysis_all_models.csv") if (TABLE_DIR / "threshold_analysis_all_models.csv").exists() else pd.DataFrame()
    threshold_main = pd.read_csv(TABLE_DIR / "threshold_analysis_main_logit.csv") if (TABLE_DIR / "threshold_analysis_main_logit.csv").exists() else pd.DataFrame()
    calibration_summary = pd.read_csv(TABLE_DIR / "calibration_error_summary.csv") if (TABLE_DIR / "calibration_error_summary.csv").exists() else pd.DataFrame()
    calibration = pd.read_csv(TABLE_DIR / "calibration_by_decile.csv") if (TABLE_DIR / "calibration_by_decile.csv").exists() else pd.DataFrame()
    hl_summary = pd.read_csv(TABLE_DIR / "hosmer_lemeshow_test.csv") if (TABLE_DIR / "hosmer_lemeshow_test.csv").exists() else pd.DataFrame()
    hl_deciles = pd.read_csv(TABLE_DIR / "hosmer_lemeshow_decile_table.csv") if (TABLE_DIR / "hosmer_lemeshow_decile_table.csv").exists() else pd.DataFrame()

    make_excel_summary(
        {
            "segment_risk_index": segment_table,
            "top_high_risk_segments": segment_table[segment_table["customer_count"] >= SMALL_SAMPLE_THRESHOLD]
            .sort_values("risk_index", ascending=False)
            .head(50),
            "top_business_priority": segment_table[segment_table["risk_difference"] > 0]
            .sort_values("business_priority_score", ascending=False)
            .head(50),
            "categorical_tests": cat_tests,
            "numeric_tests": num_tests,
            "model_comparison": model_comp,
            "specification_comparison": spec_comp,
            "algorithm_comparison": algo_comp,
            "model_applicability": model_applicability,
            "terminal_eval": terminal_eval,
            "full_eval": full_eval,
            "threshold_all_models": threshold_all,
            "threshold_main_logit": threshold_main,
            "calibration_summary": calibration_summary,
            "calibration_deciles": calibration,
            "hosmer_lemeshow": hl_summary,
            "hl_deciles": hl_deciles,
            "research_glm_fit": research_fit,
            "research_glm_coefficients": research_coef,
            "research_wald_tests": research_wald,
            "marginal_effects": research_marginal,
            "vif_diagnostics": vif_table,
            "odds_ratios": odds,
        }
    )
    main_progress.finish("Create Excel summary workbook")

    main_progress.start("Create Markdown, HTML report, and manifest")
    key_path = make_key_results_markdown(df_diag, segment_table, cat_tests, num_tests, model_comp, odds)
    report_path = make_html_report(df_diag, segment_table, cat_tests, num_tests, model_comp, odds, interaction_outputs)

    manifest = {
        "input_path": str(input_path),
        "rows": int(len(df_diag)),
        "columns": int(df_diag.shape[1]),
        "baseline_default_rate": float(df_diag[TARGET].mean()),
        "output_dir": str(OUTPUT_DIR),
        "report_html": str(report_path),
        "key_results_md": str(key_path),
        "diagnostic_dataset": str(diag_dataset_path),
        "runtime_seconds": round(time.time() - t0, 2),
    }
    (OUTPUT_DIR / "step6_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    main_progress.finish("Create Markdown, HTML report, and manifest")
    main_progress.done_all()

    log("Done.")
    log(f"HTML report: {report_path}")
    log(f"Key results for chat: {key_path}")
    log(f"Runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
