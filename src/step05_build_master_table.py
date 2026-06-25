import gc
import csv
import json
import os
import zipfile
import base64
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 250)
pd.set_option("display.max_rows", 200)
sns.set_theme(style="whitegrid", palette="Set2")


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Put Kaggle Home Credit CSV files in data/raw by default.
# Override when needed:
#   PowerShell: $env:CREDIT_RISK_RAW_DIR="D:\path\to\raw"; python src\step05_build_master_table.py
#   Bash:       CREDIT_RISK_RAW_DIR=/path/to/raw python src/step05_build_master_table.py
DATA_PATH = Path(os.environ.get("CREDIT_RISK_RAW_DIR", PROJECT_ROOT / "data" / "raw"))
OUT_PATH = Path(os.environ.get("CREDIT_RISK_OUTPUT_DIR", PROJECT_ROOT / "outputs" / "step05_master_table_run"))
FIG_PATH = OUT_PATH / "figures"
TABLE_PATH = OUT_PATH / "tables"
DATA_OUT_PATH = OUT_PATH / "data"

FIG_PATH.mkdir(parents=True, exist_ok=True)
TABLE_PATH.mkdir(parents=True, exist_ok=True)
DATA_OUT_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def save_table(df, name, display_rows=20):
    path = TABLE_PATH / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"Saved table: {path}")
    print(df.head(display_rows).to_string(index=False))
    return path


def save_fig(name):
    path = FIG_PATH / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()
    print(f"Saved figure: {path}")
    return path


def safe_divide(numerator, denominator):
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def add_history_flag(final_df, count_col, flag_col):
    final_df[flag_col] = final_df[count_col].notna().astype(int)
    return final_df


def fill_count_like_columns(df):
    count_like_tokens = [
        "_COUNT", "_SUM", "_FLAG", "_MONTH_COUNT", "_LOAN_COUNT",
        "HAS_", "_OVERDUE_COUNT", "_LATE_COUNT", "_UNDERPAYMENT_COUNT"
    ]
    for col in df.columns:
        if col in ["SK_ID_CURR", "TARGET", "IS_TRAIN"]:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if any(token in col for token in count_like_tokens):
            df[col] = df[col].fillna(0)
    return df


def format_value(value):
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{value:,.4f}"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return str(value)


def dataframe_to_html_table(df, max_rows=None):
    if df is None or df.empty:
        return "<p>No rows available.</p>"
    if max_rows is not None:
        df = df.head(max_rows)
    return df.to_html(index=False, escape=True, border=0)


def image_to_base64_html(image_path):
    if image_path is None or not image_path.exists():
        return ""
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f'<img src="data:image/png;base64,{encoded}" />'


def make_feature_source_table(feature_groups):
    rows = []
    for source, cols in feature_groups.items():
        rows.append({
            "source": source,
            "n_features": len(cols),
            "sample_features": ", ".join(cols[:12])
        })
    return pd.DataFrame(rows).sort_values("n_features", ascending=False)


def report_table_path(name):
    return TABLE_PATH / f"{name}.csv"


# ============================================================
# 1. APPLICATION BASE CLEANING
# ============================================================

print("Loading application_train/test...")
train = pd.read_csv(DATA_PATH / "application_train.csv")
test = pd.read_csv(DATA_PATH / "application_test.csv")

train["IS_TRAIN"] = 1
test["IS_TRAIN"] = 0
test["TARGET"] = np.nan

app = pd.concat([train, test], axis=0, ignore_index=True, sort=False)
del train, test
gc.collect()

initial_app_rows = len(app)
initial_app_unique = app["SK_ID_CURR"].nunique()


def clean_application_base(df):
    df = df.copy()

    df["CODE_GENDER_UNKNOWN_FLAG"] = (df["CODE_GENDER"] == "XNA").astype(int)
    df["CODE_GENDER"] = df["CODE_GENDER"].replace("XNA", np.nan)

    df["DAYS_EMPLOYED_ANOM_FLAG"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    property_cols = [
        "APARTMENTS_AVG", "BASEMENTAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG",
        "YEARS_BUILD_AVG", "COMMONAREA_AVG", "ELEVATORS_AVG", "ENTRANCES_AVG",
        "FLOORSMAX_AVG", "FLOORSMIN_AVG", "LANDAREA_AVG", "LIVINGAPARTMENTS_AVG",
        "LIVINGAREA_AVG", "NONLIVINGAPARTMENTS_AVG", "NONLIVINGAREA_AVG",
        "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "WALLSMATERIAL_MODE",
        "EMERGENCYSTATE_MODE"
    ]
    property_cols = [c for c in property_cols if c in df.columns]
    df["APP_PROPERTY_MISSING_COUNT"] = df[property_cols].isna().sum(axis=1)
    df["APP_PROPERTY_MISSING_RATIO"] = df["APP_PROPERTY_MISSING_COUNT"] / len(property_cols)
    df["APP_PROPERTY_ANY_MISSING_FLAG"] = (df["APP_PROPERTY_MISSING_COUNT"] > 0).astype(int)

    df["APP_OWN_CAR_AGE_MISSING_FLAG"] = df["OWN_CAR_AGE"].isna().astype(int)

    df["APP_AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["APP_EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25
    df["APP_REGISTRATION_YEARS"] = -df["DAYS_REGISTRATION"] / 365.25
    df["APP_ID_PUBLISH_YEARS"] = -df["DAYS_ID_PUBLISH"] / 365.25
    if "DAYS_LAST_PHONE_CHANGE" in df.columns:
        df["APP_LAST_PHONE_CHANGE_YEARS"] = -df["DAYS_LAST_PHONE_CHANGE"] / 365.25

    df["APP_CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["APP_ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["APP_GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"].replace(0, np.nan)
    df["APP_INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"].replace(0, np.nan))
    df["APP_CHILDREN_RATIO"] = df["CNT_CHILDREN"] / (df["CNT_FAM_MEMBERS"].replace(0, np.nan))

    return df


app = clean_application_base(app)

feature_groups = {
    "application_cleaned": [
        "CODE_GENDER_UNKNOWN_FLAG", "DAYS_EMPLOYED_ANOM_FLAG",
        "APP_PROPERTY_MISSING_COUNT", "APP_PROPERTY_MISSING_RATIO",
        "APP_PROPERTY_ANY_MISSING_FLAG", "APP_OWN_CAR_AGE_MISSING_FLAG",
        "APP_AGE_YEARS", "APP_EMPLOYED_YEARS", "APP_REGISTRATION_YEARS",
        "APP_ID_PUBLISH_YEARS", "APP_CREDIT_INCOME_RATIO",
        "APP_ANNUITY_INCOME_RATIO", "APP_GOODS_CREDIT_RATIO",
        "APP_INCOME_PER_PERSON", "APP_CHILDREN_RATIO"
    ]
}

aggregation_log = []

aggregation_log.append({
    "source": "application_train/test",
    "input_rows": initial_app_rows,
    "output_rows": len(app),
    "output_unique_sk_id_curr": app["SK_ID_CURR"].nunique(),
    "notes": "Application base cleaned and ratio features created"
})


# ============================================================
# 2. BUREAU AGGREGATION
# ============================================================

print("Aggregating bureau.csv...")
bureau_cols = [
    "SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "CREDIT_DAY_OVERDUE",
    "DAYS_CREDIT", "DAYS_CREDIT_ENDDATE", "AMT_CREDIT_SUM",
    "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE", "CNT_CREDIT_PROLONG"
]
bureau = pd.read_csv(DATA_PATH / "bureau.csv", usecols=bureau_cols)

bureau["BUREAU_IS_ACTIVE"] = (bureau["CREDIT_ACTIVE"] == "Active").astype(int)
bureau["BUREAU_IS_CLOSED"] = (bureau["CREDIT_ACTIVE"] == "Closed").astype(int)
bureau["BUREAU_IS_BAD_DEBT"] = (bureau["CREDIT_ACTIVE"] == "Bad debt").astype(int)
bureau["BUREAU_HAS_OVERDUE"] = (
    (bureau["CREDIT_DAY_OVERDUE"].fillna(0) > 0) |
    (bureau["AMT_CREDIT_SUM_OVERDUE"].fillna(0) > 0)
).astype(int)
bureau["BUREAU_ENDDATE_FUTURE_FLAG"] = (bureau["DAYS_CREDIT_ENDDATE"] > 0).astype(int)
bureau["BUREAU_DEBT_CREDIT_RATIO_RAW"] = bureau["AMT_CREDIT_SUM_DEBT"] / bureau["AMT_CREDIT_SUM"].replace(0, np.nan)

bureau_map = bureau[["SK_ID_BUREAU", "SK_ID_CURR"]].copy()

bureau_agg = bureau.groupby("SK_ID_CURR").agg(
    BUREAU_LOAN_COUNT=("SK_ID_BUREAU", "count"),
    BUREAU_ACTIVE_COUNT=("BUREAU_IS_ACTIVE", "sum"),
    BUREAU_CLOSED_COUNT=("BUREAU_IS_CLOSED", "sum"),
    BUREAU_BAD_DEBT_COUNT=("BUREAU_IS_BAD_DEBT", "sum"),
    BUREAU_OVERDUE_LOAN_COUNT=("BUREAU_HAS_OVERDUE", "sum"),
    BUREAU_CREDIT_DAY_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
    BUREAU_CREDIT_DAY_OVERDUE_MEAN=("CREDIT_DAY_OVERDUE", "mean"),
    BUREAU_DAYS_CREDIT_RECENT=("DAYS_CREDIT", "max"),
    BUREAU_DAYS_CREDIT_OLDEST=("DAYS_CREDIT", "min"),
    BUREAU_ENDDATE_FUTURE_COUNT=("BUREAU_ENDDATE_FUTURE_FLAG", "sum"),
    BUREAU_AMT_CREDIT_SUM_SUM=("AMT_CREDIT_SUM", "sum"),
    BUREAU_AMT_CREDIT_SUM_MEAN=("AMT_CREDIT_SUM", "mean"),
    BUREAU_AMT_CREDIT_SUM_MAX=("AMT_CREDIT_SUM", "max"),
    BUREAU_AMT_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
    BUREAU_AMT_DEBT_MEAN=("AMT_CREDIT_SUM_DEBT", "mean"),
    BUREAU_AMT_OVERDUE_SUM=("AMT_CREDIT_SUM_OVERDUE", "sum"),
    BUREAU_AMT_OVERDUE_MAX=("AMT_CREDIT_SUM_OVERDUE", "max"),
    BUREAU_CREDIT_PROLONG_SUM=("CNT_CREDIT_PROLONG", "sum"),
    BUREAU_DEBT_CREDIT_RATIO_MEAN=("BUREAU_DEBT_CREDIT_RATIO_RAW", "mean"),
    BUREAU_DEBT_CREDIT_RATIO_MAX=("BUREAU_DEBT_CREDIT_RATIO_RAW", "max"),
).reset_index()

bureau_agg["BUREAU_ACTIVE_RATE"] = safe_divide(bureau_agg["BUREAU_ACTIVE_COUNT"], bureau_agg["BUREAU_LOAN_COUNT"])
bureau_agg["BUREAU_OVERDUE_RATE"] = safe_divide(bureau_agg["BUREAU_OVERDUE_LOAN_COUNT"], bureau_agg["BUREAU_LOAN_COUNT"])
bureau_agg["BUREAU_DEBT_TO_CREDIT_SUM_RATIO"] = bureau_agg["BUREAU_AMT_DEBT_SUM"] / bureau_agg["BUREAU_AMT_CREDIT_SUM_SUM"].replace(0, np.nan)

feature_groups["bureau"] = [c for c in bureau_agg.columns if c != "SK_ID_CURR"]
aggregation_log.append({
    "source": "bureau",
    "input_rows": len(bureau),
    "output_rows": len(bureau_agg),
    "output_unique_sk_id_curr": bureau_agg["SK_ID_CURR"].nunique(),
    "notes": "Bureau loans aggregated to SK_ID_CURR"
})

del bureau
gc.collect()


# ============================================================
# 3. BUREAU_BALANCE AGGREGATION
# ============================================================

print("Aggregating bureau_balance.csv in chunks...")
status_map = {"C": 0, "X": 0, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
bb_partials = []
bb_input_rows = 0

for chunk in pd.read_csv(DATA_PATH / "bureau_balance.csv", chunksize=2_000_000):
    bb_input_rows += len(chunk)
    chunk = chunk.merge(bureau_map, on="SK_ID_BUREAU", how="inner")
    chunk["BB_STATUS_NUM"] = chunk["STATUS"].astype(str).map(status_map).fillna(0).astype(int)
    chunk["BB_OVERDUE_FLAG"] = chunk["BB_STATUS_NUM"].between(1, 5).astype(int)
    chunk["BB_BAD_STATUS_FLAG"] = (chunk["BB_STATUS_NUM"] == 5).astype(int)

    part = chunk.groupby("SK_ID_CURR").agg(
        BB_MONTH_COUNT=("MONTHS_BALANCE", "count"),
        BB_MONTHS_BALANCE_MIN=("MONTHS_BALANCE", "min"),
        BB_MONTHS_BALANCE_MAX=("MONTHS_BALANCE", "max"),
        BB_OVERDUE_MONTH_COUNT=("BB_OVERDUE_FLAG", "sum"),
        BB_BAD_STATUS_MONTH_COUNT=("BB_BAD_STATUS_FLAG", "sum"),
        BB_MAX_STATUS=("BB_STATUS_NUM", "max"),
    ).reset_index()
    bb_partials.append(part)

bb_all = pd.concat(bb_partials, axis=0, ignore_index=True)
bb_agg = bb_all.groupby("SK_ID_CURR").agg(
    BB_MONTH_COUNT=("BB_MONTH_COUNT", "sum"),
    BB_MONTHS_BALANCE_MIN=("BB_MONTHS_BALANCE_MIN", "min"),
    BB_MONTHS_BALANCE_MAX=("BB_MONTHS_BALANCE_MAX", "max"),
    BB_OVERDUE_MONTH_COUNT=("BB_OVERDUE_MONTH_COUNT", "sum"),
    BB_BAD_STATUS_MONTH_COUNT=("BB_BAD_STATUS_MONTH_COUNT", "sum"),
    BB_MAX_STATUS=("BB_MAX_STATUS", "max"),
).reset_index()

bb_agg["BB_OVERDUE_MONTH_RATE"] = safe_divide(bb_agg["BB_OVERDUE_MONTH_COUNT"], bb_agg["BB_MONTH_COUNT"])
bb_agg["BB_BAD_STATUS_MONTH_RATE"] = safe_divide(bb_agg["BB_BAD_STATUS_MONTH_COUNT"], bb_agg["BB_MONTH_COUNT"])

feature_groups["bureau_balance"] = [c for c in bb_agg.columns if c != "SK_ID_CURR"]
aggregation_log.append({
    "source": "bureau_balance",
    "input_rows": bb_input_rows,
    "output_rows": len(bb_agg),
    "output_unique_sk_id_curr": bb_agg["SK_ID_CURR"].nunique(),
    "notes": "Bureau monthly balance joined to bureau map and aggregated to SK_ID_CURR"
})

del bb_partials, bb_all
gc.collect()


# ============================================================
# 4. PREVIOUS_APPLICATION AGGREGATION
# ============================================================

print("Aggregating previous_application.csv...")
prev = pd.read_csv(DATA_PATH / "previous_application.csv")
prev_input_rows = len(prev)

prev_date_cols = [
    "DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE", "DAYS_TERMINATION"
]
for col in prev_date_cols:
    if col in prev.columns:
        prev[f"PREV_{col}_365243_FLAG"] = (prev[col] == 365243).astype(int)
        prev[col] = prev[col].replace(365243, np.nan)

prev["PREV_APPROVED_FLAG"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
prev["PREV_REFUSED_FLAG"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
prev["PREV_CANCELED_FLAG"] = (prev["NAME_CONTRACT_STATUS"] == "Canceled").astype(int)
prev["PREV_UNUSED_OFFER_FLAG"] = (prev["NAME_CONTRACT_STATUS"] == "Unused offer").astype(int)
prev["PREV_ZERO_APPLICATION_FLAG"] = (prev["AMT_APPLICATION"].fillna(0) == 0).astype(int)
prev["PREV_ZERO_CREDIT_FLAG"] = (prev["AMT_CREDIT"].fillna(0) == 0).astype(int)
prev["PREV_CREDIT_APPLICATION_RATIO"] = prev["AMT_CREDIT"] / prev["AMT_APPLICATION"].replace(0, np.nan)

prev_named_aggs = dict(
    PREV_APP_COUNT=("SK_ID_PREV", "count"),
    PREV_APPROVED_COUNT=("PREV_APPROVED_FLAG", "sum"),
    PREV_REFUSED_COUNT=("PREV_REFUSED_FLAG", "sum"),
    PREV_CANCELED_COUNT=("PREV_CANCELED_FLAG", "sum"),
    PREV_UNUSED_OFFER_COUNT=("PREV_UNUSED_OFFER_FLAG", "sum"),
    PREV_ZERO_APPLICATION_COUNT=("PREV_ZERO_APPLICATION_FLAG", "sum"),
    PREV_ZERO_CREDIT_COUNT=("PREV_ZERO_CREDIT_FLAG", "sum"),
    PREV_AMT_APPLICATION_SUM=("AMT_APPLICATION", "sum"),
    PREV_AMT_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
    PREV_AMT_APPLICATION_MAX=("AMT_APPLICATION", "max"),
    PREV_AMT_CREDIT_SUM=("AMT_CREDIT", "sum"),
    PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
    PREV_AMT_CREDIT_MAX=("AMT_CREDIT", "max"),
    PREV_AMT_ANNUITY_MEAN=("AMT_ANNUITY", "mean"),
    PREV_DAYS_DECISION_RECENT=("DAYS_DECISION", "max"),
    PREV_DAYS_DECISION_OLDEST=("DAYS_DECISION", "min"),
    PREV_DAYS_DECISION_MEAN=("DAYS_DECISION", "mean"),
    PREV_CREDIT_APPLICATION_RATIO_MEAN=("PREV_CREDIT_APPLICATION_RATIO", "mean"),
)
for col in prev_date_cols:
    flag_col = f"PREV_{col}_365243_FLAG"
    if flag_col in prev.columns:
        prev_named_aggs[f"{flag_col}_COUNT"] = (flag_col, "sum")

prev_agg = prev.groupby("SK_ID_CURR").agg(**prev_named_aggs).reset_index()
prev_agg["PREV_APPROVAL_RATE"] = safe_divide(prev_agg["PREV_APPROVED_COUNT"], prev_agg["PREV_APP_COUNT"])
prev_agg["PREV_REFUSAL_RATE"] = safe_divide(prev_agg["PREV_REFUSED_COUNT"], prev_agg["PREV_APP_COUNT"])
prev_agg["PREV_CANCELED_RATE"] = safe_divide(prev_agg["PREV_CANCELED_COUNT"], prev_agg["PREV_APP_COUNT"])

feature_groups["previous_application"] = [c for c in prev_agg.columns if c != "SK_ID_CURR"]
aggregation_log.append({
    "source": "previous_application",
    "input_rows": prev_input_rows,
    "output_rows": len(prev_agg),
    "output_unique_sk_id_curr": prev_agg["SK_ID_CURR"].nunique(),
    "notes": "Previous applications cleaned for 365243 date codes and aggregated to SK_ID_CURR"
})

del prev
gc.collect()


# ============================================================
# 5. POS_CASH_BALANCE AGGREGATION
# ============================================================

print("Aggregating POS_CASH_balance.csv...")
pos_cols = [
    "SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE", "CNT_INSTALMENT",
    "CNT_INSTALMENT_FUTURE", "NAME_CONTRACT_STATUS", "SK_DPD", "SK_DPD_DEF"
]
pos = pd.read_csv(DATA_PATH / "POS_CASH_balance.csv", usecols=pos_cols)

pos["POS_ACTIVE_FLAG"] = (pos["NAME_CONTRACT_STATUS"] == "Active").astype(int)
pos["POS_COMPLETED_FLAG"] = (pos["NAME_CONTRACT_STATUS"] == "Completed").astype(int)
pos["POS_DEMAND_FLAG"] = (pos["NAME_CONTRACT_STATUS"] == "Demand").astype(int)
pos["POS_DPD_FLAG"] = (pos["SK_DPD"].fillna(0) > 0).astype(int)
pos["POS_DPD_DEF_FLAG"] = (pos["SK_DPD_DEF"].fillna(0) > 0).astype(int)
pos["POS_FUTURE_GT_TOTAL_FLAG"] = (pos["CNT_INSTALMENT_FUTURE"] > pos["CNT_INSTALMENT"]).astype(int)

pos_duplicate_count = int(pos.duplicated(["SK_ID_PREV", "MONTHS_BALANCE"]).sum())

pos_agg = pos.groupby("SK_ID_CURR").agg(
    POS_MONTH_COUNT=("MONTHS_BALANCE", "count"),
    POS_PREV_COUNT=("SK_ID_PREV", "nunique"),
    POS_MONTHS_BALANCE_RECENT=("MONTHS_BALANCE", "max"),
    POS_MONTHS_BALANCE_OLDEST=("MONTHS_BALANCE", "min"),
    POS_ACTIVE_MONTH_COUNT=("POS_ACTIVE_FLAG", "sum"),
    POS_COMPLETED_MONTH_COUNT=("POS_COMPLETED_FLAG", "sum"),
    POS_DEMAND_MONTH_COUNT=("POS_DEMAND_FLAG", "sum"),
    POS_DPD_MONTH_COUNT=("POS_DPD_FLAG", "sum"),
    POS_DPD_DEF_MONTH_COUNT=("POS_DPD_DEF_FLAG", "sum"),
    POS_SK_DPD_MAX=("SK_DPD", "max"),
    POS_SK_DPD_MEAN=("SK_DPD", "mean"),
    POS_SK_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
    POS_SK_DPD_DEF_MEAN=("SK_DPD_DEF", "mean"),
    POS_CNT_INSTALMENT_MEAN=("CNT_INSTALMENT", "mean"),
    POS_CNT_INSTALMENT_MAX=("CNT_INSTALMENT", "max"),
    POS_CNT_INSTALMENT_FUTURE_MEAN=("CNT_INSTALMENT_FUTURE", "mean"),
    POS_CNT_INSTALMENT_FUTURE_MIN=("CNT_INSTALMENT_FUTURE", "min"),
    POS_FUTURE_GT_TOTAL_COUNT=("POS_FUTURE_GT_TOTAL_FLAG", "sum"),
).reset_index()

pos_agg["POS_DPD_MONTH_RATE"] = safe_divide(pos_agg["POS_DPD_MONTH_COUNT"], pos_agg["POS_MONTH_COUNT"])
pos_agg["POS_DPD_DEF_MONTH_RATE"] = safe_divide(pos_agg["POS_DPD_DEF_MONTH_COUNT"], pos_agg["POS_MONTH_COUNT"])

feature_groups["POS_CASH_balance"] = [c for c in pos_agg.columns if c != "SK_ID_CURR"]
aggregation_log.append({
    "source": "POS_CASH_balance",
    "input_rows": len(pos),
    "output_rows": len(pos_agg),
    "output_unique_sk_id_curr": pos_agg["SK_ID_CURR"].nunique(),
    "notes": f"POS/CASH monthly records aggregated; composite duplicate SK_ID_PREV+MONTHS_BALANCE count={pos_duplicate_count}"
})

del pos
gc.collect()


# ============================================================
# 6. INSTALLMENTS_PAYMENTS AGGREGATION
# ============================================================

print("Aggregating installments_payments.csv...")
inst_cols = [
    "SK_ID_PREV", "SK_ID_CURR", "NUM_INSTALMENT_NUMBER", "DAYS_INSTALMENT",
    "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"
]
inst = pd.read_csv(DATA_PATH / "installments_payments.csv", usecols=inst_cols)

inst["INST_DAYS_LATE"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
inst["INST_DAYS_LATE_POSITIVE"] = inst["INST_DAYS_LATE"].clip(lower=0)
inst["INST_DAYS_EARLY_POSITIVE"] = (-inst["INST_DAYS_LATE"]).clip(lower=0)
inst["INST_LATE_FLAG"] = (inst["INST_DAYS_LATE"] > 0).astype(int)
inst["INST_UNDERPAYMENT_FLAG"] = (inst["AMT_PAYMENT"] < inst["AMT_INSTALMENT"]).astype(int)
inst["INST_PAYMENT_MISSING_FLAG"] = inst["DAYS_ENTRY_PAYMENT"].isna().astype(int)
inst["INST_PAYMENT_RATIO"] = inst["AMT_PAYMENT"] / inst["AMT_INSTALMENT"].replace(0, np.nan)

inst_duplicate_count = int(inst.duplicated(["SK_ID_PREV", "NUM_INSTALMENT_NUMBER"]).sum())

inst_agg = inst.groupby("SK_ID_CURR").agg(
    INST_PAYMENT_COUNT=("NUM_INSTALMENT_NUMBER", "count"),
    INST_PREV_COUNT=("SK_ID_PREV", "nunique"),
    INST_LATE_PAYMENT_COUNT=("INST_LATE_FLAG", "sum"),
    INST_UNDERPAYMENT_COUNT=("INST_UNDERPAYMENT_FLAG", "sum"),
    INST_PAYMENT_MISSING_COUNT=("INST_PAYMENT_MISSING_FLAG", "sum"),
    INST_DAYS_LATE_POSITIVE_SUM=("INST_DAYS_LATE_POSITIVE", "sum"),
    INST_DAYS_LATE_POSITIVE_MEAN=("INST_DAYS_LATE_POSITIVE", "mean"),
    INST_DAYS_LATE_POSITIVE_MAX=("INST_DAYS_LATE_POSITIVE", "max"),
    INST_DAYS_EARLY_POSITIVE_MEAN=("INST_DAYS_EARLY_POSITIVE", "mean"),
    INST_AMT_INSTALMENT_SUM=("AMT_INSTALMENT", "sum"),
    INST_AMT_INSTALMENT_MEAN=("AMT_INSTALMENT", "mean"),
    INST_AMT_PAYMENT_SUM=("AMT_PAYMENT", "sum"),
    INST_AMT_PAYMENT_MEAN=("AMT_PAYMENT", "mean"),
    INST_PAYMENT_RATIO_MEAN=("INST_PAYMENT_RATIO", "mean"),
    INST_PAYMENT_RATIO_MIN=("INST_PAYMENT_RATIO", "min"),
).reset_index()

inst_agg["INST_LATE_PAYMENT_RATE"] = safe_divide(inst_agg["INST_LATE_PAYMENT_COUNT"], inst_agg["INST_PAYMENT_COUNT"])
inst_agg["INST_UNDERPAYMENT_RATE"] = safe_divide(inst_agg["INST_UNDERPAYMENT_COUNT"], inst_agg["INST_PAYMENT_COUNT"])
inst_agg["INST_TOTAL_PAYMENT_RATIO"] = inst_agg["INST_AMT_PAYMENT_SUM"] / inst_agg["INST_AMT_INSTALMENT_SUM"].replace(0, np.nan)

feature_groups["installments_payments"] = [c for c in inst_agg.columns if c != "SK_ID_CURR"]
aggregation_log.append({
    "source": "installments_payments",
    "input_rows": len(inst),
    "output_rows": len(inst_agg),
    "output_unique_sk_id_curr": inst_agg["SK_ID_CURR"].nunique(),
    "notes": f"Installment payment behavior aggregated; composite duplicate SK_ID_PREV+NUM_INSTALMENT_NUMBER count={inst_duplicate_count}"
})

del inst
gc.collect()


# ============================================================
# 7. CREDIT_CARD_BALANCE AGGREGATION
# ============================================================

print("Aggregating credit_card_balance.csv...")
cc_cols = [
    "SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE", "AMT_BALANCE",
    "AMT_CREDIT_LIMIT_ACTUAL", "AMT_DRAWINGS_CURRENT",
    "AMT_PAYMENT_CURRENT", "AMT_TOTAL_RECEIVABLE",
    "NAME_CONTRACT_STATUS", "SK_DPD", "SK_DPD_DEF"
]
cc = pd.read_csv(DATA_PATH / "credit_card_balance.csv", usecols=cc_cols)

cc["CC_ZERO_LIMIT_FLAG"] = (cc["AMT_CREDIT_LIMIT_ACTUAL"].fillna(0) == 0).astype(int)
cc["CC_UTILIZATION"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
cc["CC_UTILIZATION_GT_1_FLAG"] = (cc["CC_UTILIZATION"] > 1).astype(int)
cc["CC_ACTIVE_FLAG"] = (cc["NAME_CONTRACT_STATUS"] == "Active").astype(int)
cc["CC_COMPLETED_FLAG"] = (cc["NAME_CONTRACT_STATUS"] == "Completed").astype(int)
cc["CC_DPD_FLAG"] = (cc["SK_DPD"].fillna(0) > 0).astype(int)
cc["CC_DPD_DEF_FLAG"] = (cc["SK_DPD_DEF"].fillna(0) > 0).astype(int)
cc["CC_NEGATIVE_BALANCE_FLAG"] = (cc["AMT_BALANCE"] < 0).astype(int)

cc_duplicate_count = int(cc.duplicated(["SK_ID_PREV", "MONTHS_BALANCE"]).sum())

cc_agg = cc.groupby("SK_ID_CURR").agg(
    CC_MONTH_COUNT=("MONTHS_BALANCE", "count"),
    CC_PREV_COUNT=("SK_ID_PREV", "nunique"),
    CC_MONTHS_BALANCE_RECENT=("MONTHS_BALANCE", "max"),
    CC_MONTHS_BALANCE_OLDEST=("MONTHS_BALANCE", "min"),
    CC_ACTIVE_MONTH_COUNT=("CC_ACTIVE_FLAG", "sum"),
    CC_COMPLETED_MONTH_COUNT=("CC_COMPLETED_FLAG", "sum"),
    CC_ZERO_LIMIT_MONTH_COUNT=("CC_ZERO_LIMIT_FLAG", "sum"),
    CC_UTILIZATION_GT_1_MONTH_COUNT=("CC_UTILIZATION_GT_1_FLAG", "sum"),
    CC_NEGATIVE_BALANCE_MONTH_COUNT=("CC_NEGATIVE_BALANCE_FLAG", "sum"),
    CC_AMT_BALANCE_MEAN=("AMT_BALANCE", "mean"),
    CC_AMT_BALANCE_MAX=("AMT_BALANCE", "max"),
    CC_AMT_BALANCE_SUM=("AMT_BALANCE", "sum"),
    CC_CREDIT_LIMIT_MEAN=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
    CC_CREDIT_LIMIT_MAX=("AMT_CREDIT_LIMIT_ACTUAL", "max"),
    CC_UTILIZATION_MEAN=("CC_UTILIZATION", "mean"),
    CC_UTILIZATION_MAX=("CC_UTILIZATION", "max"),
    CC_DRAWINGS_CURRENT_SUM=("AMT_DRAWINGS_CURRENT", "sum"),
    CC_DRAWINGS_CURRENT_MEAN=("AMT_DRAWINGS_CURRENT", "mean"),
    CC_PAYMENT_CURRENT_SUM=("AMT_PAYMENT_CURRENT", "sum"),
    CC_PAYMENT_CURRENT_MEAN=("AMT_PAYMENT_CURRENT", "mean"),
    CC_TOTAL_RECEIVABLE_MEAN=("AMT_TOTAL_RECEIVABLE", "mean"),
    CC_TOTAL_RECEIVABLE_MAX=("AMT_TOTAL_RECEIVABLE", "max"),
    CC_DPD_MONTH_COUNT=("CC_DPD_FLAG", "sum"),
    CC_DPD_DEF_MONTH_COUNT=("CC_DPD_DEF_FLAG", "sum"),
    CC_SK_DPD_MAX=("SK_DPD", "max"),
    CC_SK_DPD_MEAN=("SK_DPD", "mean"),
    CC_SK_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
    CC_SK_DPD_DEF_MEAN=("SK_DPD_DEF", "mean"),
).reset_index()

cc_agg["CC_ZERO_LIMIT_MONTH_RATE"] = safe_divide(cc_agg["CC_ZERO_LIMIT_MONTH_COUNT"], cc_agg["CC_MONTH_COUNT"])
cc_agg["CC_UTILIZATION_GT_1_MONTH_RATE"] = safe_divide(cc_agg["CC_UTILIZATION_GT_1_MONTH_COUNT"], cc_agg["CC_MONTH_COUNT"])
cc_agg["CC_DPD_MONTH_RATE"] = safe_divide(cc_agg["CC_DPD_MONTH_COUNT"], cc_agg["CC_MONTH_COUNT"])

feature_groups["credit_card_balance"] = [c for c in cc_agg.columns if c != "SK_ID_CURR"]
aggregation_log.append({
    "source": "credit_card_balance",
    "input_rows": len(cc),
    "output_rows": len(cc_agg),
    "output_unique_sk_id_curr": cc_agg["SK_ID_CURR"].nunique(),
    "notes": f"Credit card monthly behavior aggregated with safe utilization; composite duplicate SK_ID_PREV+MONTHS_BALANCE count={cc_duplicate_count}"
})

del cc
gc.collect()


# ============================================================
# 8. MERGE ALL AGGREGATED FEATURES
# ============================================================

print("Merging aggregated features into final table...")
final = app.copy()
del app
gc.collect()

agg_tables = [
    ("bureau", bureau_agg, "BUREAU_LOAN_COUNT", "HAS_BUREAU_HISTORY"),
    ("bureau_balance", bb_agg, "BB_MONTH_COUNT", "HAS_BUREAU_BALANCE_HISTORY"),
    ("previous_application", prev_agg, "PREV_APP_COUNT", "HAS_PREVIOUS_APPLICATION"),
    ("POS_CASH_balance", pos_agg, "POS_MONTH_COUNT", "HAS_POS_CASH_HISTORY"),
    ("installments_payments", inst_agg, "INST_PAYMENT_COUNT", "HAS_INSTALLMENT_HISTORY"),
    ("credit_card_balance", cc_agg, "CC_MONTH_COUNT", "HAS_CREDIT_CARD_HISTORY"),
]

merge_log = []
for source, agg_df, count_col, flag_col in agg_tables:
    before_rows = len(final)
    before_unique = final["SK_ID_CURR"].nunique()
    final = final.merge(agg_df, on="SK_ID_CURR", how="left")
    final = add_history_flag(final, count_col, flag_col)
    after_rows = len(final)
    after_unique = final["SK_ID_CURR"].nunique()
    merge_log.append({
        "source": source,
        "rows_before": before_rows,
        "rows_after": after_rows,
        "unique_before": before_unique,
        "unique_after": after_unique,
        "history_flag": flag_col,
        "customers_with_history": int(final[flag_col].sum()),
        "coverage_pct": final[flag_col].mean() * 100,
    })
    del agg_df
    gc.collect()

final = fill_count_like_columns(final)


# ============================================================
# 9. FINAL VALIDATION
# ============================================================

print("Validating final table...")
final_rows = len(final)
final_unique = final["SK_ID_CURR"].nunique()
train_rows = int((final["IS_TRAIN"] == 1).sum())
test_rows = int((final["IS_TRAIN"] == 0).sum())
target_non_missing_train = int(final.loc[final["IS_TRAIN"] == 1, "TARGET"].notna().sum())
target_non_missing_test = int(final.loc[final["IS_TRAIN"] == 0, "TARGET"].notna().sum())

validation_df = pd.DataFrame([
    {"check": "initial_application_rows", "value": initial_app_rows, "expected": initial_app_rows, "status": "pass"},
    {"check": "final_rows", "value": final_rows, "expected": initial_app_rows, "status": "pass" if final_rows == initial_app_rows else "fail"},
    {"check": "final_unique_SK_ID_CURR", "value": final_unique, "expected": initial_app_unique, "status": "pass" if final_unique == initial_app_unique else "fail"},
    {"check": "train_rows", "value": train_rows, "expected": 307511, "status": "pass" if train_rows == 307511 else "review"},
    {"check": "test_rows", "value": test_rows, "expected": 48744, "status": "pass" if test_rows == 48744 else "review"},
    {"check": "target_non_missing_train", "value": target_non_missing_train, "expected": 307511, "status": "pass" if target_non_missing_train == 307511 else "fail"},
    {"check": "target_non_missing_test", "value": target_non_missing_test, "expected": 0, "status": "pass" if target_non_missing_test == 0 else "fail"},
])

feature_source_df = make_feature_source_table(feature_groups)
aggregation_log_df = pd.DataFrame(aggregation_log)
merge_log_df = pd.DataFrame(merge_log)

missing_final = (
    final.isna().sum()
    .reset_index()
    .rename(columns={"index": "column", 0: "missing_count"})
)
missing_final["missing_pct"] = missing_final["missing_count"] / len(final) * 100
missing_final = missing_final.sort_values("missing_pct", ascending=False)

history_flags = [
    "HAS_BUREAU_HISTORY", "HAS_BUREAU_BALANCE_HISTORY", "HAS_PREVIOUS_APPLICATION",
    "HAS_POS_CASH_HISTORY", "HAS_INSTALLMENT_HISTORY", "HAS_CREDIT_CARD_HISTORY"
]
history_coverage_df = (
    final[history_flags]
    .mean()
    .mul(100)
    .reset_index()
    .rename(columns={"index": "history_source", 0: "coverage_pct"})
    .sort_values("coverage_pct", ascending=False)
)
history_coverage_df["customers_with_history"] = history_coverage_df["history_source"].map(
    {col: int(final[col].sum()) for col in history_flags}
)

source_feature_counts = feature_source_df.copy()


# ============================================================
# 10. SAVE DATA AND REPORT TABLES
# ============================================================

print("Saving final datasets...")
final_table_path = DATA_OUT_PATH / "final_customer_analysis_table.csv.gz"
final_train_path = DATA_OUT_PATH / "final_customer_analysis_train.csv.gz"
final_test_path = DATA_OUT_PATH / "final_customer_analysis_test.csv.gz"

final.to_csv(final_table_path, index=False, compression="gzip")
final[final["IS_TRAIN"] == 1].to_csv(final_train_path, index=False, compression="gzip")
final[final["IS_TRAIN"] == 0].to_csv(final_test_path, index=False, compression="gzip")

save_table(validation_df, "01_final_table_validation")
save_table(aggregation_log_df, "02_aggregation_log")
save_table(merge_log_df, "03_merge_log")
save_table(feature_source_df, "04_feature_source_summary")
save_table(history_coverage_df, "05_history_coverage_final")
save_table(missing_final.head(50), "06_final_missing_top50")
save_table(final.head(50), "07_final_table_sample")

output_paths_df = pd.DataFrame([
    {"name": "final_customer_analysis_table", "path": str(final_table_path), "rows": len(final), "columns": final.shape[1]},
    {"name": "final_customer_analysis_train", "path": str(final_train_path), "rows": train_rows, "columns": final.shape[1]},
    {"name": "final_customer_analysis_test", "path": str(final_test_path), "rows": test_rows, "columns": final.shape[1]},
])
save_table(output_paths_df, "08_output_dataset_paths")


# ============================================================
# 11. FIGURES
# ============================================================

plt.figure(figsize=(10, 5))
sns.barplot(data=history_coverage_df, y="history_source", x="coverage_pct")
plt.xlim(0, 100)
plt.xlabel("Coverage (%)")
plt.ylabel("History source")
plt.title("Historical Data Coverage in Final Customer Table")
for i, row in history_coverage_df.reset_index(drop=True).iterrows():
    plt.text(row["coverage_pct"] + 1, i, f'{row["coverage_pct"]:.1f}%', va="center")
save_fig("01_history_coverage_final")

plt.figure(figsize=(10, 5))
sns.barplot(data=source_feature_counts.sort_values("n_features", ascending=True), y="source", x="n_features")
plt.xlabel("Number of engineered features")
plt.ylabel("Source")
plt.title("Engineered Feature Count by Source")
save_fig("02_feature_count_by_source")

plt.figure(figsize=(10, 8))
plot_df = missing_final.head(30).sort_values("missing_pct", ascending=True)
sns.barplot(data=plot_df, y="column", x="missing_pct")
plt.xlabel("Missing percentage (%)")
plt.ylabel("Column")
plt.title("Top 30 Missing Columns After Final Merge")
save_fig("03_final_missing_top30")

plt.figure(figsize=(9, 5))
target_dist = (
    final.loc[final["IS_TRAIN"] == 1, "TARGET"]
    .value_counts(normalize=True)
    .mul(100)
    .reset_index()
)
target_dist.columns = ["TARGET", "pct"]
sns.barplot(data=target_dist, x="TARGET", y="pct")
plt.ylabel("Percentage (%)")
plt.title("TARGET Distribution Preserved in Final Train Table")
save_fig("04_final_target_distribution")


# ============================================================
# 12. HTML REPORT
# ============================================================

def build_html_report():
    html_path = OUT_PATH / "step05_aggregation_feature_engineering_report.html"
    sections = [
        {
            "title": "1. Final Table Validation",
            "why": "Kiểm tra bảng cuối có giữ đúng grain một dòng = một SK_ID_CURR, không mất dòng và không nhân dòng.",
            "table": "01_final_table_validation.csv",
            "figure": None,
            "max_rows": None,
        },
        {
            "title": "2. Aggregation Log",
            "why": "Tóm tắt mỗi bảng raw đã được aggregate từ bao nhiêu dòng về bao nhiêu khách hàng.",
            "table": "02_aggregation_log.csv",
            "figure": None,
            "max_rows": None,
        },
        {
            "title": "3. Merge Log",
            "why": "Kiểm tra sau mỗi lần merge số dòng và số khách hàng unique có được giữ nguyên không.",
            "table": "03_merge_log.csv",
            "figure": None,
            "max_rows": None,
        },
        {
            "title": "4. Feature Source Summary",
            "why": "Cho biết mỗi nguồn dữ liệu đóng góp bao nhiêu feature vào bảng final.",
            "table": "04_feature_source_summary.csv",
            "figure": "02_feature_count_by_source.png",
            "max_rows": None,
        },
        {
            "title": "5. History Coverage",
            "why": "Cho biết tỷ lệ khách hàng có lịch sử ở từng nguồn dữ liệu.",
            "table": "05_history_coverage_final.csv",
            "figure": "01_history_coverage_final.png",
            "max_rows": None,
        },
        {
            "title": "6. Missing After Merge",
            "why": "Đánh giá missing sau khi merge để chuẩn bị cho EDA, dashboard và ML preprocessing.",
            "table": "06_final_missing_top50.csv",
            "figure": "03_final_missing_top30.png",
            "max_rows": 50,
        },
        {
            "title": "7. Final Table Sample",
            "why": "Kiểm tra một số dòng đầu của bảng final để đảm bảo feature đã được merge đúng.",
            "table": "07_final_table_sample.csv",
            "figure": None,
            "max_rows": 10,
        },
        {
            "title": "8. Output Dataset Paths",
            "why": "Danh sách file dữ liệu cuối cùng để dùng ở Step 5, dashboard và ML support.",
            "table": "08_output_dataset_paths.csv",
            "figure": None,
            "max_rows": None,
        },
    ]

    styles = """
    <style>
    body { font-family: Arial, sans-serif; margin: 32px; color: #1f2937; }
    h1 { color: #111827; border-bottom: 2px solid #e5e7eb; padding-bottom: 6px; }
    h2 { color: #374151; margin-top: 28px; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0 24px 0; font-size: 12px; }
    th, td { border: 1px solid #d1d5db; padding: 6px; text-align: left; vertical-align: top; }
    th { background: #f3f4f6; }
    img { max-width: 100%; margin: 12px 0 24px 0; border: 1px solid #e5e7eb; }
    .note { background: #f9fafb; border-left: 4px solid #60a5fa; padding: 12px; }
    </style>
    """
    parts = [
        "<html><head><meta charset='utf-8'>",
        styles,
        "</head><body>",
        "<h1>Step 4 - Aggregation & Feature Engineering Report</h1>",
        "<p class='note'>Report này kiểm tra việc aggregate dữ liệu lịch sử về cấp SK_ID_CURR và tạo bảng final_customer_analysis_table.</p>",
        "<h2>Executive Summary</h2>",
        "<ul>",
        "<li>Bảng cuối phải giữ grain một dòng = một SK_ID_CURR.</li>",
        "<li>Các bảng lịch sử được aggregate trước khi merge để tránh nhân dòng.</li>",
        "<li>Count-like features được fill 0 khi khách hàng không có lịch sử; mean/rate giữ NaN kèm history flag.</li>",
        "<li>Bảng final dùng cho Step 5 EDA, dashboard và ML support.</li>",
        "</ul>",
    ]

    for section in sections:
        parts.append(f"<h2>{section['title']}</h2>")
        parts.append(f"<p>{section['why']}</p>")
        table_path = TABLE_PATH / section["table"]
        if table_path.exists():
            table_df = pd.read_csv(table_path)
            parts.append(dataframe_to_html_table(table_df, max_rows=section["max_rows"]))
        if section["figure"]:
            parts.append(image_to_base64_html(FIG_PATH / section["figure"]))

    parts.append("<h2>Final Note</h2>")
    parts.append("<p>Step 4 tạo bảng dữ liệu phân tích cuối cùng. EDA và dashboard chi tiết sẽ được thực hiện ở Step 5.</p>")
    parts.append("</body></html>")
    html_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Saved HTML report: {html_path}")
    return html_path


report_path = build_html_report()


# ============================================================
# 13. ZIP OUTPUTS
# ============================================================

summary = {
    "step": "Step 4 - Aggregation & Feature Engineering",
    "output_path": str(OUT_PATH),
    "tables_path": str(TABLE_PATH),
    "figures_path": str(FIG_PATH),
    "data_path": str(DATA_OUT_PATH),
    "report_path": str(report_path),
    "final_table_path": str(final_table_path),
    "final_train_path": str(final_train_path),
    "final_test_path": str(final_test_path),
    "final_rows": int(final_rows),
    "final_columns": int(final.shape[1]),
    "n_tables_saved": len(list(TABLE_PATH.glob("*.csv"))),
    "n_figures_saved": len(list(FIG_PATH.glob("*.png"))),
}

with open(OUT_PATH / "step4_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

zip_path = OUT_PATH.parent / "step05_outputs.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for file in OUT_PATH.rglob("*"):
        z.write(file, file.relative_to(OUT_PATH.parent))

print("=" * 90)
print("STEP 4 FINISHED")
print(f"Final table: {final_table_path}")
print(f"Train table: {final_train_path}")
print(f"Test table: {final_test_path}")
print(f"Report: {report_path}")
print(f"Zip file: {zip_path}")
print("=" * 90)

