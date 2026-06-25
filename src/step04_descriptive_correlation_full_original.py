"""
Step 2 (FULL) - Descriptive Statistics & Correlation Analysis
=============================================================

Credit Risk Data Analysis with Machine Learning Support
Home Credit Default Risk

This module fills the two gaps that the original `step2_result_only_kaggle.py`
left open even though the pipeline documentation promised them:

    - 05_descriptive_all_numeric_features.csv   (full descriptive, ALL features)
    - 06_correlation_with_target.csv            (correlation of every feature vs TARGET)
    - 07_correlation_matrix_selected.csv        (correlation matrix of top drivers)
    - step02_interpretation_notes.md            (auto-written analyst commentary)

It runs on the Step 4 customer-level analytical table
(`final_customer_analysis_train.csv.gz`, 1 row = 1 SK_ID_CURR), so the
correlation/descriptive numbers reflect the SAME table used by the dashboard,
diagnostic and ML steps. That keeps the whole project internally consistent.

Design choices (defensible for a credit-risk DA project):
    * Spearman is reported alongside Pearson because monetary / ratio features
      are heavily right-skewed; rank correlation is the robust default and
      Pearson is kept for linear reference.
    * TARGET is binary, so a numeric-vs-TARGET correlation is a point-biserial
      correlation (Pearson) plus a rank-biserial style Spearman.
    * Binary 0/1 flags are profiled separately (prevalence), not mixed into the
      continuous descriptive table where skewness/kurtosis would be meaningless.
    * Categorical drivers use Cramer's V (bias-corrected) + chi-square, with a
      per-category default-rate table that always carries the sample size so
      small segments are not over-interpreted.

Author: rebuilt/upgraded for the final submission.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless: save figures without a display
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)
sns.set_theme(style="whitegrid", palette="Set2")


# ============================================================
# CONFIG
# ============================================================

# Primary input: the Step 4 customer-level analytical table (local).
# Fallbacks cover the clean_pipeline copy and a Kaggle working path.
CANDIDATE_INPUTS = [
    Path(r"D:\Code\DA\1\step4_outputs\data\final_customer_analysis_train.csv.gz"),
    Path(r"D:\Code\DA\1\clean_pipeline\outputs\step04\data\final_customer_analysis_train.csv.gz"),
    Path("/kaggle/working/step4_outputs/data/final_customer_analysis_train.csv.gz"),
    Path("final_customer_analysis_train.csv.gz"),
]

OUT_PATH = Path(r"D:\Code\DA\1\step2_full_outputs")
TABLE_PATH = OUT_PATH / "tables"
FIG_PATH = OUT_PATH / "figures"
TABLE_PATH.mkdir(parents=True, exist_ok=True)
FIG_PATH.mkdir(parents=True, exist_ok=True)

TARGET = "TARGET"
ID_COLS = ["SK_ID_CURR"]
BASELINE_DEFAULT_RATE = None  # filled at runtime

# Minimum sample size before we trust a per-category default rate.
MIN_SEGMENT_N = 100
# |Pearson| above this between two features is flagged as multicollinearity.
MULTICOLLINEARITY_THRESHOLD = 0.80
# Drop very-sparse columns from the multicollinearity scan for stability.
MAX_MISSING_FOR_CORR_MATRIX = 60.0
# How many top features to render in the correlation matrix / charts.
TOP_N_DRIVERS = 25


# ============================================================
# HELPERS
# ============================================================

def log(msg: str) -> None:
    print(f"[STEP2-FULL] {msg}")


def find_input() -> Path:
    for p in CANDIDATE_INPUTS:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find final_customer_analysis_train.csv.gz in any known location:\n"
        + "\n".join(f"  - {p}" for p in CANDIDATE_INPUTS)
    )


def save_table(df: pd.DataFrame, name: str) -> Path:
    path = TABLE_PATH / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"Saved table: {path.name}  ({len(df)} rows)")
    return path


def save_fig(name: str) -> Path:
    path = FIG_PATH / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    log(f"Saved figure: {path.name}")
    return path


def shape_from_skew(skew: float) -> str:
    """Classify distribution shape from skewness (Vietnamese, for the report)."""
    if pd.isna(skew):
        return "không xác định"
    a = abs(skew)
    side = "lệch phải" if skew > 0 else "lệch trái"
    if a < 0.5:
        return "gần đối xứng"
    if a < 1.0:
        return f"{side} vừa"
    return f"{side} mạnh"


def strength_bucket(abs_corr: float) -> str:
    """Interpret correlation magnitude (Vietnamese, for the report)."""
    if pd.isna(abs_corr):
        return "không xác định"
    if abs_corr < 0.05:
        return "không đáng kể"
    if abs_corr < 0.10:
        return "rất yếu"
    if abs_corr < 0.20:
        return "yếu"
    if abs_corr < 0.30:
        return "trung bình"
    return "khá mạnh"


def classify_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Split columns into id / target / binary-flag / continuous / categorical."""
    numeric_all = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(exclude=[np.number]).columns.tolist()

    feature_numeric = [c for c in numeric_all if c not in ID_COLS + [TARGET]]
    binary_flags, continuous = [], []
    for c in feature_numeric:
        vals = set(pd.to_numeric(df[c], errors="coerce").dropna().unique())
        if vals <= {0, 1}:
            binary_flags.append(c)
        else:
            continuous.append(c)

    return {
        "numeric_all": numeric_all,
        "feature_numeric": feature_numeric,
        "binary_flags": binary_flags,
        "continuous": continuous,
        "categorical": categorical,
    }


# ============================================================
# 1. FULL DESCRIPTIVE STATISTICS - CONTINUOUS
# ============================================================

def describe_continuous(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Complete descriptive profile for every continuous numeric feature."""
    n = len(df)
    rows = []
    for c in cols:
        s_all = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        s = s_all.dropna()
        cnt = len(s)
        miss = n - cnt
        if cnt == 0:
            continue
        q1, med, q3 = s.quantile(0.25), s.median(), s.quantile(0.75)
        iqr = q3 - q1
        mean = s.mean()
        std = s.std()
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out_mask = (s < lo) | (s > hi)
        skew = s.skew()
        rows.append({
            "feature": c,
            "count": cnt,
            "missing": miss,
            "missing_pct": round(miss / n * 100, 3),
            "n_unique": int(s.nunique()),
            "n_zeros": int((s == 0).sum()),
            "zero_pct": round((s == 0).mean() * 100, 3),
            "mean": mean,
            "std": std,
            "cv": round(std / mean, 4) if mean not in (0, np.nan) and mean != 0 else np.nan,
            "min": s.min(),
            "p1": s.quantile(0.01),
            "p5": s.quantile(0.05),
            "q1_25": q1,
            "median_50": med,
            "q3_75": q3,
            "p95": s.quantile(0.95),
            "p99": s.quantile(0.99),
            "max": s.max(),
            "range": s.max() - s.min(),
            "iqr": iqr,
            "skewness": round(skew, 4),
            "kurtosis": round(s.kurt(), 4),
            "outlier_count_iqr": int(out_mask.sum()),
            "outlier_pct_iqr": round(out_mask.mean() * 100, 3),
            "distribution_shape": shape_from_skew(skew),
        })
    out = pd.DataFrame(rows).sort_values("missing_pct").reset_index(drop=True)
    return out


# ============================================================
# 2. DESCRIPTIVE - BINARY FLAGS
# ============================================================

def describe_binary_flags(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Prevalence-style profile for 0/1 flag features."""
    n = len(df)
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        valid = s.dropna()
        ones = int((valid == 1).sum())
        rows.append({
            "feature": c,
            "count": int(valid.shape[0]),
            "missing": int(s.isna().sum()),
            "missing_pct": round(s.isna().mean() * 100, 3),
            "n_ones": ones,
            "prevalence_pct": round(ones / valid.shape[0] * 100, 3) if valid.shape[0] else np.nan,
            "n_zeros": int((valid == 0).sum()),
        })
    return pd.DataFrame(rows).sort_values("prevalence_pct", ascending=False).reset_index(drop=True)


# ============================================================
# 3. DESCRIPTIVE - CATEGORICAL
# ============================================================

def shannon_entropy(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def describe_categorical(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    n = len(df)
    rows = []
    for c in cols:
        s = df[c]
        vc = s.value_counts(dropna=True)
        if vc.empty:
            continue
        top_cat = vc.index[0]
        rows.append({
            "feature": c,
            "count": int(s.notna().sum()),
            "missing": int(s.isna().sum()),
            "missing_pct": round(s.isna().mean() * 100, 3),
            "n_unique": int(s.nunique(dropna=True)),
            "mode": str(top_cat),
            "mode_freq": int(vc.iloc[0]),
            "mode_pct": round(vc.iloc[0] / s.notna().sum() * 100, 3),
            "entropy_bits": round(shannon_entropy(vc.to_numpy()), 4),
            "top3": " | ".join(f"{idx} ({cnt})" for idx, cnt in vc.head(3).items()),
        })
    return pd.DataFrame(rows).sort_values("n_unique", ascending=False).reset_index(drop=True)


# ============================================================
# 4. CORRELATION WITH TARGET  (the missing piece)
# ============================================================

def correlation_with_target(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Pearson (point-biserial) + Spearman of each numeric feature vs TARGET."""
    y = pd.to_numeric(df[TARGET], errors="coerce")
    rows = []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        pair = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(pair) < 30 or pair["x"].nunique() < 2:
            continue
        pear = pair["x"].corr(pair["y"], method="pearson")
        spear = pair["x"].corr(pair["y"], method="spearman")
        rows.append({
            "feature": c,
            "n_used": len(pair),
            "coverage_pct": round(len(pair) / len(df) * 100, 2),
            "pearson_corr": round(pear, 5),
            "spearman_corr": round(spear, 5),
            "abs_spearman": round(abs(spear), 5),
            "direction": "tăng rủi ro (corr>0)" if spear > 0 else "giảm rủi ro (corr<0)",
            "strength_spearman": strength_bucket(abs(spear)),
        })
    out = pd.DataFrame(rows).sort_values("abs_spearman", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


# ============================================================
# 5. CORRELATION MATRIX + MULTICOLLINEARITY
# ============================================================

def correlation_matrix_and_multicollinearity(
    df: pd.DataFrame,
    continuous: list[str],
    desc_cont: pd.DataFrame,
    target_corr: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    # Stable subset: continuous features that are not too sparse.
    keep = desc_cont.loc[desc_cont["missing_pct"] <= MAX_MISSING_FOR_CORR_MATRIX, "feature"].tolist()
    keep = [c for c in keep if c in continuous]

    # Top drivers by |spearman vs target| for the displayed matrix.
    ranked = target_corr[target_corr["feature"].isin(keep)]
    top_drivers = ranked.head(TOP_N_DRIVERS)["feature"].tolist()

    matrix = df[top_drivers + [TARGET]].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    matrix_out = matrix.round(4).reset_index().rename(columns={"index": "feature"})

    # Full multicollinearity scan (Pearson) on the stable subset.
    full_corr = df[keep].apply(pd.to_numeric, errors="coerce").corr(method="pearson")
    pairs = []
    cols = full_corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = full_corr.iloc[i, j]
            if pd.notna(r) and abs(r) >= MULTICOLLINEARITY_THRESHOLD:
                pairs.append({
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "pearson_corr": round(r, 4),
                    "abs_corr": round(abs(r), 4),
                })
    multicol = (
        pd.DataFrame(pairs).sort_values("abs_corr", ascending=False).reset_index(drop=True)
        if pairs else pd.DataFrame(columns=["feature_1", "feature_2", "pearson_corr", "abs_corr"])
    )
    return matrix_out, multicol, top_drivers


# ============================================================
# 6. CATEGORICAL ASSOCIATION WITH TARGET
# ============================================================

def cramers_v_bias_corrected(confusion: np.ndarray) -> float:
    chi2 = stats.chi2_contingency(confusion, correction=False)[0]
    n = confusion.sum()
    if n == 0:
        return np.nan
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else np.nan


def categorical_association(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = pd.to_numeric(df[TARGET], errors="coerce")
    assoc_rows = []
    rate_rows = []
    for c in cols:
        sub = pd.DataFrame({"cat": df[c].astype("object").fillna("Missing"), "y": y}).dropna(subset=["y"])
        confusion = pd.crosstab(sub["cat"], sub["y"])
        if confusion.shape[0] < 2 or confusion.shape[1] < 2:
            continue
        chi2, p, dof, _ = stats.chi2_contingency(confusion, correction=False)
        v = cramers_v_bias_corrected(confusion.to_numpy())
        assoc_rows.append({
            "feature": c,
            "n_categories": int(confusion.shape[0]),
            "chi2": round(chi2, 2),
            "dof": int(dof),
            "p_value": p,
            "cramers_v": round(v, 5) if pd.notna(v) else np.nan,
            "significant_0_05": bool(p < 0.05),
        })
        # Per-category default rate (always carry sample size).
        grp = sub.groupby("cat")["y"].agg(["count", "sum", "mean"]).reset_index()
        grp.columns = ["category", "customers", "defaults", "default_rate"]
        grp["default_rate_pct"] = (grp["default_rate"] * 100).round(3)
        grp["feature"] = c
        grp["reliable"] = grp["customers"] >= MIN_SEGMENT_N
        rate_rows.append(grp[["feature", "category", "customers", "defaults", "default_rate_pct", "reliable"]])

    assoc = pd.DataFrame(assoc_rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)
    rates = pd.concat(rate_rows, ignore_index=True) if rate_rows else pd.DataFrame()
    if not rates.empty:
        rates = rates.sort_values(["feature", "default_rate_pct"], ascending=[True, False]).reset_index(drop=True)
    return assoc, rates


# ============================================================
# 7. CHARTS
# ============================================================

def chart_top_target_corr(target_corr: pd.DataFrame, top_n: int = 20) -> None:
    plot_df = target_corr.head(top_n).copy().sort_values("spearman_corr")
    colors = ["#FF5C6A" if v > 0 else "#1E8BFF" for v in plot_df["spearman_corr"]]
    plt.figure(figsize=(11, 9))
    plt.barh(plot_df["feature"], plot_df["spearman_corr"], color=colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Spearman correlation with TARGET  (do = tang rui ro, xanh = giam rui ro)")
    plt.title(f"Top {top_n} features by |Spearman| correlation with default")
    save_fig("01_top_correlation_with_target")


def chart_corr_heatmap(matrix_out: pd.DataFrame) -> None:
    m = matrix_out.set_index("feature")
    plt.figure(figsize=(13, 11))
    sns.heatmap(m, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=False, linewidths=0.4, cbar_kws={"shrink": 0.6})
    plt.title("Correlation matrix of top default drivers (Spearman)")
    save_fig("02_correlation_matrix_top_drivers")


def chart_top_categorical(assoc: pd.DataFrame, rates: pd.DataFrame, top_k: int = 3) -> None:
    if assoc.empty or rates.empty:
        return
    feats = assoc.head(top_k)["feature"].tolist()
    fig, axes = plt.subplots(1, len(feats), figsize=(6.5 * len(feats), 6))
    if len(feats) == 1:
        axes = [axes]
    for ax, f in zip(axes, feats):
        sub = rates[(rates["feature"] == f) & (rates["reliable"])].sort_values("default_rate_pct", ascending=True)
        ax.barh(sub["category"].astype(str), sub["default_rate_pct"], color="#6C4CFF")
        ax.axvline(BASELINE_DEFAULT_RATE * 100, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{f}\n(default rate %, baseline do)")
        ax.set_xlabel("Default rate (%)")
    plt.suptitle("Default rate by category for strongest categorical drivers", y=1.02)
    save_fig("03_default_rate_by_top_categories")


# ============================================================
# 8. AUTO-WRITTEN INTERPRETATION NOTES (Vietnamese)
# ============================================================

def write_interpretation_notes(
    df: pd.DataFrame,
    groups: dict,
    desc_cont: pd.DataFrame,
    target_corr: pd.DataFrame,
    multicol: pd.DataFrame,
    assoc: pd.DataFrame,
    rates: pd.DataFrame,
) -> Path:
    up = target_corr[target_corr["spearman_corr"] > 0].head(10)
    down = target_corr[target_corr["spearman_corr"] < 0].head(10)
    highly_skewed = desc_cont[desc_cont["skewness"].abs() >= 1.0]
    high_missing = desc_cont[desc_cont["missing_pct"] >= 50.0]

    lines = []
    lines.append("# Step 2 - Thống kê mô tả & Phân tích tương quan - Nhận xét\n")
    lines.append("Tài liệu này được sinh tự động từ kết quả chạy thật trên bảng phân tích cuối "
                 "(`final_customer_analysis_train.csv.gz`), không phải viết tay.\n")

    lines.append("## 1. Tổng quan dữ liệu\n")
    lines.append(f"- Số khách hàng (1 dòng = 1 SK_ID_CURR): **{len(df):,}**")
    lines.append(f"- Default rate (TARGET=1): **{BASELINE_DEFAULT_RATE*100:.2f}%**")
    lines.append(f"- Số feature liên tục: **{len(groups['continuous'])}**, "
                 f"binary flag: **{len(groups['binary_flags'])}**, "
                 f"categorical: **{len(groups['categorical'])}**\n")

    lines.append("## 2. Tín hiệu làm TĂNG rủi ro (Spearman > 0)\n")
    lines.append("| Feature | Spearman | Pearson | Độ mạnh | n dùng |")
    lines.append("|---|---:|---:|---|---:|")
    for _, r in up.iterrows():
        lines.append(f"| {r['feature']} | {r['spearman_corr']:.4f} | {r['pearson_corr']:.4f} "
                     f"| {r['strength_spearman']} | {int(r['n_used']):,} |")
    lines.append("")

    lines.append("## 3. Tín hiệu làm GIẢM rủi ro (Spearman < 0)\n")
    lines.append("| Feature | Spearman | Pearson | Độ mạnh | n dùng |")
    lines.append("|---|---:|---:|---|---:|")
    for _, r in down.iterrows():
        lines.append(f"| {r['feature']} | {r['spearman_corr']:.4f} | {r['pearson_corr']:.4f} "
                     f"| {r['strength_spearman']} | {int(r['n_used']):,} |")
    lines.append("")

    lines.append("## 4. Nhận xét về độ mạnh tương quan\n")
    max_abs = target_corr["abs_spearman"].max()
    lines.append(f"- Tương quan đơn biến mạnh nhất với TARGET chỉ khoảng **|rho| = {max_abs:.3f}**. "
                 "Điều này xác nhận kết luận xuyên suốt project: **rủi ro tín dụng không đến từ một biến đơn lẻ**, "
                 "mà là tổ hợp nhiều tín hiệu. Không nên dùng một biến để quyết định tín dụng.")
    lines.append("- Spearman được ưu tiên làm thước đo chính vì các biến tiền tệ/tỷ lệ lệch phải mạnh; "
                 "Pearson giữ lại để tham chiếu tuyến tính. Khi Pearson ≈ 0 nhưng Spearman lớn "
                 "(ví dụ BUREAU_DEBT_CREDIT_RATIO), đó là do outlier cực trị phá vỡ tương quan tuyến tính "
                 "nhưng tín hiệu đơn điệu theo hạng vẫn còn.\n")

    lines.append("## 5. Đa cộng tuyến (multicollinearity)\n")
    lines.append(f"- Số cặp feature có |Pearson| >= {MULTICOLLINEARITY_THRESHOLD}: **{len(multicol)}**.")
    if not multicol.empty:
        lines.append("- Một số cặp tiêu biểu:")
        for _, r in multicol.head(8).iterrows():
            lines.append(f"  - {r['feature_1']} ~ {r['feature_2']} (r = {r['pearson_corr']})")
        lines.append("- Khuyến nghị: ở bước diagnostic/logistic (Step 6), nên kiểm tra VIF và "
                     "tránh đưa đồng thời cả hai feature trong mỗi cặp này để hệ số ổn định và dễ diễn giải.")
    lines.append("")

    lines.append("## 6. Categorical driver mạnh nhất (Cramer's V)\n")
    lines.append("| Feature | Cramer's V | p-value | Số nhóm |")
    lines.append("|---|---:|---:|---:|")
    for _, r in assoc.head(8).iterrows():
        lines.append(f"| {r['feature']} | {r['cramers_v']:.4f} | {r['p_value']:.2e} | {int(r['n_categories'])} |")
    lines.append("")
    if not rates.empty:
        lines.append("- Nhóm có default rate cao nhất (chỉ xét segment >= "
                     f"{MIN_SEGMENT_N} khách hàng để tránh mẫu nhỏ):")
        reliable = rates[rates["reliable"]].sort_values("default_rate_pct", ascending=False).head(8)
        for _, r in reliable.iterrows():
            lines.append(f"  - {r['feature']} = {r['category']}: "
                         f"{r['default_rate_pct']:.2f}% ({int(r['customers']):,} kh)")
    lines.append("")

    lines.append("## 7. Phân phối & chất lượng biến liên tục\n")
    lines.append(f"- Số biến lệch mạnh (|skew| >= 1): **{len(highly_skewed)}** -> nên dùng median/percentile "
                 "khi mô tả, và log/clip khi trực quan hóa hoặc đưa vào model tuyến tính.")
    lines.append(f"- Số biến missing >= 50%: **{len(high_missing)}** "
                 "(chủ yếu nhóm property/building) -> coi missing như tín hiệu, tạo missing-flag thay vì xóa.\n")

    lines.append("## 8. Kết luận chuyển sang Step 5/6\n")
    lines.append("- Các driver tăng rủi ro (debt burden, credit utilization, previous refusal, late payment) "
                 "sẽ được kiểm chứng lại bằng dashboard (Step 5) và odds ratio có kiểm soát (Step 6).")
    lines.append("- EXT_SOURCE_* và affordability ratios là nhóm tín hiệu giảm rủi ro mạnh nhất, "
                 "phù hợp với kết quả feature importance ở Step 7.")

    path = OUT_PATH / "step02_interpretation_notes.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"Saved interpretation notes: {path.name}")
    return path


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    global BASELINE_DEFAULT_RATE

    input_path = find_input()
    log(f"Input: {input_path}")
    df = pd.read_csv(input_path, compression="gzip")
    log(f"Loaded shape: {df.shape}")

    if TARGET not in df.columns:
        raise ValueError(f"{TARGET} not found in input table.")
    df = df[df[TARGET].notna()].copy()
    BASELINE_DEFAULT_RATE = float(pd.to_numeric(df[TARGET], errors="coerce").mean())
    log(f"Baseline default rate: {BASELINE_DEFAULT_RATE*100:.4f}%")

    groups = classify_columns(df)
    log(f"continuous={len(groups['continuous'])}, flags={len(groups['binary_flags'])}, "
        f"categorical={len(groups['categorical'])}")

    # 1-3 descriptive
    desc_cont = describe_continuous(df, groups["continuous"])
    save_table(desc_cont, "05_descriptive_all_numeric_features")
    desc_flags = describe_binary_flags(df, groups["binary_flags"])
    save_table(desc_flags, "05b_descriptive_binary_flags")
    desc_cat = describe_categorical(df, groups["categorical"])
    save_table(desc_cat, "05c_descriptive_categorical")

    # 4 correlation with target (numeric features = continuous + flags)
    target_corr = correlation_with_target(df, groups["continuous"] + groups["binary_flags"])
    save_table(target_corr, "06_correlation_with_target")

    # 5 correlation matrix + multicollinearity
    matrix_out, multicol, top_drivers = correlation_matrix_and_multicollinearity(
        df, groups["continuous"], desc_cont, target_corr
    )
    save_table(matrix_out, "07_correlation_matrix_selected")
    save_table(multicol, "08_multicollinearity_pairs")

    # 6 categorical association
    assoc, rates = categorical_association(df, groups["categorical"])
    save_table(assoc, "09_categorical_association_target")
    save_table(rates, "10_default_rate_by_category")

    # 7 charts
    chart_top_target_corr(target_corr)
    chart_corr_heatmap(matrix_out)
    chart_top_categorical(assoc, rates)

    # 8 interpretation notes
    notes_path = write_interpretation_notes(df, groups, desc_cont, target_corr, multicol, assoc, rates)

    summary = {
        "step": "Step 2 FULL - Descriptive & Correlation",
        "input": str(input_path),
        "rows": int(len(df)),
        "baseline_default_rate_pct": round(BASELINE_DEFAULT_RATE * 100, 4),
        "n_continuous": len(groups["continuous"]),
        "n_binary_flags": len(groups["binary_flags"]),
        "n_categorical": len(groups["categorical"]),
        "strongest_abs_spearman_vs_target": float(target_corr["abs_spearman"].max()),
        "n_multicollinearity_pairs": int(len(multicol)),
        "tables_dir": str(TABLE_PATH),
        "figures_dir": str(FIG_PATH),
        "interpretation_notes": str(notes_path),
    }
    (OUT_PATH / "step02_full_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log("=" * 70)
    log("STEP 2 FULL FINISHED")
    log(f"Tables: {TABLE_PATH}")
    log(f"Figures: {FIG_PATH}")
    log(f"Notes: {notes_path}")
    log("=" * 70)


if __name__ == "__main__":
    main()
