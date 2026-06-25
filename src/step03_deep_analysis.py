# -*- coding: utf-8 -*-
"""
STEP 3 — DEEP DATA QUALITY ANALYSIS
Chạy phân tích đầy đủ trên raw data + final table
Output: bảng số liệu thật để presentation
"""
import sys, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

RAW  = Path(r"D:\Code\DA\1\clean_pipeline\data\raw")
FINAL = Path(r"D:\Code\DA\1\step4_outputs\data\final_customer_analysis_train.csv.gz")
OUT   = Path(r"D:\Code\DA\1\step3_deep_outputs")
OUT.mkdir(exist_ok=True)

print("=" * 70)
print("STEP 3 — DEEP DATA QUALITY ANALYSIS")
print("=" * 70)

# ── Load core tables ──────────────────────────────────────────────────────────
print("\n[1/8] Loading application_train + test + final table...")
app = pd.read_csv(RAW / "application_train.csv", low_memory=False)
tst = pd.read_csv(RAW / "application_test.csv",  low_memory=False)
fin = pd.read_csv(FINAL, compression="gzip")

BASE = app["TARGET"].mean()
N    = len(app)
print(f"  app_train : {len(app):,} rows x {app.shape[1]} cols")
print(f"  app_test  : {len(tst):,} rows x {tst.shape[1]} cols")
print(f"  final     : {len(fin):,} rows x {fin.shape[1]} cols")
print(f"  Baseline default rate = {BASE*100:.4f}%  ({app['TARGET'].sum():,} defaults)")

# ── 1. KEY INTEGRITY ─────────────────────────────────────────────────────────
print("\n[2/8] Key Integrity Check...")
ki = []
for name, df, keys in [
    ("application_train", app, ["SK_ID_CURR"]),
    ("application_test",  tst, ["SK_ID_CURR"]),
    ("final_train",       fin, ["SK_ID_CURR"]),
]:
    for k in keys:
        if k not in df.columns: continue
        null_n = int(df[k].isna().sum())
        uniq   = int(df[k].nunique())
        ki.append({
            "table":k + " in " + name, "total_rows":len(df),
            "null_key":null_n, "unique_vals":uniq,
            "duplicate_rows":len(df)-uniq,
            "is_unique": "✅ YES" if uniq==len(df) else "❌ NO"
        })
ki_df = pd.DataFrame(ki)
ki_df.to_csv(OUT/"01_key_integrity.csv", index=False, encoding="utf-8-sig")
print(ki_df.to_string(index=False))

# ── 2. MISSING VALUES — APPLICATION TRAIN ─────────────────────────────────────
print("\n[3/8] Missing Values Analysis (application_train)...")
miss = app.isna().sum().reset_index()
miss.columns = ["column","missing_count"]
miss["missing_pct"] = miss["missing_count"] / N * 100
miss["dtype"] = miss["column"].map(app.dtypes.astype(str).to_dict())

# Cross with TARGET
miss_target = []
for col in miss.loc[miss["missing_count"]>0,"column"].tolist():
    is_miss  = app[col].isna()
    n_miss   = int(is_miss.sum())
    n_nonmiss= int((~is_miss).sum())
    dr_miss   = app.loc[is_miss,  "TARGET"].mean()*100 if n_miss>0 else np.nan
    dr_nomiss = app.loc[~is_miss, "TARGET"].mean()*100 if n_nonmiss>0 else np.nan
    diff      = (dr_miss - dr_nomiss) if (pd.notna(dr_miss) and pd.notna(dr_nomiss)) else np.nan
    miss_target.append({
        "column":col, "missing_count":n_miss,
        "missing_pct":round(n_miss/N*100,2),
        "default_rate_when_MISSING": round(dr_miss,2) if pd.notna(dr_miss) else None,
        "default_rate_when_NOT_MISSING": round(dr_nomiss,2) if pd.notna(dr_nomiss) else None,
        "diff_pp": round(diff,2) if pd.notna(diff) else None,
        "signal": "⚠️ SIGNAL" if (pd.notna(diff) and abs(diff)>=1.5) else "—"
    })
mt_df = pd.DataFrame(miss_target).sort_values("missing_pct", ascending=False)
mt_df.to_csv(OUT/"02_missing_vs_target.csv", index=False, encoding="utf-8-sig")

# Summary stats
print(f"  Total columns: {app.shape[1]}")
print(f"  Columns with ANY missing: {(miss['missing_count']>0).sum()}")
print(f"  Columns >50% missing: {(miss['missing_pct']>50).sum()}")
print(f"  Columns >30% missing: {(miss['missing_pct']>30).sum()}")
print(f"  Columns with missing SIGNAL (diff>=1.5pp): {(mt_df['diff_pp'].abs()>=1.5).sum()}")
print("\n  TOP 20 missing columns:")
print(mt_df.head(20)[["column","missing_pct","default_rate_when_MISSING",
                        "default_rate_when_NOT_MISSING","diff_pp","signal"]].to_string(index=False))

# ── 3. MISSING — FINAL TABLE (all 271 cols) ───────────────────────────────────
print("\n[4/8] Missing in Final Table (271 cols)...")
fin_miss = fin.isna().sum().reset_index()
fin_miss.columns = ["column","missing_count"]
fin_miss["missing_pct"] = fin_miss["missing_count"] / len(fin) * 100
fin_miss["source_group"] = fin_miss["column"].apply(lambda c:
    "CC"      if c.startswith("CC_") else
    "Bureau_Balance" if c.startswith("BB_") else
    "Bureau"  if c.startswith("BUREAU_") else
    "POS"     if c.startswith("POS_") else
    "Inst"    if c.startswith("INST_") else
    "Prev"    if c.startswith("PREV_") else
    "App"     if c.startswith("APP_") else "Other")
grp = fin_miss.groupby("source_group").agg(
    n_cols=("column","count"),
    avg_missing=("missing_pct","mean"),
    max_missing=("missing_pct","max")
).reset_index().sort_values("avg_missing", ascending=False)
grp.to_csv(OUT/"03_missing_by_source_group.csv", index=False, encoding="utf-8-sig")
print(grp.to_string(index=False))

# ── 4. SPECIAL VALUES ─────────────────────────────────────────────────────────
print("\n[5/8] Special Values Check...")
sv = []
def add_sv(col, mask, label, note):
    cnt = int(mask.sum())
    pct = cnt / N * 100
    dr        = round(app.loc[mask,  "TARGET"].mean()*100, 2) if cnt > 0 else None
    dr_other  = round(app.loc[~mask, "TARGET"].mean()*100, 2) if int((~mask).sum()) > 0 else None
    diff      = round(dr - dr_other, 2) if (dr is not None and dr_other is not None) else None
    sv.append({"column":col,"special_value":label,"count":cnt,"pct":round(pct,4),
               "default_rate_special":dr,
               "default_rate_others":dr_other,
               "diff_pp":diff, "note":note})

# DAYS_EMPLOYED sentinel
add_sv("DAYS_EMPLOYED", app["DAYS_EMPLOYED"]==365243, "=365243",
       "Sentinel for unemployed/unknown — must replace NaN + flag")

# ORGANIZATION_TYPE XNA
if "ORGANIZATION_TYPE" in app.columns:
    add_sv("ORGANIZATION_TYPE", app["ORGANIZATION_TYPE"]=="XNA", "=XNA",
           "Matches DAYS_EMPLOYED anomaly group — likely not employed")

# CODE_GENDER XNA
if "CODE_GENDER" in app.columns:
    add_sv("CODE_GENDER", app["CODE_GENDER"]=="XNA", "=XNA", "Unknown gender — only 4 rows")

# NAME_FAMILY_STATUS Unknown
if "NAME_FAMILY_STATUS" in app.columns:
    add_sv("NAME_FAMILY_STATUS", app["NAME_FAMILY_STATUS"]=="Unknown", "=Unknown",
           "Only 2 rows — negligible impact")

# AMT_INCOME extreme
p99  = app["AMT_INCOME_TOTAL"].quantile(0.99)
p999 = app["AMT_INCOME_TOTAL"].quantile(0.999)
add_sv("AMT_INCOME_TOTAL", app["AMT_INCOME_TOTAL"]>p99,
       f">p99 ({p99:,.0f})", "Extreme high income outlier — valid but distorts mean")
add_sv("AMT_INCOME_TOTAL", app["AMT_INCOME_TOTAL"]>p999,
       f">p999 ({p999:,.0f})", "Very extreme income outlier")

# Zero income
add_sv("AMT_INCOME_TOTAL", app["AMT_INCOME_TOTAL"]==0, "=0", "Zero income — likely error")

sv_df = pd.DataFrame(sv)
# Add interpretations
interp = {
    "DAYS_EMPLOYED_=365243": "Home Credit sentinel cho nhom khong/chua co viec lam chinh thuc. Phai replace NaN + flag. Default rate cao hon 2.6pp — co tin hieu.",
    "ORGANIZATION_TYPE_=XNA": "Khop voi DAYS_EMPLOYED=365243. Co the la nhom khong lam viec. Giu nhu category rieng.",
    "CODE_GENDER_=XNA": "Chi 4 dong — qua hiem. Replace NaN hoac Unknown.",
    "NAME_FAMILY_STATUS_=Unknown": "Chi 2 dong — thuc te khong anh huong phan tich.",
}
sv_df.to_csv(OUT/"04_special_values.csv", index=False, encoding="utf-8-sig")
print(sv_df[["column","special_value","count","pct","default_rate_special","default_rate_others","diff_pp"]].to_string(index=False))

# ── 5. OUTLIER PROFILE ───────────────────────────────────────────────────────
print("\n[6/8] Outlier Profile (key financial vars)...")
# Create derived vars
app2 = app.copy()
app2["APP_AGE_YEARS"] = -app2["DAYS_BIRTH"] / 365.25
app2["DAYS_EMPLOYED_CLEAN"] = app2["DAYS_EMPLOYED"].replace(365243, np.nan)
app2["APP_EMPLOYED_YEARS"] = -app2["DAYS_EMPLOYED_CLEAN"] / 365.25
app2["CREDIT_INCOME_RATIO"] = app2["AMT_CREDIT"] / app2["AMT_INCOME_TOTAL"].replace(0,np.nan)
app2["ANNUITY_INCOME_RATIO"] = app2["AMT_ANNUITY"] / app2["AMT_INCOME_TOTAL"].replace(0,np.nan)
app2["GOODS_CREDIT_RATIO"] = app2["AMT_GOODS_PRICE"] / app2["AMT_CREDIT"].replace(0,np.nan)

outlier_cols = [
    "AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY",
    "APP_AGE_YEARS","APP_EMPLOYED_YEARS",
    "CREDIT_INCOME_RATIO","ANNUITY_INCOME_RATIO",
    "EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3",
]
rows_out = []
for col in outlier_cols:
    s = pd.to_numeric(app2[col], errors="coerce").replace([np.inf,-np.inf],np.nan).dropna()
    if len(s) == 0: continue
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    n_iqr_low  = int((s < q1 - 1.5*iqr).sum())
    n_iqr_high = int((s > q3 + 1.5*iqr).sum())
    rows_out.append({
        "column":col, "n":len(s), "missing":int(app2[col].isna().sum()),
        "mean":round(s.mean(),2), "median":round(s.median(),2),
        "std":round(s.std(),2),
        "skewness":round(s.skew(),3), "kurtosis":round(s.kurtosis(),3),
        "min":round(s.min(),2), "p1":round(s.quantile(0.01),2),
        "p5":round(s.quantile(0.05),2), "p25":round(q1,2),
        "p50":round(s.quantile(0.50),2), "p75":round(q3,2),
        "p95":round(s.quantile(0.95),2), "p99":round(s.quantile(0.99),2),
        "max":round(s.max(),2),
        "iqr_outliers_low":n_iqr_low, "iqr_outliers_high":n_iqr_high,
        "iqr_outlier_pct_high":round(n_iqr_high/len(s)*100,2),
    })
out_df = pd.DataFrame(rows_out)
out_df.to_csv(OUT/"05_outlier_profile.csv", index=False, encoding="utf-8-sig")
print(out_df[["column","mean","median","skewness","kurtosis","p99","max","iqr_outlier_pct_high"]].to_string(index=False))

# ── 6. PSI (Population Stability Index) ─────────────────────────────────────
print("\n[7/8] PSI — Train vs Test Drift...")

def numeric_psi(train_s, test_s, n_bins=10):
    tr = pd.to_numeric(train_s, errors="coerce").replace([np.inf,-np.inf],np.nan).dropna()
    te = pd.to_numeric(test_s,  errors="coerce").replace([np.inf,-np.inf],np.nan).dropna()
    if tr.nunique()<=1 or te.nunique()<=1 or len(te)<100: return np.nan
    qs = np.linspace(0,1,n_bins+1)
    bins = np.unique(tr.quantile(qs).values)
    if len(bins)<3: return np.nan
    bins[0]=-np.inf; bins[-1]=np.inf
    tp = pd.cut(tr,bins=bins,include_lowest=True).value_counts(sort=False)/max(len(tr),1)
    ep = pd.cut(te,bins=bins,include_lowest=True).value_counts(sort=False)/max(len(te),1)
    tp = tp.reindex(ep.index,fill_value=0).replace(0,0.0001)
    ep = ep.replace(0,0.0001)
    return float(((ep-tp)*np.log(ep/tp)).sum())

def cat_psi(train_s, test_s):
    tr = train_s.fillna("Missing").astype(str)
    te = test_s.fillna("Missing").astype(str)
    cats = sorted(set(tr.unique())|set(te.unique()))
    tp = tr.value_counts(normalize=True).reindex(cats,fill_value=0).replace(0,0.0001)
    ep = te.value_counts(normalize=True).reindex(cats,fill_value=0).replace(0,0.0001)
    return float(((ep-tp)*np.log(ep/tp)).sum())

psi_rows = []
num_cols  = ["AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","DAYS_BIRTH","DAYS_EMPLOYED",
             "EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3","AMT_GOODS_PRICE",
             "REGION_POPULATION_RELATIVE","CNT_CHILDREN","CNT_FAM_MEMBERS",
             "DAYS_REGISTRATION","DAYS_ID_PUBLISH","DAYS_LAST_PHONE_CHANGE",
             "OWN_CAR_AGE","DEF_30_CNT_SOCIAL_CIRCLE","DEF_60_CNT_SOCIAL_CIRCLE"]
cat_cols  = ["NAME_CONTRACT_TYPE","CODE_GENDER","FLAG_OWN_CAR","FLAG_OWN_REALTY",
             "NAME_TYPE_SUITE","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE",
             "NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","OCCUPATION_TYPE","ORGANIZATION_TYPE"]

for col in num_cols:
    if col in app.columns and col in tst.columns:
        tr_s = app[col].replace(365243,np.nan) if col=="DAYS_EMPLOYED" else app[col]
        te_s = tst[col].replace(365243,np.nan) if col=="DAYS_EMPLOYED" else tst[col]
        psi_val = numeric_psi(tr_s, te_s)
        lvl = "HIGH" if psi_val and psi_val>0.25 else ("MODERATE" if psi_val and psi_val>0.10 else "LOW")
        psi_rows.append({"column":col,"type":"numeric","psi":round(psi_val,4) if pd.notna(psi_val) else None,"level":lvl})
for col in cat_cols:
    if col in app.columns and col in tst.columns:
        psi_val = cat_psi(app[col], tst[col])
        lvl = "HIGH" if psi_val>0.25 else ("MODERATE" if psi_val>0.10 else "LOW")
        psi_rows.append({"column":col,"type":"categorical","psi":round(psi_val,4),"level":lvl})

psi_df = pd.DataFrame(psi_rows).sort_values("psi",ascending=False)
psi_df.to_csv(OUT/"06_psi_drift.csv", index=False, encoding="utf-8-sig")
print(psi_df.head(15).to_string(index=False))

# ── 7. PLAUSIBILITY CHECKS ───────────────────────────────────────────────────
print("\n[8/8] Plausibility / Business Logic Checks...")
pl = []
def add_pl(check, count, total, severity, note):
    pl.append({"check":check,"count":int(count),"total":int(total),
               "pct":round(count/total*100,4),"severity":severity,"note":note})

# Application checks
add_pl("DAYS_EMPLOYED sentinel (365243)", (app["DAYS_EMPLOYED"]==365243).sum(), N, "HIGH",
       "Home Credit sentinel for unknown employment — must replace NaN + flag")
add_pl("Age < 21 years", (-app["DAYS_BIRTH"]/365.25 < 21).sum(), N, "REVIEW",
       "Very young applicants — plausible but edge case")
add_pl("Age > 69 years", (-app["DAYS_BIRTH"]/365.25 > 69).sum(), N, "REVIEW",
       "Senior applicants — check if any are anomalies")
add_pl("DAYS_EMPLOYED > 0 (positive, not sentinel)",
       ((app["DAYS_EMPLOYED"]>0)&(app["DAYS_EMPLOYED"]!=365243)).sum(), N, "REVIEW",
       "Positive days_employed (unusual — normally negative or sentinel)")
add_pl("Credit-Income Ratio > 10",
       (app["AMT_CREDIT"]/app["AMT_INCOME_TOTAL"].replace(0,np.nan)>10).sum(), N, "HIGH",
       "Extreme debt burden — credit > 10x annual income")
add_pl("Credit-Income Ratio > 20",
       (app["AMT_CREDIT"]/app["AMT_INCOME_TOTAL"].replace(0,np.nan)>20).sum(), N, "HIGH",
       "Very extreme debt burden")
add_pl("Annuity-Income Ratio > 0.5",
       (app["AMT_ANNUITY"]/app["AMT_INCOME_TOTAL"].replace(0,np.nan)>0.5).sum(), N, "HIGH",
       "Monthly payment > 50% of income — extreme stress")
add_pl("AMT_INCOME_TOTAL = 0", (app["AMT_INCOME_TOTAL"]==0).sum(), N, "HIGH", "Zero income — likely error")
add_pl("AMT_CREDIT = 0", (app["AMT_CREDIT"]==0).sum(), N, "HIGH", "Zero loan amount — likely error")

# Cross-var plausibility
add_pl("AMT_GOODS_PRICE > AMT_CREDIT",
       (app["AMT_GOODS_PRICE"] > app["AMT_CREDIT"]).sum(), N, "REVIEW",
       "Goods price exceeds credit — loan doesn't cover full goods price")
add_pl("AMT_ANNUITY > AMT_INCOME_TOTAL/12",
       (app["AMT_ANNUITY"] > app["AMT_INCOME_TOTAL"]/12).sum(), N, "HIGH",
       "Monthly payment > monthly income — technically cannot afford")

# Installments — read by chunks (689 MB file)
print("  Loading installments in chunks (689 MB)...")
cols = ["DAYS_INSTALMENT","DAYS_ENTRY_PAYMENT","AMT_INSTALMENT","AMT_PAYMENT"]
cnt_dict = defaultdict(int); n_inst = 0
for chunk in pd.read_csv(RAW/"installments_payments.csv", usecols=cols,
                         chunksize=500_000, low_memory=False):
    n_inst += len(chunk)
    di = pd.to_numeric(chunk["DAYS_INSTALMENT"],  errors="coerce")
    dp = pd.to_numeric(chunk["DAYS_ENTRY_PAYMENT"],errors="coerce")
    ai = pd.to_numeric(chunk["AMT_INSTALMENT"],   errors="coerce")
    ap_ = pd.to_numeric(chunk["AMT_PAYMENT"],     errors="coerce")
    cnt_dict["late"]      += int((dp > di).sum())
    cnt_dict["underpay"]  += int((ap_ < ai).sum())
    cnt_dict["miss_date"] += int(dp.isna().sum())
    cnt_dict["zero_inst"] += int((ai == 0).sum())
    cnt_dict["zero_pay"]  += int((ap_ == 0).sum())
print(f"  installments: {n_inst:,} rows processed")
for key, label, sev, note in [
    ("late",      "[inst] Late payment rows",      "SIGNAL","Trả chậm — behavioral signal"),
    ("underpay",  "[inst] Underpayment rows",       "SIGNAL","Trả thiếu — behavioral signal"),
    ("miss_date", "[inst] Missing payment date",    "REVIEW","No payment recorded"),
    ("zero_inst", "[inst] Zero installment amount", "REVIEW","Potential system record"),
    ("zero_pay",  "[inst] Zero payment amount",     "REVIEW","Missed payment possible"),
]:
    add_pl(label, cnt_dict[key], n_inst, sev, note)

pl_df = pd.DataFrame(pl)
pl_df.to_csv(OUT/"07_plausibility_checks.csv", index=False, encoding="utf-8-sig")
print(pl_df[["check","count","pct","severity","note"]].to_string(index=False))

# ── 8. FINAL SUMMARY TABLE ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3 SUMMARY — KEY NUMBERS FOR PRESENTATION")
print("=" * 70)

n_missing_cols = int((miss["missing_count"]>0).sum())
n_signal_cols  = int((mt_df["diff_pp"].abs()>=1.5).sum())
n_high_miss    = int((miss["missing_pct"]>30).sum())
days_emp_cnt   = int((app["DAYS_EMPLOYED"]==365243).sum())
days_emp_dr    = round(app.loc[app["DAYS_EMPLOYED"]==365243,"TARGET"].mean()*100,2)
days_emp_dr_other = round(app.loc[app["DAYS_EMPLOYED"]!=365243,"TARGET"].mean()*100,2)
late_cnt       = cnt_dict["late"]
late_pct       = late_cnt/n_inst*100
underpay_cnt   = cnt_dict["underpay"]
underpay_pct   = underpay_cnt/n_inst*100
cir_extreme    = int((app["AMT_CREDIT"]/app["AMT_INCOME_TOTAL"].replace(0,np.nan)>10).sum())
annuity_stress = int((app["AMT_ANNUITY"]/app["AMT_INCOME_TOTAL"].replace(0,np.nan)>0.5).sum())
psi_high_cnt   = int((psi_df["level"]=="HIGH").sum())
psi_mod_cnt    = int((psi_df["level"]=="MODERATE").sum())

# Missing signal columns details
signal_cols = mt_df.loc[mt_df["diff_pp"].abs()>=1.5].sort_values("diff_pp",ascending=False).head(10)

summary = {
    "DATASET": f"{N:,} customers | {app['TARGET'].sum():,} defaults | {BASE*100:.2f}% default rate",
    "MISSING": f"{n_missing_cols} cột có missing | {n_high_miss} cột >30% missing | {n_signal_cols} cột có tín hiệu rủi ro khi missing",
    "SENTINEL_DAYS_EMPLOYED": f"{days_emp_cnt:,} rows ({days_emp_cnt/N*100:.2f}%) | default rate: {days_emp_dr}% vs others: {days_emp_dr_other}% | diff: +{round(days_emp_dr-days_emp_dr_other,2)}pp",
    "PAYMENT_BEHAVIOR_SIGNAL": f"Late payment: {late_cnt:,} rows ({late_pct:.2f}%) | Underpayment: {underpay_cnt:,} rows ({underpay_pct:.2f}%)",
    "EXTREME_AFFORDABILITY": f"Credit/Income >10x: {cir_extreme:,} ({cir_extreme/N*100:.2f}%) | Annuity/Income >50%: {annuity_stress:,} ({annuity_stress/N*100:.2f}%)",
    "PSI_DRIFT": f"HIGH drift: {psi_high_cnt} cols | MODERATE drift: {psi_mod_cnt} cols",
}

for k,v in summary.items():
    print(f"\n  {k}:")
    print(f"    {v}")

print("\n  TOP MISSING-SIGNAL COLUMNS (default rate khi missing cao hơn khi không missing):")
print(signal_cols[["column","missing_pct","default_rate_when_MISSING",
                    "default_rate_when_NOT_MISSING","diff_pp"]].to_string(index=False))

# Save summary
pd.DataFrame([{"key":k,"value":v} for k,v in summary.items()]).to_csv(
    OUT/"00_summary_key_numbers.csv", index=False, encoding="utf-8-sig")

print(f"\n✅ All outputs saved to: {OUT}")
print("=" * 70)
