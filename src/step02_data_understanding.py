# -*- coding: utf-8 -*-
"""STEP 2 — DEEP ANALYSIS: ERD, grain, cardinality, coverage, raw-merge explosion."""
import sys, warnings, csv
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

RAW = Path(r"D:\Code\DA\1\clean_pipeline\data\raw")
OUT = Path(r"D:\Code\DA\1\step2_deep_outputs"); OUT.mkdir(exist_ok=True)

FILES = {
    "application_train":"application_train.csv","application_test":"application_test.csv",
    "bureau":"bureau.csv","bureau_balance":"bureau_balance.csv",
    "previous_application":"previous_application.csv","installments_payments":"installments_payments.csv",
    "POS_CASH_balance":"POS_CASH_balance.csv","credit_card_balance":"credit_card_balance.csv",
}
print("="*70); print("STEP 2 — DEEP ANALYSIS"); print("="*70)

def count_rows(fp):
    with open(fp,"r",encoding="utf-8",newline="") as f:
        next(f); return sum(1 for _ in f)
def header(fp):
    with open(fp,"r",encoding="utf-8",newline="") as f: return next(csv.reader(f))
def uniq(fname, col, cs=500_000):
    s=set()
    for ch in pd.read_csv(RAW/fname, usecols=[col], chunksize=cs):
        s.update(ch[col].dropna().astype("int64").unique())
    return s

# 1. FILE PROFILE
print("\n[1] File profile + grain...")
rows=[]
for name,fn in FILES.items():
    fp=RAW/fn
    rows.append({"table":name,"rows":count_rows(fp),"columns":len(header(fp)),
                 "size_mb":round(fp.stat().st_size/1024/1024,1)})
prof=pd.DataFrame(rows).sort_values("rows",ascending=False)
prof.to_csv(OUT/"01_file_profile.csv",index=False,encoding="utf-8-sig")
print(prof.to_string(index=False))

# 2. KEY CARDINALITY (rows vs unique SK_ID_CURR)
print("\n[2] Cardinality (one-to-many evidence)...")
card=[]
specs=[("application_train","SK_ID_CURR"),("bureau","SK_ID_CURR"),("bureau","SK_ID_BUREAU"),
       ("previous_application","SK_ID_CURR"),("previous_application","SK_ID_PREV"),
       ("installments_payments","SK_ID_CURR"),("POS_CASH_balance","SK_ID_CURR"),
       ("credit_card_balance","SK_ID_CURR")]
for name,key in specs:
    fn=FILES[name]; total=count_rows(RAW/fn); u=len(uniq(fn,key))
    card.append({"table":name,"key":key,"rows":total,"unique":u,
                 "rows_per_unique":round(total/u,2) if u else None,
                 "duplicate_rows":total-u})
card_df=pd.DataFrame(card)
card_df.to_csv(OUT/"02_cardinality.csv",index=False,encoding="utf-8-sig")
print(card_df.to_string(index=False))

# 3. COVERAGE (% application customers with history in each source)
print("\n[3] Historical coverage...")
train_curr=uniq("application_train.csv","SK_ID_CURR")
test_curr=uniq("application_test.csv","SK_ID_CURR")
app_curr=train_curr|test_curr
print(f"  total application customers: {len(app_curr):,}")
cov=[]
for name in ["bureau","previous_application","POS_CASH_balance","installments_payments","credit_card_balance"]:
    c=uniq(FILES[name],"SK_ID_CURR")
    matched=len(c & app_curr)
    cov.append({"source":name,"customers_with_history":matched,
                "total_customers":len(app_curr),
                "coverage_pct":round(matched/len(app_curr)*100,2)})
cov_df=pd.DataFrame(cov).sort_values("coverage_pct",ascending=False)
cov_df.to_csv(OUT/"03_coverage.csv",index=False,encoding="utf-8-sig")
print(cov_df.to_string(index=False))

# 4. RAW MERGE EXPLOSION (application_train x bureau)
print("\n[4] Raw merge explosion demo...")
app_small=pd.read_csv(RAW/"application_train.csv",usecols=["SK_ID_CURR","TARGET"])
bureau_key=pd.read_csv(RAW/"bureau.csv",usecols=["SK_ID_CURR","SK_ID_BUREAU"])
merged=app_small.merge(bureau_key,on="SK_ID_CURR",how="inner")
# default rate correct vs wrong
dr_correct=app_small["TARGET"].mean()*100
dr_wrong=merged["TARGET"].mean()*100  # weighted by # bureau loans
explosion={
    "application_rows":len(app_small),
    "bureau_rows":len(bureau_key),
    "rows_after_inner_merge":len(merged),
    "unique_customers_after_merge":merged["SK_ID_CURR"].nunique(),
    "extra_rows_created":len(merged)-len(app_small),
    "default_rate_CORRECT_customer_level":round(dr_correct,4),
    "default_rate_WRONG_after_merge":round(dr_wrong,4),
    "distortion_pp":round(dr_wrong-dr_correct,4),
}
pd.DataFrame([explosion]).to_csv(OUT/"04_merge_explosion.csv",index=False,encoding="utf-8-sig")
for k,v in explosion.items(): print(f"  {k}: {v:,}" if isinstance(v,int) else f"  {k}: {v}")

# 5. Customers with MANY bureau loans (over-weighting evidence)
print("\n[5] Distribution of bureau loans per customer...")
bc=bureau_key.groupby("SK_ID_CURR").size()
dist={
    "customers_with_bureau":len(bc),
    "max_loans_one_customer":int(bc.max()),
    "mean_loans_per_customer":round(bc.mean(),2),
    "median_loans_per_customer":int(bc.median()),
    "p95_loans":int(bc.quantile(0.95)),
}
pd.DataFrame([dist]).to_csv(OUT/"05_bureau_loan_dist.csv",index=False,encoding="utf-8-sig")
for k,v in dist.items(): print(f"  {k}: {v}")

print("\n✅ Step 2 deep analysis done ->",OUT)
print("="*70)
