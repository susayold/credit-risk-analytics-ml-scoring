# Champion Run Environment

This file records the environment assumptions for the published Step 8 evidence.

## Published Metric Source

The headline LightGBM v3 metrics are treated as evidence-reproducible from committed CSV files:

- `outputs/final/champion_validation_metrics.csv`
- `outputs/final/model_comparison.csv`
- `outputs/final/cross_validation_metrics.csv`
- `docs/METRICS_SOURCE_OF_TRUTH.csv`

The command below rebuilds the canonical evidence layer:

```bash
python src/step08_train_champion_v3.py --mode evidence
```

The command below retrains the LightGBM v3-style model from the processed master data:

```bash
python src/step08_train_champion_v3.py --mode train
```

Exact retraining metrics can vary with package versions and hardware. The published portfolio metrics should be audited from `outputs/final/` and `docs/METRICS_SOURCE_OF_TRUTH.csv`.

## Split Design

The retraining script reads committed split ID files instead of creating a new stratified split at runtime:

- `data/splits/fit_train_ids.csv`
- `data/splits/calibration_ids.csv`
- `data/splits/final_validation_ids.csv`

This keeps the published split sizes and target rates reproducible. A fresh stratified split would push the validation default rate closer to the full portfolio baseline of 8.07%, while the published validation holdout is 9.37%.

| Split | Rows | Purpose |
|---|---:|---|
| Fit train | 196,805 | Train LightGBM and related models |
| Calibration | 49,202 | Platt probability calibration |
| Final validation | 61,504 | Final model ranking and decision-band evaluation |
| Kaggle test file | 48,744 | Unlabeled application test rows, not used for reported validation metrics |

## Environment Used For Local Evidence Verification

| Item | Value |
|---|---|
| Operating system | Windows |
| Local Python | 3.14.2 |
| CI Python | 3.11 |
| Random seed | 42 |
| Champion package | lightgbm 4.6.0 |
| scikit-learn | 1.8.0 |
| pandas | 3.0.2 |
| numpy | 2.4.4 |
| scipy | 1.17.1 |
| statsmodels | 0.14.6 |

See `requirements-lock.txt` for the pinned dependency file.

## Processed Data Checks

| File | Rows | Columns | SHA256 |
|---|---:|---:|---|
| `data/processed/final_customer_analysis_train.csv.gz` | 307,511 | 271 | `A18F480EEFA8479B339235F298E12867CA400729133567E47CA1CD0A71EA563F` |
| `data/processed/final_customer_analysis_test.csv.gz` | 48,744 | 271 | `1D525A2798FE3AF41ACDB20516988140B2D69CDBF7A1E158A27BB9850CDE27D4` |
| `data/processed/final_customer_analysis_table.csv.gz` | 356,255 | 271 | `1DABDEF20F4BC8694F1F8E4FCA3F2A7C98C0D04EC44796AA120335AB84961A44` |

## Governance Note

The score bands are review-priority analytics outputs, not production approval or rejection rules. A regulated credit decisioning system would also need policy approval, compliance controls, reason-code generation, deployment monitoring, drift alerts, and human review procedures.
