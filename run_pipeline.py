from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

REQUIRED_RAW_FILES = [
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "POS_CASH_balance.csv",
    "installments_payments.csv",
    "credit_card_balance.csv",
]


def raw_data_available() -> bool:
    return all((RAW_DIR / name).exists() for name in REQUIRED_RAW_FILES)


def run_script(script: Path) -> None:
    print(f"\n=== Running {script.relative_to(PROJECT_ROOT)} ===")
    subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Credit Risk analytics pipeline.")
    parser.add_argument("--skip-master", action="store_true", help="Skip raw-data master-table build and use processed outputs.")
    args = parser.parse_args()

    if raw_data_available() and not args.skip_master:
        run_script(PROJECT_ROOT / "src" / "step05_build_master_table.py")
    else:
        print("\n=== Skipping Step 5 master build ===")
        if not raw_data_available():
            print("Raw data not found in data/raw. Using included processed outputs where possible.")
        if not (PROCESSED_DIR / "final_customer_analysis_train.csv.gz").exists():
            raise FileNotFoundError(
                "Missing processed train file. Add raw data to data/raw or include "
                "data/processed/final_customer_analysis_train.csv.gz."
            )

    run_script(PROJECT_ROOT / "src" / "step04_descriptive_from_processed.py")

    print("\nPipeline completed.")
    print(f"Processed data: {PROCESSED_DIR}")
    print(f"Outputs: {PROJECT_ROOT / 'outputs'}")
    print(f"Dashboard: {PROJECT_ROOT / 'dashboard' / 'credit_risk_dashboard.pbix'}")


if __name__ == "__main__":
    main()

