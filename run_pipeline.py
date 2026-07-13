from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RETRAIN_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "retrain" / "latest"

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

STAGE_ORDER = ["master", "descriptive", "diagnostic", "ml", "validate"]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def raw_data_available() -> bool:
    return all((RAW_DIR / name).exists() for name in REQUIRED_RAW_FILES)


def run_command(stage: str, command: list[str]) -> None:
    logging.info("Running stage '%s': %s", stage, " ".join(command))
    try:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        logging.error("Stage '%s' failed with exit code %s", stage, exc.returncode)
        raise


def validate_processed_train_exists() -> None:
    train_path = PROCESSED_DIR / "final_customer_analysis_train.csv.gz"
    if not train_path.exists():
        raise FileNotFoundError(
            "Missing processed train file. Add raw Kaggle CSV files to data/raw "
            "or include data/processed/final_customer_analysis_train.csv.gz."
        )


def run_master(skip_master: bool) -> None:
    if raw_data_available() and not skip_master:
        run_command("master", [sys.executable, "src/step05_build_master_table.py"])
        return

    logging.info("Skipping Step 5 master build.")
    if not raw_data_available():
        logging.info("Raw data not found in data/raw; using included processed outputs.")
    validate_processed_train_exists()


def run_descriptive() -> None:
    validate_processed_train_exists()
    run_command("descriptive", [sys.executable, "src/step04_descriptive_from_processed.py"])


def run_diagnostic() -> None:
    validate_processed_train_exists()
    run_command("diagnostic", [sys.executable, "src/step07_diagnostic_analytics.py"])


def run_ml(retrain_ml: bool) -> None:
    mode = "train" if retrain_ml else "evidence"
    command = [sys.executable, "src/step08_train_champion_v3.py", "--mode", mode]
    if retrain_ml:
        command.extend(["--output-dir", str(RETRAIN_OUTPUT_DIR)])
    run_command("ml", command)


def run_validate() -> None:
    run_command("validate", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"])


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Run the Credit Risk analytics pipeline.")
    parser.add_argument(
        "--stage",
        choices=[*STAGE_ORDER, "all"],
        default="all",
        help="Pipeline stage to run. Default: all.",
    )
    parser.add_argument(
        "--skip-master",
        action="store_true",
        help="Skip the raw-data master-table build and use processed outputs.",
    )
    parser.add_argument(
        "--retrain-ml",
        action="store_true",
        help="Retrain the Step 8 champion model instead of rebuilding evidence tables.",
    )
    args = parser.parse_args()

    stages = STAGE_ORDER if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "master":
            run_master(skip_master=args.skip_master)
        elif stage == "descriptive":
            run_descriptive()
        elif stage == "diagnostic":
            run_diagnostic()
        elif stage == "ml":
            run_ml(retrain_ml=args.retrain_ml)
        elif stage == "validate":
            run_validate()

    logging.info("Pipeline completed for stage(s): %s", ", ".join(stages))
    logging.info("Processed data: %s", PROCESSED_DIR)
    logging.info("Outputs: %s", PROJECT_ROOT / "outputs")
    logging.info("Dashboard: %s", PROJECT_ROOT / "dashboard" / "dashboard.pbix")


if __name__ == "__main__":
    main()
