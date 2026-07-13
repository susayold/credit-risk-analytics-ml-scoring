import csv
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
METRICS_FILE = ROOT / "docs" / "METRICS_SOURCE_OF_TRUTH.csv"


def read_csv_rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def source_row(metric: dict[str, str]) -> dict[str, str]:
    source = ROOT / metric["source_file"]
    source_rows = read_csv_rows(source)
    selected = source_rows
    selector_column = metric["selector_column"]
    selector_value = metric["selector_value"]
    if selector_column:
        selected = [row for row in source_rows if row.get(selector_column) == selector_value]
    if len(selected) != 1:
        raise AssertionError(f"Expected one source row for {metric['metric_id']}, found {len(selected)}")
    return selected[0]


def metrics_by_id() -> dict[str, dict[str, str]]:
    return {row["metric_id"]: row for row in read_csv_rows(METRICS_FILE)}


class RepositoryContractTests(unittest.TestCase):
    def test_source_of_truth_schema_uniqueness_and_sources(self):
        rows = read_csv_rows(METRICS_FILE)
        self.assertGreaterEqual(len(rows), 30)
        required = {
            "metric_id",
            "display_metric",
            "metric_owner",
            "value_numeric",
            "value_display",
            "source_file",
            "selector_column",
            "selector_value",
            "source_column",
            "tolerance",
            "portfolio_use",
        }
        self.assertTrue(required.issubset(rows[0].keys()))
        metric_ids = [row["metric_id"] for row in rows]
        self.assertEqual(len(metric_ids), len(set(metric_ids)), "metric_id values must be unique")

        for row in rows:
            self.assertTrue(row["metric_owner"].strip(), row["metric_id"])
            self.assertTrue(row["value_display"].strip(), row["metric_id"])
            self.assertGreaterEqual(float(row["tolerance"]), 0.0, row["metric_id"])
            source = ROOT / row["source_file"]
            self.assertTrue(source.exists(), f"Missing source file for {row['metric_id']}: {source}")
            source_rows = read_csv_rows(source)
            self.assertTrue(source_rows, f"Empty source file for {row['metric_id']}: {source}")
            self.assertIn(row["source_column"], source_rows[0], row["metric_id"])
            if row["selector_column"]:
                self.assertIn(row["selector_column"], source_rows[0], row["metric_id"])
                selected = [r for r in source_rows if r.get(row["selector_column"]) == row["selector_value"]]
                self.assertEqual(len(selected), 1, row["metric_id"])

    def test_metric_values_match_source_tables(self):
        for row in read_csv_rows(METRICS_FILE):
            actual = float(source_row(row)[row["source_column"]])
            expected = float(row["value_numeric"])
            tolerance = float(row["tolerance"])
            self.assertTrue(
                math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance),
                f"{row['metric_id']} expected {expected}, got {actual}",
            )

    def test_readme_headline_metrics_are_sourced(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        metrics = metrics_by_id()
        required_metric_ids = [
            "labeled_rows",
            "final_master_columns",
            "portfolio_default_rate",
            "cc_utilization_gt100_default_rate",
            "cc_utilization_gt100_lift_vs_baseline",
            "bureau_overdue_2plus_default_rate",
            "lightgbm_v3_validation_auc",
            "lightgbm_v3_validation_pr_auc",
            "lightgbm_v3_validation_ks",
            "lightgbm_v3_lift_at_10pct",
            "lightgbm_top30_default_capture",
            "lightgbm_top30_lift",
        ]
        for metric_id in required_metric_ids:
            display = metrics[metric_id]["value_display"]
            self.assertIn(display, readme, f"README missing sourced metric {metric_id}={display}")

    def test_model_cards_match_source_of_truth(self):
        metrics = metrics_by_id()
        champion = json.loads((ROOT / "models" / "champion_model_card.json").read_text(encoding="utf-8"))
        diagnostic = json.loads((ROOT / "models" / "diagnostic_logit_model_card.json").read_text(encoding="utf-8"))

        champion_map = {
            "labeled_rows": champion["training_population"]["labeled_rows"],
            "portfolio_default_rate": champion["training_population"]["baseline_default_rate"],
            "fit_train_rows": champion["split"]["fit_train_rows"],
            "calibration_rows": champion["split"]["calibration_rows"],
            "validation_rows": champion["split"]["validation_rows"],
            "lightgbm_v3_validation_auc": champion["headline_validation_metrics"]["lightgbm_v3_roc_auc"],
            "lightgbm_v3_validation_pr_auc": champion["headline_validation_metrics"]["lightgbm_v3_pr_auc"],
            "lightgbm_v3_validation_ks": champion["headline_validation_metrics"]["lightgbm_v3_ks"],
            "lightgbm_v3_lift_at_10pct": champion["headline_validation_metrics"]["lightgbm_v3_lift_at_10pct"],
            "lightgbm_top30_default_capture": champion["headline_validation_metrics"]["lightgbm_v3_top_30pct_default_capture"],
            "lightgbm_top30_lift": champion["headline_validation_metrics"]["lightgbm_v3_top_30pct_lift"],
        }
        diagnostic_map = {
            "core_logit_auc": diagnostic["model_fit"]["roc_auc"],
            "core_logit_lift_at_10pct": diagnostic["model_fit"]["lift_at_10pct"],
            "core_logit_top10_default_rate": diagnostic["model_fit"]["top_10pct_default_rate"],
            "cc_utilization_gt100_or": diagnostic["selected_odds_ratio_drivers"][0]["odds_ratio"],
            "debt_credit_gt100_or": diagnostic["selected_odds_ratio_drivers"][1]["odds_ratio"],
        }
        for metric_id, actual in {**champion_map, **diagnostic_map}.items():
            expected = float(metrics[metric_id]["value_numeric"])
            tolerance = max(float(metrics[metric_id]["tolerance"]), 1e-12)
            self.assertTrue(math.isclose(float(actual), expected, rel_tol=0, abs_tol=tolerance), metric_id)

        for path in champion["source_files"] + diagnostic["source_files"]:
            self.assertTrue((ROOT / path).exists(), path)

    def test_final_output_manifest_references_existing_files(self):
        for row in read_csv_rows(ROOT / "outputs" / "final" / "final_outputs_manifest.csv"):
            self.assertTrue((ROOT / row["relative_path"]).exists(), row["relative_path"])

    def test_step08_split_ids_match_published_split_rates(self):
        expected = {
            "fit_train_ids.csv": (196805, 0.0774878687025228),
            "calibration_ids.csv": (49202, 0.07747652534449818),
            "final_validation_ids.csv": (61504, 0.09370122268470343),
        }
        all_ids: set[str] = set()
        for filename, (expected_rows, expected_rate) in expected.items():
            rows = read_csv_rows(ROOT / "data" / "splits" / filename)
            self.assertEqual(len(rows), expected_rows, filename)
            ids = [row["SK_ID_CURR"] for row in rows]
            self.assertEqual(len(ids), len(set(ids)), filename)
            self.assertTrue(all(id_ not in all_ids for id_ in ids), filename)
            all_ids.update(ids)
            default_rate = sum(int(float(row["TARGET"])) for row in rows) / len(rows)
            self.assertTrue(math.isclose(default_rate, expected_rate, rel_tol=0, abs_tol=1e-12), filename)
        self.assertEqual(len(all_ids), 307511)

    def test_sql_pipeline_scripts_are_present(self):
        expected = [
            "01_create_base_application.sql",
            "02_cleaning_missing_flags.sql",
            "03_feature_engineering_application.sql",
            "04_aggregate_bureau_and_bureau_balance.sql",
            "05_aggregate_previous_pos_installments_credit_card.sql",
            "06_build_customer_master_table.sql",
            "07_descriptive_statistics_and_segments.sql",
        ]
        for name in expected:
            self.assertTrue((ROOT / "sql" / name).exists(), name)

    def test_model_artifact_manifest_references_committed_evidence(self):
        manifest_rows = read_csv_rows(ROOT / "models" / "model_artifact_manifest.csv")
        self.assertGreaterEqual(len(manifest_rows), 15)
        for row in manifest_rows:
            self.assertTrue((ROOT / row["file_path"]).exists(), row["file_path"])

    def test_runner_help_is_available(self):
        result = subprocess.run(
            [sys.executable, "run_pipeline.py", "--help"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertIn("--stage", result.stdout)
        self.assertIn("--retrain-ml", result.stdout)

    def test_step08_script_uses_fixed_splits_and_current_calibration_api(self):
        text = (ROOT / "src" / "step08_train_champion_v3.py").read_text(encoding="utf-8")
        self.assertIn("load_published_split_frames", text)
        self.assertIn("FrozenEstimator(model)", text)
        self.assertIn("prepare_lgbm_frames", text)
        self.assertIn("median = fit[col].median()", text)
        self.assertNotIn('cv="prefit"', text)
        self.assertNotIn("StratifiedShuffleSplit", text)

    def test_retrain_runner_writes_outside_canonical_final(self):
        spec = importlib.util.spec_from_file_location("run_pipeline_for_test", ROOT / "run_pipeline.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        with patch.object(module, "run_command") as mocked_run:
            module.run_ml(retrain_ml=True)

        stage, command = mocked_run.call_args.args
        self.assertEqual(stage, "ml")
        self.assertIn("--output-dir", command)
        output_dir = command[command.index("--output-dir") + 1]
        self.assertTrue(output_dir.endswith("outputs\\retrain\\latest") or output_dir.endswith("outputs/retrain/latest"))
        self.assertNotIn(str(ROOT / "outputs" / "final"), command)

    def test_step08_evidence_respects_output_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "evidence"
            subprocess.run(
                [
                    sys.executable,
                    "src/step08_train_champion_v3.py",
                    "--mode",
                    "evidence",
                    "--output-dir",
                    str(out_dir),
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            manifest = out_dir / "final_outputs_manifest.csv"
            self.assertTrue(manifest.exists())
            manifest_rows = read_csv_rows(manifest)
            self.assertGreaterEqual(len(manifest_rows), 10)
            self.assertTrue((out_dir / "champion_validation_metrics.csv").exists())

    def test_no_personal_absolute_paths_in_text_artifacts(self):
        blocked = ["D:/Code" + "/DA", "D:" + "\\Code\\DA", "C:" + "\\Users"]
        suffixes = {".py", ".md", ".csv", ".json", ".ipynb", ".yml", ".yaml", ".txt"}
        ignored_parts = {".git", "node_modules", "dist", ".vinext"}
        offenders: list[str] = []
        for path in ROOT.rglob("*"):
            if path.is_dir() or path.suffix.lower() not in suffixes:
                continue
            if any(part in ignored_parts for part in path.parts):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(pattern in text for pattern in blocked):
                offenders.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(offenders, [], "Personal absolute paths found")


if __name__ == "__main__":
    unittest.main()
