import csv
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS_FILE = ROOT / "docs" / "METRICS_SOURCE_OF_TRUTH.csv"


def read_csv_rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class RepositoryContractTests(unittest.TestCase):
    def test_metric_sources_exist(self):
        rows = read_csv_rows(METRICS_FILE)
        self.assertGreaterEqual(len(rows), 20)
        for row in rows:
            source = ROOT / row["source_file"]
            self.assertTrue(source.exists(), f"Missing source file for {row['metric_id']}: {source}")

    def test_metric_values_match_source_tables(self):
        rows = read_csv_rows(METRICS_FILE)
        for row in rows:
            source = ROOT / row["source_file"]
            source_rows = read_csv_rows(source)
            selected = source_rows
            selector_column = row["selector_column"]
            selector_value = row["selector_value"]
            if selector_column:
                selected = [r for r in source_rows if r.get(selector_column) == selector_value]
            self.assertEqual(len(selected), 1, f"Expected one source row for {row['metric_id']}")
            actual = float(selected[0][row["source_column"]])
            expected = float(row["value_numeric"])
            tolerance = float(row["tolerance"])
            self.assertTrue(
                math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance),
                f"{row['metric_id']} expected {expected}, got {actual}",
            )

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

    def test_model_cards_reference_committed_evidence(self):
        manifest_rows = read_csv_rows(ROOT / "models" / "model_artifact_manifest.csv")
        self.assertGreaterEqual(len(manifest_rows), 10)
        for row in manifest_rows:
            self.assertTrue((ROOT / row["file_path"]).exists(), row["file_path"])


if __name__ == "__main__":
    unittest.main()
