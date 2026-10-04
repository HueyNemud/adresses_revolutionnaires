import csv
import json
import tempfile
import unittest
from pathlib import Path

from numrev.pipeline.extract import assign_line_keys
from numrev.pipeline.tabulate import CSV_FIELDS, process_json_to_csv


class JsonToCsvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = [
            {
                "page_index": 0,
                "data_blocks": [
                    {
                        "index": 1,
                        "bbox": [0, 0, 1, 1],
                        "label": "Section-Header",
                        "chunk_index": 0,
                        "lines": [
                            {"uid": "0.1.0", "line_index": 0, "markdown": "## TITRE"}
                        ],
                    },
                    {
                        "index": 2,
                        "bbox": [0, 0, 1, 1],
                        "label": "Text",
                        "chunk_index": 1,
                        "lines": [
                            {
                                "uid": "0.2.1",
                                "line_index": 1,
                                "markdown": "Dupont, rue A.",
                                "prediction": "ENTRY_BEGIN",
                                "provenance": "human",
                                "probability": 1.0,
                                "timestamp": "2026-09-08 10:00:00",
                            }
                        ],
                    },
                ],
            }
        ]
        assign_line_keys(self.document)
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.json_path = Path(self.temporary_directory.name) / "document.json"
        self.json_path.write_text(json.dumps(self.document), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def export_rows(self) -> list[dict[str, str]]:
        csv_path = Path(self.temporary_directory.name) / "output.csv"
        process_json_to_csv(self.json_path, csv_path, Path(self.temporary_directory.name) / "patch.csv")
        with csv_path.open(encoding="utf-8", newline="") as output_file:
            return list(csv.DictReader(output_file))

    def test_exports_the_expected_columns(self) -> None:
        self.assertEqual(list(self.export_rows()[0]), CSV_FIELDS)

    def test_flattens_lines_with_provenance(self) -> None:
        rows = self.export_rows()

        self.assertEqual(rows[0]["markdown"], "## TITRE")
        self.assertEqual(rows[0]["data_block_index"], "1")
        self.assertEqual(rows[0]["chunk_index"], "0")
        self.assertEqual(rows[0]["classe"], "")

        self.assertEqual(rows[1]["markdown"], "Dupont, rue A.")
        self.assertEqual(rows[1]["classe"], "ENTRY_BEGIN")
        self.assertEqual(rows[1]["provenance"], "human")
        self.assertEqual(rows[1]["probability"], "1.0")
        self.assertEqual(rows[1]["timestamp"], "2026-09-08 10:00:00")

    def test_rejects_a_json_missing_data_blocks(self) -> None:
        json_path = Path(self.temporary_directory.name) / "invalid.json"
        json_path.write_text(json.dumps([{"page_index": 0}]), encoding="utf-8")
        csv_path = Path(self.temporary_directory.name) / "invalid.csv"

        with self.assertRaisesRegex(ValueError, "data_blocks"):
            process_json_to_csv(json_path, csv_path)

    def test_rejects_a_json_without_line_keys(self) -> None:
        for line in self.document[0]["data_blocks"][0]["lines"]:
            del line["cle"]
        self.json_path.write_text(json.dumps(self.document), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "sans clé"):
            self.export_rows()


if __name__ == "__main__":
    unittest.main()
