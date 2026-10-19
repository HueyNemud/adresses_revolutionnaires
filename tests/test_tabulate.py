import csv
import json
import tempfile
import unittest
from pathlib import Path

from numrev.curation import read_csv, write_csv
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
                        "lines": [{"uid": "0.1.0", "line_index": 0, "markdown": "## TITRE"}],
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
                                "word_confidence": [["Dupont,", 0.99], ["rue", 0.44], ["A.", 0.97]],
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

    def test_confidence_is_the_lowest_word_score_and_flags_doubtful_lines(self) -> None:
        rows = self.export_rows()

        self.assertEqual((rows[0]["confiance"], rows[0]["douteuse"]), ("", ""))  # pas de scores
        self.assertEqual((rows[1]["confiance"], rows[1]["douteuse"]), ("0,44", "oui"))

    def test_a_line_above_the_threshold_is_not_doubtful(self) -> None:
        self.document[0]["data_blocks"][1]["lines"][0]["word_confidence"] = [["Dupont,", 0.95], ["rue", 0.9]]
        self.json_path.write_text(json.dumps(self.document), encoding="utf-8")

        self.assertEqual((self.export_rows()[1]["confiance"], self.export_rows()[1]["douteuse"]), ("0,90", ""))

    def test_confidence_columns_are_not_editable_fields(self) -> None:
        csv_path = Path(self.temporary_directory.name) / "output.csv"
        self.export_rows()
        fieldnames, rows = read_csv(csv_path)
        rows[1].update(confiance="1,00", douteuse="")
        write_csv(csv_path, fieldnames, rows)

        rows = self.export_rows()  # pas de CurationConflict : hors de l'empreinte
        self.assertEqual((rows[1]["confiance"], rows[1]["douteuse"]), ("0,44", "oui"))

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
