import csv
import json
import tempfile
import unittest
from pathlib import Path

from bs4 import BeautifulSoup
from markdownify import markdownify

from tabulate_chandra_output import CSV_FIELDS, load_document, process_json_to_csv


class TabulateChandraOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = [
            {
                "page_index": 0,
                "page_box": [0, 0, 100, 100],
                "chunks": [
                    {"bbox": [0, 0, 33, 100]},
                    {"bbox": [33, 0, 66, 100]},
                    {"bbox": [66, 0, 100, 100]},
                ],
                "raw": (
                    '<div data-bbox="0 0 330 1000" data-label="Text">Avant</div>'
                    '<div data-bbox="330 0 660 1000" data-label="Table">'
                    '<table><tr><th>En-tête</th><td><b>Gauche</b></td></tr>'
                    '<tr><td>Bas gauche</td><td>Bas<br/>droite</td></tr></table></div>'
                    '<div data-bbox="660 0 1000 1000" data-label="Text">Après</div>'
                ),
            }
        ]
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.json_path = Path(self.temporary_directory.name) / "document.json"
        self.json_path.write_text(json.dumps(self.document), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def export_rows(self, no_tables: bool = False) -> list[dict[str, str]]:
        csv_path = Path(self.temporary_directory.name) / "output.csv"
        process_json_to_csv(self.json_path, csv_path, no_tables=no_tables)
        with csv_path.open(encoding="utf-8", newline="") as output_file:
            return list(csv.DictReader(output_file))

    def test_exports_the_expected_columns(self) -> None:
        self.assertEqual(list(self.export_rows()[0]), CSV_FIELDS)

    def test_maps_data_blocks_to_chunks_by_source_order(self) -> None:
        pages = load_document(self.json_path)

        self.assertEqual(
            [block.chunk_index for block in pages[0].data_blocks], [0, 1, 2]
        )

    def test_default_output_preserves_markdown_table_formatting(self) -> None:
        rows = self.export_rows()
        expected_lines = []
        soup = BeautifulSoup(self.document[0]["raw"], "html.parser")
        for block in soup.find_all("div", recursive=False):
            expected_lines.extend(
                markdownify(str(block), heading_style="ATX").splitlines()
            )

        self.assertEqual([row["markdown"] for row in rows], expected_lines)
        self.assertTrue(any("|" in row["markdown"] for row in rows))

    def test_no_tables_explodes_cells_in_logical_reading_order(self) -> None:
        rows = self.export_rows(no_tables=True)
        table_rows = [row for row in rows if row["data_block_label"] == "Table"]

        self.assertEqual(
            [row["markdown"] for row in table_rows],
            ["En-tête", "**Gauche**", "Bas gauche", "Bas droite"],
        )
        self.assertEqual(
            [row["line_index"] for row in table_rows], ["1", "2", "3", "4"]
        )
        self.assertTrue(all("|" not in row["markdown"] for row in table_rows))

    def test_no_tables_preserves_non_table_content_and_provenance(self) -> None:
        rows = self.export_rows(no_tables=True)

        self.assertEqual(
            [row["markdown"] for row in rows],
            [
                "Avant",
                "En-tête",
                "**Gauche**",
                "Bas gauche",
                "Bas droite",
                "Après",
            ],
        )
        self.assertEqual(rows[1]["chunk_index"], "1")
        self.assertEqual(rows[1]["data_block_index"], "2")
        self.assertEqual(rows[1]["data_block_bbox"], "(330.0, 0.0, 660.0, 1000.0)")

    def test_exports_an_empty_chunk_index_when_no_chunk_contains_a_block(self) -> None:
        document = [
            {
                "page_index": 0,
                "page_box": [0, 0, 100, 100],
                "chunks": [],
                "raw": '<div data-bbox="0 0 1000 1000">Texte</div>',
            }
        ]
        json_path = Path(self.temporary_directory.name) / "no_chunks.json"
        json_path.write_text(json.dumps(document), encoding="utf-8")

        csv_path = Path(self.temporary_directory.name) / "no_chunks.csv"
        process_json_to_csv(json_path, csv_path)
        with csv_path.open(encoding="utf-8", newline="") as output_file:
            rows = list(csv.DictReader(output_file))

        self.assertEqual(rows[0]["chunk_index"], "")


if __name__ == "__main__":
    unittest.main()
