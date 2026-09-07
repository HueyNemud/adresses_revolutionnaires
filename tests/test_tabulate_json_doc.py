import csv
import json
import tempfile
import unittest
from pathlib import Path

from bs4 import BeautifulSoup
from markdownify import markdownify

from tabulate_json_doc import CSV_FIELDS, load_document, process_json_to_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_JSON = PROJECT_ROOT / "delete_me_asap.json"


class TabulateJsonDocTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_pages = json.loads(SAMPLE_JSON.read_text(encoding="utf-8"))
        cls.pages = load_document(SAMPLE_JSON)

        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.output_path = Path(cls.temporary_directory.name) / "output.csv"
        process_json_to_csv(SAMPLE_JSON, cls.output_path)

        with cls.output_path.open(encoding="utf-8", newline="") as output_file:
            cls.rows = list(csv.DictReader(output_file))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_directory.cleanup()

    def rows_for_page(self, page_index: int) -> list[dict[str, str]]:
        return [row for row in self.rows if row["page_index"] == str(page_index)]

    def test_exports_the_expected_columns(self) -> None:
        self.assertEqual(list(self.rows[0]), CSV_FIELDS)

    def test_maps_data_blocks_to_chunks_by_source_order(self) -> None:
        for page in self.pages:
            self.assertEqual(
                [block.chunk_index for block in page.data_blocks],
                list(range(len(page.data_blocks))),
            )

    def test_exports_every_line_of_each_data_block_markdown(self) -> None:
        expected_lines = []
        for raw_page in self.raw_pages:
            soup = BeautifulSoup(raw_page["raw"], "html.parser")
            expected_lines.extend(
                markdownify(str(block), heading_style="ATX").splitlines()
                for block in soup.find_all("div", recursive=False)
            )

        actual_lines = [row["markdown"] for row in self.rows]
        self.assertEqual(
            actual_lines,
            [line for block_lines in expected_lines for line in block_lines],
        )

    def test_keeps_block_metadata_for_each_markdown_line(self) -> None:
        rows = self.rows_for_page(0)
        self.assertEqual(
            [
                (row["chunk_index"], row["data_block_index"], row["line_index"])
                for row in rows[:6]
            ],
            [
                ("0", "1", "0"),
                ("0", "1", "1"),
                ("0", "1", "2"),
                ("0", "1", "3"),
                ("0", "1", "4"),
                ("1", "2", "5"),
            ],
        )

    def test_exports_an_empty_chunk_index_when_no_chunk_contains_a_block(self) -> None:
        document = [
            {
                "page_index": 0,
                "page_box": [0, 0, 100, 100],
                "chunks": [],
                "raw": '<div data-bbox="0 0 1000 1000">Texte</div>',
            }
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            json_path = Path(temporary_directory) / "document.json"
            csv_path = Path(temporary_directory) / "document.csv"
            json_path.write_text(json.dumps(document), encoding="utf-8")
            process_json_to_csv(json_path, csv_path)

            with csv_path.open(encoding="utf-8", newline="") as output_file:
                rows = list(csv.DictReader(output_file))
            self.assertEqual(rows[0]["chunk_index"], "")


if __name__ == "__main__":
    unittest.main()
