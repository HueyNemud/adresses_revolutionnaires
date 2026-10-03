import json
import tempfile
import unittest
from pathlib import Path


from extract_chandra_lines import load_document, process_json_to_json


class ExtractChandraLinesTests(unittest.TestCase):
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

    def export_pages(self) -> list[dict]:
        output_path = Path(self.temporary_directory.name) / "output.json"
        process_json_to_json(self.json_path, output_path)
        return json.loads(output_path.read_text(encoding="utf-8"))

    def test_output_is_a_copy_of_the_input_with_data_blocks_added(self) -> None:
        pages = self.export_pages()

        self.assertEqual(len(pages), 1)
        page = pages[0]
        for key, value in self.document[0].items():
            self.assertEqual(page[key], value)
        self.assertIn("data_blocks", page)
        self.assertEqual(
            [block["label"] for block in page["data_blocks"]],
            ["Text", "Table", "Text"],
        )

    def test_maps_data_blocks_to_chunks_by_source_order(self) -> None:
        pages = load_document(self.json_path)

        self.assertEqual(
            [block.chunk_index for block in pages[0].data_blocks], [0, 1, 2]
        )

    def test_tables_are_exploded_into_cells_and_their_lines_in_reading_order(self) -> None:
        pages = self.export_pages()
        table_block = next(
            block for block in pages[0]["data_blocks"] if block["label"] == "Table"
        )

        self.assertEqual(
            [line["markdown"] for line in table_block["lines"]],
            # `Bas<br/>droite` : chaque ligne d'une cellule devient une ligne.
            ["En-tête", "**Gauche**", "Bas gauche", "Bas", "droite"],
        )
        self.assertEqual(
            [line["line_index"] for line in table_block["lines"]], [0, 1, 2, 3, 4]
        )
        self.assertTrue(
            all("|" not in line["markdown"] for line in table_block["lines"])
        )

    def test_table_explosion_preserves_non_table_content_and_provenance(self) -> None:
        pages = self.export_pages()
        blocks = pages[0]["data_blocks"]

        self.assertEqual(
            [line["markdown"] for block in blocks for line in block["lines"]],
            [
                "Avant",
                "En-tête",
                "**Gauche**",
                "Bas gauche",
                "Bas",
                "droite",
                "Après",
            ],
        )
        table_block = blocks[1]
        self.assertEqual(table_block["chunk_index"], 1)
        self.assertEqual(table_block["index"], 2)
        self.assertEqual(table_block["bbox"], [330.0, 0.0, 660.0, 1000.0])

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

        output_path = Path(self.temporary_directory.name) / "no_chunks.json.out"
        process_json_to_json(json_path, output_path)
        pages = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertIsNone(pages[0]["data_blocks"][0]["chunk_index"])


if __name__ == "__main__":
    unittest.main()
