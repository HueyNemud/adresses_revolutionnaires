import argparse
import json
import tempfile
import unittest
from pathlib import Path

from numrev.command import CommandError
from numrev.pipeline import extract
from numrev.pipeline.extract import load_document, process_json_to_json


def span(text: str, confidence: str | None = None, bbox: str | None = None) -> str:
    """Une ligne au format de latynna : un `<span>`, avec ses attributs facultatifs."""
    attributes = ""
    if confidence is not None:
        attributes += f' data-word-confidence="{confidence}"'
    if bbox is not None:
        attributes += f' data-line-bbox-estimate="{bbox}"'
    return f"<span{attributes}>{text}</span>"


def page(raw: str, chunks: int = 0, page_index: int = 7) -> dict:
    return {
        "page_index": page_index,
        "page_box": [0, 0, 100, 100],
        "chunks": [{"bbox": [0, 0, 100, 100]}] * chunks,
        "raw": raw,
    }


class ExtractLatynnaLinesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write(self, pages: list[dict]) -> Path:
        json_path = self.directory / "document.ocr.json"
        json_path.write_text(json.dumps(pages), encoding="utf-8")
        return json_path

    def export(self, pages: list[dict]) -> list[dict]:
        output_path = self.directory / "document.lines.json"
        process_json_to_json(self.write(pages), output_path)
        return json.loads(output_path.read_text(encoding="utf-8"))

    def lines(self, raw: str) -> list[dict]:
        return [line for block in self.export([page(raw)])[0]["data_blocks"] for line in block["lines"]]

    def test_output_is_a_copy_of_the_input_with_data_blocks_added(self) -> None:
        document = [
            page(
                '<div data-bbox="0 0 500 1000" data-label="Section-Header"><h2>' + span("TITRE") + "</h2></div>"
                '<div data-bbox="500 0 1000 1000" data-label="Text"><p>' + span("Brou, rue A.") + "</p></div>",
                chunks=2,
            )
        ]
        pages = self.export(document)

        for key, value in document[0].items():
            self.assertEqual(pages[0][key], value)
        self.assertEqual([block["label"] for block in pages[0]["data_blocks"]], ["Section-Header", "Text"])
        self.assertEqual([block["chunk_index"] for block in pages[0]["data_blocks"]], [0, 1])

    def test_one_span_is_one_line_with_one_based_indices(self) -> None:
        lines = self.lines(
            '<div data-bbox="0 0 1000 1000" data-label="List-Group"><p>'
            + span("Brou, R. Mouffertard, 24.")
            + "<br/>"
            + span("Brunet, palais du Luxembourg, 1201.")
            + "</p></div>"
        )

        self.assertEqual([line["markdown"] for line in lines], ["Brou, R. Mouffertard, 24.", "Brunet, palais du Luxembourg, 1201."])
        self.assertEqual([line["line_index"] for line in lines], [1, 2])
        self.assertEqual([line["uid"] for line in lines], ["7.1.1", "7.1.2"])
        self.assertTrue(all(line["cle"] for line in lines))

    def test_heading_level_comes_from_the_enclosing_h_tag(self) -> None:
        lines = self.lines('<div data-bbox="0 0 1000 1000" data-label="Section-Header"><h3>' + span("NÉGOCIANS.") + "</h3></div>")

        self.assertEqual([line["markdown"] for line in lines], ["### NÉGOCIANS."])

    def test_table_cells_are_lines_like_any_other(self) -> None:
        lines = self.lines(
            '<div data-bbox="0 0 1000 1000" data-label="Table"><table>'
            "<tr><td>" + span("Gauche") + "</td><td>" + span("Bas") + "<br/>" + span("droite") + "</td></tr>"
            "</table></div>"
        )

        self.assertEqual([line["markdown"] for line in lines], ["Gauche", "Bas", "droite"])

    def test_each_line_carries_its_own_formatting(self) -> None:
        lines = self.lines(
            '<div data-bbox="0 0 1000 1000" data-label="Text"><p>'
            + span("Foucher, (<i>de la Société des inventions et</i>")
            + "<br/>"
            + span("<i>découvertes</i>) rue de la Ferronnerie, 4.")
            + "</p></div>"
        )

        self.assertEqual(
            [line["markdown"] for line in lines],
            ["Foucher, (*de la Société des inventions et*", "*découvertes*) rue de la Ferronnerie, 4."],
        )

    def test_a_span_inside_a_line_span_is_not_a_line(self) -> None:
        lines = self.lines('<div data-bbox="0 0 1000 1000" data-label="Text"><p>' + span("Brou, <span>rue</span> A.") + "</p></div>")

        self.assertEqual([line["markdown"] for line in lines], ["Brou, rue A."])

    def test_empty_spans_give_no_line(self) -> None:
        lines = self.lines('<div data-bbox="0 0 1000 1000" data-label="Text"><p>' + span(" ") + "<br/>" + span("Brou.") + "</p></div>")

        self.assertEqual([(line["line_index"], line["markdown"]) for line in lines], [(1, "Brou.")])

    def test_image_blocks_give_no_line_but_keep_block_numbering(self) -> None:
        blocks = self.export(
            [
                page(
                    '<div data-bbox="0 0 1000 500" data-label="Image">' + span("Gravure d'un vaisseau") + "</div>"
                    '<div data-bbox="0 500 1000 1000" data-label="Text"><p>' + span("Brou.") + "</p></div>"
                )
            ]
        )[0]["data_blocks"]

        self.assertEqual([block["lines"] for block in blocks][0], [])
        self.assertEqual(blocks[1]["lines"][0]["uid"], "7.2.1")

    def test_word_confidence_pairs_visible_words_with_their_scores(self) -> None:
        lines = self.lines(
            '<div data-bbox="0 0 1000 1000" data-label="Text"><p>'
            + span("Brou, <i>R.</i> Mouffertard,&nbsp;24.", confidence="0.99 0.97 0.44 0.98", bbox="91 241 702 268")
            + "</p></div>"
        )

        self.assertEqual(lines[0]["word_confidence"], [["Brou,", 0.99], ["R.", 0.97], ["Mouffertard,", 0.44], ["24.", 0.98]])
        self.assertEqual(lines[0]["line_bbox_estimate"], [91, 241, 702, 268])

    def test_scores_are_dropped_when_their_count_differs_from_the_words(self) -> None:
        lines = self.lines(
            '<div data-bbox="0 0 1000 1000" data-label="Text"><p>' + span("Brou, rue A.", confidence="0.99 0.97") + "</p></div>"
        )

        self.assertNotIn("word_confidence", lines[0])
        self.assertNotIn("line_bbox_estimate", lines[0])

    def test_text_outside_any_span_is_refused_with_its_location(self) -> None:
        with self.assertRaisesRegex(ValueError, r"page 7, bloc 1 : texte hors de toute balise <span> : « LISTES PARTICULIÈRES »"):
            self.export([page('<div data-bbox="0 0 1000 1000" data-label="Text"><p>LISTES PARTICULIÈRES</p></div>')])

    def test_exports_an_empty_chunk_index_when_no_chunk_contains_a_block(self) -> None:
        document = [page('<div data-bbox="0 0 1000 1000" data-label="Text">' + span("Texte") + "</div>", chunks=0)]

        self.assertIsNone(load_document(self.write(document))[0].data_blocks[0].chunk_index)


class ExtractVolumeTests(unittest.TestCase):
    def test_a_failing_range_leaves_the_whole_volume_unwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            volume = Path(tmp) / "V"
            good = '<div data-bbox="0 0 1000 1000" data-label="Text"><p>' + span("Brou.") + "</p></div>"
            bad = '<div data-bbox="0 0 1000 1000" data-label="Text"><p>Brou.</p></div>'
            for name, raw in (("7-9", good), ("10-12", bad)):
                (volume / name).mkdir(parents=True)
                (volume / name / f"V.{name}.ocr.json").write_text(json.dumps([page(raw)]), encoding="utf-8")

            with self.assertRaisesRegex(CommandError, "V.10-12.ocr.json : page 7, bloc 1"):
                extract.run(argparse.Namespace(folder=volume, apply=True))
            self.assertFalse((volume / "7-9" / "V.7-9.lines.json").exists())


if __name__ == "__main__":
    unittest.main()
