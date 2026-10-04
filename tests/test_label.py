import json
import tempfile
import unittest
from pathlib import Path

from numrev.crf.active_learning import ActiveCRF, SourceLine, load_json_lines
from numrev.crf.features import extract_features, get_heuristic_label, normalize_ocr_label
from numrev.crf.labels import CLASSES, AnnotationLabel
from numrev.pipeline.label import load_session_state, save_session


def make_line(uid: str, line_index: int, markdown: str) -> dict:
    return {"uid": uid, "line_index": line_index, "markdown": markdown}


def make_document() -> list[dict]:
    """Build a tabulate_chandra_output-shaped document for the four test records."""
    return [
        {
            "page_index": 0,
            "data_blocks": [
                {
                    "index": 1,
                    "bbox": [0, 0, 1, 1],
                    "label": "Section-Header",
                    "chunk_index": 0,
                    "lines": [make_line("0.1.0", 0, "## AGENS D'AFFAIRES")],
                },
                {
                    "index": 2,
                    "bbox": [0, 0, 1, 1],
                    "label": "Text",
                    "chunk_index": 1,
                    "lines": [make_line("0.2.5", 5, "Dupont, rue A, 1.")],
                },
                {
                    "index": 3,
                    "bbox": [0, 0, 1, 1],
                    "label": "Text",
                    "chunk_index": 2,
                    "lines": [make_line("0.3.6", 6, "continuation de l'entrée")],
                },
            ],
        },
        {
            "page_index": 1,
            "data_blocks": [
                {
                    "index": 1,
                    "bbox": [0, 0, 1, 1],
                    "label": "Text",
                    "chunk_index": 0,
                    "lines": [make_line("1.1.0", 0, "{12}----------------")],
                },
            ],
        },
    ]


class ActiveCRFTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = make_document()
        self.records = [
            SourceLine(
                "0.1.0", 1, "## AGENS D'AFFAIRES", "0", "0", "1", "0",
                page_pos=0, block_pos=0, line_pos=0,
            ),
            SourceLine(
                "0.2.5", 3, "Dupont, rue A, 1.", "0", "1", "2", "5",
                page_pos=0, block_pos=1, line_pos=0,
            ),
            SourceLine(
                "0.3.6", 5, "continuation de l'entrée", "0", "2", "3", "6",
                page_pos=0, block_pos=2, line_pos=0,
            ),
            SourceLine(
                "1.1.0", 7, "{12}----------------", "1", "0", "1", "0",
                page_pos=1, block_pos=0, line_pos=0,
            ),
        ]

    def test_load_json_lines_reads_chandra_provenance_and_skips_blank_markdown(self) -> None:
        document = [
            {
                "page_index": 0,
                "data_blocks": [
                    {
                        "index": 2,
                        "bbox": [1, 2, 3, 4],
                        "label": "Text",
                        "chunk_index": 1,
                        "lines": [
                            make_line("0.2.5", 5, "  Dupont, rue A.  "),
                            make_line("0.2.6", 6, ""),
                        ],
                    }
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "output.json"
            source.write_text(json.dumps(document), encoding="utf-8")
            records, raw_document, document_hash = load_json_lines(source)

        self.assertEqual(len(document_hash), 64)
        self.assertEqual(raw_document, document)
        self.assertEqual(
            ActiveCRF(records).features[0]["ocr_data_block_label"], "text"
        )
        self.assertEqual(records, [
            SourceLine(
                uid="0.2.5",
                source_row=1,
                text="Dupont, rue A.",
                page_index="0",
                chunk_index="1",
                data_block_index="2",
                line_index="5",
                data_block_bbox="[1, 2, 3, 4]",
                data_block_label="Text",
                page_pos=0,
                block_pos=0,
                line_pos=0,
            )
        ])

    def test_load_json_lines_rejects_an_unrelated_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "other.json"
            source.write_text(json.dumps([{"page_index": 0}]), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "data_blocks"):
                load_json_lines(source)

    def test_heuristic_only_treats_separator_lines_as_noise(self) -> None:
        self.assertEqual(get_heuristic_label("----------------"), "OUT OF SCOPE")
        self.assertEqual(get_heuristic_label("**—————**"), "OUT OF SCOPE")
        self.assertEqual(get_heuristic_label("Texte avec {une accolade}"), "B-ENTRY")

    def test_features_use_four_boundary_shapes_without_shape_ngrams(self) -> None:
        features = extract_features(["**Didier, R. du Bac, 12."])[0]

        # Les marqueurs d'emphase Markdown sont ignorés par les formes de tokens.
        self.assertEqual(features["shape_start_0"], "TITLE")
        self.assertEqual(features["shape_start_1"], "PUNCT")
        self.assertEqual(features["shape_start_2"], "UPPER")
        self.assertEqual(features["shape_start_3"], "PUNCT")
        self.assertIn("token_count", features)
        self.assertNotIn("starts_upper", features)
        self.assertNotIn("digit_count", features)
        self.assertFalse(any(key.startswith(("bg_", "tg_")) for key in features))

    def test_features_include_the_normalized_ocr_data_block_label(self) -> None:
        features = extract_features(
            ["## AGENS D'AFFAIRES", "Dupont, rue A, 1."],
            ocr_labels=["Section-Header", " Text "],
        )

        self.assertEqual(normalize_ocr_label("Section Header"), "section_header")
        self.assertEqual(normalize_ocr_label(""), "missing")
        self.assertEqual(features[0]["ocr_data_block_label"], "section_header")
        self.assertEqual(features[1]["ocr_data_block_label"], "text")

    def test_features_reject_misaligned_ocr_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "classe de bloc OCR"):
            extract_features(["Dupont"], ocr_labels=[])

    def test_features_reject_misaligned_source_line_numbers(self) -> None:
        with self.assertRaisesRegex(ValueError, "numéro de ligne source"):
            extract_features(["Dupont"], source_line_numbers=[])

    def test_annotation_labels_have_one_canonical_definition(self) -> None:
        self.assertEqual(
            CLASSES,
            tuple(label.value for label in AnnotationLabel),
        )

    def test_annotations_are_selected_and_trained_in_two_line_blocks(self) -> None:
        crf = ActiveCRF(self.records, self.document, seed_size=2)

        self.assertEqual(crf.next_block(), (1, [0, 1]))
        crf.set_labels({0: "B-TITLE", 1: "B-ENTRY"})
        self.assertIsNotNone(crf.tagger)
        self.assertEqual(set(crf.tagger.labels()), {"B-ENTRY", "B-TITLE"})
        self.assertEqual(crf.next_block(), (2, [2, 3]))

    def test_last_unannotated_line_forms_a_singleton_block(self) -> None:
        crf = ActiveCRF(self.records, self.document, seed_size=2)
        crf.set_labels({0: "B-TITLE", 1: "B-ENTRY", 2: "I-ENTRY"})

        self.assertEqual(crf.next_block(), (3, [3]))

    def test_undo_removes_the_latest_label_and_reoffers_its_line(self) -> None:
        crf = ActiveCRF(self.records, self.document, seed_size=2)
        crf.set_labels({0: "B-TITLE", 1: "B-ENTRY"})
        first_timestamp = crf.label_timestamps[1]

        self.assertEqual(crf.undo_last_label(), 1)
        self.assertEqual(crf.labels, ["B-TITLE", None, None, None])
        self.assertEqual(crf.annotation_history, [0])
        self.assertNotIn(1, crf.label_timestamps)
        selection = crf.next_block(preferred_index=1)

        self.assertIsNotNone(selection)
        target_index, block_indices = selection
        self.assertEqual(target_index, 1)
        # La ligne annulée est reproposée seule, sans voisine, pour ne pas
        # laisser croire que l'annulation a été ignorée.
        self.assertEqual(block_indices, [1])

        crf.set_labels({1: "B-ENTRY"})
        self.assertNotEqual(crf.label_timestamps[1], "")
        self.assertGreaterEqual(crf.label_timestamps[1], first_timestamp)

    def test_json_preserves_chandra_provenance_and_predictions(self) -> None:
        crf = ActiveCRF(self.records, self.document, seed_size=2)
        crf.set_labels({0: "B-TITLE", 1: "B-ENTRY"})

        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "predictions.json"
            crf.export_json(str(output))
            pages = json.loads(output.read_text(encoding="utf-8"))

        first_line = pages[0]["data_blocks"][0]["lines"][0]
        third_line = pages[0]["data_blocks"][2]["lines"][0]

        self.assertEqual(first_line["markdown"], "## AGENS D'AFFAIRES")
        self.assertEqual(first_line["prediction"], "B-TITLE")
        self.assertEqual(first_line["provenance"], "human")
        self.assertEqual(first_line["timestamp"], crf.label_timestamps[0])
        self.assertEqual(third_line["line_index"], 6)
        self.assertEqual(third_line["markdown"], "continuation de l'entrée")
        self.assertTrue(third_line["timestamp"])
        self.assertEqual(third_line["provenance"], "model")

    def test_model_confidences_are_normalized_before_json_export(self) -> None:
        crf = ActiveCRF(self.records, self.document, seed_size=2)
        crf.set_labels({0: "B-TITLE", 1: "B-ENTRY"})

        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "predictions.json"
            crf.export_json(str(output))
            pages = json.loads(output.read_text(encoding="utf-8"))

        model_lines = [
            line
            for page in pages
            for block in page["data_blocks"]
            for line in block["lines"]
            if line.get("provenance") == "model"
        ]
        self.assertTrue(model_lines)
        self.assertTrue(
            all(0.0 <= line["probability"] <= 1.0 for line in model_lines)
        )

    def test_close_removes_the_temporary_model_file(self) -> None:
        crf = ActiveCRF(self.records, self.document, seed_size=2)
        crf.set_labels({0: "B-TITLE", 1: "B-ENTRY"})

        self.assertTrue(crf.model_path.exists())
        crf.close()
        self.assertFalse(crf.model_path.exists())

    def test_session_is_bound_to_the_exact_document_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            document = [
                {
                    "page_index": 0,
                    "data_blocks": [
                        {
                            "index": 1,
                            "bbox": [0, 0, 1, 1],
                            "label": "",
                            "chunk_index": 0,
                            "lines": [make_line("0.1.0", 0, "## TITRE")],
                        },
                        {
                            "index": 2,
                            "bbox": [0, 0, 1, 1],
                            "label": "",
                            "chunk_index": 1,
                            "lines": [make_line("0.2.1", 1, "Dupont, rue A.")],
                        },
                    ],
                }
            ]
            source = Path(tmp_dir) / "annuaire.json"
            source.write_text(json.dumps(document), encoding="utf-8")
            records, raw_document, document_hash = load_json_lines(source)
            crf = ActiveCRF(records, raw_document)
            crf.set_labels({0: "B-TITLE"})
            session = Path(tmp_dir) / "annuaire.label-session.json"
            save_session(session, document_hash, crf)

            self.assertEqual(
                load_session_state(session, document_hash),
                (["B-TITLE", None], [0], {0: crf.label_timestamps[0]}),
            )
            with self.assertRaises(ValueError):
                load_session_state(session, "another-document")


if __name__ == "__main__":
    unittest.main()
