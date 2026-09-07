import csv
import tempfile
import unittest
from pathlib import Path

from cli_crf import (
    ActiveCRF,
    INPUT_CSV_FIELDS,
    SourceLine,
    extract_features,
    get_heuristic_label,
    load_csv_lines,
    load_session,
    load_session_state,
    normalize_ocr_label,
    save_session,
)


class ActiveCRFTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            SourceLine(1, "## AGENS D'AFFAIRES", "0", "0", "1", "0"),
            SourceLine(3, "Dupont, rue A, 1.", "0", "1", "2", "5"),
            SourceLine(5, "continuation de l'entrée", "0", "2", "3", "6"),
            SourceLine(7, "{12}----------------", "1", "0", "1", "0"),
        ]

    def test_load_csv_lines_reads_chandra_provenance_and_skips_blank_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "output.csv"
            with source.open("w", encoding="utf-8", newline="") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=INPUT_CSV_FIELDS)
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "uid": "0.2.5",
                            "page_index": "0",
                            "chunk_index": "1",
                            "data_block_index": "2",
                            "line_index": "5",
                            "data_block_bbox": "(1, 2, 3, 4)",
                            "data_block_label": "Text",
                            "markdown": "  Dupont, rue A.  ",
                        },
                        {
                            "uid": "0.2.6",
                            "page_index": "0",
                            "chunk_index": "1",
                            "data_block_index": "2",
                            "line_index": "6",
                            "data_block_bbox": "(1, 2, 3, 4)",
                            "data_block_label": "Text",
                            "markdown": "",
                        },
                    ]
                )
            records, document_hash = load_csv_lines(source)

        self.assertEqual(len(document_hash), 64)
        self.assertEqual(
            ActiveCRF(records).features[0]["ocr_data_block_label"], "text"
        )
        self.assertEqual(records, [
            SourceLine(
                source_row=1,
                text="Dupont, rue A.",
                page_index="0",
                chunk_index="1",
                data_block_index="2",
                line_index="5",
                data_block_bbox="(1, 2, 3, 4)",
                data_block_label="Text",
            )
        ])

    def test_load_csv_lines_rejects_an_unrelated_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "other.csv"
            source.write_text("text\nDupont\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "colonnes manquantes"):
                load_csv_lines(source)

    def test_heuristic_only_treats_structural_page_markers_as_noise(self) -> None:
        self.assertEqual(get_heuristic_label("{12}----------------"), "OUT_OF_SCOPE")
        self.assertEqual(
            get_heuristic_label("Texte avec {une accolade}"), "ENTRY_BEGIN"
        )

    def test_features_use_four_boundary_shapes_without_shape_ngrams(self) -> None:
        features = extract_features(["**Didier, R. du Bac, 12."])[0]

        self.assertEqual(features["shape_start_0"], "SYM")
        self.assertEqual(features["shape_start_1"], "SYM")
        self.assertEqual(features["shape_start_2"], "TITLE")
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

    def test_annotations_are_selected_and_trained_in_two_line_blocks(self) -> None:
        crf = ActiveCRF(self.records, seed_size=2)

        self.assertEqual(crf.next_block(), (1, [0, 1]))
        crf.set_labels({0: "TITLE", 1: "ENTRY_BEGIN"})
        self.assertIsNotNone(crf.tagger)
        self.assertEqual(set(crf.tagger.labels()), {"ENTRY_BEGIN", "TITLE"})
        self.assertEqual(crf.features[1]["previous_source_gap"], "True")
        self.assertEqual(crf.next_block(), (2, [2, 3]))

    def test_last_unannotated_line_forms_a_singleton_block(self) -> None:
        crf = ActiveCRF(self.records, seed_size=2)
        crf.set_labels({0: "TITLE", 1: "ENTRY_BEGIN", 2: "ENTRY_INSIDE"})

        self.assertEqual(crf.next_block(), (3, [3]))

    def test_undo_removes_the_latest_label_and_reoffers_its_line(self) -> None:
        crf = ActiveCRF(self.records, seed_size=2)
        crf.set_labels({0: "TITLE", 1: "ENTRY_BEGIN"})

        self.assertEqual(crf.undo_last_label(), 1)
        self.assertEqual(crf.labels, ["TITLE", None, None, None])
        self.assertEqual(crf.annotation_history, [0])
        selection = crf.next_block(preferred_index=1)

        self.assertIsNotNone(selection)
        target_index, block_indices = selection
        self.assertEqual(target_index, 1)
        self.assertIn(1, block_indices)

    def test_csv_preserves_chandra_provenance_and_predictions(self) -> None:
        crf = ActiveCRF(self.records, seed_size=2)
        crf.set_labels({0: "TITLE", 1: "ENTRY_BEGIN"})

        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "predictions.csv"
            crf.export_csv(str(output))
            with output.open(encoding="utf-8", newline="") as output_file:
                rows = list(csv.DictReader(output_file))

        self.assertEqual(
            list(rows[0]),
            [*INPUT_CSV_FIELDS, "prediction", "provenance", "probability"],
        )
        self.assertEqual(rows[0]["page_index"], "0")
        self.assertEqual(rows[0]["data_block_index"], "1")
        self.assertEqual(rows[0]["markdown"], "## AGENS D'AFFAIRES")
        self.assertEqual(rows[0]["prediction"], "TITLE")
        self.assertEqual(rows[0]["provenance"], "human")
        self.assertEqual(rows[2]["line_index"], "6")
        self.assertEqual(rows[2]["markdown"], "continuation de l'entrée")

    def test_model_confidences_are_normalized_before_csv_export(self) -> None:
        crf = ActiveCRF(self.records, seed_size=2)
        crf.set_labels({0: "TITLE", 1: "ENTRY_BEGIN"})

        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "predictions.csv"
            crf.export_csv(str(output))
            with output.open(encoding="utf-8", newline="") as output_file:
                rows = list(csv.DictReader(output_file))

        model_rows = [row for row in rows if row["provenance"] == "model"]
        self.assertTrue(model_rows)
        self.assertTrue(
            all(0.0 <= float(row["probability"]) <= 1.0 for row in model_rows)
        )

    def test_session_is_bound_to_the_exact_document_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "annuaire.csv"
            with source.open("w", encoding="utf-8", newline="") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=INPUT_CSV_FIELDS)
                writer.writeheader()
                writer.writerows(
                    [
                        dict(zip(INPUT_CSV_FIELDS, ["0.1.0", "0", "0", "1", "0", "", "", "## TITRE"])),
                        dict(zip(INPUT_CSV_FIELDS, ["0.2.1", "0", "1", "2", "1", "", "", "Dupont, rue A."])),
                    ]
                )
            records, document_hash = load_csv_lines(source)
            crf = ActiveCRF(records)
            crf.set_labels({0: "TITLE"})
            session = Path(tmp_dir) / "annuaire.crf-session.json"
            save_session(session, document_hash, crf)

            self.assertEqual(load_session(session, document_hash), ["TITLE", None])
            self.assertEqual(
                load_session_state(session, document_hash), (["TITLE", None], [0])
            )
            with self.assertRaises(ValueError):
                load_session(session, "another-document")


if __name__ == "__main__":
    unittest.main()
