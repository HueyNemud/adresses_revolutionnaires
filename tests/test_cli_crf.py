import csv
import tempfile
import unittest
from pathlib import Path

from cli_crf import (
    ActiveCRF,
    SourceLine,
    extract_features,
    get_heuristic_label,
    load_session,
    load_source_lines,
    save_session,
)


class ActiveCRFTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            SourceLine(1, "## AGENS D'AFFAIRES"),
            SourceLine(3, "Dupont, rue A, 1."),
            SourceLine(5, "continuation de l'entrée"),
            SourceLine(7, "{12}----------------"),
        ]

    def test_heuristic_only_treats_structural_page_markers_as_noise(self) -> None:
        self.assertEqual(get_heuristic_label("{12}----------------"), "BRUIT")
        self.assertEqual(
            get_heuristic_label("Texte avec {une accolade}"), "DEBUT_ENTREE"
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

    def test_annotations_are_selected_and_trained_in_two_line_blocks(self) -> None:
        crf = ActiveCRF(self.records, seed_size=2)

        self.assertEqual(crf.next_block(), (3, [2, 3]))
        crf.set_labels({2: "SUITE_ENTREE", 3: "BRUIT"})
        self.assertIsNotNone(crf.tagger)
        self.assertEqual(set(crf.tagger.labels()), {"BRUIT", "SUITE_ENTREE"})
        self.assertEqual(crf.features[1]["previous_source_gap"], "True")
        self.assertEqual(crf.next_block(), (1, [0, 1]))

    def test_last_unannotated_line_forms_a_singleton_block(self) -> None:
        crf = ActiveCRF(self.records, seed_size=2)
        crf.set_labels({0: "TITRE", 1: "DEBUT_ENTREE", 2: "SUITE_ENTREE"})

        self.assertEqual(crf.next_block(), (3, [3]))

    def test_csv_preserves_source_lines_and_provenance(self) -> None:
        crf = ActiveCRF(self.records, seed_size=2)
        crf.set_labels({0: "TITRE", 1: "DEBUT_ENTREE"})

        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "predictions.csv"
            crf.export_csv(str(output))
            rows = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(rows[0], "source_line,ligne,classe,provenance,proba")
        self.assertTrue(rows[1].startswith("1,## AGENS D'AFFAIRES,TITRE,human,1.0000"))
        self.assertTrue(rows[3].startswith("5,continuation de l'entrée,"))

    def test_model_confidences_are_normalized_before_csv_export(self) -> None:
        crf = ActiveCRF(self.records, seed_size=2)
        crf.set_labels({0: "TITRE", 1: "DEBUT_ENTREE"})

        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "predictions.csv"
            crf.export_csv(str(output))
            with output.open(encoding="utf-8", newline="") as output_file:
                rows = list(csv.DictReader(output_file))

        model_rows = [row for row in rows if row["provenance"] == "model"]
        self.assertTrue(model_rows)
        self.assertTrue(
            all(0.0 <= float(row["proba"]) <= 1.0 for row in model_rows)
        )

    def test_session_is_bound_to_the_exact_document_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "annuaire.md"
            source.write_text("\n## TITRE\n\nDupont, rue A.\n", encoding="utf-8")
            records, document_hash = load_source_lines(source)
            crf = ActiveCRF(records)
            crf.set_labels({0: "TITRE"})
            session = Path(tmp_dir) / "annuaire.crf-session.json"
            save_session(session, document_hash, crf)

            self.assertEqual(load_session(session, document_hash), ["TITRE", None])
            with self.assertRaises(ValueError):
                load_session(session, "another-document")


if __name__ == "__main__":
    unittest.main()
