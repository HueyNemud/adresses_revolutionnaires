import csv
import tempfile
import unittest
from pathlib import Path

from build_entity_tree import (
    ROOT_UUID,
    alpha_sort_key,
    assign_entity_ids,
    document_name,
    format_title_tree,
    process_csv,
)

FIELDS = ["uid", "markdown", "prediction_curated"]
ROWS = [
    ("1.1.0", "Dupont, rue A, 1.", "B-ENTRY"),
    ("1.1.1", "suite", "I-ENTRY"),
    ("1.1.2", "Boulanger, rue B, 2.", "B-ENTRY"),
    ("1.1.2", "Boulanger (Ve.), rue C, 3.", "B-ENTRY"),  # ligne dupliquée à la curation
]


def merge(directory: Path, rows=ROWS) -> list[dict[str, str]]:
    source = directory / "Vol.1-9.ocr.lines.annotated.curated.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    output = directory / "out.csv"
    process_csv(source, output, "prediction_curated", "uid")
    with output.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class EntityIdTests(unittest.TestCase):
    def test_document_name(self):
        self.assertEqual(document_name(Path("1808_AD75-PER292.6-185.ocr.lines.annotated.curated.csv")), "1808_AD75-PER292.6-185")
        self.assertEqual(document_name(Path("autre.csv")), "autre")

    def test_ids_are_stable_unique_and_independent_of_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = merge(Path(tmp))
            second = merge(Path(tmp))
            edited = merge(Path(tmp), [(uid, text.replace("rue", "R."), label) for uid, text, label in ROWS])
        ids = [row["uuid"] for row in first]
        self.assertEqual(len(ids), 3)
        self.assertEqual(len(set(ids)), 3)
        self.assertEqual(ids, [row["uuid"] for row in second])
        self.assertEqual(ids, [row["uuid"] for row in edited])

    def test_document_distinguishes_identical_uids(self):
        a, b = [{"uid": "1.1.0"}], [{"uid": "1.1.0"}]
        assign_entity_ids(a, "vol-A", "uid")
        assign_entity_ids(b, "vol-B", "uid")
        self.assertNotEqual(a[0]["uuid"], b[0]["uuid"])


class AlphaSortKeyTests(unittest.TestCase):
    def test_two_first_words_truncated(self):
        self.assertEqual(alpha_sort_key("Brunet ( J. B. ) , R. Jacob , 24."), "BRUNE")
        self.assertEqual(alpha_sort_key("**Coste** ( Louis ), R. de l'Université"), "COSTE")
        self.assertEqual(alpha_sort_key("Le Roux, R. S. Denis"), "LEROU")
        self.assertEqual(alpha_sort_key("D'Erlach (Me.), R. de Turenne"), "DERLA")
        self.assertEqual(alpha_sort_key("Adam, Pont-Neuf, 2."), "ADAM")
        self.assertEqual(alpha_sort_key("Lyon, *marchand de bijoux*"), "LYON")

    def test_homonyms_are_not_violations(self):
        rows = [
            ("1.1.0", "Brunet (P.), rue A, 1.", "B-ENTRY"),
            ("1.1.1", "Brunet (J. B.), rue B, 2.", "B-ENTRY"),
            ("1.1.2", "Bonnet (Ant.), rue C, 3.", "B-ENTRY"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "in.csv"
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(FIELDS)
                writer.writerows(rows)
            report = process_csv(source, Path(tmp) / "out.csv", "prediction_curated", "uid")
        self.assertEqual([uid for uid, *_ in report.alpha_violations], ["1.1.2"])


class ParentIdTests(unittest.TestCase):
    def test_title_tree_and_entry_parents(self):
        rows = [
            ("1.1.0", "Avant tout titre, rue A, 1.", "B-ENTRY"),
            ("1.1.1", "# LIVRE", "B-TITLE"),
            ("1.1.2", "## Section A", "B-TITLE"),
            ("1.1.3", "(suite du titre)", "I-TITLE"),
            ("1.1.4", "Page 12", "OUT OF SCOPE"),
            ("1.1.5", "Adam, rue B, 2.", "B-ENTRY"),
            ("1.1.6", "Sous-titre sans dièse", "B-TITLE"),
            ("1.1.7", "Bernard, rue C, 3.", "B-ENTRY"),
            ("1.1.8", "## Section B", "B-TITLE"),
            ("1.1.9", "Coste, rue D, 4.", "B-ENTRY"),
            ("1.2.0", "# AUTRE LIVRE", "B-TITLE"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = {row["uid"].split(",")[0]: row for row in merge(Path(tmp), rows)}
        uuid_of = {uid: row["uuid"] for uid, row in out.items()}
        expected = {
            "1.1.0": ROOT_UUID,
            "1.1.1": ROOT_UUID,
            "1.1.2": uuid_of["1.1.1"],
            "1.1.4": "",
            "1.1.5": uuid_of["1.1.2"],
            "1.1.6": uuid_of["1.1.2"],
            "1.1.7": uuid_of["1.1.6"],
            "1.1.8": uuid_of["1.1.1"],
            "1.1.9": uuid_of["1.1.8"],
            "1.2.0": ROOT_UUID,
        }
        self.assertEqual({uid: row["parent_uuid"] for uid, row in out.items()}, expected)
        self.assertEqual(ROOT_UUID, "00000000-0000-0000-0000-000000000000")


class TitleTreeReportTests(unittest.TestCase):
    def test_tree_counts_direct_and_recursive_entries(self):
        rows = [
            ("1.1.0", "Avant tout titre, rue A, 1.", "B-ENTRY"),
            ("1.1.1", "# LIVRE", "B-TITLE"),
            ("1.1.2", "## Section A", "B-TITLE"),
            ("1.1.3", "Adam, rue B, 2.", "B-ENTRY"),
            ("1.1.4", "Bernard, rue C, 3.", "B-ENTRY"),
            ("1.1.5", "### Sous-section", "B-TITLE"),
            ("1.1.6", "Coste, rue D, 4.", "B-ENTRY"),
            ("1.1.7", "## Section B", "B-TITLE"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "in.csv"
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(FIELDS)
                writer.writerows(rows)
            report = process_csv(source, Path(tmp) / "out.csv", "prediction_curated", "uid")
        self.assertEqual(
            format_title_tree(report),
            [
                "[racine] — 4 (1)",
                "  # LIVRE — 3 (0)",
                "    ## Section A — 3 (2)",
                "      ### Sous-section — 1 (1)",
                "    ## Section B — 0 (0)",
            ],
        )


if __name__ == "__main__":
    unittest.main()
