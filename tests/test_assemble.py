import csv
import tempfile
import unittest
from pathlib import Path

from numrev.pipeline.assemble import (
    alpha_sort_key,
    assign_entity_ids,
    format_title_tree,
    process_csv,
)
from numrev.titles import ROOT_UUID

FIELDS = ["cle", "uid", "markdown", "classe", "corrige", "empreinte"]
ROWS = [
    ("1.1.0", "Dupont, rue A, 1.", "B-ENTRY"),
    ("1.1.1", "suite", "I-ENTRY"),
    ("1.1.2", "Boulanger, rue B, 2.", "B-ENTRY"),
    ("1.1.2", "Boulanger (Ve.), rue C, 3.", "B-ENTRY", "k1.1.2+1"),  # ligne dupliquée à la curation
]


def with_keys(rows) -> list[tuple[str, ...]]:
    """(cle, uid, texte, classe, corrige, empreinte) ; clé `k<uid>` par défaut."""
    return [(row[3] if len(row) > 3 else f"k{row[0]}", *row[:3], "", "") for row in rows]


def write_source(source: Path, rows) -> None:
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(with_keys(rows))


def merge(directory: Path, rows=ROWS) -> list[dict[str, str]]:
    source = directory / "Vol.1-9.lines.csv"
    write_source(source, rows)
    output = directory / "out.csv"
    process_csv(source, output)
    with output.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class EntityIdTests(unittest.TestCase):
    def test_ids_are_stable_unique_and_independent_of_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = merge(Path(tmp))
            second = merge(Path(tmp))
            edited = merge(Path(tmp), [(uid, text.replace("rue", "R."), label, *rest) for uid, text, label, *rest in ROWS])
        ids = [row["uuid"] for row in first]
        self.assertEqual(len(ids), 3)
        self.assertEqual(len(set(ids)), 3)
        self.assertEqual(ids, [row["uuid"] for row in second])
        self.assertEqual(ids, [row["uuid"] for row in edited])

    def test_document_distinguishes_identical_keys(self):
        a, b = [{"cle": "abc"}], [{"cle": "abc"}]
        assign_entity_ids(a, "vol-A")
        assign_entity_ids(b, "vol-B")
        self.assertNotEqual(a[0]["uuid"], b[0]["uuid"])

    def test_id_depends_on_root_line_only(self):
        """Détacher une continuation ne change pas l'identité de l'entrée ;
        les uid (page, bloc) n'interviennent pas."""
        detached = [ROWS[0], ("1.1.1", "suite", "OUT OF SCOPE"), *ROWS[2:]]
        renumbered = [(f"9.{uid}", text, label, *rest) for uid, text, label, *rest in ROWS]
        with tempfile.TemporaryDirectory() as tmp:
            first = merge(Path(tmp))
            other = merge(Path(tmp), detached)
            moved = merge(Path(tmp), renumbered)
        self.assertEqual(first[0]["uuid"], other[0]["uuid"])
        self.assertEqual(first[0]["cle"], "k1.1.0,k1.1.1")
        self.assertEqual(other[0]["cle"], "k1.1.0")
        self.assertNotEqual([row["uuid"] for row in first], [row["uuid"] for row in moved])

    def test_deleted_lines_are_skipped_and_curation_columns_dropped(self):
        rows = [*ROWS[:2], ("1.1.2", "Boulanger, rue B, 2.", "SUPPRIMÉE")]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "Vol.1-9.lines.csv"
            write_source(source, rows)
            report = process_csv(source, Path(tmp) / "out.csv")
            with (Path(tmp) / "out.csv").open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                out = list(reader)
        self.assertEqual(report.rows_deleted, 1)
        self.assertEqual(len(out), 1)
        self.assertNotIn("corrige", reader.fieldnames)
        self.assertNotIn("empreinte", reader.fieldnames)


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
            source = Path(tmp) / "Vol.1-9.lines.csv"
            write_source(source, rows)
            report = process_csv(source, Path(tmp) / "out.csv")
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
            "1.1.4": uuid_of["1.1.2"],
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
            source = Path(tmp) / "Vol.1-9.lines.csv"
            write_source(source, rows)
            report = process_csv(source, Path(tmp) / "out.csv")
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
