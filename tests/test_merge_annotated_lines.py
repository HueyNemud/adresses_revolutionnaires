import csv
import tempfile
import unittest
from pathlib import Path

from merge_annotated_lines import assign_entity_ids, document_name, process_csv

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


if __name__ == "__main__":
    unittest.main()
