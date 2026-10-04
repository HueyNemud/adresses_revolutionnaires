import tempfile
import unittest
from pathlib import Path

from numrev.paths import ENTITIES, LINES_CSV, NER, Pair, discover, document_name, join_output, raw_alignment, step_path, volume_of


class DocumentNameTests(unittest.TestCase):
    def test_same_name_at_every_step(self):
        for suffix in (".ocr.json", ".lines.json", ".labeled.json", ".lines.csv", ".entities.csv", ".entities.report.txt", ".ner.csv"):
            self.assertEqual(document_name(Path(f"1808_AD75-PER292.7-186{suffix}")), "1808_AD75-PER292.7-186")

    def test_unknown_suffix_refused(self):
        for name in ("autre.csv", ".ner.csv", "1808.7-186.ocr.lines.annotated.csv"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                document_name(Path(name))

    def test_step_path_and_volume(self):
        path = Path("annuaires/V/7-9/V.7-9.lines.csv")
        self.assertEqual(step_path(path, ENTITIES), Path("annuaires/V/7-9/V.7-9.entities.csv"))
        self.assertEqual(volume_of("1808_AD75-PER292.7-186"), "1808_AD75-PER292")

    def test_discover_ignores_other_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("V/7-9/V.7-9.ner.csv", "V/7-9/V.7-9.copie.ner.csv", "V/10-12/V.10-12.ner.csv", "V/notes/V.notes.ner.csv"):
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).touch()
            self.assertEqual(discover(root, NER), [root / "V/10-12/V.10-12.ner.csv", root / "V/7-9/V.7-9.ner.csv"])
            self.assertEqual(discover(root, LINES_CSV), [])


class PairTests(unittest.TestCase):
    def test_of_file(self):
        for name in ("A__B.nw.csv", "A__B.raw-sections.dedupe.csv", "A__B.gold-inversions.csv", "A__B.csv"):
            with self.subTest(name=name):
                self.assertEqual(Pair.of_file(Path(name)).name, "A__B")
        with self.assertRaises(ValueError):
            Pair.of_file(Path("A.nw.csv"))

    def test_of_dirs_keeps_root(self):
        pair = Pair.of_dirs(Path("ailleurs/A"), Path("ailleurs/B"))
        self.assertEqual(pair.left_dir, Path("ailleurs/A"))
        with self.assertRaises(ValueError):
            Pair.of_dirs(Path("x/A"), Path("y/B"))

    def test_files(self):
        pair = Pair("A", "B")
        self.assertEqual(pair.entry_patch, Path("data/alignment/A__B.patch.csv"))
        self.assertEqual(pair.section_patch, Path("data/alignment/A__B.sections.csv"))
        self.assertEqual(pair.output(".nw.csv"), Path("annuaires/alignments/A__B.nw.csv"))
        self.assertEqual(raw_alignment(pair.output()), Path("annuaires/alignments/A__B.dedupe.csv"))
        self.assertEqual(join_output(pair.output(".nw.csv")), Path("annuaires/alignments/A__B.nw.join.csv"))


if __name__ == "__main__":
    unittest.main()
