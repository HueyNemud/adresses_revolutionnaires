import os
import tempfile
import unittest
from pathlib import Path

from numrev.curation import write_csv
from numrev.devtools.status import DONE, MISSING, STALE, range_status


class StatusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.range_dir = Path(self.tmp.name) / "V" / "7-9"
        self.range_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def touch(self, suffix: str, mtime: int) -> Path:
        path = self.range_dir / f"V.7-9{suffix}"
        path.touch()
        os.utime(path, (mtime, mtime))
        return path

    def test_first_missing_step_gives_the_next_command(self) -> None:
        self.touch(".ocr.json", 100)
        self.touch(".lines.json", 200)

        status = range_status(self.range_dir)

        self.assertEqual(status.states["ocr"], DONE)
        self.assertEqual(status.states["extract"], DONE)
        self.assertEqual(status.states["label"], MISSING)
        self.assertEqual(status.next_command, f"uv run numrev label {self.range_dir}")  # label exporte sans --apply

    def test_output_older_than_its_input_is_stale(self) -> None:
        self.touch(".ocr.json", 300)
        self.touch(".lines.json", 200)

        status = range_status(self.range_dir)

        self.assertEqual(status.states["extract"], STALE)
        self.assertEqual(status.next_command, f"uv run numrev extract {self.range_dir} --apply")

    def test_tabulate_depends_on_the_raw_lines_without_annotation(self) -> None:
        self.touch(".ocr.json", 100)
        self.touch(".lines.json", 300)
        self.touch(".lines.csv", 200)

        self.assertEqual(range_status(self.range_dir).states["tabulate"], STALE)

    def test_counts_corrected_and_doubtful_lines(self) -> None:
        path = self.range_dir / "V.7-9.lines.csv"
        write_csv(
            path,
            ["cle", "douteuse", "corrige"],
            [{"cle": "a", "douteuse": "oui", "corrige": "oui"}, {"cle": "b", "douteuse": "", "corrige": ""}],
        )

        status = range_status(self.range_dir)

        self.assertEqual((status.corrected_lines, status.doubtful_lines), (1, 1))


if __name__ == "__main__":
    unittest.main()
