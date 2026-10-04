import argparse
import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from numrev import paths
from numrev.command import CommandError
from numrev.devtools import publish_viewer
from numrev.devtools.publish_viewer import ALLOWED, ENTRY_POINT, MANIFEST, module_closure, run

FIELDS = ["uuid", "parent_uuid", "entity", "markdown", "tagged_text", "page_index"]


def write_volume(root: Path, volume: str) -> None:
    directory = root / volume / "1-9"
    directory.mkdir(parents=True)
    with (directory / f"{volume}.1-9{paths.NER}").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerow([f"{volume}-e", "", "ENTRY", "Martin, rue A.", "<SUBJ>Martin</SUBJ>, rue A.", "1"])


class BoundaryTests(unittest.TestCase):
    def test_viewer_only_needs_light_libraries(self):
        files, third_party = module_closure()
        self.assertLessEqual(third_party, set(ALLOWED), "le viewer déployé ne doit pas dépendre de torch, GLiNER, Dedupe…")
        names = {path.name for path in files}
        self.assertIn("alignment.py", names)
        self.assertNotIn("gliner.py", names)

    def test_published_layout_matches_path_defaults(self):
        self.assertEqual(paths.ALIGNMENTS_DIR, publish_viewer.PUBLISHED_ALIGNMENTS)
        self.assertEqual(paths.ALIGNMENT_DATA_DIR, publish_viewer.PUBLISHED_ALIGNMENT_DATA)


class PublishTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.annuaires, self.data, self.output = root / "src" / "annuaires", root / "src" / "data", root / "deploy"
        for volume in ("1807_A", "1808_A"):
            write_volume(self.annuaires, volume)
        (self.annuaires / "alignments").mkdir()
        self.alignment = self.annuaires / "alignments" / f"1807_A__1808_A{paths.NW_SUFFIX}"
        self.alignment.write_text("left_file,left_uuid,right_uuid,right_file,score,source\n")
        self.data.mkdir(parents=True)
        (self.data / "1807_A__1808_A.sections.csv").write_text("left_uuid,right_uuid\n")
        patches = mock.patch.multiple(
            paths, ANNUAIRES_DIR=self.annuaires, ALIGNMENTS_DIR=self.annuaires / "alignments", ALIGNMENT_DATA_DIR=self.data
        )
        patches.start()
        self.addCleanup(patches.stop)
        commit = mock.patch.object(publish_viewer, "source_commit", return_value=("abc1234", False))
        commit.start()
        self.addCleanup(commit.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def publish(self, apply: bool = True, **options) -> str:
        args = argparse.Namespace(output=self.output, alignment=None, force=False, apply=apply, **options)
        buffer = io.StringIO()
        with mock.patch.object(publish_viewer, "console") as console, contextlib.redirect_stdout(buffer):
            console.print.side_effect = lambda *items, **kwargs: buffer.write(" ".join(map(str, items)) + "\n")
            run(args)
        return buffer.getvalue()

    def test_dry_run_writes_nothing(self):
        self.publish(apply=False)
        self.assertFalse(self.output.exists())

    def test_publishes_code_data_and_generated_files(self):
        self.publish()
        for name in (
            ENTRY_POINT,
            "requirements.txt",
            "README.md",
            "numrev/viewers/alignment.py",
            "numrev/viewers/assets/context.js",
            "annuaires/1807_A/1-9/1807_A.1-9.ner.csv",
            "annuaires/1808_A/1-9/1808_A.1-9.ner.csv",
            f"annuaires/alignments/1807_A__1808_A{paths.NW_SUFFIX}",
            "data/alignment/1807_A__1808_A.sections.csv",
        ):
            self.assertTrue((self.output / name).exists(), name)
        self.assertIn("abc1234", (self.output / "README.md").read_text())
        self.assertEqual({line.split("==")[0] for line in (self.output / "requirements.txt").read_text().split()}, set(ALLOWED))
        self.assertFalse((self.output / "numrev/pipeline").exists())

    def test_only_managed_files_are_removed(self):
        self.publish()
        (self.output / "notes.txt").write_text("à moi")
        (self.output / ".git").mkdir()
        (self.output / ".git" / "HEAD").write_text("ref")
        patch = self.data / "1807_A__1808_A.patch.csv"
        patch.write_text("left_uuid\n")
        self.publish()
        self.assertTrue((self.output / "data/alignment/1807_A__1808_A.patch.csv").exists())
        patch.unlink()
        report = self.publish()
        self.assertFalse((self.output / "data/alignment/1807_A__1808_A.patch.csv").exists())
        self.assertTrue((self.output / "notes.txt").exists())
        self.assertTrue((self.output / ".git" / "HEAD").exists())
        self.assertIn("notes.txt", report)
        manifest = json.loads((self.output / MANIFEST).read_text())
        self.assertEqual(manifest["source_commit"], "abc1234")
        self.assertNotIn("data/alignment/1807_A__1808_A.patch.csv", manifest["files"])

    def test_refusals(self):
        with mock.patch.object(publish_viewer, "source_commit", return_value=("abc1234", True)):
            with self.assertRaises(CommandError):
                self.publish()
        with self.assertRaises(CommandError):
            run(argparse.Namespace(output=Path.cwd() / "deploy", alignment=None, force=False, apply=False))


if __name__ == "__main__":
    unittest.main()
