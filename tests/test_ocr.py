import argparse
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from numrev.command import CommandError
from numrev.pipeline import ocr


class FakeLatynna:
    """Le paquet latynna, sans serveur vLLM : pages vides numérotées comme latynna (rang dans la sortie)."""

    def __init__(self, pages_in_pdf: int = 20) -> None:
        self.calls: list[list[int]] = []
        self.pages_in_pdf = pages_in_pdf

    def ocr_document(self, source: Path, pages: list[int], workers: int) -> list[dict]:
        self.calls.append(pages)
        return [{"raw": "", "page_index": rank} for rank in range(len(pages))]

    def modules(self) -> dict[str, types.ModuleType]:
        package = types.ModuleType("latynna")
        package.ocr_document = self.ocr_document
        render = types.ModuleType("latynna.render")
        render.page_count = lambda source: self.pages_in_pdf
        package.render = render
        return {"latynna": package, "latynna.render": render}


class OcrCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.volume = Path(self.tmp.name) / "Vol"
        self.volume.mkdir()
        (self.volume / "Vol.pdf").write_bytes(b"%PDF")
        self.latynna = FakeLatynna()
        patcher = mock.patch.dict(sys.modules, self.latynna.modules())
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_ocr(self, range_name: str, apply: bool = True, force: bool = False) -> Path:
        range_dir = self.volume / range_name
        ocr.run(argparse.Namespace(range_dir=range_dir, apply=apply, force=force, workers=1))
        return range_dir / f"Vol.{range_name}.ocr.json"

    def test_page_index_is_the_one_based_page_number_in_the_pdf(self) -> None:
        output = self.run_ocr("7-9")

        self.assertEqual(self.latynna.calls, [[6, 7, 8]])
        self.assertEqual([page["page_index"] for page in json.loads(output.read_text(encoding="utf-8"))], [7, 8, 9])

    def test_dry_run_checks_but_does_not_run_the_ocr(self) -> None:
        output = self.run_ocr("7-9", apply=False)

        self.assertEqual(self.latynna.calls, [])
        self.assertFalse(output.exists())

    def test_existing_ocr_is_replaced_only_with_force(self) -> None:
        self.run_ocr("7-9")
        with self.assertRaisesRegex(CommandError, "existe déjà"):
            self.run_ocr("7-9")
        self.run_ocr("7-9", force=True)
        self.assertEqual(len(self.latynna.calls), 2)

    def test_refuses_a_range_beyond_the_pdf(self) -> None:
        with self.assertRaisesRegex(CommandError, "a 20 page"):
            self.run_ocr("19-21")

    def test_refuses_a_folder_that_is_not_a_page_range(self) -> None:
        for name in ("pages", "9-7", "0-3"):
            with self.subTest(name=name), self.assertRaisesRegex(CommandError, "première page"):
                self.run_ocr(name)

    def test_refuses_a_missing_pdf(self) -> None:
        (self.volume / "Vol.pdf").unlink()
        with self.assertRaisesRegex(CommandError, "Vol.pdf"):
            self.run_ocr("7-9")


if __name__ == "__main__":
    unittest.main()
