import argparse
import socket
import tempfile
import unittest
from pathlib import Path

from numrev.command import CommandError
from numrev.viewers import byte_range, directory_files, ocr_files, port_number, serve


class DirectoryFilesTests(unittest.TestCase):
    def test_serves_the_deliverable_then_the_report_and_pdf_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            volume = Path(tmp) / "Vol"
            volume.mkdir()
            for suffix in (".document.json", ".structure.json"):
                (volume / f"Vol{suffix}").write_text("{}")
            self.assertEqual(sorted(directory_files(volume)), ["document", "structure"])

            (volume / "Vol.export.report.txt").write_text("")
            (volume / "Vol.pdf").write_bytes(b"%PDF")
            files = directory_files(volume)
            self.assertEqual(sorted(files), ["document", "pdf", "report", "structure"])
            self.assertEqual(files["pdf"], volume / "Vol.pdf")

    def test_refuses_a_volume_without_deliverable_or_a_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            volume = Path(tmp) / "Vol"
            (volume / "7-12").mkdir(parents=True)
            (volume / "Vol.document.json").write_text("{}")
            with self.assertRaisesRegex(CommandError, "livrable absent : .*Vol.structure.json .*numrev export"):
                directory_files(volume)
            with self.assertRaisesRegex(CommandError, "est une plage"):
                directory_files(volume / "7-12")


class OcrFilesTests(unittest.TestCase):
    def test_opens_the_first_page_of_the_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            range_dir = Path(tmp) / "Vol" / "7-12"
            range_dir.mkdir(parents=True)
            (range_dir / "Vol.7-12.ocr.json").write_text("[]")
            (range_dir.parent / "Vol.pdf").write_bytes(b"%PDF")
            files, fragment = ocr_files(range_dir)
            self.assertEqual((sorted(files), fragment), (["json", "pdf"], "p=7"))


class ByteRangeTests(unittest.TestCase):
    def test_ranges(self):
        self.assertEqual(byte_range("bytes=0-99", 1000), (0, 99))
        self.assertEqual(byte_range("bytes=900-", 1000), (900, 999))
        self.assertEqual(byte_range("bytes=900-5000", 1000), (900, 999))
        self.assertEqual(byte_range("bytes=-100", 1000), (900, 999))

    def test_whole_file_when_absent_unsupported_or_outside(self):
        for header in (None, "", "bytes=-", "bytes=0-9,20-29", "items=0-9", "bytes=2000-"):
            with self.subTest(header=header):
                self.assertIsNone(byte_range(header, 1000))


if __name__ == "__main__":
    unittest.main()


class PortTests(unittest.TestCase):
    def test_accepts_only_a_tcp_port(self):
        self.assertEqual(port_number("8765"), 8765)
        for text in ("0", "65536", "-1", "abc", ""):
            with self.assertRaisesRegex(argparse.ArgumentTypeError, "port invalide"):
                port_number(text)

    def test_refuses_a_busy_port_before_opening_the_browser(self):
        with socket.socket() as busy, tempfile.TemporaryDirectory() as tmp:
            busy.bind(("127.0.0.1", 0))
            busy.listen()
            page = Path(tmp) / "page.html"
            page.write_text("")
            with self.assertRaisesRegex(CommandError, f"port {busy.getsockname()[1]} sur 127.0.0.1 : .*--port"):
                serve(page, {}, "", "Viewer", busy.getsockname()[1])
