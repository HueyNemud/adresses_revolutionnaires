import csv
import tempfile
import unittest
from pathlib import Path

from lib.alignment import clean_title, curated_csv, dedupe_records, load_volume, range_dirs, subject_text
from lib.ner.corpus import CURATED_NER_SUFFIX

FIELDS = ["uuid", "parent_uuid", "entity", "markdown", "tagged_text", "page_index"]
ROOT = "00000000-0000-0000-0000-000000000000"


def write_range(volume: Path, pages: str, rows: list[tuple[str, ...]]) -> None:
    directory = volume / pages
    directory.mkdir(parents=True)
    with (directory / f"{volume.name}.{pages}{CURATED_NER_SUFFIX}").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(rows)


class SubjectTextTests(unittest.TestCase):
    def test_markdown_removed_inside_span(self):
        self.assertEqual(subject_text("**<SUBJ>A**rchédéacon (Edmont)</SUBJ>, <ADDR>rue Duphot, 4.</ADDR>"), "Archédéacon (Edmont)")

    def test_several_subjects_joined(self):
        self.assertEqual(subject_text("<SUBJ>Dupont</SUBJ> et <SUBJ>*Durand*</SUBJ>, <ADDR>rue A</ADDR>"), "Dupont Durand")

    def test_missing_or_invalid(self):
        self.assertEqual(subject_text(""), "")
        self.assertEqual(subject_text("<ADDR>rue A</ADDR>"), "")
        self.assertEqual(subject_text("<SUBJ>Dupont, rue A"), "")


class CleanTitleTests(unittest.TestCase):
    def test_variants_share_a_key(self):
        self.assertEqual(clean_title("## BOIS. ( MARCHANDS DE )"), clean_title("##\xa0**Bois (Marchands de)**"))
        self.assertEqual(clean_title("## HÔTELS GARNIS."), "hotels garnis")


class LoadVolumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.volume = Path(self.tmp.name) / "1807_TEST"
        # Plage 10-20 : pas de `##`, rubrique = le `#` ; sous-titre `###` ignoré.
        write_range(
            self.volume,
            "10-20",
            [
                ("t1", ROOT, "TITLE", "# MÉDECINS.", "", "10"),
                ("t2", "t1", "TITLE", "### Adjoints", "", "10"),
                ("e3", "t2", "ENTRY", "Martin, rue C.", "<SUBJ>Martin</SUBJ>, <ADDR>rue C.</ADDR>", "11"),
            ],
        )
        # Plage 2-9 : `##` prioritaire sur `#`, entrée avant tout titre, OUT OF SCOPE exclu.
        write_range(
            self.volume,
            "2-9",
            [
                ("e0", ROOT, "ENTRY", "Avant, rue Z.", "<SUBJ>Avant</SUBJ>, <ADDR>rue Z.</ADDR>", "2"),
                ("t0", ROOT, "TITLE", "# LISTES", "", "2"),
                ("t1b", "t0", "TITLE", "## AGENS DE CHANGE.", "", "2"),
                ("e1", "t1b", "ENTRY", "**Dupont**, rue A.  ", "<SUBJ>**Dupont**</SUBJ>, <ADDR>rue A.</ADDR>", "3,4"),
                ("o1", "t1b", "OUT OF SCOPE", "3", "", "3"),
                ("e2", "t1b", "ENTRY", "   ", "", "3"),
            ],
        )
        (self.volume / "notes.txt").write_text("ignoré")

    def tearDown(self):
        self.tmp.cleanup()

    def test_ranges_sorted_by_first_page(self):
        self.assertEqual([path.name for path in range_dirs(self.volume)], ["2-9", "10-20"])

    def test_records(self):
        records = load_volume(self.volume)
        self.assertEqual([record.uuid for record in records], ["e0", "e1", "e3"])
        self.assertEqual([record.order for record in records], [0, 1, 2])
        self.assertEqual([record.section for record in records], ["", "agens de change", "medecins"])
        self.assertEqual([record.section_title for record in records], ["", "AGENS DE CHANGE.", "MÉDECINS."])
        self.assertEqual(records[1].text, "Dupont, rue A.")
        self.assertEqual(records[1].subj, "Dupont")
        self.assertEqual(records[1].page, "3")

    def test_dedupe_records(self):
        data = dedupe_records(load_volume(self.volume))
        self.assertEqual(data["e0"], {"section": None, "subj": "avant", "text": "avant, rue z."})

    def test_missing_curated_csv(self):
        (self.volume / "30-40").mkdir()
        with self.assertRaises(FileNotFoundError):
            curated_csv(self.volume / "30-40")


if __name__ == "__main__":
    unittest.main()
