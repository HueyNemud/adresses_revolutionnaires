import csv
import tempfile
import unittest
from pathlib import Path

from numrev.alignment.records import clean_title, dedupe_records, load_volume, ner_csv, subject_text
from numrev.paths import NER, range_dirs

FIELDS = ["uuid", "parent_uuid", "entity", "markdown", "tagged_text", "page_index"]
ROOT = "00000000-0000-0000-0000-000000000000"


def write_range(volume: Path, pages: str, rows: list[tuple[str, ...]], fields: list[str] = FIELDS) -> None:
    directory = volume / pages
    directory.mkdir(parents=True)
    with (directory / f"{volume.name}.{pages}{NER}").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
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
        self.assertEqual([record.section_uuid for record in records], ["", "t1b", "t1"])
        self.assertEqual(records[1].text, "Dupont, rue A.")
        self.assertEqual(records[1].subj, "Dupont")
        self.assertEqual(records[1].page, "3")

    def test_dedupe_records(self):
        data = dedupe_records(load_volume(self.volume))
        self.assertEqual(data["e0"], {"section": None, "subj": "avant", "text": "avant, rue z."})

    def test_missing_ner_csv(self):
        (self.volume / "30-40").mkdir()
        with self.assertRaises(FileNotFoundError):
            ner_csv(self.volume / "30-40")


class ContinuedSectionTests(unittest.TestCase):
    def test_only_a_title_added_by_hand_continues_the_previous_section(self):
        """Titre recopié en tête de plage (clé `debut+1`) : même rubrique que la
        fin de la plage précédente. Un titre imprimé (plage 30-39) ou d'une autre
        clé (plage 40-49) ouvre une autre rubrique."""
        fields = ["uuid", "parent_uuid", "cle", "entity", "markdown", "page_index"]
        ranges = {
            "10-19": [("t1", ROOT, "a1", "TITLE", "## BOUCHERS.", "10"), ("e1", "t1", "a2", "ENTRY", "Adam, rue A.", "10")],
            "20-29": [("t2", ROOT, "debut+1", "TITLE", "## Bouchers", "20"), ("e2", "t2", "b1", "ENTRY", "Bert, rue B.", "20")],
            "30-39": [("t3", ROOT, "c1", "TITLE", "## BOUCHERS.", "30"), ("e3", "t3", "c2", "ENTRY", "Caron, rue C.", "30")],
            "40-49": [("t4", ROOT, "debut+1,d0", "TITLE", "## BOULANGERS.", "40"), ("e4", "t4", "d1", "ENTRY", "Dumas, rue D.", "40")],
        }
        with tempfile.TemporaryDirectory() as tmp:
            volume = Path(tmp) / "1807_TEST"
            for pages, rows in ranges.items():
                write_range(volume, pages, rows, fields)
            records = load_volume(volume)
        self.assertEqual([record.section_inserted for record in records], [False, True, False, True])
        self.assertEqual([record.section_uuid for record in records], ["t1", "t1", "t3", "t4"])


if __name__ == "__main__":
    unittest.main()
