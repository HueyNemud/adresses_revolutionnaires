import csv
import tempfile
import unittest
from pathlib import Path

from numrev.devtools.typeset import Line, Stats, body, compose, folio, inline, latex_escape, read_volume, running_title


def line(entity, markdown, tagged_text="", page=0, label="Text"):
    return Line(entity, markdown, tagged_text, page, label)


class InlineTests(unittest.TestCase):
    def test_escape(self):
        self.assertEqual(latex_escape("50 % & #1_a~\xa0{x}"), r"50 \% \& \#1\_a\textasciitilde{}~\{x\}")

    def test_emphasis_and_spans_overlap(self):
        tagged = "**<SUBJ>A**rchédéacon</SUBJ>, <ADDR>rue Duphot, 4.</ADDR>"
        self.assertEqual(
            inline("**A**rchédéacon, rue Duphot, 4.  ", tagged),
            r"\nom{\textbf{A}}\nom{rchédéacon}, rue Duphot, 4.",
        )

    def test_italic_description(self):
        tagged = "<SUBJ>Demours</SUBJ>, *<DESC>ocul.</DESC>*, <ADDR>rue de l'Université, 19.</ADDR>"
        self.assertEqual(inline("", tagged), r"\nom{Demours}, \textit{ocul.}, rue de l'Université, 19.")

    def test_unclosed_emphasis_ends_with_line(self):
        self.assertEqual(inline("*Fabricans  d'instrument {*"), r"\textit{Fabricans d'instrument \{}")

    def test_markdown_escape_and_image(self):
        self.assertEqual(inline(r"(\*) note ![tampon]()"), "(*) note")

    def test_invalid_tagging_falls_back_to_markdown(self):
        self.assertEqual(inline("Bou, rue X.", "<SUBJ>Bou, rue X."), "Bou, rue X.")


class BlockTests(unittest.TestCase):
    def test_running_title(self):
        self.assertEqual(running_title("##\xa0**AGENS DE CHANGE.**"), "Agens de change")

    def test_folio(self):
        self.assertEqual(folio(line("OUT OF SCOPE", "116 *Arquebusiers.—PARIS.*", label="Page-Header")), "116")
        self.assertEqual(folio(line("OUT OF SCOPE", "152 *Bronzes.—PARIS.*", label="Section-Header")), "152")
        self.assertIsNone(folio(line("OUT OF SCOPE", "8 \\*", label="Page-Footer")))
        self.assertIsNone(folio(line("OUT OF SCOPE", "SOCIÉTÉ GALVANIQUE.", label="Section-Header")))

    def test_body(self):
        lines = [
            line("OUT OF SCOPE", "Agens.- PARIS.", page=5, label="Page-Header"),
            line("OUT OF SCOPE", "114", page=5, label="Page-Header"),
            line("TITLE", "## AGENS DE CHANGE", page=5),
            line("ENTRY", "Bou, rue X.", "<SUBJ>Bou</SUBJ>, <ADDR>rue X.</ADDR>", page=5),
            line("OUT OF SCOPE", "PORTS", page=6, label="Table"),
            line("OUT OF SCOPE", "jours", page=6, label="Table"),
            line("OUT OF SCOPE", "", page=6, label="List-Group"),
            line("ENTRY", "Cou, rue Y.", "<SUBJ>Cou</SUBJ>, <ADDR>rue Y.</ADDR>", page=6),
        ]
        stats = Stats()
        self.assertEqual(
            body(lines, stats),
            [
                r"\rubrique{AGENS DE CHANGE}{Agens de change}",
                r"\entree{\folio{114}\nom{Bou}, rue X.}",
                r"\avis{\folio{p.~7}PORTS · jours}",
                r"\entree{\nom{Cou}, rue Y.}",
            ],
        )
        self.assertEqual((stats.pages, stats.folios, stats.entries), (2, 1, 2))
        self.assertEqual(stats.skipped["Page-Header"], 2)


class VolumeTests(unittest.TestCase):
    def test_read_and_compose(self):
        fields = ["entity", "markdown", "tagged_text", "page_index", "data_block_label"]
        with tempfile.TemporaryDirectory() as tmp:
            volume = Path(tmp) / "1807_X"
            for name, rows in {
                "9-10": [["ENTRY", "Cou, rue Y.", "<SUBJ>Cou</SUBJ>, rue Y.", "9,10", "Text,Text"]],
                "1-8": [["TITLE", "# LISTES", "", "0", "Section-Header"], ["ENTRY", "Bou", "<SUBJ>Bou</SUBJ>", "1", "Text"]],
            }.items():
                (volume / name).mkdir(parents=True)
                with open(volume / name / f"1807_X.{name}.ner.csv", "w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(fields)
                    writer.writerows(rows)
            lines = read_volume(volume)
            self.assertEqual([(item.markdown, item.page) for item in lines], [("# LISTES", 0), ("Bou", 1), ("Cou, rue Y.", 9)])
            document, stats = compose("1807_X", lines)
        self.assertIn(r"\partie{LISTES}{LISTES}", document)
        self.assertIn("{\\Large 1807\\par}", document)
        self.assertTrue(document.rstrip().endswith(r"\end{document}"))
        self.assertEqual(stats.entries, 2)


if __name__ == "__main__":
    unittest.main()
