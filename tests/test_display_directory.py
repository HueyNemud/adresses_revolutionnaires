import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from display_directory import add_derived_columns, render_tagged_html, section_paths, title_level  # noqa: E402


class DisplayDirectoryTests(unittest.TestCase):
    def test_title_level(self):
        self.assertEqual(title_level("## AGENS DE CHANGE"), (2, "AGENS DE CHANGE"))
        self.assertEqual(title_level("##\xa0**BAIGNEURS**"), (2, "BAIGNEURS"))
        self.assertEqual(title_level("Sans dièse"), (9, "Sans dièse"))

    def test_section_paths_follow_heading_levels(self):
        entities = pd.Series(["ENTRY", "TITLE", "TITLE", "ENTRY", "TITLE", "ENTRY", "TITLE", "ENTRY"])
        markdown = pd.Series(["x", "# LISTES", "## AGENS", "a", "### Syndics", "b", "## ARCHITECTES", "c"])
        self.assertEqual(
            section_paths(entities, markdown),
            [
                "(avant le premier titre)",
                "LISTES",
                "LISTES › AGENS",
                "LISTES › AGENS",
                "LISTES › AGENS › Syndics",
                "LISTES › AGENS › Syndics",
                "LISTES › ARCHITECTES",
                "LISTES › ARCHITECTES",
            ],
        )

    def test_render_escapes_text_and_marks_spans(self):
        rendered = render_tagged_html("<SUBJ>A &lt;b&gt;</SUBJ>, <ADDR>rue &amp; C</ADDR>")
        self.assertIn("A &lt;b&gt;", rendered)
        self.assertIn('title="SUBJ"', rendered)
        self.assertIn("rue &amp;amp; C", rendered)  # « &amp; » est du texte littéral dans tagged_text
        self.assertIn("balisage invalide", render_tagged_html("<SUBJ>A"))

    def test_derived_columns_recompute_reasons_without_flag_columns(self):
        df = pd.DataFrame(
            {
                "entity": ["TITLE", "ENTRY", "ENTRY"],
                "markdown": ["## RUBRIQUE", "Dupont, rue A, 1.", "Durand, rue B, 2."],
                "page_index": ["6", "6,7", "7"],
                "tagged_text": ["", "<SUBJ>Dupont</SUBJ>, <ADDR>rue A, 1.</ADDR>", "<SUBJ>Durand</SUBJ>, rue B, 2."],
            }
        )
        derived = add_derived_columns(df)
        self.assertEqual(derived["page"].tolist(), [6, 6, 7])
        self.assertEqual(derived["signature"].tolist(), ["", "SUBJ,ADDR", "SUBJ"])
        self.assertEqual(derived["ner_suspect"].tolist(), ["", "", "texte non couvert"])
        self.assertEqual(derived["rubrique"].tolist(), ["RUBRIQUE"] * 3)


if __name__ == "__main__":
    unittest.main()
