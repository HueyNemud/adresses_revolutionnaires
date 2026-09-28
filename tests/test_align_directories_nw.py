import unittest

import numpy as np

from align_directories_nw import Params, align, needleman_wunsch, residual_pairs, windows
from lib.alignment import SOURCE_NW, SOURCE_NW_RESIDUAL, Record
from lib.section_alignment import SectionPatchEntry, align_sections


def records(side: str, entries: list[tuple[str, str]]) -> list[Record]:
    """(rubrique, texte) → Record ; le SUBJ est le nom avant la virgule."""
    return [
        Record(
            document=f"{side}.csv",
            uuid=f"{side}{order}",
            order=order,
            page="1",
            section=section,
            section_title=section.upper(),
            subj=text.split(",")[0],
            text=text,
            markdown=text,
            tagged_text="",
            section_uuid=f"{side}:{section}",
        )
        for order, (section, text) in enumerate(entries)
    ]


def pairs(result) -> set[tuple[str, str]]:
    return {(link.left_uuid, link.right_uuid) for link in result.links}


class NeedlemanWunschTests(unittest.TestCase):
    def test_diagonal(self):
        self.assertEqual(needleman_wunsch(np.eye(3), 0.5), [(0, 0), (1, 1), (2, 2)])

    def test_insertion_and_deletion_cost_nothing(self):
        similarity = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertEqual(needleman_wunsch(similarity, 0.5), [(0, 0), (2, 2)])

    def test_below_threshold_left_alone(self):
        self.assertEqual(needleman_wunsch(np.array([[0.4]]), 0.5), [])

    def test_order_kept_and_residual_recovers_inversion(self):
        similarity = np.array([[0.0, 0.9], [0.95, 0.0]])
        ordered = needleman_wunsch(similarity, 0.5)
        self.assertEqual(ordered, [(1, 0)])
        self.assertEqual(residual_pairs(similarity, ordered, 0.85), [(0, 1)])
        self.assertEqual(residual_pairs(similarity, ordered, 0.95), [])


class WindowsTests(unittest.TestCase):
    def test_between_anchors_with_virtual_bounds(self):
        self.assertEqual(windows([(1, 0), (3, 3)], 5, 4), [([0], []), ([2], [1, 2]), ([4], [])])

    def test_residual_entries_excluded(self):
        self.assertEqual(windows([(1, 1)], 4, 4, [(3, 2)]), [([0], [0]), ([2], [3])])


class AlignTests(unittest.TestCase):
    def test_renamed_section_and_missing_section(self):
        left = records(
            "g",
            [
                ("bouchers", "Dupont, rue Saint-Denis, 12."),
                ("bouchers", "Durand, rue du Bac, 3."),
                ("liste de non commercans", "Martin, rue de Grammont, 14."),
                ("liste de non commercans", "Moreau, rue Charlot, 18."),
            ],
        )
        right = records(
            "d",
            [
                ("bouchers", "Dupont, rue St.-Denis, 12."),
                ("bouchers", "Durand, rue du Bac, 3."),
                ("cordiers", "Martin, rue de Grammont, 14."),
                ("listes de non commercans", "Martin, rue de Grammont, 14."),
                ("listes de non commercans", "Moreau, rue Charlot, 18."),
            ],
        )
        result = align(left, right)
        self.assertEqual(pairs(result), {("g0", "d0"), ("g1", "d1"), ("g2", "d3"), ("g3", "d4")})
        self.assertEqual([section.key for section in result.sections.unmatched_right], ["cordiers"])
        self.assertTrue(all(link.source == SOURCE_NW for link in result.links))

    def test_local_inversion_is_residual(self):
        left = records("g", [("graveurs", "Aubert, rue de la Loi, 5."), ("graveurs", "Aubry, quai Voltaire, 2."), ("graveurs", "Bernard, rue du Bac, 1.")])
        right = records("d", [("graveurs", "Aubry, quai Voltaire, 2."), ("graveurs", "Aubert, rue de la Loi, 5."), ("graveurs", "Bernard, rue du Bac, 1.")])
        result = align(left, right)
        self.assertEqual(pairs(result), {("g0", "d1"), ("g1", "d0"), ("g2", "d2")})
        self.assertEqual(sum(link.source == SOURCE_NW_RESIDUAL for link in result.links), 1)

    def test_gap_between_anchors_is_aligned(self):
        """Rubriques trop différentes pour s'aligner, entre deux ancres : leurs
        entrées sont alignées ensemble."""
        left = records("g", [("architectes", "Brongniart, rue Monsieur, 1."), ("jardiniers fleuristes", "Vilmorin, quai de la Megisserie, 30."), ("vins", "Bardet, en gros, 5.")])
        right = records("d", [("architectes", "Brongniart, rue Monsieur, 1."), ("marchands d arbres", "Vilmorin, quai de la Mégisserie, 30."), ("vins", "Bardet, en gros, 5.")])
        result = align(left, right)
        self.assertEqual(pairs(result), {("g0", "d0"), ("g1", "d1"), ("g2", "d2")})
        self.assertEqual(len(result.sections.groups), 2)

    def test_manual_section_group_out_of_order(self):
        """Rubrique renommée et déplacée dans l'ordre alphabétique, liée par
        le patch : ses entrées sont alignées."""
        left = records(
            "g",
            [("architectes", "Brongniart, rue Monsieur, 1."), ("jardiniers fleuristes", "Vilmorin, quai de la Megisserie, 30."), ("vins", "Bardet, en gros, 5.")],
        )
        right = records(
            "d",
            [("architectes", "Brongniart, rue Monsieur, 1."), ("vins", "Bardet, en gros, 5."), ("zz marchands d arbres", "Vilmorin, quai de la Mégisserie, 30.")],
        )
        self.assertNotIn(("g1", "d2"), pairs(align(left, right)))
        patch = [SectionPatchEntry(left_uuid="g:jardiniers fleuristes", right_uuid="d:zz marchands d arbres")]
        sections = align_sections(left, right, patch)
        result = align(left, right, sections=sections)
        self.assertEqual(pairs(result), {("g0", "d0"), ("g1", "d2"), ("g2", "d1")})

    def test_no_context_is_plain_needleman_wunsch(self):
        left = records("g", [("vins", "Dupont, rue A, 1."), ("vins", "Pagès, rue de l'Echiquier, 33."), ("vins", "Péan, place des Vosges, 6.")])
        right = records("d", [("vins", "Dupont, rue A, 1."), ("vins", "Pagès, boulevard Montmartre, 14."), ("vins", "Péan, place des Vosges, 6.")])
        result = align(left, right, Params(context=False))
        self.assertIsNone(result.fit)
        self.assertTrue({link.source for link in result.links} <= {SOURCE_NW, SOURCE_NW_RESIDUAL})

    def test_one_to_one(self):
        left = records("g", [("vins", "Dupont, rue A, 1."), ("vins", "Dupont, rue A, 1.")])
        right = records("d", [("vins", "Dupont, rue A, 1.")])
        result = align(left, right)
        self.assertEqual(len(result.links), 1)


if __name__ == "__main__":
    unittest.main()
