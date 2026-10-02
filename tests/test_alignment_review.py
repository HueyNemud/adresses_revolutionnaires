import unittest

from lib.alignment import SOURCE_NW, SOURCE_NW_CONTEXT, SOURCE_PROPOSAL, Link, Record
from lib.alignment_review import REASON_CONTEXT, REASON_HOMONYM, REASON_PROPOSAL, review
from lib.section_alignment import align_sections


def records(side: str, texts: list[str]) -> list[Record]:
    """Une seule rubrique ; le SUBJ est le nom avant la virgule."""
    return [
        Record(
            document=f"{side}.csv",
            uuid=f"{side}{order}",
            order=order,
            page="1",
            section="vins",
            section_title="VINS",
            subj=text.split(",")[0],
            text=text,
            markdown=text,
            tagged_text="",
            section_uuid=f"{side}:vins",
        )
        for order, text in enumerate(texts)
    ]


def run(left, right, links, **options):
    sections = align_sections(left, right)
    return review(links, left, right, sections, low=options.pop("low", 0.6), high=0.85, subj_weight=0.5, **options)


class ReviewTests(unittest.TestCase):
    def test_close_homonym_is_flagged(self):
        left = records("g", ["Martin, rue A, 1.", "Martin, rue A, 2."])
        right = records("d", ["Martin, rue A, 1."])
        result = run(left, right, [Link("g0", "d0", 1.0, SOURCE_NW)])
        self.assertEqual(result.reviews["g0", "d0"].reasons, (REASON_HOMONYM,))
        self.assertEqual(result.reviews["g0", "d0"].level, 1)

    def test_distinct_pair_is_not_flagged(self):
        left = records("g", ["Martin, rue A, 1.", "Bernard, quai Voltaire, 30."])
        right = records("d", ["Martin, rue A, 1."])
        self.assertEqual(run(left, right, [Link("g0", "d0", 1.0, SOURCE_NW)]).reviews, {})

    def test_uncertain_context(self):
        left = records("g", ["Pagès, rue de l'Echiquier, 33."])
        right = records("d", ["Pagès, boulevard Montmartre, 14."])
        result = run(left, right, [Link("g0", "d0", 0.8, SOURCE_NW_CONTEXT)])
        self.assertEqual(result.reviews["g0", "d0"].reasons, (REASON_CONTEXT,))
        self.assertEqual(result.reviews["g0", "d0"].level, 1)
        self.assertEqual(run(left, right, [Link("g0", "d0", 0.6, SOURCE_NW_CONTEXT)]).reviews["g0", "d0"].level, 2)
        self.assertEqual(run(left, right, [Link("g0", "d0", 0.95, SOURCE_NW_CONTEXT)]).reviews, {})

    def test_proposal_needs_mutual_best_free_partners_in_the_grey_zone(self):
        left = records("g", ["Pagès, rue de l'Echiquier, 33.", "Bernard, quai Voltaire, 30."])
        right = records("d", ["Pagès, boulevard Montmartre, 14.", "Bernard, quai Voltaire, 30."])
        result = run(left, right, [Link("g1", "d1", 1.0, SOURCE_NW)])
        self.assertEqual([(link.left_uuid, link.right_uuid, link.source) for link in result.proposals], [("g0", "d0", SOURCE_PROPOSAL)])
        self.assertEqual(result.reviews["g0", "d0"].reasons, (REASON_PROPOSAL,))
        self.assertEqual(result.reviews["g0", "d0"].level, 2)
        # Sous le seuil bas, déclarée sans correspondance, ou déjà appariée : pas de proposition.
        self.assertEqual(run(left, right, [Link("g1", "d1", 1.0, SOURCE_NW)], low=0.84).proposals, [])
        self.assertEqual(run(left, right, [Link("g1", "d1", 1.0, SOURCE_NW)], declared={"g0"}).proposals, [])
        self.assertEqual(run(left, right, [Link("g1", "d1", 1.0, SOURCE_NW), Link("g0", "d0", None, "manuel")]).proposals, [])

    def test_no_proposal_above_the_residual_threshold(self):
        left = records("g", ["Bernard, quai Voltaire, 30."])
        right = records("d", ["Bernard, quai Voltaire, 30."])
        self.assertEqual(run(left, right, []).proposals, [])


if __name__ == "__main__":
    unittest.main()
