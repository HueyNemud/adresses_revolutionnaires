import unittest

from numrev.alignment.records import DocLine, Record
from numrev.viewers.context import EntryState, centers, documents, payload, window
from numrev.viewers.focus import alternatives, char_diff


def line(uuid: str, entity: str = "ENTRY", markdown: str = "x") -> DocLine:
    return DocLine(uuid, entity, "1", markdown, f"<SUBJ>{markdown}</SUBJ>" if entity == "ENTRY" else "", 2 if entity == "TITLE" else None)


def record(uuid: str, text: str) -> Record:
    return Record("v.ner.csv", uuid, 0, "1", "vins", "VINS", text.split(",")[0], text, text, "")


class DocumentsTests(unittest.TestCase):
    def setUp(self):
        left = [line("tg", "TITLE", "## VINS"), line("g0"), line("o", "OUT OF SCOPE"), line("g1"), line("g2")]
        right = [line("td", "TITLE", "## VINS"), line("d0"), line("d1")]
        states = {
            "left": {"g0": EntryState("pair", "d0"), "g1": EntryState("alone"), "g2": EntryState("pair", "d1", manual=True)},
            "right": {"d0": EntryState("pair", "g0"), "d1": EntryState("pair", "g2", manual=True)},
        }
        self.docs = documents({"left": left, "right": right}, states)

    def test_centers_of_a_pair(self):
        self.assertEqual(centers(self.docs, "g2", "d1"), {"left": 4, "right": 2})

    def test_lone_entry_is_anchored_on_the_previous_pair(self):
        # g1 seule : à droite, on se place sur le partenaire de g0, la paire qui la précède.
        self.assertEqual(centers(self.docs, "g1", ""), {"left": 3, "right": 1})
        # Sans paire avant, on prend la suivante ; d0 seule côté droit → partenaire g0.
        self.assertEqual(centers(self.docs, "", "d0")["left"], 1)

    def test_window_and_payload(self):
        bounds = window(self.docs, {"left": 4, "right": 2}, 1, 1)
        self.assertEqual(bounds, {"left": (3, 5), "right": (1, 3)})
        data = payload(self.docs, bounds, ("g2", "d1"), {}, 400)
        self.assertEqual([item["u"] for item in data["left"]], ["g1", "g2"])
        self.assertEqual(data["more"]["left"], [True, False])
        bounds = window(self.docs, {"left": 4, "right": 2}, 0, 0)
        data = payload(self.docs, bounds, ("g2", "d1"), {}, 400)
        self.assertEqual(data["left"][0]["o"], 0)  # partenaire d1 visible
        bounds = {"left": (0, 2), "right": (2, 3)}
        data = payload(self.docs, bounds, ("g0", ""), {}, 400)
        self.assertEqual(data["left"][1]["o"], -1)  # partenaire d0 au-dessus de la fenêtre de droite
        self.assertEqual(data["left"][0]["t"], "title")
        self.assertNotIn("s", data["left"][0])


class FocusTests(unittest.TestCase):
    def test_char_diff_marks_only_differences(self):
        left, right = char_diff("Martin, rue A, 1.", "Martin, rue B, 1.")
        self.assertEqual(left, "Martin, rue <mark class='diff'>A</mark>, 1.")
        self.assertEqual(right, "Martin, rue <mark class='diff'>B</mark>, 1.")
        self.assertEqual(char_diff("a<b", "a<b"), ("a&lt;b", "a&lt;b"))

    def test_alternatives_are_ranked_and_exclude_the_partner(self):
        current = record("g0", "Martin, rue A, 1.")
        others = [record("d0", "Martin, rue A, 1."), record("d1", "Martin, rue A, 2."), record("d2", "Bernard, quai Voltaire, 30.")]
        found = alternatives(current, "left", others, exclude="d0", partners={"d1": record("g9", "Martin")}, subj_weight=0.5, rivals={"d1"})
        self.assertEqual([alternative.record.uuid for alternative in found], ["d1", "d2"])
        self.assertTrue(found[0].rival)
        self.assertEqual(found[0].partner.uuid, "g9")
        self.assertIsNone(found[1].partner)
        right = alternatives(others[0], "right", [current], exclude="", partners={}, subj_weight=0.5)
        self.assertEqual(right[0].record.uuid, "g0")


if __name__ == "__main__":
    unittest.main()
