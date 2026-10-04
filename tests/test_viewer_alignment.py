import unittest

import pandas as pd

from numrev.alignment.export import CANDIDATE, LEFT_ONLY, PAIR, RIGHT_ONLY
from numrev.alignment.records import SOURCE_MANUAL, SOURCE_MANUAL_UNCERTAIN, DocLine, Record
from numrev.viewers import theme_options
from numrev.viewers.alignment import EntryStates, TaskOptions, task_levels, task_marks, task_queue
from numrev.viewers.context import EntryState, Marks, centers, documents, payload, section_before, window
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
        data = payload(self.docs, bounds, ("g2", "d1"), Marks(), "400px")
        self.assertEqual([item["u"] for item in data["left"]], ["g1", "g2"])
        self.assertEqual(data["more"]["left"], [True, False])
        bounds = window(self.docs, {"left": 4, "right": 2}, 0, 0)
        data = payload(self.docs, bounds, ("g2", "d1"), Marks(), "400px")
        self.assertEqual(data["left"][0]["o"], 0)  # partenaire d1 visible
        bounds = {"left": (0, 2), "right": (2, 3)}
        data = payload(self.docs, bounds, ("g0", ""), Marks(), "400px")
        self.assertEqual(data["left"][1]["o"], -1)  # partenaire d0 au-dessus de la fenêtre de droite
        self.assertEqual(data["left"][0]["t"], "title")
        self.assertNotIn("s", data["left"][0])

    def test_section_above_the_window(self):
        def title(uuid: str, markdown: str) -> DocLine:
            return DocLine(uuid, "TITLE", "1", markdown, "", markdown.count("#"))

        lines = [title("t1", "# LISTES"), title("t2", "## VINS"), line("g0"), title("t3", "### Détail"), line("g1")]
        self.assertEqual(section_before(lines, 0), "")
        self.assertEqual(section_before(lines, 2), "VINS")
        self.assertEqual(section_before(lines, 5), "VINS")  # un titre de niveau 3 n'est pas une rubrique
        data = payload(self.docs, {"left": (1, 5), "right": (0, 3)}, ("g2", "d1"), Marks(), "400px")
        self.assertEqual(data["sections"], {"left": "VINS", "right": ""})

    def test_marks(self):
        marks = Marks(
            tasks={"left": {"g1": 2}, "right": {}}, hits={"left": set(), "right": {"d0"}}, eligible={"left": set(), "right": {"d0"}}
        )
        data = payload(self.docs, {"left": (0, 5), "right": (0, 3)}, ("g1", ""), marks, "400px")
        by_uuid = {item["u"]: item for item in data["left"] + data["right"]}
        self.assertEqual(by_uuid["g1"]["k"], 2)
        self.assertTrue(by_uuid["d0"]["f"])
        self.assertTrue(data["pairing"])
        self.assertTrue(by_uuid["d0"]["e"])
        self.assertFalse(by_uuid["d1"]["e"])
        self.assertNotIn("e", by_uuid["tg"])  # un titre n'est jamais cliquable


class EntryStatesTests(unittest.TestCase):
    def test_uncertain_pairs_are_dashed(self):
        # Trait en tirets : paire relue « incertaine », ou automatique d'incertitude moyenne ou forte.
        columns = {
            "left_uuid": ["g0", "g1", "g2", "g3", "g4"],
            "right_uuid": ["d0", "d1", "d2", "d3", ""],
            "kind": [PAIR, PAIR, PAIR, CANDIDATE, LEFT_ONLY],
            "source": ["nw", "nw", SOURCE_MANUAL_UNCERTAIN, "candidate", ""],
            "level": [0, 1, 0, 2, 0],
        }
        index = {uuid: row for row, uuid in enumerate(columns["left_uuid"])}
        states = EntryStates("left", columns, index, set(), {"g2"})
        self.assertEqual([states[uuid].uncertain for uuid in index], [False, True, True, False, False])
        self.assertEqual(states["g2"], EntryState("pair", "d2", manual=True, local=True, uncertain=True))
        self.assertEqual(states["g3"].kind, "candidate")
        self.assertEqual(states["g4"], EntryState("alone", ""))


class TaskQueueTests(unittest.TestCase):
    def setUp(self):
        self.rows = pd.DataFrame(
            {
                "kind": [PAIR, PAIR, CANDIDATE, LEFT_ONLY, RIGHT_ONLY, PAIR, PAIR],
                "level": [0, 1, 2, 0, 0, 2, 0],
                "source": ["nw", "nw", "candidate", "", "", "nw", SOURCE_MANUAL],
                "score": [0.95, 0.8, 0.7, None, None, 0.6, None],
                "left_uuid": ["g0", "g1", "g2", "g3", "", "g5", "g6"],
                "right_uuid": ["d0", "d1", "d2", "", "d4", "d5", "d6"],
                "left_section": ["vins"] * 6 + ["bois"],
                "right_section": ["vins"] * 6 + ["bois"],
            }
        )

    def test_default_tasks_are_candidates_and_uncertain_pairs(self):
        levels = task_levels(self.rows, set(), TaskOptions())
        self.assertEqual(levels.tolist(), [0, 1, 2, 0, 0, 2, 0])
        self.assertEqual(task_queue(levels, by_level=False), [1, 2, 5])
        self.assertEqual(task_queue(levels, by_level=True), [2, 5, 1])
        self.assertEqual(task_marks(self.rows, levels), {"left": {"g1": 1, "g2": 2, "g5": 2}, "right": {"d1": 1, "d2": 2, "d5": 2}})

    def test_optional_tasks(self):
        options = TaskOptions(candidates=False, medium=False, high=False, left_only=True, right_only=True, below=0.97)
        # g6 : paire relue (patch), jamais une tâche de score faible
        self.assertEqual(task_levels(self.rows, set(), options).tolist(), [1, 1, 0, 1, 1, 1, 0])

    def test_decided_rows(self):
        levels = task_levels(self.rows, {"g3"}, TaskOptions(left_only=True))
        self.assertEqual(levels[3], 0)  # confirmée seule : décidée
        levels = task_levels(self.rows, {"g3"}, TaskOptions(include_decided=True))
        self.assertEqual(levels[[3, 6]].tolist(), [1, 1])

    def test_section(self):
        self.assertEqual(task_queue(task_levels(self.rows, set(), TaskOptions(section="bois", include_decided=True)), False), [6])


class ThemeTests(unittest.TestCase):
    def test_theme_becomes_streamlit_options(self):
        options = theme_options()
        self.assertIn("--theme.dark.backgroundColor=#0d1117", options)
        self.assertIn("--theme.light.backgroundColor=#ffffff", options)
        self.assertIn("--theme.showWidgetBorder=true", options)
        self.assertTrue(all(option.startswith("--theme.") for option in options))


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
