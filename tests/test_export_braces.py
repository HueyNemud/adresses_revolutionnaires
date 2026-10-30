import unittest

from numrev.export.braces import Zone, group_members


def grouping(*lines: str):
    """Membres d'une entrée dont chaque ligne est un segment (lignes jointes par un saut de ligne)."""
    text = "\n".join(lines)
    segments, position = [], 0
    for line in lines:
        segments.append(Zone(position, position + len(line)))
        position += len(line) + 1
    return text, group_members(text, segments)


def texts(text: str, member) -> tuple[str, list[str]]:
    return text[member.own.start : member.own.end], [text[zone.start : zone.end] for zone in member.inherited]


class GroupMembersTests(unittest.TestCase):
    def test_closing_brace_shares_the_text_on_the_right_of_the_first_line(self):
        text, result = grouping("Fieffé. } *Membres du conseil.*", "Pastoret. }", "Thouret. }")

        self.assertEqual(result.problems, [])
        self.assertEqual(
            [texts(text, member) for member in result.members],
            [("Fieffé.", ["*Membres du conseil.*"]), ("Pastoret.", ["*Membres du conseil.*"]), ("Thouret.", ["*Membres du conseil.*"])],
        )

    def test_opening_brace_shares_the_text_on_the_left_of_the_first_line(self):
        text, result = grouping("Commissaires. { MM. Touzet.", "{ Lalande, rue Mandar.")

        self.assertEqual(result.problems, [])
        self.assertEqual(
            [texts(text, member) for member in result.members],
            [("MM. Touzet.", ["Commissaires."]), ("Lalande, rue Mandar.", ["Commissaires."])],
        )

    def test_a_new_common_text_opens_a_new_run_under_the_same_brace(self):
        text, result = grouping(
            "Ports { S.-Bernard..... } Rive gauche.", "{ SS.-Pères..... }", "{ S.-Paul..... } Rive droite.", "{ au Blé..... }"
        )

        self.assertEqual(result.problems, [])
        self.assertEqual(
            [texts(text, member) for member in result.members],
            [
                ("S.-Bernard.....", ["Ports", "Rive gauche."]),
                ("SS.-Pères.....", ["Ports", "Rive gauche."]),
                ("S.-Paul.....", ["Ports", "Rive droite."]),
                ("au Blé.....", ["Ports", "Rive droite."]),
            ],
        )
        self.assertEqual([text[zone.start : zone.end] for zone in result.commons], ["Ports", "Rive gauche.", "Rive droite."])

    def test_a_segment_without_closing_brace_ends_the_closing_run(self):
        text, result = grouping("I.er { Tuileries } Cuynet.", "{ Roule", "{ Mail } Barat.")

        self.assertEqual(result.problems, [])
        self.assertEqual(
            [texts(text, member) for member in result.members],
            [("Tuileries", ["I.er", "Cuynet."]), ("Roule", ["I.er"]), ("Mail", ["I.er", "Barat."])],
        )

    def test_a_single_segment_or_a_text_without_sign_is_no_group(self):
        self.assertIsNone(grouping("Igonel, rue des Noyers, 28. } *Huissiers.*")[1])
        self.assertIsNone(grouping("Masson, rue de l'Aiguillerie,", "Son bureau est à la halle.")[1])

    def test_unresolved_cases_are_reported_by_segment(self):
        _, result = grouping("Leroux,", "Caron, } *adjoints*.", "Martin, }")
        self.assertEqual(result.problems, [(0, "ligne sans accolade dans un groupe")])

        _, result = grouping("{ Lalande", "{ Perrin")
        self.assertEqual(result.problems, [(0, "« { » sans texte commun à sa gauche")])

        _, result = grouping("Halles { à la Viande } Places. { d'Aval.", "{ au Beurre }")
        self.assertEqual(result.problems[0], (0, "signes d'accolade inattendus (2 « { », 1 « } »)"))


if __name__ == "__main__":
    unittest.main()
