import csv
import io
import unittest

from lib.alignment import SOURCE_MANUAL, SOURCE_MANUAL_UNCERTAIN, SOURCE_CANDIDATE, Link, Record
from lib.alignment_export import EXPORT_FIELDS, CANDIDATE, export_csv, export_row, natural_rows, span_texts
from lib.alignment_review import Review


def record(uuid: str, order: int, tagged_text: str = "", section_uuid: str = "s") -> Record:
    return Record(
        document=f"VOL_{uuid[0]}.1-9.csv",
        uuid=uuid,
        order=order,
        page="3",
        section="rubrique",
        section_title="RUBRIQUE",
        subj="",
        text=f"texte {uuid}",
        markdown=f"texte {uuid}",
        tagged_text=tagged_text,
        section_uuid=section_uuid,
    )


def side(records: list[Record]) -> dict[str, Record]:
    return {r.uuid: r for r in records}


def uuids(rows) -> list[tuple[str, str]]:
    return [(row.left.uuid if row.left else "", row.right.uuid if row.right else "") for row in rows]


class NaturalRowsTests(unittest.TestCase):
    def test_right_only_after_preceding_pair_and_before_first(self):
        left = side([record("l0", 0), record("l1", 1), record("l2", 2), record("l3", 3)])
        right = side([record("r0", 0), record("r1", 1), record("r2", 2), record("r3", 3), record("r4", 4)])
        # r0 précède toute paire : juste avant la paire de r1, dont la gauche est l2.
        links = [Link("l2", "r1", 0.9), Link("l3", "r3", 0.8)]
        rows, missing = natural_rows(links, left, right)
        self.assertEqual(
            uuids(rows),
            [("l0", ""), ("l1", ""), ("", "r0"), ("l2", "r1"), ("", "r2"), ("l3", "r3"), ("", "r4")],
        )
        self.assertEqual(missing, 0)

    def test_crossed_pair_anchors_on_right_order(self):
        left = side([record("l0", 0), record("l1", 1)])
        right = side([record("r0", 0), record("r1", 1), record("r2", 2)])
        rows, _ = natural_rows([Link("l0", "r1", 0.9), Link("l1", "r0", 0.9)], left, right)
        self.assertEqual(uuids(rows), [("l0", "r1"), ("", "r2"), ("l1", "r0")])

    def test_no_pair_and_missing_links(self):
        left = side([record("l0", 0)])
        right = side([record("r0", 0)])
        rows, missing = natural_rows([Link("l0", "gone", 0.9)], left, right)
        self.assertEqual(uuids(rows), [("l0", ""), ("", "r0")])
        self.assertEqual(missing, 1)


class ExportTests(unittest.TestCase):
    def test_span_texts(self):
        spans = span_texts("**<SUBJ>D**upont</SUBJ> et <SUBJ>Durand</SUBJ>, <DESC>notaires</DESC>, <ADDR>rue A</ADDR>")
        self.assertEqual(spans, {"SUBJ": "Dupont | Durand", "DESC": "notaires", "ADDR": "rue A"})
        self.assertEqual(span_texts("<SUBJ>Dupont, rue A"), {"SUBJ": "", "DESC": "", "ADDR": ""})

    def test_export_row(self):
        left = record("l0", 0, "<SUBJ>Dupont</SUBJ>, <ADDR>rue A</ADDR>", section_uuid="a")
        right = record("r0", 0, section_uuid="b")
        rows, _ = natural_rows([Link("l0", "r0", 0.91234, "nw")], side([left]), side([right]))
        values = export_row(rows[0])
        self.assertEqual(values["statut"], "apparié")
        self.assertEqual(values["score"], "0.9123")
        self.assertNotIn("rubriques_correspondantes", values)
        self.assertEqual(values["gauche_volume"], "VOL_l")
        self.assertEqual(values["gauche_sujet"], "Dupont")
        self.assertEqual(values["droite_sujet"], "")

    def test_certainty_level_and_reasons(self):
        left = side([record("l0", 0), record("l1", 1), record("l2", 2), record("l3", 3)])
        right = side([record("r0", 0), record("r1", 1), record("r2", 2), record("r3", 3)])
        links = [
            Link("l0", "r0", 0.95, "nw"),
            Link("l1", "r1", 0.6, "nw-contexte"),
            Link("l2", "r2", None, SOURCE_MANUAL_UNCERTAIN),
            Link("l3", "r3", 0.8, SOURCE_CANDIDATE),
        ]
        reviews = {("l1", "r1"): Review(("déduite des voisines (p < 0,9)",), 2), ("l3", "r3"): Review(("candidate non appariée",), 2)}
        rows, _ = natural_rows(links, left, right)
        values = [export_row(row, reviews) for row in rows]
        columns = [(v["statut"], v["certitude"], v["niveau_incertitude"], v["motifs_relecture"]) for v in values]
        self.assertEqual(
            columns,
            [
                ("apparié", "automatique", "faible", ""),
                ("apparié", "automatique", "forte", "déduite des voisines (p < 0,9)"),
                ("apparié", "incertaine", "", ""),
                ("candidate", "", "forte", "candidate non appariée"),
            ],
        )
        self.assertEqual(rows[3].kind, CANDIDATE)
        manual, _ = natural_rows([Link("l0", "r0", None, SOURCE_MANUAL)], left, right)
        self.assertEqual(export_row(manual[0])["certitude"], "relue")
        self.assertEqual(export_row(rows[0])["niveau_incertitude"], "")

    def test_export_csv_excel(self):
        rows, _ = natural_rows([], side([record("l0", 0)]), {})
        content = export_csv(rows, excel=True)
        parsed = list(csv.DictReader(io.StringIO(content), delimiter=";"))
        self.assertEqual(list(parsed[0]), EXPORT_FIELDS)
        self.assertEqual((parsed[0]["statut"], parsed[0]["score"], parsed[0]["droite_uuid"]), ("gauche seulement", "", ""))


if __name__ == "__main__":
    unittest.main()
