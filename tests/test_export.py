import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path

import jsonschema

from numrev.command import CommandError
from numrev.curation import CORRECTED, CORRECTED_COLUMN, line_keys, read_csv, write_csv
from numrev.devtools.status import DONE, MISSING, STALE, export_state
from numrev.export.layers import ORIGIN_ADDED, ExportReport, build_range, volume_layers
from numrev.ner.spans import Span, render_tagged_text
from numrev.paths import DOCUMENT_JSON, STRUCTURE_JSON, volume_file
from numrev.pipeline import export
from numrev.pipeline.assemble import assemble_document
from numrev.pipeline.tabulate import CSV_FIELDS, iter_csv_rows
from numrev.titles import ROOT_UUID

SCHEMAS = Path(export.__file__).resolve().parents[1] / "export" / "schemas"

# (page, texte, classe) des lignes OCR de la plage 7-8.
OCR_LINES = [
    (7, "## MARCHANDS.", "B-TITLE"),
    (7, "Dupont, *boulanger*, rue de la", "B-ENTRY"),
    (7, "Paix, 12.", "I-ENTRY"),
    (7, "Durand, rue Neuve, 3.", "B-ENTRY"),
    (7, "Ligne parasite", "OUT OF SCOPE"),
    (7, "doublon", "B-ENTRY"),
    (8, "Fieffé. } *Membres du conseil.*", "B-ENTRY"),
    (8, "Pastoret. }", "SUB-ENTRY"),
    (8, "Martin, rue A, 1.", "B-ENTRY"),
]
# Empans de chaque entrée : (classe, sous-chaîne du texte de l'entrée).
SPANS = {
    "Dupont": [("SUBJ", "Dupont"), ("DESC", "*boulanger*"), ("ADDR", "rue de la Paix, 12.")],
    "Durand": [("SUBJ", "Durand"), ("ADDR", "rue Neuve, 3.")],
    "Fieffé": [("SUBJ", "Fieffé"), ("DESC", "*Membres du conseil.*"), ("SUBJ", "Pastoret")],
    "Martin": [("SUBJ", "Martin"), ("ADDR", "rue A, 1.")],
}


def lines_json() -> list[dict]:
    """JSON des lignes (`numrev extract`, annoté) : une page par numéro, un bloc par page."""
    keys = line_keys(text for _, text, _ in OCR_LINES)
    pages: dict[int, dict] = {}
    for number, ((page, text, label), key) in enumerate(zip(OCR_LINES, keys), start=1):
        block = pages.setdefault(
            page,
            {
                "page_index": page,
                "page_box": [0, 0, 1400, 2200],
                "data_blocks": [{"index": 1, "bbox": [50, 60, 950, 940], "label": "List-Group", "lines": []}],
            },
        )["data_blocks"][0]
        line = {"uid": f"{page}.1.{len(block['lines']) + 1}", "markdown": text, "cle": key, "prediction": label, "probability": 0.99}
        if number != 4:  # « Durand » : sans confiance de mots
            line["word_confidence"] = [[word, 0.8 if word == "Dupont," else 0.99] for word in text.split()]
        if number != 2:  # « Dupont » : sans boîte estimée
            line["line_bbox_estimate"] = [60, 60 + 20 * number, 900, 78 + 20 * number]
        block["lines"].append(line)
    return list(pages.values())


def tag(row: dict[str, str]) -> str:
    text = row["markdown"].strip()
    spans, cursor = [], 0
    for label, fragment in SPANS[text.split(",")[0].split(".")[0]]:
        start = text.index(fragment, cursor)
        spans.append(Span(start, start + len(fragment), label))
        cursor = start + len(fragment)
    return render_tagged_text(text, spans)


def write_range(range_dir: Path) -> None:
    """Les fichiers des étapes 1 à 5 d'une plage, construits avec les fonctions de `tabulate` et `assemble` :
    un titre restitué à la main en tête (`debut+1`), une ligne supprimée."""
    document = f"{range_dir.parent.name}.{range_dir.name}"
    pages = lines_json()
    (range_dir / f"{document}.lines.json").write_text(json.dumps(pages), encoding="utf-8")
    rows = list(iter_csv_rows(pages))
    for row in rows:
        row |= {CORRECTED_COLUMN: "", "empreinte": ""}
        if row["markdown"] == "doublon":
            row |= {"classe": "SUPPRIMÉE", CORRECTED_COLUMN: CORRECTED}
    added = {**rows[0], "cle": "debut+1", "uid": "", "markdown": "# CONSEIL", "classe": "B-TITLE", CORRECTED_COLUMN: CORRECTED}
    added |= dict.fromkeys(("confiance", "douteuse", "provenance", "probability", "timestamp"), "")
    write_csv(range_dir / f"{document}.lines.csv", CSV_FIELDS, [added, *rows])
    assemble_document(range_dir / f"{document}.lines.csv", range_dir / f"{document}.entities.csv")
    fieldnames, entities = read_csv(range_dir / f"{document}.entities.csv")
    for row in entities:
        entry = row["entity"] == "ENTRY"
        row |= {
            "tagged_text": tag(row) if entry else "",
            "ner_confidence": "0.9500" if entry else "",
            "ner_suspect": "",
            CORRECTED_COLUMN: "",
        }
    write_csv(range_dir / f"{document}.ner.csv", [*fieldnames, "tagged_text", "ner_confidence", "ner_suspect", CORRECTED_COLUMN], entities)


def build(range_dir: Path, ner_edit=None):
    document = f"{range_dir.parent.name}.{range_dir.name}"
    ner_rows = read_csv(range_dir / f"{document}.ner.csv")[1]
    if ner_edit:
        ner_edit(ner_rows)
    report = ExportReport()
    layers = build_range(
        range_dir.name,
        json.loads((range_dir / f"{document}.lines.json").read_text(encoding="utf-8")),
        read_csv(range_dir / f"{document}.lines.csv")[1],
        ner_rows,
        report,
    )
    return layers, report


class ExportTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.volume_dir = Path(self.tmp.name) / "V"
        self.range_dir = self.volume_dir / "7-8"
        self.range_dir.mkdir(parents=True)
        write_range(self.range_dir)

    def tearDown(self) -> None:
        self.tmp.cleanup()


class BuildRangeTests(ExportTestCase):
    def test_document_layer_keeps_every_line_with_its_origin_and_geometry(self):
        layers, report = build(self.range_dir)
        lines = [line for page in layers.pages for line in page["lines"]]

        self.assertEqual([page["id"] for page in layers.pages], ["7-8:7", "7-8:8"])
        self.assertEqual(len(lines), len(OCR_LINES) + 1)
        added = lines[0]
        self.assertEqual(
            (added["origin"], added["uid"], added["bbox_estimate"], added["ocr"], added["ocr_text"]), (ORIGIN_ADDED, None, None, None, None)
        )
        deleted = next(line for line in lines if line["text"] == "doublon")
        self.assertEqual((deleted["class"], deleted["corrected"]), ("SUPPRIMÉE", True))
        dupont = lines[2]
        self.assertEqual((dupont["block"], dupont["bbox_estimate"], dupont["ocr"]["doubtful"]), ("7-8:7.1", None, True))
        self.assertEqual(dupont["labeling"], {"probability": 0.99, "provenance": None})
        self.assertEqual((report.lines, report.added_lines, report.deleted_lines), (10, 1, 1))
        self.assertEqual(report.lines_without_box, ["7-8 p. 7, ligne 7.1.2"])

    def test_titles_and_entries_point_back_to_their_lines(self):
        layers, _ = build(self.range_dir)
        lines = {line["id"]: line for page in layers.pages for line in page["lines"]}

        council, merchants = layers.titles
        self.assertEqual((council["parent"], council["level"], council["heading"]), (ROOT_UUID, 1, "CONSEIL"))
        self.assertEqual((merchants["parent"], merchants["level"]), (council["id"], 2))
        self.assertEqual(
            [entry["text"].split(",")[0] for entry in layers.entries],
            ["Dupont", "Durand", "Fieffé. } *Membres du conseil.*\nPastoret. }", "Martin"],
        )
        for item in layers.titles + layers.entries:
            for ref in item["lines"]:
                self.assertEqual(lines[ref["line"]]["text"].strip(), item["text"][ref["start"] : ref["end"]])

    def test_out_of_scope_lines_are_items_under_the_open_title(self):
        layers, report = build(self.range_dir)
        lines = {line["id"]: line for page in layers.pages for line in page["lines"]}

        (item,) = layers.out_of_scope
        self.assertEqual((item["parent"], item["text"], item["pages"]), (layers.titles[1]["id"], "Ligne parasite", [7]))
        (ref,) = item["lines"]
        self.assertEqual((lines[ref["line"]]["class"], ref["start"], ref["end"]), ("OUT OF SCOPE", 0, len("Ligne parasite")))
        self.assertEqual(report.out_of_scope, 1)

    def test_a_span_over_two_lines_has_one_piece_per_line(self):
        layers, _ = build(self.range_dir)
        lines = {line["id"]: line for page in layers.pages for line in page["lines"]}
        dupont = layers.entries[0]

        address = next(span for span in dupont["spans"] if span["label"] == "ADDR")
        self.assertEqual(address["text"], "rue de la Paix, 12.")
        self.assertEqual(
            [lines[piece["line"]]["text"][piece["start"] : piece["end"]] for piece in address["pieces"]], ["rue de la", "Paix, 12."]
        )
        self.assertEqual(dupont["pages"], [7])
        self.assertEqual(dupont["quality"]["ocr_confidence"], 0.8)
        self.assertTrue(dupont["quality"]["ocr_doubtful"])
        self.assertEqual(dupont["quality"]["ner_confidence"], 0.95)

    def test_brace_group_members_inherit_the_common_text(self):
        layers, report = build(self.range_dir)
        group = layers.entries[2]
        spans = {span["id"]: span["text"] for span in group["spans"]}

        self.assertEqual(group["kind"], "group")
        self.assertEqual([[spans[i] for i in member["spans"]] for member in group["members"]], [["Fieffé"], ["Pastoret"]])
        self.assertEqual([[spans[i] for i in member["inherited_spans"]] for member in group["members"]], [["*Membres du conseil.*"]] * 2)
        self.assertEqual([entry["members"] for entry in layers.entries if entry["kind"] == "single"], [None, None, None])
        self.assertEqual((report.groups, report.members, report.brace_problems), (1, 2, []))

    def test_span_ids_survive_corrections_of_other_spans(self):
        before, _ = build(self.range_dir)

        def drop_description(rows):
            row = next(row for row in rows if row["markdown"].startswith("Dupont"))
            row["tagged_text"] = row["tagged_text"].replace("<DESC>", "").replace("</DESC>", "")

        after, _ = build(self.range_dir, drop_description)
        ids = lambda layers: {span["text"]: span["id"] for span in layers.entries[0]["spans"]}  # noqa: E731
        self.assertEqual(ids(after), {text: id_ for text, id_ in ids(before).items() if text != "*boulanger*"})

    def test_tagged_text_that_no_longer_matches_the_lines_is_refused(self):
        def edit(rows):
            row = next(row for row in rows if row["markdown"].startswith("Durand"))
            row["tagged_text"] = row["tagged_text"].replace("Neuve", "Vieille")

        with self.assertRaisesRegex(ValueError, "ne redonne pas son texte"):
            build(self.range_dir, edit)

    def test_output_follows_the_schemas(self):
        layers, _ = build(self.range_dir)
        header = {
            "volume": "V",
            "source": {"pdf": "V.pdf", "sha256": None},
            "ranges": [{"name": "7-8", "first_page": 7, "last_page": 8}],
            "coordinates": export.COORDINATES,
            "provenance": {
                "numrev": {"commit": None, "uncommitted_changes": None},
                "exported_at": "2026-10-29T00:00:00+00:00",
                "inputs": {},
            },
        }
        for content, name in zip(volume_layers(header, [layers]), ("document", "structure")):
            schema = json.loads((SCHEMAS / f"{name}.schema.json").read_text(encoding="utf-8"))
            jsonschema.Draft202012Validator.check_schema(schema)
            jsonschema.validate(content, schema, cls=jsonschema.Draft202012Validator)

    def test_both_schemas_share_the_header_definitions(self):
        document, structure = (
            json.loads((SCHEMAS / f"{name}.schema.json").read_text(encoding="utf-8")) for name in ("document", "structure")
        )
        for name in ("source", "range", "coordinates", "provenance"):
            self.assertEqual(document["$defs"][name], structure["$defs"][name], name)


class ExportCommandTests(ExportTestCase):
    def setUp(self) -> None:
        super().setUp()
        # Sorties des étapes 0 à 5, chacune plus récente que son entrée (numrev status).
        document = self.range_dir / "V.7-8"
        (self.range_dir / "V.7-8.ocr.json").write_text("[]", encoding="utf-8")
        (self.range_dir / "V.7-8.labeled.json").write_text("[]", encoding="utf-8")
        for mtime, suffix in enumerate((".ocr.json", ".lines.json", ".labeled.json", ".lines.csv", ".entities.csv", ".ner.csv"), start=1):
            os.utime(f"{document}{suffix}", (mtime * 100, mtime * 100))

    def run_export(self, apply: bool = True) -> None:
        export.run(argparse.Namespace(folder=self.volume_dir, apply=apply))

    def test_writes_both_layers_and_the_report(self):
        self.assertEqual(export_state(self.volume_dir, [self.range_dir]), MISSING)

        self.run_export()

        structure = json.loads(volume_file(self.volume_dir, STRUCTURE_JSON).read_text(encoding="utf-8"))
        document = json.loads(volume_file(self.volume_dir, DOCUMENT_JSON).read_text(encoding="utf-8"))
        self.assertEqual((structure["schema"], document["schema"]), ("numrev-structure/2", "numrev-document/1"))
        self.assertEqual(structure["ranges"], [{"name": "7-8", "first_page": 7, "last_page": 8}])
        self.assertEqual(
            sorted(structure["provenance"]["inputs"]),
            sorted(str(self.range_dir / f"V.7-8{s}") for s in (".lines.json", ".lines.csv", ".ner.csv")),
        )
        self.assertTrue((self.volume_dir / "V.export.report.txt").exists())
        self.assertEqual(export_state(self.volume_dir, [self.range_dir]), DONE)

        os.utime(self.range_dir / "V.7-8.ner.csv")
        self.assertEqual(export_state(self.volume_dir, [self.range_dir]), STALE)

    def test_dry_run_writes_nothing(self):
        self.run_export(apply=False)

        self.assertFalse(volume_file(self.volume_dir, DOCUMENT_JSON).exists())

    def test_a_stale_range_is_refused(self):
        os.utime(self.range_dir / "V.7-8.entities.csv", (10_000, 10_000))

        with self.assertRaisesRegex(CommandError, "tag périmé"):
            self.run_export()

    def test_a_range_folder_is_refused(self):
        with self.assertRaisesRegex(CommandError, "prend un volume entier"):
            export.run(argparse.Namespace(folder=self.range_dir, apply=False))


if __name__ == "__main__":
    unittest.main()
