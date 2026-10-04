import json
import tempfile
import unittest
from pathlib import Path

from numrev.curation import (
    CORRECTED,
    Curation,
    CurationConflict,
    Step,
    fingerprint,
    line_keys,
    read_csv,
    write_csv,
)
from numrev.pipeline.extract import assign_line_keys
from numrev.pipeline.tabulate import process_json_to_csv
from numrev.pipeline.tag import NER_STEP, apply_corrections
from numrev.titles import ROOT_UUID as ROOT


def page(index: int, blocks: list[list[tuple[str, str]]]) -> dict:
    """Page Chandra annotée : blocs de lignes (texte, classe)."""
    return {
        "page_index": index,
        "data_blocks": [
            {
                "index": block_index,
                "bbox": [0, 0, 1, 1],
                "label": "Text",
                "chunk_index": block_index,
                "lines": [
                    {
                        "uid": f"{index}.{block_index}.{line_index}",
                        "line_index": line_index,
                        "markdown": text,
                        "prediction": label,
                        "provenance": "model",
                        "probability": 0.9,
                        "timestamp": "",
                    }
                    for line_index, (text, label) in enumerate(lines)
                ],
            }
            for block_index, lines in enumerate(blocks, start=1)
        ],
    }


DOCUMENT = [
    page(
        6,
        [
            [("## PAPETIERS.", "B-TITLE")],
            [("Auzou, rue d'Anjou, 19.", "B-ENTRY"), ("Badet, rue Helvétius, 37.", "B-ENTRY"), ("et cartier", "B-ENTRY")],
        ],
    ),
    page(7, [[("---", "OUT OF SCOPE"), ("Bertaux, rue St.-Jacques.", "B-ENTRY"), ("---", "OUT OF SCOPE")]]),
]
assign_line_keys(DOCUMENT)


class LineKeyTests(unittest.TestCase):
    def test_keys_depend_on_text_and_occurrence_only(self):
        keys = line_keys(["a", "---", "b", "---", " a "])
        self.assertEqual(len(set(keys)), 5)
        self.assertEqual(keys[3], keys[1] + "~2")
        self.assertEqual(keys[4], keys[0] + "~2")  # blancs normalisés
        self.assertEqual(line_keys(["x", "b"])[1], keys[2])

    def test_export_keeps_the_keys_written_at_extraction(self):
        texts = [line["markdown"] for page_ in DOCUMENT for block in page_["data_blocks"] for line in block["lines"]]
        with tempfile.TemporaryDirectory() as tmp:
            json_path, csv_path = Path(tmp) / "doc.json", Path(tmp) / "doc.csv"
            json_path.write_text(json.dumps(DOCUMENT), encoding="utf-8")
            process_json_to_csv(json_path, csv_path, Path(tmp) / "doc.patch.csv", capture=False)
            self.assertEqual([row["cle"] for row in read_csv(csv_path)[1]], line_keys(texts))

    def test_fingerprint_ignores_whitespace_changes(self):
        self.assertEqual(fingerprint({"a": "x  y "}, ["a"]), fingerprint({"a": "x y"}, ["a"]))
        self.assertNotEqual(fingerprint({"a": "x", "b": ""}, ["a", "b"]), fingerprint({"a": "", "b": "x"}, ["a", "b"]))


class LinesCurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.csv = self.dir / "Vol.6-7.lines.csv"
        self.patch = self.dir / "Vol.6-7.lines.patch.csv"
        self.export(DOCUMENT)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def export(self, document, **options) -> Curation:
        """Exporte le document tel que l'extraction l'aurait écrit (clés
        recalculées sur son texte : une ré-OCR change les clés)."""
        document = json.loads(json.dumps(document))
        assign_line_keys(document)
        json_path = self.dir / "doc.json"
        json_path.write_text(json.dumps(document), encoding="utf-8")
        return process_json_to_csv(json_path, self.csv, self.patch, **options)

    def rows(self) -> list[dict[str, str]]:
        return read_csv(self.csv)[1]

    def edit(self, change) -> None:
        fieldnames, rows = read_csv(self.csv)
        rows = change(rows) or rows
        write_csv(self.csv, fieldnames, rows)

    def test_first_export_has_no_correction(self):
        rows = self.rows()
        self.assertEqual([row["corrige"] for row in rows], [""] * 7)
        self.assertTrue(all(row["empreinte"] for row in rows))
        self.assertFalse(self.patch.exists())

    def test_marked_corrections_survive_a_new_run(self):
        def change(rows):
            rows[3].update(classe="I-ENTRY", markdown="*et cartier*", corrige="oui")
            rows[1]["corrige"] = "oui"  # validée telle quelle

        self.edit(change)
        self.export(DOCUMENT)
        rows = self.rows()
        self.assertEqual((rows[3]["classe"], rows[3]["markdown"], rows[3]["corrige"]), ("I-ENTRY", "*et cartier*", CORRECTED))
        self.assertEqual(rows[1]["corrige"], CORRECTED)
        self.assertEqual(len(read_csv(self.patch)[1]), 2)
        first_patch = self.patch.read_text()
        self.export(DOCUMENT)  # capture idempotente
        self.assertEqual(self.patch.read_text(), first_patch)

    def test_unmarked_edit_panics_without_writing(self):
        self.edit(lambda rows: rows[2].update(classe="I-ENTRY"))
        before = self.csv.read_text()
        with self.assertRaises(CurationConflict) as raised:
            self.export(DOCUMENT)
        self.assertIn("modifiée sans", raised.exception.problems[0])
        self.assertEqual(self.csv.read_text(), before)
        self.assertFalse(self.patch.exists())

    def test_force_takes_the_machine_output_and_cancels_a_correction(self):
        self.edit(lambda rows: rows[2].update(classe="I-ENTRY", corrige="oui"))
        self.export(DOCUMENT)
        self.edit(lambda rows: rows[2].update(corrige=""))  # annuler la correction
        with self.assertRaises(CurationConflict):
            self.export(DOCUMENT)
        curation = self.export(DOCUMENT, force=True)
        self.assertEqual(self.rows()[2]["classe"], "B-ENTRY")
        self.assertEqual(curation.report.unmarked_discarded, [self.rows()[2]["cle"]])
        self.assertEqual(read_csv(self.patch)[1], [])

    def test_added_and_duplicated_lines_keep_their_place(self):
        def change(rows):
            added = {**rows[1], "cle": "", "markdown": "Auzou fils, rue d'Anjou, 21.", "classe": "B-ENTRY"}
            duplicate = {**rows[5], "markdown": "Bertaux (Ve.), rue St.-Jacques."}
            return [*rows[:2], added, *rows[2:6], duplicate, *rows[6:]]

        self.edit(change)
        self.export(DOCUMENT)
        rows = self.rows()
        self.assertEqual(len(rows), 9)
        self.assertEqual(rows[2]["markdown"], "Auzou fils, rue d'Anjou, 21.")
        self.assertEqual(rows[2]["cle"], rows[1]["cle"] + "+1")
        self.assertEqual(rows[7]["cle"], rows[6]["cle"] + "+1")
        self.assertEqual({row["corrige"] for row in (rows[2], rows[7])}, {CORRECTED})
        patch = read_csv(self.patch)[1]
        self.assertEqual([row["apres"] for row in patch], [rows[1]["cle"], rows[6]["cle"]])
        self.export(DOCUMENT, capture=False)  # depuis le seul patch
        self.assertEqual([row["markdown"] for row in self.rows()], [row["markdown"] for row in rows])

    def test_erased_line_panics_deleted_class_does_not(self):
        self.edit(lambda rows: [row for row in rows if row["uid"] != "7.1.1"])
        with self.assertRaisesRegex(CurationConflict, "problème"):
            self.export(DOCUMENT)
        self.export(DOCUMENT, capture=False)  # repartir du patch : la ligne revient
        self.edit(lambda rows: rows[5].update(classe="SUPPRIMÉE", corrige="oui"))
        self.export(DOCUMENT)
        self.assertEqual(self.rows()[5]["classe"], "SUPPRIMÉE")

    def test_moved_machine_line_panics(self):
        self.edit(lambda rows: [rows[0], rows[2], rows[1], *rows[3:]])
        with self.assertRaises(CurationConflict) as raised:
            self.export(DOCUMENT)
        self.assertIn("déplacée", raised.exception.problems[0])
        self.export(DOCUMENT, force=True)  # retour à l'ordre de l'OCR
        self.assertTrue(self.rows()[1]["markdown"].startswith("Auzou"))

    def test_keys_survive_page_insertion_and_resegmentation(self):
        self.edit(lambda rows: rows[2].update(classe="I-ENTRY", corrige="oui"))
        self.export(DOCUMENT)
        resegmented = [
            page(5, [[("Page de garde", "OUT OF SCOPE")]]),
            page(
                7,
                [
                    [("## PAPETIERS.", "B-TITLE"), ("Auzou, rue d'Anjou, 19.", "B-ENTRY")],
                    [("Badet, rue Helvétius, 37.", "B-ENTRY"), ("et cartier", "B-ENTRY")],
                ],
            ),
            DOCUMENT[1],
        ]
        with self.assertRaises(CurationConflict):  # nouvelle ligne absente du fichier édité
            self.export(resegmented)
        self.export(resegmented, force=True)
        badet = next(row for row in self.rows() if row["markdown"].startswith("Badet"))
        self.assertEqual((badet["uid"], badet["classe"], badet["corrige"]), ("7.2.0", "I-ENTRY", CORRECTED))

    def test_correction_of_a_vanished_line_panics(self):
        self.edit(lambda rows: rows[2].update(classe="I-ENTRY", corrige="oui"))
        self.export(DOCUMENT)
        reocr = json.loads(json.dumps(DOCUMENT, ensure_ascii=False).replace("Helvétius", "Helvetius"))
        with self.assertRaisesRegex(CurationConflict, "problème") as raised:
            self.export(reocr, capture=False)
        self.assertIn("introuvable", raised.exception.problems[0])
        curation = self.export(reocr, capture=False, force=True)
        self.assertEqual(len(curation.report.dropped), 1)
        self.assertEqual(read_csv(self.patch)[1], [])

    def test_old_format_file_panics(self):
        write_csv(self.csv, ["uid", "markdown", "prediction"], [{"uid": "1", "markdown": "x", "prediction": "B-ENTRY"}])
        with self.assertRaisesRegex(CurationConflict, "problème"):
            self.export(DOCUMENT)

    def test_raw_document_is_exported_without_patch(self):
        raw = json.loads(json.dumps(DOCUMENT))
        for p in raw:
            for block in p["data_blocks"]:
                for line in block["lines"]:
                    for name in ("prediction", "provenance", "probability", "timestamp"):
                        del line[name]
        self.assertIsNone(self.export(raw))
        self.assertEqual({row["classe"] for row in self.rows()}, {""})


class CheckTests(unittest.TestCase):
    def test_check_refuses_a_stale_correction(self):
        step = Step("ner", "uuid", ("tagged_text",))
        with tempfile.TemporaryDirectory() as tmp:
            patch = Path(tmp) / "p.csv"
            write_csv(patch, step.patch_fields, [{"uuid": "u1", "tagged_text": "<SUBJ>A</SUBJ>"}])
            curation = Curation(step, Path(tmp) / "absent.csv", patch)
            with self.assertRaises(CurationConflict):
                curation.apply([{"uuid": "u1", "tagged_text": ""}], check=lambda correction, row: "texte modifié")
            rows = Curation(step, Path(tmp) / "absent.csv", patch).apply([{"uuid": "u1", "tagged_text": ""}])
        self.assertEqual(rows[0]["tagged_text"], "<SUBJ>A</SUBJ>")


if __name__ == "__main__":
    unittest.main()


class NerCurationTests(unittest.TestCase):
    """Réapplication des corrections NER (numrev tag), sans modèle."""

    def machine(self) -> list[dict[str, str]]:
        return [
            {"uuid": "t1", "parent_uuid": ROOT, "uid": "1", "entity": "TITLE", "markdown": "## A", "tagged_text": ""},
            {"uuid": "t2", "parent_uuid": ROOT, "uid": "2", "entity": "TITLE", "markdown": "## B", "tagged_text": ""},
            {
                "uuid": "e1",
                "parent_uuid": "t2",
                "uid": "3",
                "entity": "ENTRY",
                "markdown": "Dupont, rue A.",
                "tagged_text": "<SUBJ>Dupont, rue A.</SUBJ>",
                "subject_count": "1",
                "address_count": "0",
            },
        ]

    def run_ner(self, directory: Path, rows, **options) -> list[dict[str, str]]:
        output, patch = directory / "Vol.ocr.ner.csv", directory / "Vol.ner.patch.csv"
        curation = Curation(NER_STEP, output, patch, **options)
        rows = apply_corrections(curation, rows)
        curation.save_patch()
        write_csv(output, list(dict.fromkeys(name for row in rows for name in row)), rows)
        return read_csv(output)[1]

    def correct(self, directory: Path, **values) -> None:
        output = directory / "Vol.ocr.ner.csv"
        fieldnames, rows = read_csv(output)
        rows[2].update(values, corrige="oui")
        write_csv(output, fieldnames, rows)

    def test_spans_and_parent_corrections_are_reapplied_and_recounted(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.run_ner(directory, self.machine())
            self.correct(directory, tagged_text="<SUBJ>Dupont</SUBJ>, <ADDR>rue A</ADDR>.", parent_uuid="t1")
            rows = self.run_ner(directory, self.machine())
        self.assertEqual((rows[2]["parent_uuid"], rows[2]["subject_count"], rows[2]["address_count"]), ("t1", "1", "1"))

    def test_correction_on_changed_text_or_missing_parent_panics(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.run_ner(directory, self.machine())
            self.correct(directory, tagged_text="<SUBJ>Dupont</SUBJ>, rue A.")
            self.run_ner(directory, self.machine())
            changed = self.machine()
            changed[2]["markdown"] = "Dupont, rue B."
            with self.assertRaisesRegex(CurationConflict, "problème") as raised:
                self.run_ner(directory, changed)
            self.assertIn("texte de l'entité modifié", raised.exception.problems[0])
            rows = self.run_ner(directory, changed, force=True)
        self.assertEqual(rows[2]["tagged_text"], "<SUBJ>Dupont, rue A.</SUBJ>")

    def test_correction_whose_parent_vanished_panics(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.run_ner(directory, self.machine())
            self.correct(directory, parent_uuid="t1")
            self.run_ner(directory, self.machine())
            without_t1 = self.machine()[1:]
            with self.assertRaisesRegex(CurationConflict, "problème") as raised:
                self.run_ner(directory, without_t1)
        self.assertIn("titre parent", raised.exception.problems[0])
