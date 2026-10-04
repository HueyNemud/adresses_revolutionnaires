import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from numrev.crf.evaluation import (
    N_CLASSES,
    Experiment,
    cross_volume_splits,
    group_entities,
    macro_f1,
    page_confusions,
    page_entity_counts,
    prf,
    run_experiments,
    score_lines,
    within_document_splits,
)
from numrev.crf.features import PRODUCTION_GROUPS
from numrev.crf.labels import CLASSES
from numrev.crf.silver import document_names, load_silver_document
from numrev.pipeline.extract import assign_line_keys
from numrev.stats import roc_auc

FIELDS = [
    "cle",
    "uid",
    "page_index",
    "chunk_index",
    "data_block_index",
    "line_index",
    "data_block_bbox",
    "data_block_label",
    "markdown",
    "classe",
    "provenance",
]


def write_volume(folder: Path, name: str, pages: int, edit_first_title: bool = False) -> Path:
    """Écrit le JSON vu par l'annotateur, celui qu'il a exporté et le CSV curé."""
    document, rows = [], []
    for page in range(pages):
        lines = [
            ("# TITRE", "B-TITLE"),
            ("Dupont, rue A, 1.", "B-ENTRY"),
            ("rue B, 2.", "I-ENTRY"),
            ("Durand, rue C, 3.", "B-ENTRY"),
            (str(page), "OUT OF SCOPE"),
        ]
        block_lines = []
        for index, (text, label) in enumerate(lines):
            uid = f"{page}.1.{index}"
            block_lines.append({"uid": uid, "line_index": index, "markdown": text, "prediction": label})
            curated_text = "## TITRE" if edit_first_title and index == 0 else text
            rows.append(
                {
                    "uid": uid,
                    "page_index": page,
                    "chunk_index": 0,
                    "data_block_index": 1,
                    "line_index": index,
                    "data_block_bbox": "[0, 0, 1, 1]",
                    "data_block_label": "Text",
                    "markdown": curated_text,
                    "classe": label,
                    "provenance": "model",
                }
            )
        document.append(
            {
                "page_index": page,
                "data_blocks": [{"index": 1, "bbox": [0, 0, 1, 1], "label": "Text", "chunk_index": 0, "lines": block_lines}],
            }
        )
    assign_line_keys(document)
    for row, line in zip(rows, (line for page in document for block in page["data_blocks"] for line in block["lines"])):
        row["cle"] = line["cle"]
    (folder / f"{name}.labeled.json").write_text(json.dumps(document), encoding="utf-8")
    for page in document:
        for block in page["data_blocks"]:
            for line in block["lines"]:
                del line["prediction"]
    (folder / f"{name}.lines.json").write_text(json.dumps(document), encoding="utf-8")
    csv_path = folder / f"{name}.lines.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


class SilverTests(unittest.TestCase):
    def test_document_names(self):
        self.assertEqual(
            document_names(Path("1808_AD75-PER292.6-185.lines.csv")),
            ("1808_AD75-PER292.6-185", "1808_AD75-PER292"),
        )

    def test_observations_come_from_source_json_not_curated_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_volume(Path(tmp), "V1.1-3", pages=3, edit_first_title=True)
            document = load_silver_document(path)
        self.assertEqual(document.records[0].text, "# TITRE")
        self.assertEqual(document.gold[:3], ["B-TITLE", "B-ENTRY", "I-ENTRY"])
        self.assertEqual(document.stats.text_edited, 3)
        self.assertEqual(document.stats.heading_marker_changed, 3)
        self.assertEqual(document.stats.unmatched_source_lines, 0)
        self.assertEqual(document.stats.label_changed_vs_original, 0)


class EvaluationTests(unittest.TestCase):
    def test_group_entities_follows_merge_rules(self):
        labels = ["I-ENTRY", "B-TITLE", "B-ENTRY", "OUT OF SCOPE", "I-TITLE", "I-ENTRY", "SUB-ENTRY", None, "B-ENTRY"]
        self.assertEqual(
            group_entities(labels),
            [("ENTRY", (0,)), ("TITLE", (1, 4)), ("ENTRY", (2, 5, 6)), ("ENTRY", (8,))],
        )

    def test_prf_and_macro_f1(self):
        cm = np.zeros((N_CLASSES, N_CLASSES))
        cm[0, 0], cm[0, 1], cm[1, 1] = 8, 2, 5
        precision, recall, f1, support = prf(cm)
        self.assertAlmostEqual(precision[1], 5 / 7)
        self.assertAlmostEqual(recall[0], 0.8)
        mask = support > 0
        self.assertAlmostEqual(float(macro_f1(cm, mask)), (f1[0] + f1[1]) / 2)

    def test_roc_auc(self):
        self.assertEqual(roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([False, False, True, True])), 1.0)
        self.assertEqual(roc_auc(np.array([0.5, 0.5]), np.array([False, True])), 0.5)

    def test_cross_validation_predicts_each_line_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            documents = [
                load_silver_document(write_volume(Path(tmp), "V1.1-6", pages=6)),
                load_silver_document(write_volume(Path(tmp), "V2.1-5", pages=5)),
            ]
        splits = within_document_splits(documents, folds=3) + cross_volume_splits(documents)
        for protocol in ("intra-document", "inter-volumes"):
            covered = [np.zeros(len(d), int) for d in documents]
            for split in splits:
                if split.protocol == protocol:
                    for doc, start, end in split.test:
                        covered[doc][start:end] += 1
            self.assertTrue(all((c == 1).all() for c in covered), protocol)

        predictions = run_experiments(documents, [Experiment("production", PRODUCTION_GROUPS)], splits, workers=1)
        scored = score_lines(predictions[("production", "intra-document")], documents)
        self.assertEqual(len(scored), sum(len(d) for d in documents))
        self.assertEqual(page_confusions(scored).sum(), len(scored))
        counts = page_entity_counts(scored, documents).sum(axis=0)
        self.assertEqual(counts[1, 2], 11)  # un titre attendu par page
        self.assertGreater((scored.gold == scored.pred).mean(), 0.9)
        self.assertEqual(scored.marginals.shape[1], len(CLASSES))


if __name__ == "__main__":
    unittest.main()
