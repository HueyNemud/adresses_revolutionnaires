"""Chargement des « silver datasets » : CSV d'annotations CRF vérifiées et
corrigées à la main (`*.ocr.lines.annotated.csv`, colonne `classe` ; les
lignes de classe `SUPPRIMÉE` sont ignorées).

Observations et vérité viennent de deux fichiers distincts :

- les **observations** (texte, bloc OCR, page...) sont relues dans le JSON
  `<nom>.ocr.lines.json` voisin, c'est-à-dire exactement ce qu'a vu
  annotate_lines_crf.py. La curation a en effet aussi corrigé le texte
  (ajout / retrait de marqueurs de titre « # », ponctuation, ordre de
  quelques lignes) : calculer les features sur le texte curé ferait fuiter
  la vérité dans les features (« # » ajouté sur les titres) ;
- la **vérité** (`classe`) est rattachée à ces lignes par leur clé stable
  `cle` (lib/curation.py), présente dans le JSON et dans le CSV. Une ligne
  du JSON sans correspondant (supprimée ou vidée à la curation) reste sans
  label : elle fait partie de la séquence observée mais n'est ni apprise ni
  évaluée ; une ligne ajoutée à la main (clé `<clé>+n`) n'a pas
  d'observation et est ignorée.

La classe d'origine (sortie de l'annotateur, pour mesurer ce que la curation
a changé) est relue dans `<nom>.ocr.lines.annotated.json`.
"""

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from lib.chandra_document import iter_line_locations
from lib.crf.active_learning import SourceLine, context_from_records, load_json_lines
from lib.crf.features import SequenceContext
from lib.crf.labels import CLASSES
from lib.curation import DELETED_CLASS

SILVER_GLOB = "*.ocr.lines.annotated.csv"
GOLD_COLUMN = "classe"


@dataclass(frozen=True)
class CurationStats:
    """Écarts entre les lignes vues par l'annotateur et le CSV curé."""

    source_lines: int
    matched: int
    unmatched_source_lines: int
    added_lines: int
    text_edited: int
    heading_marker_changed: int
    label_changed_vs_original: int
    original_model_lines: int
    original_human_lines: int


@dataclass
class SilverDocument:
    """Un document : lignes observées, vérité curée (None si non alignée),
    et annotation d'origine produite par l'annotateur."""

    name: str
    volume: str
    path: Path
    records: list[SourceLine]
    gold: list[str | None]
    original_prediction: list[str]
    original_provenance: list[str]
    stats: CurationStats
    context: SequenceContext = field(init=False)

    def __post_init__(self) -> None:
        self.context = context_from_records(self.records)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def page_ids(self) -> list[str]:
        """Identifiant de page unique dans tout le corpus (unité de rééchantillonnage)."""
        return [f"{self.name}#{record.page_index}" for record in self.records]

    @property
    def labeled_count(self) -> int:
        return sum(label is not None for label in self.gold)


def document_names(csv_path: Path) -> tuple[str, str]:
    """(nom du document, nom du volume) déduits du nom de fichier.

    « 1808_AD75-PER292.6-185.ocr.lines.annotated.csv »
    → (« 1808_AD75-PER292.6-185 », « 1808_AD75-PER292 »).
    """
    name = csv_path.name.removesuffix(SILVER_GLOB.lstrip("*"))
    return name, name.split(".", 1)[0]


def _heading_prefix(text: str) -> str:
    stripped = text.replace("\xa0", " ").strip()
    return stripped[: len(stripped) - len(stripped.lstrip("#"))]


def _machine_labels(json_path: Path) -> dict[str, str]:
    """cle → classe écrite par l'annotateur."""
    document = json.loads(json_path.read_text(encoding="utf-8"))
    return {str(location.line.get("cle", "")): location.line.get("prediction", "") for location in iter_line_locations(document)}


def load_silver_document(csv_path: Path) -> SilverDocument:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if (row.get("markdown") or "").strip() and (row.get(GOLD_COLUMN) or "").strip() != DELETED_CLASS
        ]
    if rows and GOLD_COLUMN not in rows[0]:
        raise ValueError(f"{csv_path} : colonne '{GOLD_COLUMN}' absente.")
    for row in rows:
        if (row.get(GOLD_COLUMN) or "").strip() not in CLASSES:
            raise ValueError(
                f"{csv_path} : ligne {row.get('uid')!r} sans classe curée valide "
                f"({row.get(GOLD_COLUMN)!r})."
            )

    name, volume = document_names(csv_path)
    json_path = csv_path.with_name(f"{name}.ocr.lines.json")
    annotated_path = csv_path.with_name(f"{name}.ocr.lines.annotated.json")
    for path in (json_path, annotated_path):
        if not path.exists():
            raise ValueError(f"{path} introuvable : il faut le JSON vu par l'annotateur et celui qu'il a exporté.")
    records, _, _ = load_json_lines(json_path)
    if any(not record.cle for record in records):
        raise ValueError(f"{json_path} : lignes sans clé `cle` (relancez extract_chandra_lines.py).")

    by_key = {row["cle"]: row for row in rows}
    matched: list[dict[str, str] | None] = [by_key.get(record.cle) for record in records]
    used = {id(row) for row in matched if row is not None}

    gold = [row[GOLD_COLUMN].strip() if row else None for row in matched]
    machine = _machine_labels(annotated_path)

    def original(row: dict[str, str]) -> str:
        return (machine.get(row["cle"]) or "").strip()

    original_prediction = [original(row) if row else "" for row in matched]
    original_provenance = [(row.get("provenance") or "").strip() if row else "" for row in matched]
    pairs = [(record, row) for record, row in zip(records, matched) if row is not None]
    stats = CurationStats(
        source_lines=len(records),
        matched=len(pairs),
        unmatched_source_lines=len(records) - len(pairs),
        added_lines=len(rows) - len(used),
        text_edited=sum(record.text != row["markdown"].strip() for record, row in pairs),
        heading_marker_changed=sum(
            _heading_prefix(record.text) != _heading_prefix(row["markdown"])
            for record, row in pairs
        ),
        label_changed_vs_original=sum(
            original(row) != row[GOLD_COLUMN].strip() for _, row in pairs
        ),
        original_model_lines=sum(row.get("provenance") == "model" for _, row in pairs),
        original_human_lines=sum(row.get("provenance") == "human" for _, row in pairs),
    )
    return SilverDocument(
        name=name,
        volume=volume,
        path=csv_path,
        records=records,
        gold=gold,
        original_prediction=original_prediction,
        original_provenance=original_provenance,
        stats=stats,
    )


def discover_silver_csvs(root: Path) -> list[Path]:
    return sorted(root.rglob(SILVER_GLOB))


def load_silver_corpus(paths: Sequence[Path]) -> list[SilverDocument]:
    return [load_silver_document(path) for path in paths]
