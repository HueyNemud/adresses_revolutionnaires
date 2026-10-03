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
- la **vérité** (`classe`) est alignée sur ces lignes par `uid`,
  puis, pour les lignes restantes, par (page, texte) unique. Une ligne du
  JSON sans correspondant fiable reste sans label : elle fait partie de la
  séquence observée mais n'est ni apprise ni évaluée.

Si le JSON est absent, le texte du CSV curé est utilisé à défaut, et le
document est signalé (`observations_from_curated_text`).

La classe d'origine (sortie de l'annotateur, pour mesurer ce que la curation
a changé) est relue dans `<nom>.ocr.lines.annotated.json` ; à défaut, c'est
la classe du CSV pour une ligne non corrigée (`corrige` vide : elle est
restée celle de la machine), inconnue pour une ligne corrigée.
"""

import csv
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from lib.chandra_document import iter_line_locations
from lib.crf.active_learning import SourceLine, context_from_records, load_json_lines
from lib.crf.features import SequenceContext
from lib.crf.labels import CLASSES
from lib.curation import DELETED_CLASS, is_corrected

SILVER_GLOB = "*.ocr.lines.annotated.csv"
GOLD_COLUMN = "classe"


@dataclass(frozen=True)
class CurationStats:
    """Écarts entre les lignes vues par l'annotateur et le CSV curé."""

    source_lines: int
    matched_by_uid: int
    matched_by_text: int
    unmatched_source_lines: int
    curated_lines_without_source: int
    ambiguous_uids: int
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
    observations_from_curated_text: bool = False
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


def _records_from_curated_rows(rows: list[dict[str, str]]) -> list[SourceLine]:
    """Reconstruit les lignes depuis le CSV (repli quand le JSON est absent)."""
    records: list[SourceLine] = []
    page_positions: dict[str, int] = {}
    block_positions: dict[tuple[str, str], int] = {}
    previous_blank = False
    for row in rows:
        text = (row.get("markdown") or "").strip()
        if not text:
            previous_blank = True
            continue
        page = row.get("page_index", "")
        block_key = (page, row.get("data_block_index", ""))
        records.append(
            SourceLine(
                uid=row.get("uid", ""),
                source_row=len(records) + 1,
                text=text,
                page_index=page,
                chunk_index=row.get("chunk_index", ""),
                data_block_index=row.get("data_block_index", ""),
                line_index=row.get("line_index", ""),
                data_block_bbox=row.get("data_block_bbox", ""),
                data_block_label=row.get("data_block_label", ""),
                page_pos=page_positions.setdefault(page, len(page_positions)),
                block_pos=block_positions.setdefault(block_key, len(block_positions)),
                line_pos=int(row.get("line_index") or 0),
                follows_blank=previous_blank,
            )
        )
        previous_blank = False
    return records


def _machine_annotations(json_path: Path) -> dict[str, tuple[str, str]] | None:
    """uid → (classe, provenance) écrites par l'annotateur ; None sans JSON."""
    if not json_path.exists():
        return None
    document = json.loads(json_path.read_text(encoding="utf-8"))
    return {
        str(location.line.get("uid", "")): (location.line.get("prediction", ""), location.line.get("provenance", ""))
        for location in iter_line_locations(document)
    }


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
    from_curated_text = not json_path.exists()
    if from_curated_text:
        records = _records_from_curated_rows(rows)
    else:
        records, _, _ = load_json_lines(json_path)

    # Alignement 1 : par uid (les uids dupliqués dans le CSV sont écartés).
    uid_counts = Counter(row["uid"] for row in rows)
    by_uid = {row["uid"]: row for row in rows if uid_counts[row["uid"]] == 1}
    matched: list[dict[str, str] | None] = [by_uid.get(record.uid) for record in records]
    used = {id(row) for row in matched if row is not None}

    # Alignement 2 : par (page, texte) unique parmi les lignes restantes.
    remaining = [row for row in rows if id(row) not in used and uid_counts[row["uid"]] == 1]
    text_counts = Counter((row["page_index"], row["markdown"].strip()) for row in remaining)
    by_text = {
        (row["page_index"], row["markdown"].strip()): row
        for row in remaining
        if text_counts[(row["page_index"], row["markdown"].strip())] == 1
    }
    matched_by_text = 0
    for index, record in enumerate(records):
        if matched[index] is None:
            row = by_text.pop((record.page_index, record.text), None)
            if row is not None:
                matched[index] = row
                used.add(id(row))
                matched_by_text += 1

    gold = [row[GOLD_COLUMN].strip() if row else None for row in matched]
    machine = _machine_annotations(csv_path.with_name(f"{name}.ocr.lines.annotated.json"))

    def original(row: dict[str, str]) -> str:
        if machine is not None:
            return (machine.get(row.get("uid", ""), ("", ""))[0] or "").strip()
        return "" if is_corrected(row) else row[GOLD_COLUMN].strip()

    original_prediction = [original(row) if row else "" for row in matched]
    original_provenance = [(row.get("provenance") or "").strip() if row else "" for row in matched]
    pairs = [(record, row) for record, row in zip(records, matched) if row is not None]
    stats = CurationStats(
        source_lines=len(records),
        matched_by_uid=len(pairs) - matched_by_text,
        matched_by_text=matched_by_text,
        unmatched_source_lines=len(records) - len(pairs),
        curated_lines_without_source=len(rows) - len(used),
        ambiguous_uids=sum(count > 1 for count in uid_counts.values()),
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
        observations_from_curated_text=from_curated_text,
    )


def discover_silver_csvs(root: Path) -> list[Path]:
    return sorted(root.rglob(SILVER_GLOB))


def load_silver_corpus(paths: Sequence[Path]) -> list[SilverDocument]:
    return [load_silver_document(path) for path in paths]
