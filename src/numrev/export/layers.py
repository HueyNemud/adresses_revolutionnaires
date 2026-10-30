"""Construction des deux couches du livrable d'un volume, plage par plage,
à partir des sorties des étapes 1 à 5 (format : `docs/format_export.md`).

Sources d'une plage :
- `<document>.lines.json` (`numrev extract`) : pages, blocs, boîtes, texte
  OCR et confiance des mots, retrouvés par la clé `cle` de chaque ligne ;
- `<document>.lines.csv` (`numrev tabulate`, corrigé à la main) : l'ordre des
  lignes, leur texte retenu, leur classe, les lignes ajoutées à la main ;
- `<document>.ner.csv` (`numrev tag`, corrigé à la main) : titres et entrées,
  leur parent, leurs empans et leurs indicateurs de qualité.

Fonctions pures : aucune lecture ni écriture de fichier. Une incohérence
entre ces sources (sortie d'une étape plus ancienne que son entrée, en
pratique) lève ValueError avec un message qui dit ce qui a été observé.
"""

import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from numrev.crf.labels import AnnotationLabel
from numrev.curation import DELETED_CLASS, is_corrected, is_inserted
from numrev.document import iter_line_locations
from numrev.export.braces import Zone, group_members
from numrev.ner.spans import Span, normalize_markdown, parse_tagged_text
from numrev.ner.suspicion import SEPARATOR
from numrev.pipeline.tabulate import CLASS_COLUMN, KEY_COLUMN, OCR_DOUBTFUL_THRESHOLD
from numrev.titles import ROOT_UUID, title_level, title_text

DOCUMENT_SCHEMA = "numrev-document/1"
STRUCTURE_SCHEMA = "numrev-structure/1"

# Espaces de noms des identifiants d'empans et de membres : les changer
# changerait tous les identifiants déjà publiés.
SPAN_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "adresses_revolutionnaires/empans")
MEMBER_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "adresses_revolutionnaires/membres")

ORIGIN_OCR = "ocr"
ORIGIN_ADDED = "added"
KIND_SINGLE = "single"
KIND_GROUP = "group"
SUB_ENTRY = AnnotationLabel.SUBENTRY.value
EXCERPT_LENGTH = 60


@dataclass
class ExportReport:
    """Comptes et cas à vérifier, accumulés plage par plage."""

    ranges: list[str] = field(default_factory=list)
    pages: int = 0
    lines: int = 0
    added_lines: int = 0
    deleted_lines: int = 0
    lines_without_box: list[str] = field(default_factory=list)  # lignes OCR sans boîte estimée
    titles: int = 0
    entries: int = 0
    groups: int = 0
    members: int = 0
    spans: Counter[str] = field(default_factory=Counter)
    brace_problems: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Line:
    """Ce que la couche structurée retient d'une ligne de la couche documentaire."""

    id: str
    uid: str
    text: str
    label: str
    page: int
    confidence: float | None
    doubtful: bool
    corrected: bool


def line_id(range_name: str, key: str) -> str:
    """Identifiant d'une ligne dans le volume : `<plage>:<cle>` (une clé n'est unique que dans sa plage)."""
    return f"{range_name}:{key}"


def excerpt(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= EXCERPT_LENGTH else text[: EXCERPT_LENGTH - 1] + "…"


# ----------------------------------------------------------------------
# Couche documentaire
# ----------------------------------------------------------------------
def _optional_float(value: str | None) -> float | None:
    value = (value or "").strip().replace(",", ".")
    return float(value) if value else None


def _ocr_quality(line: dict[str, Any]) -> dict[str, Any] | None:
    """Mots et confiances d'une ligne OCR ; `confidence` = plus faible score de mot,
    `doubtful` sous `OCR_DOUBTFUL_THRESHOLD` (comme les colonnes de `numrev tabulate`)."""
    words = line.get("word_confidence")
    if not words:
        return None
    lowest = min(score for _, score in words)
    return {"confidence": lowest, "doubtful": lowest < OCR_DOUBTFUL_THRESHOLD, "words": words}


def document_pages(
    range_name: str, json_pages: list[dict[str, Any]], line_rows: list[dict[str, str]], report: ExportReport
) -> tuple[list[dict[str, Any]], dict[str, Line]]:
    """Pages de la plage, chacune avec ses blocs et ses lignes dans l'ordre de
    la table des lignes, et les lignes par clé pour la couche structurée."""
    pages: dict[int, dict[str, Any]] = {}
    ocr_lines: dict[str, tuple[int, int, dict[str, Any]]] = {}  # clé → (page, bloc, ligne JSON)
    for page in json_pages:
        number = page["page_index"]
        x0, y0, x1, y1 = page["page_box"]
        pages[number] = {
            "id": f"{range_name}:{number}",
            "page": number,
            "range": range_name,
            "size": [x1 - x0, y1 - y0],
            "blocks": [
                {"id": f"{range_name}:{number}.{block['index']}", "label": block.get("label"), "bbox": block.get("bbox")}
                for block in page["data_blocks"]
            ],
            "lines": [],
        }
    for location in iter_line_locations(json_pages):
        ocr_lines[location.line[KEY_COLUMN]] = (location.page["page_index"], location.block["index"], location.line)

    lines: dict[str, Line] = {}
    for row in line_rows:
        key = row[KEY_COLUMN]
        added = is_inserted(key)
        if added:
            # Ligne ajoutée à la main : la table des lignes lui a donné la page et le bloc de sa voisine.
            page_number, block_index, ocr = int(row["page_index"]), int(row["data_block_index"]), None
        elif key in ocr_lines:
            page_number, block_index, ocr = ocr_lines[key]
        else:
            raise ValueError(f"ligne {key} (uid {row.get('uid', '?')}) de la table des lignes absente du JSON des lignes")
        if page_number not in pages:
            raise ValueError(f"ligne {key} (uid {row.get('uid', '?')}) : page {page_number} absente du JSON des lignes")
        quality = _ocr_quality(ocr) if ocr else None
        bbox = ocr.get("line_bbox_estimate") if ocr else None
        probability, provenance = _optional_float(row.get("probability")), (row.get("provenance") or "").strip()
        entry = {
            "id": line_id(range_name, key),
            "key": key,
            "uid": ocr["uid"] if ocr else None,
            "block": f"{range_name}:{page_number}.{block_index}",
            "origin": ORIGIN_ADDED if added else ORIGIN_OCR,
            "class": row[CLASS_COLUMN],
            "text": row["markdown"],
            "ocr_text": ocr["markdown"] if ocr else None,
            "bbox_estimate": bbox,
            "ocr": quality,
            "labeling": {"probability": probability, "provenance": provenance or None} if probability is not None or provenance else None,
            "corrected": is_corrected(row),
        }
        pages[page_number]["lines"].append(entry)
        lines[key] = Line(
            entry["id"],
            row.get("uid", ""),
            row["markdown"],
            row[CLASS_COLUMN],
            page_number,
            quality["confidence"] if quality else None,
            bool(quality and quality["doubtful"]),
            entry["corrected"],
        )
        report.lines += 1
        report.added_lines += added
        report.deleted_lines += row[CLASS_COLUMN] == DELETED_CLASS
        if ocr and bbox is None:
            report.lines_without_box.append(f"{range_name} p. {page_number}, ligne {ocr['uid']}")
    report.pages += len(pages)
    return list(pages.values()), lines


# ----------------------------------------------------------------------
# Couche structurée
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Piece:
    """Place d'une ligne dans le texte d'une entité : `[start, end)` de l'entité
    ← texte de la ligne sans ses blancs de bord, qui commence à `lead` dans la ligne."""

    line: Line
    start: int
    end: int
    lead: int


def entity_text(keys: list[str], lines: dict[str, Line]) -> tuple[str, list[Piece]]:
    """Texte d'une entité reconstitué comme `numrev assemble` (lignes sans blancs
    de bord, jointes par un saut de ligne devant une SUB-ENTRY, une espace sinon),
    et la place de chaque ligne dans ce texte."""
    text = ""
    pieces: list[Piece] = []
    for index, key in enumerate(keys):
        line = lines[key]
        stripped = line.text.strip()
        if index:
            text = text.rstrip() + ("\n" if line.label == SUB_ENTRY else " ")
        pieces.append(Piece(line, len(text), len(text) + len(stripped), len(line.text) - len(line.text.lstrip())))
        text += stripped
    return text, pieces


def _line_refs(pieces: list[Piece]) -> list[dict[str, Any]]:
    return [{"line": piece.line.id, "start": piece.start, "end": piece.end} for piece in pieces]


def _span_pieces(start: int, end: int, pieces: list[Piece]) -> list[dict[str, Any]]:
    """Morceaux d'un empan dans chacune des lignes qu'il couvre, en positions du texte de la ligne."""
    result = []
    for piece in pieces:
        low, high = max(start, piece.start), min(end, piece.end)
        if high > low:
            offset = piece.lead - piece.start
            result.append({"line": piece.line.id, "start": low + offset, "end": high + offset})
    return result


def _entity_lines(row: dict[str, str], lines: dict[str, Line]) -> tuple[str, list[Piece]]:
    keys = row[KEY_COLUMN].split(",")
    missing = [key for key in keys if key not in lines]
    if missing:
        raise ValueError(
            f"entité {row['uuid']} (uid {row.get('uid', '?')}) : ligne(s) {', '.join(missing)} absente(s) de la table des lignes"
        )
    text, pieces = entity_text(keys, lines)
    if text != row["markdown"].strip():
        raise ValueError(
            f"entité {row['uuid']} (uid {row.get('uid', '?')}) : son texte diffère de celui de ses lignes dans la table des lignes"
        )
    return text, pieces


def _pages(pieces: list[Piece]) -> list[int]:
    return sorted({piece.line.page for piece in pieces})


def title_object(range_name: str, row: dict[str, str], lines: dict[str, Line]) -> dict[str, Any]:
    text, pieces = _entity_lines(row, lines)
    return {
        "id": row["uuid"],
        "parent": row["parent_uuid"],
        "range": range_name,
        "pages": _pages(pieces),
        "level": title_level(text),
        "heading": title_text(text),
        "text": text,
        "lines": _line_refs(pieces),
    }


def span_objects(entry_id: str, text: str, spans: list[Span], pieces: list[Piece]) -> list[dict[str, Any]]:
    """Empans d'une entrée, chacun avec son identifiant stable : `uuid5` de
    l'entrée, de la classe, du texte normalisé et de son rang parmi les empans
    de même classe et de même texte. Il survit aux corrections des autres
    empans et change si le texte de l'empan change."""
    occurrences: Counter[tuple[str, str]] = Counter()
    objects = []
    for span in sorted(spans, key=lambda span: span.start):
        raw = text[span.start : span.end]
        normalized = " ".join(normalize_markdown(raw).text.split())
        occurrences[(span.label, normalized)] += 1
        name = f"{entry_id}#{span.label}#{normalized}#{occurrences[(span.label, normalized)]}"
        objects.append(
            {
                "id": str(uuid.uuid5(SPAN_ID_NAMESPACE, name)),
                "label": span.label,
                "start": span.start,
                "end": span.end,
                "text": raw,
                "normalized": normalized,
                "pieces": _span_pieces(span.start, span.end, pieces),
            }
        )
    return objects


def _segments(pieces: list[Piece]) -> list[list[Piece]]:
    """Lignes de chaque membre potentiel : une nouvelle à chaque SUB-ENTRY, les I-ENTRY avec la précédente."""
    segments: list[list[Piece]] = []
    for piece in pieces:
        if not segments or piece.line.label == SUB_ENTRY:
            segments.append([])
        segments[-1].append(piece)
    return segments


def member_objects(
    range_name: str, entry_id: str, text: str, pieces: list[Piece], spans: list[dict[str, Any]], report: ExportReport
) -> list[dict[str, Any]] | None:
    """Membres d'un groupe à accolade (`numrev/export/braces.py`), None pour une entrée simple.
    Les cas non résolus vont dans `report.brace_problems`."""
    segments = _segments(pieces)
    grouping = group_members(text, [Zone(segment[0].start, segment[-1].end) for segment in segments])
    if grouping is None:
        return None

    def problem(segment: list[Piece], message: str) -> None:
        first = segment[0]
        report.brace_problems.append(
            f"{range_name} p. {first.line.page}, ligne {first.line.uid} : {message} — « {excerpt(text[first.start : segment[-1].end])} »"
        )

    for index, message in grouping.problems:
        problem(segments[index], message)
    zones = [member.own for member in grouping.members] + grouping.commons
    for span in spans:
        if not any(zone.contains(span["start"], span["end"]) for zone in zones):
            segment = next(s for s in segments if s[0].start <= span["start"] <= s[-1].end)
            problem(segment, f"empan {span['label']} « {excerpt(span['text'])} » à cheval sur une accolade")

    members = []
    for member in grouping.members:
        segment = segments[member.segment]
        own = [span["id"] for span in spans if member.own.contains(span["start"], span["end"])]
        inherited = [span["id"] for span in spans if any(zone.contains(span["start"], span["end"]) for zone in member.inherited)]
        labels = {span["label"] for span in spans if span["id"] in own or span["id"] in inherited}
        if spans and "SUBJ" not in labels:
            problem(segment, "membre sans SUBJ, ni propre ni hérité")
        members.append(
            {
                "id": str(uuid.uuid5(MEMBER_ID_NAMESPACE, f"{entry_id}#{segment[0].line.id}")),
                "lines": [piece.line.id for piece in segment],
                "own": {"start": member.own.start, "end": member.own.end},
                "inherited": [{"start": zone.start, "end": zone.end} for zone in member.inherited],
                "spans": own,
                "inherited_spans": inherited,
            }
        )
    report.groups += 1
    report.members += len(members)
    return members


def entry_object(range_name: str, row: dict[str, str], lines: dict[str, Line], report: ExportReport) -> dict[str, Any]:
    text, pieces = _entity_lines(row, lines)
    spans: list[Span] = []
    if row.get("tagged_text"):
        tagged, spans = parse_tagged_text(row["tagged_text"])
        if tagged != text:
            raise ValueError(f"entrée {row['uuid']} (uid {row.get('uid', '?')}) : son texte balisé (tagged_text) ne redonne pas son texte")
    span_list = span_objects(row["uuid"], text, spans, pieces)
    report.spans.update(span["label"] for span in span_list)
    members = member_objects(range_name, row["uuid"], text, pieces, span_list, report)
    confidences = [piece.line.confidence for piece in pieces if piece.line.confidence is not None]
    suspect = (row.get("ner_suspect") or "").strip()
    return {
        "id": row["uuid"],
        "parent": row["parent_uuid"],
        "range": range_name,
        "pages": _pages(pieces),
        "kind": KIND_SINGLE if members is None else KIND_GROUP,
        "text": text,
        "lines": _line_refs(pieces),
        "spans": span_list,
        "members": members,
        "quality": {
            "ocr_confidence": min(confidences) if confidences else None,
            "ocr_doubtful": any(piece.line.doubtful for piece in pieces),
            "ner_confidence": _optional_float(row.get("ner_confidence")) if row.get("tagged_text") else None,
            "ner_suspect": suspect.split(SEPARATOR) if suspect else [],
            "corrected_lines": any(piece.line.corrected for piece in pieces),
            "corrected_ner": is_corrected(row),
        },
    }


@dataclass
class RangeLayers:
    pages: list[dict[str, Any]]
    titles: list[dict[str, Any]]
    entries: list[dict[str, Any]]


def build_range(
    range_name: str,
    json_pages: list[dict[str, Any]],
    line_rows: list[dict[str, str]],
    ner_rows: list[dict[str, str]],
    report: ExportReport,
) -> RangeLayers:
    """Les deux couches d'une plage. Les lignes hors champ ne vont que dans la couche documentaire."""
    pages, lines = document_pages(range_name, json_pages, line_rows, report)
    titles, entries = [], []
    for row in ner_rows:
        if row.get("entity") == "TITLE":
            titles.append(title_object(range_name, row, lines))
        elif row.get("entity") == "ENTRY":
            entries.append(entry_object(range_name, row, lines, report))
    known = {title["id"] for title in titles} | {ROOT_UUID}
    for item in titles + entries:
        if item["parent"] not in known:
            raise ValueError(f"{item['id']} : titre parent {item['parent']} introuvable dans la plage")
    report.ranges.append(range_name)
    report.titles += len(titles)
    report.entries += len(entries)
    return RangeLayers(pages, titles, entries)


def volume_layers(header: dict[str, Any], ranges: list[RangeLayers]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Les deux fichiers d'un volume. `header` : volume, source, plages, provenance, communs aux deux."""
    document = {"schema": DOCUMENT_SCHEMA, **header, "pages": [page for layers in ranges for page in layers.pages]}
    structure = {
        "schema": STRUCTURE_SCHEMA,
        **header,
        "root": ROOT_UUID,
        "titles": [title for layers in ranges for title in layers.titles],
        "entries": [entry for layers in ranges for entry in layers.entries],
    }
    return document, structure
