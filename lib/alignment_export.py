"""Jointure de deux annuaires alignés, pour la lecture et l'export
(`tools/display_alignment.py`, `tools/export_alignment.py`).

Une ligne par correspondance ou par entrée sans correspondance, dans
l'**ordre naturel** des listes : les correspondances et les entrées de
gauche sans correspondance dans l'ordre de l'annuaire de gauche ; chaque
entrée de droite sans correspondance après la paire qui contient l'entrée
de droite appariée qui la précède (avant la première paire s'il n'y en a
pas).

L'export CSV s'adresse aux utilisateurs des données (historiens) : colonnes
en français, texte sans Markdown, empans NER éclatés en colonnes
`sujet` / `description` / `adresse`.
"""

import csv
import io
from collections import defaultdict
from dataclasses import dataclass

from lib.alignment import Link, Record, clean_text
from lib.ner.spans import normalize_markdown, parse_tagged_text, project_spans

PAIR, LEFT_ONLY, RIGHT_ONLY = "pair", "left", "right"
STATUS_LABELS = {PAIR: "apparié", LEFT_ONLY: "gauche seulement", RIGHT_ONLY: "droite seulement"}
SPAN_COLUMNS = {"SUBJ": "sujet", "DESC": "description", "ADDR": "adresse"}
SPAN_SEPARATOR = " | "  # entre plusieurs empans de même classe
SIDE_COLUMNS = ["volume", "page", "rubrique", "texte", *SPAN_COLUMNS.values(), "texte_balise", "uuid"]
SIDES = {"left": "gauche", "right": "droite"}
EXPORT_FIELDS = [
    "statut",
    "score",
    "methode",
    "rubriques_correspondantes",
    *(f"{side}_{column}" for side in SIDES.values() for column in SIDE_COLUMNS),
]


@dataclass
class JoinedRow:
    left: Record | None
    right: Record | None
    link: Link | None  # None pour une entrée sans correspondance

    @property
    def kind(self) -> str:
        if self.link is not None:
            return PAIR
        return LEFT_ONLY if self.left is not None else RIGHT_ONLY


def natural_rows(links: list[Link], left: dict[str, Record], right: dict[str, Record]) -> tuple[list[JoinedRow], int]:
    """Lignes de la jointure dans l'ordre naturel, et nombre de liens ignorés
    faute d'entrée (étapes amont modifiées depuis l'alignement)."""
    kept = [link for link in links if link.left_uuid in left and link.right_uuid in right]
    by_left = {link.left_uuid: link for link in kept}
    by_right = {link.right_uuid: link for link in kept}

    # Entrées de droite seules, rattachées à la paire qui les précède (clé :
    # uuid de gauche de la paire) ; celles d'avant la première paire vont
    # juste avant elle.
    after: dict[str, list[Record]] = defaultdict(list)
    before: list[Record] = []
    first, anchor = None, None
    for record in sorted(right.values(), key=lambda record: record.order):
        link = by_right.get(record.uuid)
        if link is not None:
            anchor = link.left_uuid
            first = first or anchor
        elif anchor is None:
            before.append(record)
        else:
            after[anchor].append(record)

    rows: list[JoinedRow] = []
    for record in sorted(left.values(), key=lambda record: record.order):
        if record.uuid == first:
            rows += [JoinedRow(None, alone, None) for alone in before]
        link = by_left.get(record.uuid)
        rows.append(JoinedRow(record, right[link.right_uuid] if link else None, link))
        rows += [JoinedRow(None, alone, None) for alone in after.get(record.uuid, [])]
    if first is None:
        rows += [JoinedRow(None, alone, None) for alone in before]
    return rows, len(links) - len(kept)


# ----------------------------------------------------------------------
# Export CSV
# ----------------------------------------------------------------------
def span_texts(tagged_text: str) -> dict[str, str]:
    """Texte des empans de chaque classe, sans Markdown, joints par
    `SPAN_SEPARATOR` ; vides si aucun balisage ou balisage invalide."""
    texts: dict[str, list[str]] = {label: [] for label in SPAN_COLUMNS}
    if tagged_text:
        try:
            text, spans = parse_tagged_text(tagged_text)
        except ValueError:
            spans = []
        if spans:
            normalized = normalize_markdown(text)
            for span in sorted(project_spans(spans, normalized), key=lambda span: span.start):
                if span.label in texts:
                    texts[span.label].append(clean_text(normalized.text[span.start : span.end]))
    return {label: SPAN_SEPARATOR.join(values) for label, values in texts.items()}


def side_values(record: Record | None) -> dict[str, str]:
    if record is None:
        return {column: "" for column in SIDE_COLUMNS}
    spans = span_texts(record.tagged_text)
    return {
        "volume": record.document.split(".")[0],
        "page": record.page,
        "rubrique": record.section_title,
        "texte": record.text,
        **{column: spans[label] for label, column in SPAN_COLUMNS.items()},
        "texte_balise": record.tagged_text,
        "uuid": record.uuid,
    }


def export_row(row: JoinedRow, matching: set[tuple[str, str]] | None) -> dict[str, str]:
    """Ligne d'export. `matching` : paires d'uuid de rubriques qui se
    correspondent (`lib.section_alignment.corresponding`) ; None pour ne pas
    remplir `rubriques_correspondantes`."""
    values = {
        "statut": STATUS_LABELS[row.kind],
        "score": "" if row.link is None or row.link.score is None else f"{row.link.score:.4f}",
        "methode": row.link.source if row.link else "",
        "rubriques_correspondantes": "",
    }
    if row.kind == PAIR and matching is not None:
        values["rubriques_correspondantes"] = "oui" if (row.left.section_uuid, row.right.section_uuid) in matching else "non"
    for side, name in SIDES.items():
        values |= {f"{name}_{column}": value for column, value in side_values(getattr(row, side)).items()}
    return values


def export_csv(rows: list[JoinedRow], matching: set[tuple[str, str]] | None = None, excel: bool = False) -> str:
    """CSV de la jointure. `excel` : séparateur `;` (tableur réglé en
    français) ; l'appelant écrit alors en `utf-8-sig`, que le tableur
    reconnaît."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=EXPORT_FIELDS, delimiter=";" if excel else ",", lineterminator="\n")
    writer.writeheader()
    writer.writerows(export_row(row, matching) for row in rows)
    return buffer.getvalue()


def export_encoding(excel: bool) -> str:
    return "utf-8-sig" if excel else "utf-8"
