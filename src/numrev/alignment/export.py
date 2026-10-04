"""Jointure de deux annuaires alignés, pour la lecture et l'export
(`numrev view alignment`, `numrev join`).

Une ligne par correspondance ou par entrée sans correspondance, dans
l'**ordre naturel** des listes : les correspondances et les entrées de
gauche sans correspondance dans l'ordre de l'annuaire de gauche ; chaque
entrée de droite sans correspondance après la paire qui contient l'entrée
de droite appariée qui la précède (avant la première paire s'il n'y en a
pas).

Une **candidate non appariée** (`numrev/alignment/review.py`) n'est pas un
lien : deux entrées sans correspondance soumises au relecteur. Passée avec
les liens à `natural_rows`, elle occupe une ligne à elle (statut
`candidate`).

L'export CSV s'adresse aux utilisateurs des données (historiens) : colonnes
en français, texte sans Markdown, empans NER éclatés en colonnes
`sujet` / `description` / `adresse`, et, pour chaque paire, sa `certitude`
(`relue`, `incertaine` à la relecture, ou `automatique`), son
`niveau_incertitude` (faible, moyenne ou forte) et ses `motifs_relecture`.
"""

import csv
import io
from dataclasses import dataclass

import numpy as np

from numrev.alignment.records import SOURCE_CANDIDATE, SOURCE_MANUAL, SOURCE_MANUAL_UNCERTAIN, Link, Record, clean_text
from numrev.alignment.review import LEVEL_LABELS, SEPARATOR, Review
from numrev.ner.spans import normalize_markdown, parse_tagged_text, project_spans

PAIR, LEFT_ONLY, RIGHT_ONLY, CANDIDATE = "pair", "left", "right", "candidate"
STATUS_LABELS = {PAIR: "apparié", LEFT_ONLY: "gauche seulement", RIGHT_ONLY: "droite seulement", CANDIDATE: "candidate"}
CERTAINTY_LABELS = {SOURCE_MANUAL: "relue", SOURCE_MANUAL_UNCERTAIN: "incertaine"}  # autres liens : AUTOMATIC
AUTOMATIC = "automatique"
SPAN_COLUMNS = {"SUBJ": "sujet", "DESC": "description", "ADDR": "adresse"}
SPAN_SEPARATOR = " | "  # entre plusieurs empans de même classe
SIDE_COLUMNS = ["volume", "page", "rubrique", "texte", *SPAN_COLUMNS.values(), "texte_balise", "uuid"]
SIDES = {"left": "gauche", "right": "droite"}
EXPORT_FIELDS = [
    "statut",
    "score",
    "methode",
    "certitude",
    "niveau_incertitude",
    "motifs_relecture",
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
            return CANDIDATE if self.link.source == SOURCE_CANDIDATE else PAIR
        return LEFT_ONLY if self.left is not None else RIGHT_ONLY


@dataclass
class NaturalOrder:
    """L'ordre naturel sous forme de tableaux, une case par ligne de la
    jointure : rang de l'entrée de gauche et de droite dans leur annuaire
    (-1 si absente) et indice du lien dans `links` (-1 sans lien)."""

    left: np.ndarray
    right: np.ndarray
    link: np.ndarray
    links: list[Link]  # liens dont les deux entrées existent
    missing: int  # liens ignorés faute d'entrée


def natural_order(links: list[Link], left_rank: dict[str, int], right_rank: dict[str, int]) -> NaturalOrder:
    """Ordre naturel (voir la docstring du module), calculé par un tri
    numpy plutôt qu'une boucle (le viewer le recalcule après chaque
    décision, sur des annuaires de 100 000 entrées et plus). `*_rank` : uuid
    → rang de l'entrée dans son annuaire (0, 1, 2…)."""
    kept = [link for link in links if link.left_uuid in left_rank and link.right_uuid in right_rank]
    n_left, n_right = len(left_rank), len(right_rank)
    link_left = np.fromiter((left_rank[link.left_uuid] for link in kept), dtype=np.int64, count=len(kept))
    link_right = np.fromiter((right_rank[link.right_uuid] for link in kept), dtype=np.int64, count=len(kept))
    of_left, of_right = np.full(n_left, -1, dtype=np.int64), np.full(n_right, -1, dtype=np.int64)
    of_left[link_left] = np.arange(len(kept))
    of_right[link_right] = np.arange(len(kept))

    # Une entrée de droite seule suit la paire de l'entrée de droite appariée
    # qui la précède (clé : rang de gauche de la paire, puis 2) ; avant la
    # première, elle précède la première paire (clé : son rang de gauche, 0) ;
    # sans aucune paire, elle va à la fin.
    paired = of_right >= 0
    previous = np.maximum.accumulate(np.where(paired, np.arange(n_right), -1)) if n_right else np.zeros(0, dtype=np.int64)
    alone = np.flatnonzero(~paired)
    anchors = previous[alone]
    first = link_left[of_right[np.argmax(paired)]] if paired.any() else n_left
    alone_anchor = np.where(anchors >= 0, link_left[of_right[np.maximum(anchors, 0)]] if len(kept) else first, first)
    alone_place = np.where(anchors >= 0, 2, 0)

    # Une ligne par entrée de gauche (clé : son rang, 1), avec son lien.
    lefts = np.arange(n_left)
    primary = np.concatenate([lefts, alone_anchor])
    secondary = np.concatenate([np.ones(n_left, dtype=np.int64), alone_place])
    tertiary = np.concatenate([np.full(n_left, -1, dtype=np.int64), alone])
    order = np.lexsort((tertiary, secondary, primary))
    row_link = np.concatenate([of_left, np.full(len(alone), -1, dtype=np.int64)])[order]
    row_left = np.concatenate([lefts, np.full(len(alone), -1, dtype=np.int64)])[order]
    row_right = np.where(
        row_link >= 0, link_right[np.maximum(row_link, 0)] if len(kept) else -1, np.concatenate([np.full(n_left, -1), alone])[order]
    )
    return NaturalOrder(row_left, row_right, row_link, kept, len(links) - len(kept))


def natural_rows(links: list[Link], left: dict[str, Record], right: dict[str, Record]) -> tuple[list[JoinedRow], int]:
    """Lignes de la jointure dans l'ordre naturel, et nombre de liens ignorés
    faute d'entrée (étapes amont modifiées depuis l'alignement)."""
    left_records = sorted(left.values(), key=lambda record: record.order)
    right_records = sorted(right.values(), key=lambda record: record.order)
    found = natural_order(
        links,
        {record.uuid: rank for rank, record in enumerate(left_records)},
        {record.uuid: rank for rank, record in enumerate(right_records)},
    )
    rows = [
        JoinedRow(left_records[i] if i >= 0 else None, right_records[j] if j >= 0 else None, found.links[k] if k >= 0 else None)
        for i, j, k in zip(found.left.tolist(), found.right.tolist(), found.link.tolist())
    ]
    return rows, found.missing


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


def review_values(row: JoinedRow, reviews: dict[tuple[str, str], Review] | None) -> dict[str, str]:
    """`certitude`, `niveau_incertitude`, `motifs_relecture` d'une ligne ;
    niveau et motifs vides sans `reviews` et pour une paire relue."""
    values = {"certitude": "", "niveau_incertitude": "", "motifs_relecture": ""}
    if row.link is None:
        return values
    if row.kind == PAIR:
        values["certitude"] = CERTAINTY_LABELS.get(row.link.source, AUTOMATIC)
    if reviews is not None and values["certitude"] in ("", AUTOMATIC):
        found = reviews.get((row.link.left_uuid, row.link.right_uuid))
        values["niveau_incertitude"] = LEVEL_LABELS[found.level if found else 0]
        values["motifs_relecture"] = SEPARATOR.join(found.reasons) if found else ""
    return values


def export_row(row: JoinedRow, reviews: dict[tuple[str, str], Review] | None = None) -> dict[str, str]:
    """Ligne d'export. `reviews` : motifs de relecture
    (`numrev.alignment.review.review`) ; None pour ne pas remplir le niveau."""
    values = {
        "statut": STATUS_LABELS[row.kind],
        "score": "" if row.link is None or row.link.score is None else f"{row.link.score:.4f}",
        "methode": row.link.source if row.link else "",
        **review_values(row, reviews),
    }
    for side, name in SIDES.items():
        values |= {f"{name}_{column}": value for column, value in side_values(getattr(row, side)).items()}
    return values


def export_csv(rows: list[JoinedRow], excel: bool = False, reviews: dict[tuple[str, str], Review] | None = None) -> str:
    """CSV de la jointure. `excel` : séparateur `;` (tableur réglé en
    français) ; l'appelant écrit alors en `utf-8-sig`, que le tableur
    reconnaît."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=EXPORT_FIELDS, delimiter=";" if excel else ",", lineterminator="\n")
    writer.writeheader()
    writer.writerows(export_row(row, reviews) for row in rows)
    return buffer.getvalue()


def export_encoding(excel: bool) -> str:
    return "utf-8-sig" if excel else "utf-8"
