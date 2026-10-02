"""Chargement d'un annuaire complet pour l'alignement entre éditions
(`align_directories.py`, `tools/display_alignment.py`).

Un annuaire `annuaires/<volume>/` est découpé en plages de pages traitées
séparément (`<volume>/<première>-<dernière>/`). On lit, dans chaque plage, son
propre `<volume>.<plage>….merged.ner.curated.csv` et on concatène les plages
dans l'ordre des pages, pour comparer des annuaires entiers.

Chaque ENTRY devient un `Record` décrit par trois champs de comparaison :

- `section` : la rubrique, c'est-à-dire le titre ancêtre de niveau `##`, à
  défaut celui de niveau `#` (d'une édition à l'autre, une même rubrique
  change parfois de niveau). L'arbre des titres est celui de `parent_uuid`
  (`build_entity_tree.py`), propre à chaque fichier : une plage n'hérite pas
  des titres de la précédente. `section` est la clé comparée (sans accents
  ni ponctuation) ; `section_title` garde le titre lisible, pour l'affichage
  et la relecture ;
- `text` : le texte complet, sans Markdown ;
- `subj` : le texte des empans SUBJ, sans Markdown.
"""

import csv
import re
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Indel, JaroWinkler
from rapidfuzz.process import cdist

from lib.ner.corpus import CURATED_NER_SUFFIX
from lib.ner.spans import normalize_markdown, parse_tagged_text, project_spans
from lib.titles import title_level, title_text

RANGE_PATTERN = re.compile(r"^(\d+)-(\d+)$")
SECTION_LEVELS = (2, 1)  # niveau de titre préféré pour la rubrique, puis repli
NON_WORD = re.compile(r"[\W_]+")


@dataclass
class Record:
    document: str  # nom du fichier CSV source
    uuid: str
    order: int  # rang de l'entrée dans l'annuaire complet
    page: str
    section: str  # clé de rubrique comparée (`clean_title`)
    section_title: str  # titre de la rubrique, lisible
    subj: str
    text: str
    markdown: str
    tagged_text: str
    section_uuid: str = ""  # uuid du TITLE de la rubrique, vide sans rubrique


def range_dirs(volume_dir: Path) -> list[Path]:
    """Sous-dossiers de plages (`7-177`, `179-186`…), triés par première page."""
    ranges = [path for path in volume_dir.iterdir() if path.is_dir() and RANGE_PATTERN.match(path.name)]
    return sorted(ranges, key=lambda path: int(RANGE_PATTERN.match(path.name).group(1)))


def curated_csv(range_dir: Path) -> Path:
    """Le CSV NER corrigé propre à la plage (nommé d'après le volume et la plage)."""
    path = range_dir / f"{range_dir.parent.name}.{range_dir.name}{CURATED_NER_SUFFIX}"
    if not path.exists():
        raise FileNotFoundError(f"CSV NER corrigé introuvable : {path}")
    return path


def clean_text(text: str) -> str:
    return " ".join(text.split())


def clean_title(markdown: str) -> str:
    """Clé de rubrique : titre sans `#`, emphase, accents ni ponctuation, en
    minuscules. Les éditions varient sur ces points pour une même rubrique :
    « BOIS. ( MARCHANDS DE ) » / « BOIS (Marchands de) » → « bois marchands de »,
    « HÔTELS GARNIS » / « HOTELS GARNIS » → « hotels garnis »."""
    text = normalize_markdown(markdown.lstrip().lstrip("#")).text
    unaccented = "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))
    return clean_text(NON_WORD.sub(" ", unaccented)).lower()


def subject_text(tagged_text: str) -> str:
    """Texte des empans SUBJ (joints par une espace), sans Markdown ; vide si
    aucun SUBJ ou balisage invalide."""
    if not tagged_text:
        return ""
    try:
        text, spans = parse_tagged_text(tagged_text)
    except ValueError:
        return ""
    normalized = normalize_markdown(text)
    subjects = [span for span in project_spans(spans, normalized) if span.label == "SUBJ"]
    return clean_text(" ".join(normalized.text[span.start : span.end] for span in subjects))


def section_of(parent_uuid: str, titles: dict[str, tuple[str, str]]) -> tuple[str, str]:
    """Rubrique d'une ligne : (uuid, Markdown) de son ancêtre de niveau
    préféré (`SECTION_LEVELS`), ("", "") sans tel ancêtre. `titles` : uuid →
    (parent_uuid, markdown) des TITLE du fichier."""
    by_level: dict[int, tuple[str, str]] = {}
    current, seen = parent_uuid, set()
    while current in titles and current not in seen:
        seen.add(current)
        parent, markdown = titles[current]
        by_level.setdefault(title_level(markdown), (current, markdown))
        current = parent
    for level in SECTION_LEVELS:
        if level in by_level:
            return by_level[level]
    return "", ""


def load_document(path: Path, start: int = 0) -> list[Record]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    titles = {row["uuid"]: (row.get("parent_uuid", ""), row.get("markdown", "")) for row in rows if row.get("entity") == "TITLE"}
    records = []
    for row in rows:
        if row.get("entity") != "ENTRY":
            continue
        text = clean_text(normalize_markdown(row.get("markdown", "")).text)
        if not text:
            continue
        section_uuid, section = section_of(row.get("parent_uuid", ""), titles)
        records.append(
            Record(
                document=path.name,
                uuid=row["uuid"],
                order=start + len(records),
                page=row.get("page_index", "").split(",")[0],
                section=clean_title(section),
                section_title=title_text(section),
                subj=subject_text(row.get("tagged_text", "")),
                text=text,
                markdown=row.get("markdown", ""),
                tagged_text=row.get("tagged_text", ""),
                section_uuid=section_uuid,
            )
        )
    return records


def load_volume(volume_dir: Path) -> list[Record]:
    """Toutes les ENTRY d'un annuaire, plages concaténées dans l'ordre des pages."""
    ranges = range_dirs(volume_dir)
    if not ranges:
        raise FileNotFoundError(f"Aucun dossier de plage (`<début>-<fin>`) dans {volume_dir}")
    records: list[Record] = []
    for range_dir in ranges:
        records += load_document(curated_csv(range_dir), start=len(records))
    duplicates = len(records) - len({record.uuid for record in records})
    if duplicates:
        warnings.warn(f"{volume_dir.name} : {duplicates} uuid en double, seule la dernière occurrence est comparée")
    return records


def dedupe_records(records: list[Record], section_keys: dict[str, str] | None = None) -> dict[str, dict[str, str | None]]:
    """Données au format Dedupe (uuid → champs), en minuscules (la casse des
    noms varie d'une édition à l'autre : « ARCHÉDÉACON » / « Archédéacon »),
    `None` pour un champ vide. `section_keys` : clé de rubrique → clé
    canonique commune aux deux annuaires (`lib/section_alignment.py`)."""
    section_keys = section_keys or {}
    return {
        record.uuid: {
            "section": section_keys.get(record.section, record.section) or None,
            "subj": record.subj.lower() or None,
            "text": record.text.lower(),
        }
        for record in records
    }


def similarity_matrix(left: list[Record], right: list[Record], subj_weight: float) -> np.ndarray:
    """Similarité de chaque entrée de gauche à chaque entrée de droite, sur
    les champs de Dedupe : `w · JaroWinkler(subj) + (1 − w) · Indel(text)`,
    le texte seul si l'une des deux n'a pas de SUBJ (`align_directories_nw.py`,
    `lib/alignment_review.py`)."""
    left_fields, right_fields = dedupe_records(left), dedupe_records(right)
    left_values = [left_fields[record.uuid] for record in left]
    right_values = [right_fields[record.uuid] for record in right]

    def matrix(name: str, scorer) -> np.ndarray:
        return cdist(
            [values[name] or "" for values in left_values],
            [values[name] or "" for values in right_values],
            scorer=scorer.normalized_similarity,
            dtype=np.float32,
            workers=-1,
        )

    text = matrix("text", Indel)
    subj = matrix("subj", JaroWinkler)
    both = np.outer([bool(values["subj"]) for values in left_values], [bool(values["subj"]) for values in right_values])
    return np.where(both, subj_weight * subj + (1 - subj_weight) * text, text)


# ----------------------------------------------------------------------
# Correspondances (CSV `annuaires/alignements/…`)
# ----------------------------------------------------------------------
SOURCE_DEDUPE = "dedupe"
SOURCE_MANUAL = "manuel"
SOURCE_MANUAL_UNCERTAIN = "manuel-incertain"  # paire du patch marquée `certitude=incertaine`
SOURCE_PROPOSAL = "proposition"  # paire non retenue proposée à la relecture (`lib/alignment_review.py`), jamais un lien
SOURCE_NW = "nw"  # `align_directories_nw.py` : alignement ordonné
SOURCE_NW_CONTEXT = "nw-contexte"  # idem, décidée par le pair-HMM entre deux ancres
SOURCE_NW_RESIDUAL = "nw-residuel"  # idem, passe résiduelle (inversions locales)
LINK_FIELDS = [
    "left_file",
    "left_uuid",
    "right_uuid",
    "right_file",
    "score",
    "source",
    "left_section",
    "right_section",
    "left_tagged_text",
    "right_tagged_text",
]


@dataclass(frozen=True)
class Link:
    left_uuid: str
    right_uuid: str
    score: float | None  # None : paire saisie à la main
    source: str = SOURCE_DEDUPE


def display_text(record: Record) -> str:
    """Texte balisé de l'entrée, à défaut son Markdown."""
    return record.tagged_text or record.markdown.strip()


def write_links(path: Path, links: list[Link], left: dict[str, Record], right: dict[str, Record]) -> None:
    """CSV des correspondances, dans l'ordre de l'annuaire de gauche. Les
    colonnes de rubrique et de texte sont un instantané pour la relecture ;
    seules `*_file` / `*_uuid` identifient les entrées."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(LINK_FIELDS)
        for link in sorted(links, key=lambda link: left[link.left_uuid].order):
            left_record, right_record = left[link.left_uuid], right[link.right_uuid]
            writer.writerow(
                [
                    left_record.document,
                    left_record.uuid,
                    right_record.uuid,
                    right_record.document,
                    "" if link.score is None else f"{link.score:.4f}",
                    link.source,
                    left_record.section_title,
                    right_record.section_title,
                    display_text(left_record),
                    display_text(right_record),
                ]
            )


def read_links(path: Path) -> list[Link]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            Link(
                row["left_uuid"],
                row["right_uuid"],
                float(row["score"]) if row.get("score") else None,
                row.get("source") or SOURCE_DEDUPE,
            )
            for row in csv.DictReader(handle)
        ]
