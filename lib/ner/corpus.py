"""Lecture des entrées ENTRY des CSV fusionnés et de leurs annotations NER.

Pour chaque document `annuaires/<volume>/<plage>/<doc>.….merged.csv`, on
lit, quand ils existent, les deux fichiers NER voisins :

- `….merged.ner.csv` (source `ner`) : sortie brute de `infer_gliner.py` ;
- `….merged.ner.curated.csv` (source `ner_curated`) : la même, corrigée à la
  main.

Tout est ramené sur le **texte normalisé** (`normalize_markdown` : emphase
Markdown retirée), qui est le texte de référence des annotations gold. Une
annotation dont le texte ne correspond plus à celui de l'entrée (le
curateur a parfois modifié le texte) ou dont le balisage est mal formé est
ignorée (`None`) plutôt que réalignée approximativement.

Jointure par `uid` (lignes sources de l'entrée) : les fichiers fusionnés
avant l'identifiant déterministe portent des `uuid` aléatoires.
"""

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from lib.ner.shapes import coarse_shape
from lib.ner.spans import NormalizedText, Span, normalize_markdown, parse_tagged_text, project_spans

MERGED_SUFFIX = ".ocr.lines.annotated.curated.merged.csv"
RAW_NER_SUFFIX = ".ocr.lines.annotated.curated.merged.ner.csv"
CURATED_NER_SUFFIX = ".ocr.lines.annotated.curated.merged.ner.curated.csv"


@dataclass
class Entry:
    document: str
    volume: str
    uid: str
    page: str
    raw_text: str  # colonne `markdown`, blancs de bord retirés
    normalized: NormalizedText
    shape: str
    annotations: dict[str, list[Span] | None] = field(default_factory=dict)  # source → empans (texte normalisé)

    @property
    def text(self) -> str:
        return self.normalized.text

    @property
    def key(self) -> str:
        return f"{self.document}#{self.uid}"


def document_names(merged_path: Path) -> tuple[str, str]:
    """« 1808_AD75-PER292.6-185.ocr.lines.annotated.curated.merged.csv »
    → (« 1808_AD75-PER292.6-185 », « 1808_AD75-PER292 »)."""
    name = merged_path.name.removesuffix(MERGED_SUFFIX)
    return name, name.split(".", 1)[0]


def discover_documents(root: Path) -> list[Path]:
    return sorted(root.glob(f"**/*{MERGED_SUFFIX}"))


def spans_from_tagged(tagged: str, raw_text: str, normalized: NormalizedText) -> list[Span] | None:
    try:
        text, spans = parse_tagged_text(tagged)
    except ValueError:
        return None
    if text != raw_text:
        return None
    return project_spans(spans, normalized)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_document(merged_path: Path) -> list[Entry]:
    document, volume = document_names(merged_path)
    base = str(merged_path).removesuffix(MERGED_SUFFIX)
    sources = {
        "ner": Path(base + RAW_NER_SUFFIX),
        "ner_curated": Path(base + CURATED_NER_SUFFIX),
    }
    tagged_by_source = {
        name: {row["uid"]: row.get("tagged_text", "") for row in _read_rows(path) if row.get("entity") == "ENTRY"}
        for name, path in sources.items()
        if path.exists()
    }

    entries = []
    for row in _read_rows(merged_path):
        if row.get("entity") != "ENTRY":
            continue
        raw_text = row.get("markdown", "").strip()
        if not raw_text:
            continue
        normalized = normalize_markdown(raw_text)
        entry = Entry(
            document=document,
            volume=volume,
            uid=row["uid"],
            page=row["page_index"].split(",")[0],
            raw_text=raw_text,
            normalized=normalized,
            shape=coarse_shape(normalized.text),
        )
        for name, tagged in tagged_by_source.items():
            if row["uid"] in tagged:
                entry.annotations[name] = spans_from_tagged(tagged[row["uid"]], raw_text, normalized)
        entries.append(entry)
    return entries


def ls_texts(path: Path) -> set[str]:
    """Textes normalisés des tâches d'un JSON Label Studio (gold, jeu
    d'entraînement) : sert à exclure ces textes d'un autre jeu. La
    comparaison se fait sur le texte, les uid n'étant pas comparables d'un
    fichier à l'autre."""
    tasks = json.loads(path.read_text(encoding="utf-8"))
    texts = {normalize_markdown(task.get("data", {}).get("markdown") or task.get("data", {}).get("text", "")).text for task in tasks}
    texts.discard("")
    return texts


def iter_corpus(root: Path) -> Iterator[Entry]:
    for path in discover_documents(root):
        yield from load_document(path)
