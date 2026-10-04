"""Lecture des entrées ENTRY des entités et de leurs annotations NER.

Pour chaque document `annuaires/<volume>/<plage>/<document>.entities.csv`,
on lit, s'il existe, le fichier NER voisin `<document>.ner.csv` : sortie de
`numrev tag`, où les lignes corrigées à la main portent `corrige = oui`
(numrev/curation.py). Chaque entrée reçoit l'annotation de source `ner` ; une
entrée corrigée reçoit aussi la source `ner_curated` (mêmes empans).

Tout est ramené sur le **texte normalisé** (`normalize_markdown` : emphase
Markdown retirée), qui est le texte de référence des annotations gold. Une
annotation dont le texte ne correspond plus à celui de l'entrée (le
curateur a parfois modifié le texte) ou dont le balisage est mal formé est
ignorée (`None`) plutôt que réalignée approximativement.

Jointure par `uid` (lignes sources de l'entrée) : les fichiers fusionnés
avant l'identifiant déterministe portent des `uuid` aléatoires.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from numrev.curation import is_corrected, read_csv
from numrev.ner.shapes import coarse_shape
from numrev.ner.spans import NormalizedText, Span, normalize_markdown, parse_tagged_text, project_spans
from numrev.paths import ENTITIES, NER, discover, document_name, step_path, volume_of


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


def discover_documents(root: Path) -> list[Path]:
    return discover(root, ENTITIES)


def spans_from_tagged(tagged: str, raw_text: str, normalized: NormalizedText) -> list[Span] | None:
    try:
        text, spans = parse_tagged_text(tagged)
    except ValueError:
        return None
    if text != raw_text:
        return None
    return project_spans(spans, normalized)


def load_document(merged_path: Path) -> list[Entry]:
    document = document_name(merged_path)
    volume = volume_of(document)
    ner_path = step_path(merged_path, NER)
    ner_rows = [row for row in read_csv(ner_path)[1] if row.get("entity") == "ENTRY"] if ner_path.exists() else []
    tagged_by_source = {
        "ner": {row["uid"]: row.get("tagged_text", "") for row in ner_rows},
        "ner_curated": {row["uid"]: row.get("tagged_text", "") for row in ner_rows if is_corrected(row)},
    }

    entries = []
    for row in read_csv(merged_path)[1]:
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
    fichier à l'autre. Fichier absent : aucun texte (à l'appelant de le
    signaler)."""
    if not path.exists():
        return set()
    tasks = json.loads(path.read_text(encoding="utf-8"))
    texts = {normalize_markdown(task.get("data", {}).get("markdown") or task.get("data", {}).get("text", "")).text for task in tasks}
    texts.discard("")
    return texts


def iter_corpus(root: Path) -> Iterator[Entry]:
    for path in discover_documents(root):
        yield from load_document(path)
