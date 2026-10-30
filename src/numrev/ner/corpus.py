"""Les ENTRY de `annuaires/` et leurs annotations NER (jeu d'entraînement,
gold, audit).

Pour chaque `<document>.entities.csv`, on lit le `<document>.ner.csv` voisin
s'il existe : chaque entrée reçoit l'annotation de source `ner`, et une
entrée corrigée à la main (`corrige = oui`) aussi celle de source
`ner_curated`. Les empans sont ramenés sur le texte sans Markdown
(`normalize_markdown`), celui du gold ; une annotation dont le texte ne
correspond plus à l'entrée, ou au balisage invalide, vaut None plutôt que
d'être réalignée approximativement. Les deux fichiers sont joints par `uid`.
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
    """Empans d'un `tagged_text` sur le texte sans Markdown ; None si le balisage est invalide ou ne redonne pas `raw_text`."""
    try:
        text, spans = parse_tagged_text(tagged)
    except ValueError:
        return None
    if text != raw_text:
        return None
    return project_spans(spans, normalized)


def load_document(entities_path: Path) -> list[Entry]:
    """ENTRY non vides d'un CSV d'entités, avec leurs annotations NER."""
    document = document_name(entities_path)
    volume = volume_of(document)
    ner_path = step_path(entities_path, NER)
    ner_rows = [row for row in read_csv(ner_path)[1] if row.get("entity") == "ENTRY"] if ner_path.exists() else []
    tagged_by_source = {
        "ner": {row["uid"]: row.get("tagged_text", "") for row in ner_rows},
        "ner_curated": {row["uid"]: row.get("tagged_text", "") for row in ner_rows if is_corrected(row)},
    }

    entries = []
    for row in read_csv(entities_path)[1]:
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
