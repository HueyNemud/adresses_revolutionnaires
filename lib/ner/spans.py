"""Empans NER en caractères et leurs représentations.

Un empan est un triplet (début, fin exclue, classe) sur un texte donné. Trois
représentations circulent dans le pipeline :

- `tagged_text` (colonne CSV de `infer_gliner.py`, corrigée à la main) :
  le texte intégral avec des balises `<SUBJ>…</SUBJ>` ; `<` et `>` du texte
  sont échappés en `&lt;` / `&gt;` ;
- les tâches Label Studio (`data.text` + `annotations`/`predictions`) ;
- les tokens GLiNER (découpage par espaces, empans en indices de mots
  inclusifs).

Normalisation Markdown : l'emphase (`*`, `**`, `_`) produite par l'OCR n'est
que de la typographie, mais une frontière d'empan tombe souvent à
l'intérieur d'une paire de marqueurs (`<SUBJ>**Cetto</SUBJ> (…)**`).
`normalize_markdown` retire ces marqueurs en conservant, pour chaque
caractère du texte obtenu, sa position dans le texte d'origine :
`project_spans` transporte les empans d'un texte à l'autre.
"""

import re
from collections.abc import Iterable, Sequence
from typing import NamedTuple

from lib.crf.features import EMPHASIS_PATTERN

LABELS = ("SUBJ", "DESC", "ADDR")
TAG_PATTERN = re.compile(r"<(/?)(SUBJ|DESC|ADDR)>")

# Ponctuation de liaison : jamais en bord d'empan canonique (voir `canonical_spans`).
EDGE_PUNCTUATION = " \t\n,;:.—–-"


class Span(NamedTuple):
    start: int
    end: int  # exclu
    label: str
    score: float | None = None


# ----------------------------------------------------------------------
# tagged_text
# ----------------------------------------------------------------------
def _escape(fragment: str) -> str:
    return fragment.replace("<", "&lt;").replace(">", "&gt;")


def _unescape(fragment: str) -> str:
    return fragment.replace("&lt;", "<").replace("&gt;", ">")


def render_tagged_text(text: str, spans: Iterable[Span]) -> str:
    """Texte intégral, chaque empan entouré de ses balises à sa position
    exacte ; le texte de liaison est conservé tel quel. Un empan qui
    chevauche un empan déjà rendu est ignoré (balisage toujours bien formé).
    """
    pieces: list[str] = []
    cursor = 0
    for span in sorted(spans, key=lambda span: span.start):
        if span.start < cursor:
            continue
        pieces.append(_escape(text[cursor : span.start]))
        pieces.append(f"<{span.label}>{_escape(text[span.start : span.end])}</{span.label}>")
        cursor = span.end
    pieces.append(_escape(text[cursor:]))
    return "".join(pieces)


def parse_tagged_text(tagged: str) -> tuple[str, list[Span]]:
    """Inverse de `render_tagged_text`. Lève ValueError sur un balisage mal
    formé (balise non fermée, imbriquée ou fermée par une autre classe)."""
    text_parts: list[str] = []
    spans: list[Span] = []
    length = 0
    open_label: str | None = None
    open_start = 0
    cursor = 0
    for match in TAG_PATTERN.finditer(tagged):
        chunk = _unescape(tagged[cursor : match.start()])
        text_parts.append(chunk)
        length += len(chunk)
        cursor = match.end()
        closing, label = bool(match.group(1)), match.group(2)
        if closing:
            if open_label != label:
                raise ValueError(f"balise </{label}> sans <{label}> ouvrante")
            spans.append(Span(open_start, length, label))
            open_label = None
        else:
            if open_label is not None:
                raise ValueError(f"balise <{label}> dans <{open_label}> non fermée")
            open_label, open_start = label, length
    if open_label is not None:
        raise ValueError(f"balise <{open_label}> non fermée")
    text_parts.append(_unescape(tagged[cursor:]))
    return "".join(text_parts), spans


def signature(spans: Iterable[Span]) -> str:
    """Suite des classes dans l'ordre du texte, ex. « SUBJ,DESC,ADDR »."""
    return ",".join(span.label for span in sorted(spans, key=lambda span: span.start))


# ----------------------------------------------------------------------
# Normalisation Markdown et transport d'empans
# ----------------------------------------------------------------------
class NormalizedText(NamedTuple):
    text: str
    origin: tuple[int, ...]  # origin[i] = position dans le texte source du caractère i


def normalize_markdown(text: str) -> NormalizedText:
    """Retire les marqueurs d'emphase et les blancs de bord, en conservant la
    position d'origine de chaque caractère restant."""
    kept: list[str] = []
    origin: list[int] = []
    position = 0
    for match in EMPHASIS_PATTERN.finditer(text):
        for index in range(position, match.start()):
            kept.append(text[index])
            origin.append(index)
        position = match.end()
    for index in range(position, len(text)):
        kept.append(text[index])
        origin.append(index)
    start, end = 0, len(kept)
    while start < end and kept[start].isspace():
        start += 1
    while end > start and kept[end - 1].isspace():
        end -= 1
    return NormalizedText("".join(kept[start:end]), tuple(origin[start:end]))


def project_spans(spans: Iterable[Span], normalized: NormalizedText) -> list[Span]:
    """Transporte des empans du texte source vers le texte normalisé : un
    caractère conservé appartient à l'empan qui contenait sa position
    d'origine. Les empans qui ne contiennent plus rien sont abandonnés."""
    projected = []
    for span in spans:
        inside = [i for i, position in enumerate(normalized.origin) if span.start <= position < span.end]
        if inside:
            projected.append(span._replace(start=inside[0], end=inside[-1] + 1))
    return trim_spans(normalized.text, projected)


def unproject_spans(spans: Iterable[Span], normalized: NormalizedText) -> list[Span]:
    """Inverse de `project_spans` : empans du texte normalisé → texte source."""
    return [
        span._replace(start=normalized.origin[span.start], end=normalized.origin[span.end - 1] + 1)
        for span in spans
        if span.end > span.start
    ]


# ----------------------------------------------------------------------
# Bornes
# ----------------------------------------------------------------------
def trim_spans(text: str, spans: Iterable[Span], characters: str = " \t\n") -> list[Span]:
    """Retire `characters` en bord d'empan ; abandonne les empans vides."""
    trimmed = []
    for span in spans:
        start, end = span.start, span.end
        while start < end and text[start] in characters:
            start += 1
        while end > start and text[end - 1] in characters:
            end -= 1
        if end > start:
            trimmed.append(span._replace(start=start, end=end))
    return sorted(trimmed, key=lambda span: span.start)


def canonical_spans(text: str, spans: Iterable[Span]) -> tuple[tuple[str, int, int], ...]:
    """Forme de comparaison : empans sans ponctuation de liaison ni blancs en
    bord (« R. Vivienne, 44. » ≡ « R. Vivienne, 44 »), sans score. Deux
    annotations de même forme canonique ne demandent aucune correction."""
    return tuple((span.label, span.start, span.end) for span in trim_spans(text, spans, EDGE_PUNCTUATION))


# ----------------------------------------------------------------------
# Label Studio
# ----------------------------------------------------------------------
def spans_from_ls_result(result: Sequence[dict]) -> list[Span]:
    spans = []
    for item in result:
        value = item.get("value", {})
        labels = value.get("labels") or []
        if labels and labels[0] in LABELS:
            spans.append(Span(int(value["start"]), int(value["end"]), labels[0], item.get("score")))
    return sorted(spans, key=lambda span: span.start)


def ls_task_spans(task: dict, prefer: str = "annotations") -> list[Span] | None:
    """Empans d'une tâche : la dernière annotation humaine non annulée si
    `prefer="annotations"` (repli sur les prédictions sinon), ou la première
    prédiction si `prefer="predictions"`. None si la tâche n'a ni l'une ni
    l'autre.

    Deux formats de prédictions : celui de l'import (`predictions` = liste
    de dicts) et celui de l'export JSON de Label Studio, où `predictions`
    ne contient que des identifiants et où le contenu de la prédiction
    relue est recopié dans `annotations[].prediction`."""
    annotations = [a for a in task.get("annotations") or [] if not a.get("was_cancelled")]
    predictions = [p for p in task.get("predictions") or [] if isinstance(p, dict)]
    predictions += [a["prediction"] for a in task.get("annotations") or [] if isinstance(a.get("prediction"), dict)]
    if prefer == "annotations" and annotations:
        return spans_from_ls_result(annotations[-1].get("result", []))
    if predictions:
        return spans_from_ls_result(predictions[0].get("result", []))
    return None


def ls_result(text: str, spans: Iterable[Span], from_name: str = "label", to_name: str = "text") -> list[dict]:
    result = []
    for span in sorted(spans, key=lambda span: span.start):
        item = {
            "from_name": from_name,
            "to_name": to_name,
            "type": "labels",
            "value": {"start": span.start, "end": span.end, "text": text[span.start : span.end], "labels": [span.label]},
        }
        if span.score is not None:
            item["score"] = span.score
        result.append(item)
    return result


# ----------------------------------------------------------------------
# Tokens GLiNER (découpage par espaces)
# ----------------------------------------------------------------------
class Token(NamedTuple):
    text: str
    start_char: int
    end_char: int  # exclu


def tokenize_with_offsets(text: str) -> list[Token]:
    """Tokenisation par espaces avec offsets, identique à la convention
    `words_splitter_type="whitespace"` par défaut de GLiNER."""
    return [Token(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def char_span_to_word_span(tokens: Sequence[Token], start_char: int, end_char: int) -> tuple[int, int] | None:
    """Empan en caractères (fin exclue) → empan en mots (fin incluse) : les
    mots retenus sont ceux qui *chevauchent* l'empan, ce qui élargit à la
    frontière de mot la plus proche (limitation inhérente à la NER au niveau
    mot). None si l'empan ne chevauche aucun mot."""
    overlapping = [
        index for index, token in enumerate(tokens) if token.start_char < end_char and token.end_char > start_char
    ]
    if not overlapping:
        return None
    return overlapping[0], overlapping[-1]
