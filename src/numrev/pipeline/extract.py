"""Étape 1 · lignes de la sortie OCR (`<document>.ocr.json` → `<document>.lines.json`).

Une copie du JSON de latynna où chaque page reçoit ses blocs de données
(`data_blocks`, un par `div` de premier niveau du `raw`), chacun avec ses
lignes : un `<span>` = une ligne (format de latynna, tableaux compris).
Chaque ligne porte `uid` (`page.bloc.ligne`, tout compté à partir de 1),
`line_index`, `markdown`, `cle` (clé stable, `numrev/curation.py`) et, si le
`<span>` les donne, `word_confidence` (`[[mot, score], …]`) et
`line_bbox_estimate` (`[x0, y0, x1, y1]`, repère 0–1000 de la page).
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Comment, Tag
from markdownify import markdownify

from numrev.command import (
    CommandError,
    Writes,
    add_apply_argument,
    add_folder_argument,
    console,
    require_input,
    require_ranges,
    write_or_plan,
)
from numrev.curation import line_keys
from numrev.document import iter_line_locations, load_pages
from numrev.paths import LINES, OCR, range_document

DESCRIPTION = """Étape 1 · découpage de la sortie OCR en lignes (<document>.ocr.json → <document>.lines.json).

Chaque ligne lue par l'OCR devient une ligne, avec sa clé stable (colonne cle des tables suivantes)
et, si l'OCR les donne, la confiance de ses mots. Rien à faire à la main."""

BBox = tuple[float, float, float, float]

IMAGE_LABELS = ("Image", "Figure")  # blocs dont le texte décrit une image : pas de lignes
HEADING_TAG = re.compile(r"^h[1-6]$")


def parse_bbox(value: object) -> BBox:
    """Boîte à quatre coordonnées, donnée en liste JSON ou en attribut HTML (« x0 y0 x1 y1 »)."""
    if isinstance(value, str):
        coordinates = tuple(float(number) for number in value.split())
    elif isinstance(value, list):
        coordinates = tuple(float(number) for number in value)
    else:
        raise ValueError(f"boîte attendue en chaîne ou en liste, trouvé : {value!r}")
    if len(coordinates) != 4:
        raise ValueError(f"boîte à 4 coordonnées attendue, trouvé : {value!r}")
    return coordinates[0], coordinates[1], coordinates[2], coordinates[3]


def to_page_frame(bbox: BBox, page_box: BBox) -> BBox:
    """Boîte d'un chunk ramenée au repère 0–1000 de la page, celui des blocs."""
    x0, y0, x1, y1 = page_box
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0:
        raise ValueError(f"boîte de page invalide : {page_box!r}")
    return (bbox[0] - x0) / width * 1000, (bbox[1] - y0) / height * 1000, (bbox[2] - x0) / width * 1000, (bbox[3] - y0) / height * 1000


def containing_chunk(block: BBox, chunks: list[BBox]) -> int | None:
    """Rang (à partir de 0) du premier chunk qui contient le bloc, None s'il n'y en a pas."""
    for index, (x0, y0, x1, y1) in enumerate(chunks):
        if x0 <= block[0] <= block[2] <= x1 and y0 <= block[1] <= block[3] <= y1:
            return index
    return None


def top_level_spans(container: Tag) -> list[Tag]:
    """Les `<span>` de ligne du bloc, dans l'ordre : ceux qui ne sont pas dans
    un autre `<span>` (latynna permet un `span` en ligne à l'intérieur)."""
    return [span for span in container.find_all("span") if span.find_parent("span") is None]


def text_outside_spans(container: Tag) -> str:
    """Texte visible du bloc qui n'est dans aucun `<span>` (vide si tout le texte est dans des `<span>`)."""
    pieces = [text for text in container.find_all(string=True) if not isinstance(text, Comment) and text.find_parent("span") is None]
    return " ".join(" ".join(pieces).split())


def heading_level(span: Tag) -> int | None:
    """Niveau du titre `<h1>`…`<h6>` qui contient la ligne, None hors titre."""
    heading = span.find_parent(HEADING_TAG)
    return int(heading.name[1]) if heading is not None else None


def word_confidence(span: Tag) -> list[list[str | float]] | None:
    """`[[mot, score], …]` : un score de `data-word-confidence` par mot du
    texte visible du `<span>`. None sans attribut, ou si le nombre de scores
    n'est pas le nombre de mots : on n'apparie jamais des scores décalés."""
    value = span.get("data-word-confidence")
    if not isinstance(value, str):
        return None
    words = span.get_text().split()
    scores = [float(score) for score in value.split()]
    if len(words) != len(scores):
        return None
    return [[word, score] for word, score in zip(words, scores)]


def line_bbox_estimate(span: Tag) -> list[int] | None:
    """Boîte estimée de la ligne `[x0, y0, x1, y1]` (repère 0–1000 de la page), None sans attribut."""
    value = span.get("data-line-bbox-estimate")
    return [int(number) for number in value.split()] if isinstance(value, str) else None


def block_lines(page_index: int, block_index: int, block: Tag) -> list[dict[str, object]]:
    """Lignes d'un bloc : Markdown du seul contenu de chaque `<span>`, précédé
    de `#` × niveau dans un titre `<hN>`. Un bloc d'image et un `<span>` vide
    ne donnent aucune ligne. ValueError si du texte du bloc n'est dans aucun `<span>`."""
    outside = text_outside_spans(block)
    if outside:
        excerpt = outside if len(outside) <= 40 else outside[:40] + "…"
        raise ValueError(f"page {page_index}, bloc {block_index} : texte hors de toute balise <span> : « {excerpt} »")
    if str(block.get("data-label", "")) in IMAGE_LABELS:
        return []
    lines: list[dict[str, object]] = []
    for span in top_level_spans(block):
        markdown = " ".join(markdownify(span.decode_contents()).split())
        if not markdown:
            continue
        level = heading_level(span)
        if level is not None:
            markdown = "#" * level + " " + markdown
        line_index = len(lines) + 1
        line: dict[str, object] = {"uid": f"{page_index}.{block_index}.{line_index}", "line_index": line_index, "markdown": markdown}
        if (confidence := word_confidence(span)) is not None:
            line["word_confidence"] = confidence
        if (bbox := line_bbox_estimate(span)) is not None:
            line["line_bbox_estimate"] = bbox
        lines.append(line)
    return lines


def page_blocks(page: dict[str, Any]) -> list[dict[str, object]]:
    """Blocs de données d'une page : un par `div` de premier niveau du `raw`,
    avec sa boîte, sa classe, son chunk (`chunk_index`) et ses lignes."""
    chunks = page.get("chunks", [])
    if not isinstance(chunks, list) or not all(isinstance(chunk, dict) for chunk in chunks):
        raise ValueError("la clé 'chunks' d'une page doit être une liste d'objets JSON.")
    page_box = parse_bbox(page["page_box"])
    chunk_boxes = [to_page_frame(parse_bbox(chunk["bbox"]), page_box) for chunk in chunks]
    divs = BeautifulSoup(str(page.get("raw", "")), "html.parser").find_all("div", recursive=False)
    blocks: list[dict[str, object]] = []
    for index, div in enumerate(divs, start=1):
        value = div.get("data-bbox")
        if not isinstance(value, str):
            raise ValueError(f"page {page['page_index']}, bloc {index} : attribut 'data-bbox' absent ou invalide")
        bbox = parse_bbox(value)
        blocks.append(
            {
                "index": index,
                "bbox": list(bbox),
                "label": str(div.get("data-label", "")),
                # Autant de blocs que de chunks : ils se correspondent dans l'ordre ; sinon, chunk qui contient le bloc.
                "chunk_index": index - 1 if len(divs) == len(chunks) else containing_chunk(bbox, chunk_boxes),
                "lines": block_lines(int(page["page_index"]), index, div),
            }
        )
    return blocks


def assign_line_keys(pages: list[dict[str, Any]]) -> None:
    """Ajoute à chaque ligne sa clé stable `cle` (hash du texte OCR, `numrev/curation.py`)."""
    lines = [location.line for location in iter_line_locations(pages)]
    for line, key in zip(lines, line_keys(line["markdown"] for line in lines)):
        line["cle"] = key


def extract_document(ocr_path: Path, output_path: Path, writes: Writes | None = None) -> tuple[int, int, int]:
    """Écrit (ou planifie) le JSON des lignes d'un document ; rend (pages, blocs, lignes)."""
    pages = [dict(page, data_blocks=page_blocks(page)) for page in load_pages(ocr_path)]
    assign_line_keys(pages)
    write_or_plan(writes, output_path, lambda: output_path.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8"))
    blocks = [block for page in pages for block in page["data_blocks"]]
    return len(pages), len(blocks), sum(len(block["lines"]) for block in blocks)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    add_folder_argument(parser)
    add_apply_argument(parser)


def run(args: argparse.Namespace) -> None:
    writes = Writes(args.apply)
    for range_dir in require_ranges(args.folder):
        input_path = require_input(range_dir, OCR, "ocr")
        try:
            page_count, block_count, line_count = extract_document(input_path, range_document(range_dir, LINES), writes)
        except (UnicodeDecodeError, ValueError, KeyError) as error:
            raise CommandError(f"{input_path} : {error}") from error
        console.print(f"{input_path.name} : {page_count} pages, {block_count} blocs, {line_count} lignes.")
    writes.finish(console)
