"""Étape 1 · lignes Markdown de la sortie OCR latynna (`<document>.ocr.json` →
`<document>.lines.json`).

Le JSON de sortie est une copie du JSON d'entrée : chaque page reçoit en plus
une liste de blocs de données (« data_blocks », un par `div` de premier
niveau du `raw`), et chaque bloc la liste de ses lignes. latynna met chaque
ligne dans un `<span>` (HTML complet) : une ligne = un `<span>`, tableaux
compris. Chaque ligne porte `uid` (`page.bloc.ligne`, tout en 1-based),
`line_index`, `markdown`, `cle` et, si le `<span>` les a, `word_confidence`
(`[[mot, score], …]`) et `line_bbox_estimate` (`[x0, y0, x1, y1]`, repère
0–1000 de la page, estimée).
"""

import argparse
import json
import re
from dataclasses import dataclass
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
from numrev.document import iter_line_locations
from numrev.paths import LINES, OCR, range_document

BBox = tuple[float, float, float, float]

IMAGE_LABELS = ("Image", "Figure")  # blocs dont le texte décrit une image : pas de lignes
HEADING_TAG = re.compile(r"^h[1-6]$")


def load_raw_pages(json_path: str | Path) -> list[dict[str, Any]]:
    """Parse the document root and validate that it is a JSON array of page objects."""
    raw_document: Any = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if not isinstance(raw_document, list):
        raise ValueError("Le JSON doit être une liste de pages.")
    for raw_page in raw_document:
        if not isinstance(raw_page, dict):
            raise ValueError("Chaque page doit être un objet JSON.")
    return raw_document


@dataclass(frozen=True)
class DataBlock:
    index: int
    bbox: BBox
    label: str
    raw_html: str
    chunk_index: int | None


@dataclass(frozen=True)
class Page:
    index: int
    data_blocks: list[DataBlock]


def parse_bbox(value: object) -> BBox:
    """Return a four-coordinate bounding box from a JSON array or HTML attribute."""
    if isinstance(value, str):
        coordinates = tuple(float(number) for number in value.split())
    elif isinstance(value, list):
        coordinates = tuple(float(number) for number in value)
    else:
        raise ValueError(f"Bounding box must be a string or list, got {value!r}")

    if len(coordinates) != 4:
        raise ValueError(f"Bounding box must have 4 values, got {value!r}")
    return coordinates[0], coordinates[1], coordinates[2], coordinates[3]


def get_required_attribute(element: Tag, name: str) -> str:
    value = element.get(name)
    if not isinstance(value, str):
        raise ValueError(f"Missing or invalid {name!r} attribute in {element}")
    return value


def normalize_bbox(bbox: BBox, page_bbox: BBox) -> BBox:
    """Convert a chunk bounding box to the 0--1000 coordinate system of raw blocks."""
    x0, y0, x1, y1 = page_bbox
    width = x1 - x0
    height = y1 - y0
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid page bounding box: {page_bbox!r}")

    return (
        (bbox[0] - x0) / width * 1000,
        (bbox[1] - y0) / height * 1000,
        (bbox[2] - x0) / width * 1000,
        (bbox[3] - y0) / height * 1000,
    )


def find_containing_chunk(block_bbox: BBox, chunks: list[BBox]) -> int | None:
    """Return the first chunk whose normalized bounding box contains a raw block."""
    for index, chunk_bbox in enumerate(chunks):
        chunk_x0, chunk_y0, chunk_x1, chunk_y1 = chunk_bbox
        block_x0, block_y0, block_x1, block_y1 = block_bbox
        if chunk_x0 <= block_x0 <= block_x1 <= chunk_x1 and chunk_y0 <= block_y0 <= block_y1 <= chunk_y1:
            return index
    return None


def parse_page(raw_page: dict[str, Any]) -> Page:
    raw_chunks = raw_page.get("chunks", [])
    if not isinstance(raw_chunks, list):
        raise ValueError("Page chunks must be a JSON array.")
    if not all(isinstance(chunk, dict) for chunk in raw_chunks):
        raise ValueError("Each chunk must be a JSON object.")

    page_bbox = parse_bbox(raw_page["page_box"])
    chunk_bboxes = [normalize_bbox(parse_bbox(chunk["bbox"]), page_bbox) for chunk in raw_chunks]

    raw_soup = BeautifulSoup(str(raw_page.get("raw", "")), "html.parser")
    block_elements = raw_soup.find_all("div", recursive=False)
    data_blocks = []
    map_by_order = len(block_elements) == len(raw_chunks)
    for index, element in enumerate(block_elements, start=1):
        block_bbox = parse_bbox(get_required_attribute(element, "data-bbox"))
        chunk_index = index - 1 if map_by_order else find_containing_chunk(block_bbox, chunk_bboxes)
        data_blocks.append(
            DataBlock(
                index=index,
                bbox=block_bbox,
                label=str(element.get("data-label", "")),
                raw_html=str(element),
                chunk_index=chunk_index,
            )
        )
    return Page(
        index=int(raw_page["page_index"]),
        data_blocks=data_blocks,
    )


def parse_document(raw_pages: list[dict[str, Any]]) -> list[Page]:
    """Parse a full document's worth of raw pages already loaded from disk."""
    return [parse_page(raw_page) for raw_page in raw_pages]


def load_document(json_path: str | Path) -> list[Page]:
    """Load the Chandra OCR JSON document and its block-to-chunk provenance."""
    return parse_document(load_raw_pages(json_path))


def generate_line_uid(page_index: int, block_index: int, line_index: int, sep=".") -> str:
    """Return a unique identifier for a Markdown line in a data block."""
    return sep.join(str(i) for i in (page_index, block_index, line_index))


def top_level_spans(container: Tag) -> list[Tag]:
    """Les `<span>` de ligne du bloc, dans l'ordre : ceux qui ne sont pas
    dans un autre `<span>` (latynna permet un `span` en ligne à l'intérieur)."""
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
    texte visible du `<span>` (format de latynna). None sans attribut, ou si
    le nombre de scores n'est pas le nombre de mots : on n'apparie jamais des
    scores décalés."""
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


def span_lines(page: Page, block: DataBlock) -> list[dict[str, object]]:
    """Lignes d'un bloc : un `<span>` = une ligne (garantie de latynna).

    Texte = Markdown du seul contenu du `<span>`, précédé de `#` × niveau
    dans un titre `<hN>`. Un bloc d'image (`IMAGE_LABELS`) et un `<span>`
    sans texte ne donnent aucune ligne. ValueError si du texte du bloc
    n'est dans aucun `<span>`."""
    container = BeautifulSoup(block.raw_html, "html.parser").find("div")
    if container is None:
        return []
    outside = text_outside_spans(container)
    if outside:
        excerpt = outside if len(outside) <= 40 else outside[:40] + "…"
        raise ValueError(f"page {page.index}, bloc {block.index} : texte hors de toute balise <span> : « {excerpt} »")
    if block.label in IMAGE_LABELS:
        return []

    lines: list[dict[str, object]] = []
    for span in top_level_spans(container):
        markdown = " ".join(markdownify(span.decode_contents()).split())
        if not markdown:
            continue
        level = heading_level(span)
        if level is not None:
            markdown = "#" * level + " " + markdown
        line_index = len(lines) + 1
        line: dict[str, object] = {
            "uid": generate_line_uid(page.index, block.index, line_index),
            "line_index": line_index,
            "markdown": markdown,
        }
        if (confidence := word_confidence(span)) is not None:
            line["word_confidence"] = confidence
        if (bbox := line_bbox_estimate(span)) is not None:
            line["line_bbox_estimate"] = bbox
        lines.append(line)
    return lines


def assign_line_keys(pages: list[dict[str, Any]]) -> None:
    """Ajoute à chaque ligne sa clé stable `cle` (numrev/curation.py : hash du
    texte OCR, indépendant de la page et du bloc), recopiée telle quelle par
    les étapes suivantes jusqu'au CSV corrigé à la main."""
    lines = [location.line for location in iter_line_locations(pages)]
    for line, key in zip(lines, line_keys(line["markdown"] for line in lines)):
        line["cle"] = key


def build_data_block_record(page: Page, block: DataBlock) -> dict[str, object]:
    """Build one JSON record for a data block, including its Markdown lines."""
    return {
        "index": block.index,
        "bbox": list(block.bbox),
        "label": block.label,
        "chunk_index": block.chunk_index,
        "lines": span_lines(page, block),
    }


def process_json_to_json(json_path: str | Path, output_path: str | Path, writes: Writes | None = None) -> tuple[int, int, int]:
    """Copy a Chandra OCR JSON document, adding parsed data blocks and lines to each page.

    Retourne (nombre de pages, nombre de blocs de données, nombre de lignes).
    """
    raw_pages = load_raw_pages(json_path)
    pages = parse_document(raw_pages)

    output_pages = []
    block_count = 0
    line_count = 0
    for raw_page, page in zip(raw_pages, pages):
        block_records = [build_data_block_record(page, block) for block in page.data_blocks]
        output_page = dict(raw_page)
        output_page["data_blocks"] = block_records
        output_pages.append(output_page)
        block_count += len(block_records)
        line_count += sum(len(record["lines"]) for record in block_records)
    assign_line_keys(output_pages)

    output_path = Path(output_path)
    write_or_plan(
        writes, output_path, lambda: output_path.write_text(json.dumps(output_pages, ensure_ascii=False, indent=2), encoding="utf-8")
    )
    return len(output_pages), block_count, line_count


def add_arguments(parser: argparse.ArgumentParser) -> None:
    add_folder_argument(parser)
    add_apply_argument(parser)


def run(args: argparse.Namespace) -> None:
    writes = Writes(args.apply)
    for range_dir in require_ranges(args.folder):
        input_path = require_input(range_dir, OCR, "ocr")
        try:
            page_count, block_count, line_count = process_json_to_json(input_path, range_document(range_dir, LINES), writes)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError, KeyError) as error:
            raise CommandError(f"{input_path} : {error}") from error
        console.print(f"{input_path.name} : {page_count} pages, {block_count} blocs, {line_count} lignes.")
    writes.finish(console)
