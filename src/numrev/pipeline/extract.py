"""Étape 1 · lignes Markdown d'une sortie OCR Chandra (`<document>.ocr.json` →
`<document>.lines.json`).

Le JSON de sortie est une copie du JSON d'entrée : chaque page reçoit en plus
une liste de blocs de données (« data_blocks »), et chaque bloc contient la
liste de ses lignes Markdown avec leur provenance (chunk, position). Les
tableaux sont éclatés : chaque cellule (et chaque ligne d'une cellule)
devient une ligne, sans syntaxe Markdown de tableau.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from bs4 import BeautifulSoup, Tag
from markdownify import markdownify

from numrev.command import CommandError, Writes, add_apply_argument, console, default_output, require_file
from numrev.curation import line_keys
from numrev.document import iter_line_locations
from numrev.paths import LINES

BBox = tuple[float, float, float, float]


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


def table_cell_lines(table: Tag) -> Iterator[str]:
    """Yield table cells in visual reading order without Markdown table syntax."""
    for row in table.find_all("tr"):
        if row.find_parent("table") is not table:
            continue
        for cell in row.find_all(["th", "td"], recursive=False):
            markdown = markdownify(cell.decode_contents(), heading_style="ATX")
            # As a table cell might contain multiple lines,
            # we yield each line separately,
            # after cleaning whitespace.
            for line in markdown.splitlines():
                cleaned = " ".join(line.split())
                if cleaned:
                    yield cleaned


def _markdown_fragment_lines(html_fragments: list[str]) -> Iterator[str]:
    """Convert accumulated non-table HTML fragments into Markdown lines."""
    if not html_fragments:
        return
    yield from markdownify("".join(html_fragments), heading_style="ATX").splitlines()


def no_table_markdown_lines(raw_html: str) -> Iterator[str]:
    """Yield normal Markdown lines and table cells, retaining document order."""
    soup = BeautifulSoup(raw_html, "html.parser")
    container = soup.find("div")
    if container is None:
        return

    non_table_fragments: list[str] = []
    for child in container.contents:
        if isinstance(child, Tag) and child.name == "table":
            yield from _markdown_fragment_lines(non_table_fragments)
            non_table_fragments = []
            yield from table_cell_lines(child)
        else:
            non_table_fragments.append(str(child))
    yield from _markdown_fragment_lines(non_table_fragments)


def _line_record(page: Page, block: DataBlock, line_index: int, line: str) -> dict[str, object]:
    """Build one JSON record for a Markdown line and its Chandra provenance."""
    return {
        "uid": generate_line_uid(page.index, block.index, line_index),
        "line_index": line_index,
        "markdown": line,
    }


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
        "lines": [_line_record(page, block, line_index, line) for line_index, line in enumerate(no_table_markdown_lines(block.raw_html))],
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
    (writes or Writes()).add(
        output_path, lambda: output_path.write_text(json.dumps(output_pages, ensure_ascii=False, indent=2), encoding="utf-8")
    )
    return len(output_pages), block_count, line_count


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, help="JSON OCR brut de Chandra (<document>.ocr.json).")
    parser.add_argument("-o", "--output", type=Path, default=None, help="JSON de sortie (défaut : <document>.lines.json).")
    add_apply_argument(parser)


def run(args: argparse.Namespace) -> None:
    output_path = args.output or default_output(args.input, LINES)
    writes = Writes(args.apply)
    try:
        page_count, block_count, line_count = process_json_to_json(require_file(args.input), output_path, writes)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError, KeyError) as error:
        raise CommandError(f"{args.input} : {error}") from error
    console.print(f"{page_count} pages, {block_count} blocs, {line_count} lignes.")
    writes.finish(console)
