"""Convertit une sortie JSON Datalab/Chandra en JSON enrichi de blocs de données.

Le JSON de sortie est une copie du JSON d'entrée : chaque page reçoit en plus
une liste de blocs de données (« data_blocks »), et chaque bloc contient la
liste de ses lignes Markdown (ou, avec --notables, de ses cellules de
tableau) avec leur provenance (chunk, position).
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from bs4 import BeautifulSoup, Tag
from markdownify import markdownify

BBox = tuple[float, float, float, float]


def load_raw_pages(json_path: str | Path) -> list[dict[str, Any]]:
    """Parse the document root and validate that it is a JSON array of page objects."""
    raw_document: Any = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if not isinstance(raw_document, list):
        raise ValueError("The document root must be a JSON array of pages.")
    for raw_page in raw_document:
        if not isinstance(raw_page, dict):
            raise ValueError("Each page must be a JSON object.")
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
        if (
            chunk_x0 <= block_x0 <= block_x1 <= chunk_x1
            and chunk_y0 <= block_y0 <= block_y1 <= chunk_y1
        ):
            return index
    return None


def parse_page(raw_page: dict[str, Any]) -> Page:
    raw_chunks = raw_page.get("chunks", [])
    if not isinstance(raw_chunks, list):
        raise ValueError("Page chunks must be a JSON array.")
    if not all(isinstance(chunk, dict) for chunk in raw_chunks):
        raise ValueError("Each chunk must be a JSON object.")

    page_bbox = parse_bbox(raw_page["page_box"])
    chunk_bboxes = [
        normalize_bbox(parse_bbox(chunk["bbox"]), page_bbox) for chunk in raw_chunks
    ]

    raw_soup = BeautifulSoup(str(raw_page.get("raw", "")), "html.parser")
    block_elements = raw_soup.find_all("div", recursive=False)
    data_blocks = []
    map_by_order = len(block_elements) == len(raw_chunks)
    for index, element in enumerate(block_elements, start=1):
        block_bbox = parse_bbox(get_required_attribute(element, "data-bbox"))
        chunk_index = (
            index - 1
            if map_by_order
            else find_containing_chunk(block_bbox, chunk_bboxes)
        )
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


def load_document(json_path: Path) -> list[Page]:
    """Load the Datalab JSON document and its block-to-chunk provenance."""
    return [parse_page(raw_page) for raw_page in load_raw_pages(json_path)]


def generate_line_uid(
    page_index: int, block_index: int, line_index: int, sep="."
) -> str:
    """Return a unique identifier for a Markdown line in a data block."""
    return sep.join(str(i) for i in (page_index, block_index, line_index))


def table_cell_lines(table: Tag) -> Iterator[str]:
    """Yield table cells in visual reading order without Markdown table syntax."""
    for row in table.find_all("tr"):
        if row.find_parent("table") is not table:
            continue
        for cell in row.find_all(["th", "td"], recursive=False):
            markdown = markdownify(cell.decode_contents(), heading_style="ATX")
            yield " ".join(markdown.split())


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


def _line_record(
    page: Page, block: DataBlock, line_index: int, line: str
) -> dict[str, object]:
    """Build one JSON record for a Markdown line and its Datalab provenance."""
    return {
        "uid": generate_line_uid(page.index, block.index, line_index),
        "line_index": line_index,
        "markdown": line,
    }


def data_block_lines(
    block: DataBlock, no_tables: bool = False
) -> Iterator[str]:
    """Yield Markdown lines or, with no-tables, table cells, for one data block."""
    if no_tables:
        yield from no_table_markdown_lines(block.raw_html)
    else:
        yield from markdownify(block.raw_html, heading_style="ATX").splitlines()


def build_data_block_record(
    page: Page, block: DataBlock, no_tables: bool = False
) -> dict[str, object]:
    """Build one JSON record for a data block, including its Markdown lines."""
    return {
        "index": block.index,
        "bbox": list(block.bbox),
        "label": block.label,
        "chunk_index": block.chunk_index,
        "lines": [
            _line_record(page, block, line_index, line)
            for line_index, line in enumerate(
                data_block_lines(block, no_tables=no_tables)
            )
        ],
    }


def process_json_to_json(
    json_path: str | Path, output_path: str | Path, no_tables: bool = False
) -> None:
    """Copy a Datalab JSON document, adding parsed data blocks and lines to each page."""
    output_pages = []
    for raw_page in load_raw_pages(json_path):
        page = parse_page(raw_page)
        output_page = dict(raw_page)
        output_page["data_blocks"] = [
            build_data_block_record(page, block, no_tables=no_tables)
            for block in page.data_blocks
        ]
        output_pages.append(output_page)

    Path(output_path).write_text(
        json.dumps(output_pages, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a copy of the Datalab JSON document with data blocks and "
            "their Markdown lines attached to each page."
        )
    )
    parser.add_argument(
        "json_path", help="Path to the input JSON file (Datalab output)"
    )
    parser.add_argument(
        "output_path",
        help="Path to the output JSON file (pages with data blocks and lines)",
    )
    parser.add_argument(
        "--notables",
        action="store_true",
        help="Exporte chaque cellule de tableau séparément, sans syntaxe Markdown de tableau.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_json_to_json(args.json_path, args.output_path, no_tables=args.notables)


if __name__ == "__main__":
    main()
