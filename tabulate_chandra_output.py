import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from bs4 import BeautifulSoup, Tag
from markdownify import markdownify


BBox = tuple[float, float, float, float]

CSV_FIELDS = [
    "page_index",
    "chunk_index",
    "data_block_index",
    "line_index",
    "data_block_bbox",
    "data_block_label",
    "markdown",
]


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
        normalize_bbox(parse_bbox(chunk["bbox"]), page_bbox)
        for chunk in raw_chunks
    ]

    raw_soup = BeautifulSoup(str(raw_page.get("raw", "")), "html.parser")
    block_elements = raw_soup.find_all("div", recursive=False)
    data_blocks = []
    map_by_order = len(block_elements) == len(raw_chunks)
    for index, element in enumerate(block_elements, start=1):
        block_bbox = parse_bbox(get_required_attribute(element, "data-bbox"))
        chunk_index = index - 1 if map_by_order else find_containing_chunk(
            block_bbox, chunk_bboxes
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
    raw_document: Any = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(raw_document, list):
        raise ValueError("The document root must be a JSON array of pages.")

    pages = []
    for raw_page in raw_document:
        if not isinstance(raw_page, dict):
            raise ValueError("Each page must be a JSON object.")
        pages.append(parse_page(raw_page))
    return pages


def markdown_lines(page: Page) -> Iterator[dict[str, object]]:
    """Yield one CSV record for each Markdown line produced from a data block."""
    line_index = 0
    for block in page.data_blocks:
        markdown = markdownify(block.raw_html, heading_style="ATX")
        for line in markdown.splitlines():
            yield {
                "page_index": page.index,
                "chunk_index": block.chunk_index,
                "data_block_index": block.index,
                "line_index": line_index,
                "data_block_bbox": block.bbox,
                "data_block_label": block.label,
                "markdown": line,
            }
            line_index += 1


def process_json_to_csv(json_path: str | Path, csv_path: str | Path) -> None:
    """Convert a Datalab JSON document to Markdown lines with source provenance."""
    pages = load_document(Path(json_path))

    with Path(csv_path).open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for page in pages:
            writer.writerows(markdown_lines(page))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Markdown lines with their Datalab chunk and data-block provenance."
    )
    parser.add_argument("json_path", nargs="?", default="delete_me_asap.json")
    parser.add_argument("csv_path", nargs="?", default="output.csv")
    args = parser.parse_args()
    process_json_to_csv(args.json_path, args.csv_path)


if __name__ == "__main__":
    main()
