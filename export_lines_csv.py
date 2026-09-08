"""Convertit un JSON Chandra (extract_chandra_lines.py ou annotate_lines_crf.py) en CSV.

Chaque ligne du CSV correspond à une ligne Markdown d'un bloc de données,
avec sa provenance (page, chunk, bloc de données, position) et, si le JSON
en dispose, l'annotation produite par annotate_lines_crf.py (prediction,
provenance, probability, timestamp).
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterator

from chandra_document import iter_line_locations

CSV_FIELDS = [
    "uid",
    "page_index",
    "chunk_index",
    "data_block_index",
    "line_index",
    "data_block_bbox",
    "data_block_label",
    "markdown",
    "prediction",
    "provenance",
    "probability",
    "timestamp",
]


def iter_csv_rows(document: list[dict[str, Any]]) -> Iterator[dict[str, object]]:
    """Yield one CSV row per Markdown line found in the JSON document."""
    for location in iter_line_locations(document):
        yield {
            "uid": location.line.get("uid", ""),
            "page_index": location.page.get("page_index", ""),
            "chunk_index": location.block.get("chunk_index", ""),
            "data_block_index": location.block.get("index", ""),
            "line_index": location.line.get("line_index", ""),
            "data_block_bbox": location.block.get("bbox", ""),
            "data_block_label": location.block.get("label", ""),
            "markdown": location.line.get("markdown", ""),
            "prediction": location.line.get("prediction", ""),
            "provenance": location.line.get("provenance", ""),
            "probability": location.line.get("probability", ""),
            "timestamp": location.line.get("timestamp", ""),
        }


def process_json_to_csv(json_path: str | Path, csv_path: str | Path) -> None:
    """Flatten a Chandra JSON document (extract or annotate output) into a CSV."""
    document: Any = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if not isinstance(document, list):
        raise ValueError("Le document JSON doit être une liste de pages.")

    with Path(csv_path).open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(iter_csv_rows(document))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convertit un JSON Chandra (extract_chandra_lines.py ou "
            "annotate_lines_crf.py) en CSV de lignes Markdown avec provenance."
        )
    )
    parser.add_argument(
        "json_path",
        help="Chemin du JSON d'entrée (sortie de extract_chandra_lines.py ou annotate_lines_crf.py)",
    )
    parser.add_argument("csv_path", help="Chemin du CSV de sortie")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_json_to_csv(args.json_path, args.csv_path)


if __name__ == "__main__":
    main()
