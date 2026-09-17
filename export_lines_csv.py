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

from rich.console import Console

from chandra_document import iter_line_locations

console = Console()

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


def process_json_to_csv(json_path: Path, csv_path: Path) -> int:
    """Aplatit un JSON Chandra (extraction ou annotation) en CSV.

    Retourne le nombre de lignes écrites.
    """
    document: Any = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(document, list):
        raise ValueError("Le document JSON doit être une liste de pages.")

    row_count = 0
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in iter_csv_rows(document):
            writer.writerow(row)
            row_count += 1
    return row_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convertit un JSON Chandra (extract_chandra_lines.py ou "
            "annotate_lines_crf.py) en CSV de lignes Markdown avec provenance."
        )
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="JSON produit par extract_chandra_lines.py ou annotate_lines_crf.py.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Chemin du CSV de sortie (défaut : <entrée>.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.json_path.exists():
        console.print(
            f"[bold red]Erreur :[/bold red] Fichier '{args.json_path}' introuvable."
        )
        return

    output_path = args.output or args.json_path.with_suffix(".csv")

    try:
        row_count = process_json_to_csv(args.json_path, output_path)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        console.print(f"[bold red]Erreur de JSON :[/bold red] {error}")
        return

    console.print(
        "\n[bold green]✅ Export CSV réussi :[/bold green] "
        f"[yellow]{output_path}[/yellow] ({row_count} lignes)"
    )


if __name__ == "__main__":
    main()
