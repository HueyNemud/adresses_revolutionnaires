"""Convertit un JSON Chandra (extract_chandra_lines.py ou annotate_lines_crf.py) en CSV.

Chaque ligne du CSV correspond à une ligne Markdown d'un bloc de données,
avec sa clé stable (`cle`), sa provenance (page, chunk, bloc de données,
position) et, si le JSON en dispose, l'annotation produite par
annotate_lines_crf.py (`classe`, provenance, probability, timestamp).

Le CSV d'un document annoté est aussi le fichier qu'on corrige à la main
(classe, texte, niveaux `#`) : il suit le protocole de `lib/curation.py`.
Les lignes marquées `corrige = oui` du CSV existant sont capturées dans
`data/curation/<document>.lignes.patch.csv` (versionné), puis réappliquées
sur le nouvel export. Une ligne se supprime en lui donnant la classe
`SUPPRIMÉE` ; une ligne ajoutée (ou dupliquée) dans le tableur est gardée à
sa place.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

from rich.console import Console

from lib.chandra_document import iter_line_locations
from lib.curation import (
    CORRECTED_COLUMN,
    FINGERPRINT_COLUMN,
    Curation,
    CurationConflict,
    Step,
    line_keys,
    patch_path,
    print_conflict,
    print_report,
    write_csv,
)

console = Console()

KEY_COLUMN = "cle"
CLASS_COLUMN = "classe"

CSV_FIELDS = [
    KEY_COLUMN,
    "uid",
    "page_index",
    "chunk_index",
    "data_block_index",
    "line_index",
    "data_block_bbox",
    "data_block_label",
    "markdown",
    CLASS_COLUMN,
    CORRECTED_COLUMN,
    "provenance",
    "probability",
    "timestamp",
    FINGERPRINT_COLUMN,
]

LINES_STEP = Step("lignes", KEY_COLUMN, (CLASS_COLUMN, "markdown"), context=("uid",), insertions=True)


def iter_csv_rows(document: list[dict[str, Any]]) -> Iterator[dict[str, str]]:
    """Yield one CSV row per Markdown line found in the JSON document.

    La clé est celle écrite par extract_chandra_lines.py ; un JSON plus ancien
    qui n'en a pas reçoit les mêmes clés, calculées ici sur son texte OCR."""
    locations = list(iter_line_locations(document))
    stored = [location.line.get(KEY_COLUMN) for location in locations]
    keys = stored if all(stored) else line_keys(location.line.get("markdown", "") for location in locations)
    for location, key in zip(locations, keys):
        row = {
            KEY_COLUMN: key,
            "uid": location.line.get("uid", ""),
            "page_index": location.page.get("page_index", ""),
            "chunk_index": location.block.get("chunk_index", ""),
            "data_block_index": location.block.get("index", ""),
            "line_index": location.line.get("line_index", ""),
            "data_block_bbox": location.block.get("bbox", ""),
            "data_block_label": location.block.get("label", ""),
            "markdown": location.line.get("markdown", ""),
            CLASS_COLUMN: location.line.get("prediction", ""),
            "provenance": location.line.get("provenance", ""),
            "probability": location.line.get("probability", ""),
            "timestamp": location.line.get("timestamp", ""),
        }
        yield {name: "" if value is None else str(value) for name, value in row.items()}


def is_annotated(document: list[dict[str, Any]]) -> bool:
    return any("prediction" in location.line for location in iter_line_locations(document))


def process_json_to_csv(
    json_path: Path,
    csv_path: Path,
    patch_file: Path | None = None,
    *,
    force: bool = False,
    capture: bool = True,
) -> Curation | None:
    """Aplatit un JSON Chandra (extraction ou annotation) en CSV.

    Pour un document annoté, applique le protocole de curation (lève
    CurationConflict sans rien écrire) et retourne la Curation ; None pour
    un document brut, exporté tel quel.
    """
    document: Any = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(document, list):
        raise ValueError("Le document JSON doit être une liste de pages.")

    rows = list(iter_csv_rows(document))
    curation = None
    if is_annotated(document):
        curation = Curation(LINES_STEP, csv_path, patch_file or patch_path(csv_path, LINES_STEP.name), force=force, capture=capture)
        rows = curation.apply(rows)
        curation.save_patch()
    write_csv(csv_path, CSV_FIELDS, rows)
    return curation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convertit un JSON Chandra (extract_chandra_lines.py ou "
            "annotate_lines_crf.py) en CSV de lignes Markdown avec provenance ; "
            "pour un document annoté, capture et réapplique les corrections "
            "humaines (lib/curation.py)."
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
    parser.add_argument(
        "--patch",
        type=Path,
        default=None,
        help="Patch des corrections (défaut : data/curation/<document>.lignes.patch.csv).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprend la sortie machine pour les lignes modifiées sans « corrige = oui » et abandonne les corrections inapplicables.",
    )
    parser.add_argument(
        "--sans-capture",
        action="store_true",
        help="Ignore le CSV existant et repart du patch versionné.",
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
        curation = process_json_to_csv(
            args.json_path, output_path, args.patch, force=args.force, capture=not args.sans_capture
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        console.print(f"[bold red]Erreur de JSON :[/bold red] {error}")
        raise SystemExit(1)
    except CurationConflict as conflict:
        print_conflict(console, conflict)
        raise SystemExit(1)

    if curation is not None:
        print_report(console, curation)
    console.print(
        "\n[bold green]✅ Export CSV réussi :[/bold green] "
        f"[yellow]{output_path}[/yellow]"
    )


if __name__ == "__main__":
    main()
