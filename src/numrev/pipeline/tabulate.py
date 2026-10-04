"""Étape 3 · table des lignes, corrigée à la main (`<document>.labeled.json`
→ `<document>.lines.csv`).

Accepte aussi le JSON brut de `numrev extract` (`<document>.lines.json`) :
classes vides, aucun patch lu ni écrit, pour inspecter l'extraction. Chaque
ligne du CSV correspond à une ligne Markdown d'un bloc de données,
avec sa clé stable (`cle`), sa provenance (page, chunk, bloc de données,
position) et, si le JSON en dispose, l'annotation produite par
`numrev label` (`classe`, provenance, probability, timestamp).

Le CSV d'un document annoté est aussi le fichier qu'on corrige à la main
(classe, texte, niveaux `#`) : il suit le protocole de `numrev/curation.py`.
Les lignes marquées `corrige = oui` du CSV existant sont capturées dans
`data/curation/<document>.lines.patch.csv` (versionné), puis réappliquées
sur le nouvel export. Une ligne se supprime en lui donnant la classe
`SUPPRIMÉE` ; une ligne ajoutée (ou dupliquée) dans le tableur est gardée à
sa place.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

from numrev.command import (
    CommandError,
    Writes,
    add_apply_argument,
    add_capture_arguments,
    console,
    default_output,
    require_file,
)
from numrev.curation import (
    CORRECTED_COLUMN,
    FINGERPRINT_COLUMN,
    Curation,
    Step,
    print_report,
    write_csv,
)
from numrev.document import iter_line_locations
from numrev.paths import LINES_CSV

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

LINES_STEP = Step("lines", KEY_COLUMN, (CLASS_COLUMN, "markdown"), context=("uid",), insertions=True)


def iter_csv_rows(document: list[dict[str, Any]]) -> Iterator[dict[str, str]]:
    """Yield one CSV row per Markdown line found in the JSON document.

    La clé est celle écrite par numrev extract ; un JSON qui n'en
    a pas est refusé."""
    for location in iter_line_locations(document):
        if not location.line.get(KEY_COLUMN):
            raise ValueError(f"ligne {location.line.get('uid', '?')} sans clé `{KEY_COLUMN}` : relancez numrev extract.")
        row = {
            KEY_COLUMN: location.line[KEY_COLUMN],
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
    writes: Writes | None = None,
) -> Curation | None:
    """Aplatit un JSON Chandra (extraction ou annotation) en CSV.

    Pour un document annoté, applique le protocole de curation (lève
    CurationConflict sans rien écrire) et retourne la Curation ; None pour
    un document brut, exporté tel quel. `writes` décide si les fichiers sont
    écrits (numrev/command.py ; défaut : oui).
    """
    writes = writes or Writes()
    document: Any = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(document, list):
        raise ValueError("Le document JSON doit être une liste de pages.")

    rows = list(iter_csv_rows(document))
    curation = None
    if is_annotated(document):
        curation = Curation(LINES_STEP, csv_path, patch_file, force=force, capture=capture)
        rows = curation.apply(rows)
        curation.save_patch(writes)
    writes.add(csv_path, lambda: write_csv(csv_path, CSV_FIELDS, rows))
    return curation


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, help="JSON annoté (<document>.labeled.json) ou brut (<document>.lines.json).")
    parser.add_argument("-o", "--output", type=Path, default=None, help="CSV de sortie (défaut : <document>.lines.csv).")
    add_capture_arguments(parser)
    add_apply_argument(parser)


def run(args: argparse.Namespace) -> None:
    output_path = args.output or default_output(args.input, LINES_CSV)
    writes = Writes(args.apply)
    try:
        curation = process_json_to_csv(require_file(args.input), output_path, force=args.force, capture=not args.no_capture, writes=writes)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise CommandError(f"JSON {args.input} : {error}") from error
    if curation is not None:
        print_report(console, curation)
    writes.finish(console)
