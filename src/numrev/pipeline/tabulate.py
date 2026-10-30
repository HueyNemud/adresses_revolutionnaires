"""Étape 3 · table des lignes, corrigée à la main (`<document>.labeled.json`
→ `<document>.lines.csv`).

Une ligne CSV par ligne du JSON : clé `cle`, position (page, chunk, bloc,
ligne), texte, confiance OCR (`confiance` = plus faible score de ses mots,
`douteuse` = `oui` sous `OCR_DOUBTFUL_THRESHOLD` ; ni l'une ni l'autre
éditable) et annotation de `numrev label` (`classe`, provenance,
probabilité, horodatage). Un document pas encore annoté (`<document>.lines.json`)
est exporté sans patch, classes vides, pour inspecter l'extraction.

Pour un document annoté, ce CSV est aussi le fichier corrigé à la main
(classe, texte, niveaux `#`), selon le protocole de `numrev/curation.py` :
corrections capturées dans `data/curation/<document>.lines.patch.csv` puis
réappliquées. Une correction de classe vide ou inconnue fait paniquer.
"""

import argparse
from pathlib import Path
from typing import Any, Iterator

from numrev.command import (
    CommandError,
    Writes,
    add_apply_argument,
    add_capture_arguments,
    add_folder_argument,
    console,
    fine_tuning,
    require_input,
    require_ranges,
    write_or_plan,
)
from numrev.crf.labels import CLASSES
from numrev.curation import (
    CORRECTED_COLUMN,
    DELETED_CLASS,
    FINGERPRINT_COLUMN,
    Curation,
    Step,
    print_report,
    write_csv,
)
from numrev.document import iter_line_locations, load_pages
from numrev.paths import LABELED, LINES, LINES_CSV, range_document

KEY_COLUMN = "cle"
CLASS_COLUMN = "classe"
CONFIDENCE_COLUMN = "confiance"
DOUBTFUL_COLUMN = "douteuse"

# Une ligne est douteuse si l'un de ses mots a un score OCR sous ce seuil (repère du README de
# latynna : sur latynna-v1, les mots sous 0,9 font 7,5 % des mots et 74 % des erreurs ; à recalibrer).
OCR_DOUBTFUL_THRESHOLD = 0.9

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
    CONFIDENCE_COLUMN,
    DOUBTFUL_COLUMN,
    CLASS_COLUMN,
    CORRECTED_COLUMN,
    "provenance",
    "probability",
    "timestamp",
    FINGERPRINT_COLUMN,
]

# Une ligne ajoutée à la main n'a ni score OCR ni annotation du modèle : elle ne les hérite pas de sa voisine.
LINE_MACHINE_COLUMNS = (CONFIDENCE_COLUMN, DOUBTFUL_COLUMN, "provenance", "probability", "timestamp")
LINES_STEP = Step("lines", KEY_COLUMN, (CLASS_COLUMN, "markdown"), context=("uid",), insertions=True, machine_only=LINE_MACHINE_COLUMNS)
CORRECTION_CLASSES = (*CLASSES, DELETED_CLASS)


def class_check(correction: dict[str, str], row: dict[str, str]) -> str | None:
    """Une correction doit porter une classe connue : `numrev assemble` ferait
    d'une faute de frappe (`b-entry`) ou d'une classe vide une ligne hors champ."""
    value = (correction.get(CLASS_COLUMN) or "").strip()
    if value in CORRECTION_CLASSES:
        return None
    observed = f"classe inconnue {value!r}" if value else "classe vide"
    return f"{observed} (classes possibles : {', '.join(CORRECTION_CLASSES)})"


def lowest_word_score(line: dict[str, Any]) -> float | None:
    """Plus faible score OCR des mots d'une ligne du JSON, None sans scores."""
    scores = [score for _, score in line.get("word_confidence") or []]
    return min(scores) if scores else None


def confidence_columns(line: dict[str, Any]) -> dict[str, str]:
    """`confiance` (`0,44`) et `douteuse` (`oui` sous `OCR_DOUBTFUL_THRESHOLD`), vides sans scores."""
    lowest = lowest_word_score(line)
    if lowest is None:
        return {CONFIDENCE_COLUMN: "", DOUBTFUL_COLUMN: ""}
    return {
        CONFIDENCE_COLUMN: f"{lowest:.2f}".replace(".", ","),
        DOUBTFUL_COLUMN: "oui" if lowest < OCR_DOUBTFUL_THRESHOLD else "",
    }


def iter_csv_rows(document: list[dict[str, Any]]) -> Iterator[dict[str, str]]:
    """Une ligne CSV par ligne du JSON, dans l'ordre ; ValueError sur une ligne sans clé `cle`."""
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
            **confidence_columns(location.line),
            CLASS_COLUMN: location.line.get("prediction", ""),
            "provenance": location.line.get("provenance", ""),
            "probability": location.line.get("probability", ""),
            "timestamp": location.line.get("timestamp", ""),
        }
        yield {name: "" if value is None else str(value) for name, value in row.items()}


def is_annotated(document: list[dict[str, Any]]) -> bool:
    return any("prediction" in location.line for location in iter_line_locations(document))


def tabulate_document(
    json_path: Path,
    csv_path: Path,
    patch_file: Path | None = None,
    *,
    force: bool = False,
    capture: bool = True,
    writes: Writes | None = None,
) -> Curation | None:
    """Écrit (ou planifie, via `writes`) la table des lignes d'un JSON. Pour un
    document annoté, applique le protocole de curation (CurationConflict :
    rien n'est écrit) et rend la `Curation` ; None pour un document brut."""
    document = load_pages(json_path)
    rows = list(iter_csv_rows(document))
    curation = None
    if is_annotated(document):
        curation = Curation(LINES_STEP, csv_path, patch_file, force=force, capture=capture)
        rows = curation.apply(rows, check=class_check)
        curation.save_patch(writes)
    write_or_plan(writes, csv_path, lambda: write_csv(csv_path, CSV_FIELDS, rows))
    return curation


DESCRIPTION = """Étape 3 · table des lignes, à corriger dans un tableur (<document>.labeled.json → <document>.lines.csv ;
sur une plage pas encore annotée, <document>.lines.json, pour inspecter l'extraction).

On corrige les colonnes classe et markdown en mettant corrige = oui ; à chaque relance, la commande garde
ces corrections dans data/curation/<document>.lines.patch.csv et les réapplique. Elle s'arrête sans rien
écrire si une correction ne s'applique plus. Guide : docs/guide_curation.md."""


def add_arguments(parser: argparse.ArgumentParser) -> None:
    add_folder_argument(parser)
    add_capture_arguments(parser, fine_tuning(parser))
    add_apply_argument(parser)


def table_input(range_dir: Path) -> Path:
    """Le JSON annoté (`numrev label`) s'il existe, sinon le JSON brut de `numrev extract` (export sans patch)."""
    labeled = range_document(range_dir, LABELED)
    return labeled if labeled.exists() else require_input(range_dir, LINES, "extract")


def run(args: argparse.Namespace) -> None:
    writes = Writes(args.apply)
    for range_dir in require_ranges(args.folder):
        input_path = table_input(range_dir)
        annotated = input_path.name.endswith(LABELED)
        console.print(f"Entrée : [yellow]{input_path}[/yellow]" + ("" if annotated else " (lignes non annotées : export brut, sans patch)"))
        try:
            curation = tabulate_document(
                input_path, range_document(range_dir, LINES_CSV), force=args.force, capture=not args.no_capture, writes=writes
            )
        except (UnicodeDecodeError, ValueError) as error:
            raise CommandError(f"JSON {input_path} : {error}") from error
        if curation is not None:
            print_report(console, curation)
    writes.finish(console)
