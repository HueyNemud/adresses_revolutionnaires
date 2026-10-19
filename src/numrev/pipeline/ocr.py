"""Étape 0 · OCR latynna d'une plage de pages (`annuaires/<volume>/<volume>.pdf`
→ `<document>.ocr.json`).

latynna (Chandra 2 affiné, servi par vLLM) lit les pages de la plage et rend
le JSON de Chandra, où chaque ligne est un `<span>` portant la confiance de
ses mots et sa boîte estimée (format : README du paquet latynna). Seule
différence avec la sortie de `latynna ocr` : `page_index` est le numéro de
la page dans le PDF (1-based, comme le nom de la plage), pas son rang dans
la sortie.

Le serveur vLLM se lance à part : `uv run latynna serve --detach && uv run
latynna serve --wait`. Sans --apply, la commande vérifie la plage et le PDF
mais ne lance pas l'OCR (plusieurs minutes de GPU).
"""

import argparse
import json
from pathlib import Path

from numrev.command import CommandError, Writes, add_apply_argument, add_force_argument, console, require_file
from numrev.paths import OCR, page_range, range_document, volume_pdf

DESCRIPTION = __doc__.split("\n\n")[0]


def ocr_pages(pdf: Path, first: int, last: int, workers: int) -> list[dict]:
    """Pages `first`…`last` (1-based) du PDF lues par latynna, `page_index` = numéro de la page dans le PDF."""
    from latynna import ocr_document

    # latynna compte les pages du PDF à partir de 0, et numérote page_index selon le rang dans sa sortie.
    pages = ocr_document(pdf, pages=list(range(first - 1, last)), workers=workers)
    for number, page in zip(range(first, last + 1), pages, strict=True):
        page["page_index"] = number
    return pages


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "range_dir",
        type=Path,
        metavar="plage",
        help="Dossier de la plage, annuaires/<volume>/<première page>-<dernière page> (pages du PDF comptées à partir de 1) ; "
        "créé s'il n'existe pas.",
    )
    add_force_argument(parser, "Relance l'OCR et remplace un <document>.ocr.json existant.")
    parser.add_argument("--workers", type=int, default=8, help="Pages lues en parallèle par le serveur vLLM (défaut : 8).")
    add_apply_argument(parser)


def run(args: argparse.Namespace) -> None:
    try:
        first, last = page_range(args.range_dir)
    except ValueError as error:
        raise CommandError(str(error)) from error
    pdf = require_file(volume_pdf(args.range_dir.parent))
    output = range_document(args.range_dir, OCR)
    if output.exists() and not args.force:
        raise CommandError(f"{output} existe déjà ; --force pour relancer l'OCR et le remplacer.")
    try:
        from latynna.render import page_count
    except ImportError as error:
        raise CommandError("paquet latynna absent : uv sync --group ocr") from error
    pages_in_pdf = page_count(pdf)
    if last > pages_in_pdf:
        raise CommandError(f"plage {first}-{last} : {pdf} a {pages_in_pdf} page(s).")

    def write() -> None:
        pages = ocr_pages(pdf, first, last, args.workers)
        output.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
        without_confidence = sum('data-word-confidence="' not in page["raw"] for page in pages)
        console.print(f"{len(pages)} pages lues, dont {without_confidence} sans scores de confiance.")

    console.print(f"Pages {first} à {last} de {pdf}.")
    writes = Writes(args.apply)
    writes.add(output, write)
    writes.finish(console)
