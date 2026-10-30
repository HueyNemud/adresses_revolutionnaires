"""Étape 6 · livrable final d'un volume (toutes ses plages →
`<volume>.document.json`, `<volume>.structure.json`, `<volume>.export.report.txt`).

Deux fichiers JSON (format : `docs/format_export.md`, construction :
`numrev/export/layers.py`) : la couche documentaire (pages, blocs, lignes,
boîtes, texte OCR et texte retenu) et la couche structurée (titres, entrées,
empans, membres des groupes à accolade), reliées par les identifiants de
lignes. Étape dérivée, sans correction propre : elle se recalcule à
l'identique (date d'export mise à part) depuis les sorties des étapes 1 à 5.
Elle refuse une plage dont une étape manque ou est périmée ; un volume
partiellement traité est exporté avec la liste de ses plages. Le rapport
liste les groupes à accolade que la règle ne résout pas et les lignes OCR
sans boîte.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from numrev.command import CommandError, Writes, add_apply_argument, add_folder_argument, console, require_volume
from numrev.curation import read_csv
from numrev.devtools.status import DONE, range_status
from numrev.document import load_pages
from numrev.export.layers import ExportReport, build_range, volume_layers
from numrev.paths import (
    DOCUMENT_JSON,
    EXPORT_REPORT,
    LINES,
    LINES_CSV,
    NER,
    STRUCTURE_JSON,
    page_range,
    range_dirs,
    range_document,
    volume_file,
    volume_pdf,
)

# Repère de toutes les boîtes (blocs et lignes) : celui de latynna, 0–1000 sur
# la largeur et la hauteur de l'image de la page, origine en haut à gauche.
COORDINATES = {"width": 1000, "height": 1000, "origin": "top-left"}
REPORT_LIST_LIMIT = 200

DESCRIPTION = """Étape 6 · livrable final d'un volume (toutes ses plages → <volume>.document.json et <volume>.structure.json).

Rassemble les plages du volume en deux fichiers JSON : la couche documentaire (pages, blocs, lignes avec leur
boîte, leur texte OCR et leur texte corrigé) et la couche structurée (titres, entrées, sujets, descriptions,
adresses, membres des groupes à accolade), où chaque information renvoie aux lignes dont elle vient.
Rien ne se corrige à cette étape : on corrige les tables des lignes et les CSV NER, puis on relance.
Refuse une plage dont une étape manque ou est périmée (voir numrev status). Format : docs/format_export.md."""


def add_arguments(parser: argparse.ArgumentParser) -> None:
    add_folder_argument(parser, "Un volume, annuaires/<volume> (toutes ses plages).")
    add_apply_argument(parser)


def sha256(path: Path) -> str:
    """Empreinte d'un fichier, lu par blocs (le PDF d'un volume fait des centaines de Mo)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def numrev_commit() -> dict[str, Any]:
    """Commit de numrev et présence de modifications non commitées dans son code ; None hors d'un dépôt git."""
    source = Path(__file__).resolve().parents[1]

    def git(*arguments: str) -> str | None:
        try:
            result = subprocess.run(["git", *arguments], cwd=source, capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    commit = git("rev-parse", "HEAD")
    changes = git("status", "--porcelain", "--", ".") if commit else None
    return {"commit": commit, "uncommitted_changes": bool(changes) if commit else None}


def require_up_to_date(volume_dir: Path, ranges: list[Path]) -> None:
    """CommandError si une étape d'une plage manque ou est périmée."""
    pending = [status for status in map(range_status, ranges) if status.next_command]
    if pending:
        details = "\n".join(
            f"  {status.range_dir.name} : "
            + ", ".join(f"{command} {state}" for command, state in status.states.items() if state != DONE)
            + f"\n    ensuite : {status.next_command}"
            for status in pending
        )
        raise CommandError(f"{volume_dir} : plage(s) pas à jour (étape absente « — » ou périmée) :\n{details}")


def write_json(path: Path, content: dict[str, Any]) -> None:
    """Écriture atomique (fichier temporaire puis renommage), compacte : sans
    indentation, les fichiers d'un gros volume font 40 % de moins."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def format_report(volume_dir: Path, report: ExportReport) -> str:
    """Texte de `<volume>.export.report.txt` : comptes, puis cas à vérifier (au plus `REPORT_LIST_LIMIT` par liste)."""

    def listing(title: str, items: list[str]) -> list[str]:
        if not items:
            return [f"{title} : aucun.", ""]
        shown = [f"  - {item}" for item in items[:REPORT_LIST_LIMIT]]
        more = [f"  … et {len(items) - REPORT_LIST_LIMIT} autre(s)."] if len(items) > REPORT_LIST_LIMIT else []
        return [f"{title} ({len(items)}) :", *shown, *more, ""]

    spans = ", ".join(f"{label} {count}" for label, count in sorted(report.spans.items()))
    lines = [
        f"Export du volume {volume_dir.name}",
        f"Plages : {', '.join(report.ranges)}",
        "",
        f"Pages                         : {report.pages}",
        f"Lignes                        : {report.lines} (dont {report.added_lines} ajoutée(s) à la main, "
        f"{report.deleted_lines} supprimée(s))",
        f"Titres                        : {report.titles}",
        f"Entrées                       : {report.entries}",
        f"Groupes à accolade            : {report.groups} ({report.members} membres)",
        f"Empans                        : {spans or 'aucun'}",
        "",
        *listing(
            "Groupes à accolade à vérifier (à corriger dans la table des lignes ou le CSV NER, voir docs/format_export.md)",
            report.brace_problems,
        ),
        *listing("Lignes OCR sans boîte estimée (latynna n'a pas pu les placer)", report.lines_without_box),
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    volume_dir = require_volume(args.folder, "export")
    ranges = range_dirs(volume_dir)
    if not ranges:
        raise CommandError(f"{volume_dir} : aucune plage (<première page>-<dernière page>) dans ce dossier.")
    require_up_to_date(volume_dir, ranges)

    report = ExportReport()
    layers = []
    inputs: dict[str, str] = {}
    for range_dir in ranges:
        files = [range_document(range_dir, suffix) for suffix in (LINES, LINES_CSV, NER)]
        try:
            json_pages = load_pages(files[0])
            line_rows = read_csv(files[1])[1]
            ner_rows = read_csv(files[2])[1]
            layers.append(build_range(range_dir.name, json_pages, line_rows, ner_rows, report))
        except (csv.Error, ValueError) as error:
            raise CommandError(f"{range_dir} : {error}") from error
        except KeyError as error:
            raise CommandError(f"{range_dir} : colonne ou champ absent : {error}") from error
        inputs |= {path.as_posix(): sha256(path) for path in files}

    pdf = volume_pdf(volume_dir)
    header = {
        "volume": volume_dir.name,
        "source": {"pdf": pdf.name, "sha256": sha256(pdf) if pdf.exists() else None},
        "ranges": [
            {"name": range_dir.name, "first_page": first, "last_page": last}
            for range_dir, (first, last) in ((range_dir, page_range(range_dir)) for range_dir in ranges)
        ],
        "coordinates": COORDINATES,
        "provenance": {
            "numrev": numrev_commit(),
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "inputs": inputs,
        },
    }
    document, structure = volume_layers(header, layers)
    document_path, structure_path = volume_file(volume_dir, DOCUMENT_JSON), volume_file(volume_dir, STRUCTURE_JSON)
    report_path = volume_file(volume_dir, EXPORT_REPORT)
    report_text = format_report(volume_dir, report)

    writes = Writes(args.apply)
    writes.add(document_path, lambda: write_json(document_path, document))
    writes.add(structure_path, lambda: write_json(structure_path, structure))
    writes.add(report_path, lambda: report_path.write_text(report_text, encoding="utf-8"))

    console.print(
        f"{volume_dir.name} : {len(ranges)} plage(s), {report.pages} pages, {report.lines} lignes → "
        f"{report.titles} titres, {report.entries} entrées ({report.groups} groupe(s) à accolade, {report.members} membre(s))."
    )
    if report.brace_problems:
        console.print(f"[yellow]⚠ {len(report.brace_problems)} cas de groupe à accolade à vérifier (voir {report_path.name}).[/yellow]")
    if report.lines_without_box:
        console.print(f"{len(report.lines_without_box)} ligne(s) OCR sans boîte estimée (liste dans {report_path.name}).")
    writes.finish(console)
