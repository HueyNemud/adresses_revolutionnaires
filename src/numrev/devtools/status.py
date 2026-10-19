"""État de la chaîne par plage : étapes faites, absentes ou périmées, corrections
et lignes douteuses, et la prochaine commande à lancer.

Une sortie est **périmée** quand elle est plus ancienne que son entrée (date
de modification) : l'entrée a changé depuis, l'étape est à relancer. Lecture
seule.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from rich import box
from rich.table import Table

from numrev.command import CommandError, console, require_dir
from numrev.curation import is_corrected, read_csv
from numrev.paths import (
    ALIGNMENTS_DIR,
    ANNUAIRES_DIR,
    ENTITIES,
    LABELED,
    LINES,
    LINES_CSV,
    NER,
    OCR,
    RANGE_PATTERN,
    range_dirs,
    range_document,
    volume_pdf,
)
from numrev.pipeline.tabulate import DOUBTFUL_COLUMN

DESCRIPTION = __doc__.split("\n\n")[0]

DONE, MISSING, STALE = "✓", "—", "périmé"


@dataclass(frozen=True)
class ChainStep:
    command: str  # commande numrev qui écrit la sortie
    suffix: str  # sortie
    apply: bool = True  # la commande simule sans --apply


# Dans l'ordre de la chaîne.
STEPS = (
    ChainStep("ocr", OCR),
    ChainStep("extract", LINES),
    ChainStep("label", LABELED, apply=False),
    ChainStep("tabulate", LINES_CSV),
    ChainStep("assemble", ENTITIES),
    ChainStep("tag", NER),
)


@dataclass
class RangeStatus:
    range_dir: Path
    states: dict[str, str] = field(default_factory=dict)  # commande → DONE / MISSING / STALE
    corrected_lines: int = 0
    doubtful_lines: int = 0
    corrected_entries: int = 0
    next_command: str = ""


def step_input(range_dir: Path, step: ChainStep) -> Path:
    """Entrée dont dépend la sortie d'une étape (`tabulate` lit le JSON annoté s'il existe, sinon le brut)."""
    inputs = {
        "ocr": volume_pdf(range_dir.parent),
        "extract": range_document(range_dir, OCR),
        "label": range_document(range_dir, LINES),
        "tabulate": range_document(range_dir, LABELED),
        "assemble": range_document(range_dir, LINES_CSV),
        "tag": range_document(range_dir, ENTITIES),
    }
    path = inputs[step.command]
    if step.command == "tabulate" and not path.exists():
        return range_document(range_dir, LINES)
    return path


def step_state(range_dir: Path, step: ChainStep) -> str:
    output = range_document(range_dir, step.suffix)
    if not output.exists():
        return MISSING
    source = step_input(range_dir, step)
    return STALE if source.exists() and source.stat().st_mtime > output.stat().st_mtime else DONE


def range_status(range_dir: Path) -> RangeStatus:
    status = RangeStatus(range_dir)
    for step in STEPS:
        status.states[step.command] = step_state(range_dir, step)
    lines_csv, ner_csv = range_document(range_dir, LINES_CSV), range_document(range_dir, NER)
    if lines_csv.exists():
        _, rows = read_csv(lines_csv)
        status.corrected_lines = sum(is_corrected(row) for row in rows)
        status.doubtful_lines = sum((row.get(DOUBTFUL_COLUMN) or "").strip().lower() == "oui" for row in rows)
    if ner_csv.exists():
        status.corrected_entries = sum(is_corrected(row) for row in read_csv(ner_csv)[1])
    pending = next((step for step in STEPS if status.states[step.command] != DONE), None)
    if pending is not None:
        status.next_command = f"uv run numrev {pending.command} {range_dir}" + (" --apply" if pending.apply else "")
    return status


def volume_dirs(root: Path) -> list[Path]:
    """Dossiers de volumes : ceux qui ont des plages ou leur PDF (`<volume>/<volume>.pdf`)."""
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.resolve() != ALIGNMENTS_DIR.resolve() and (range_dirs(path) or volume_pdf(path).exists())
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "folder",
        type=Path,
        nargs="?",
        default=ANNUAIRES_DIR,
        metavar="dossier",
        help=f"Un volume (annuaires/<volume>), une plage, ou le dossier des volumes (défaut : {ANNUAIRES_DIR}, tous les volumes).",
    )


def print_volume(volume_dir: Path, ranges: list[Path]) -> None:
    """Tableau des étapes par plage, puis pour chaque plage ses comptes et sa
    prochaine commande, hors du tableau pour pouvoir la copier entière."""
    table = Table(title=str(volume_dir), title_justify="left", box=box.SIMPLE_HEAD)
    table.add_column("plage")
    for step in STEPS:
        table.add_column(step.command, justify="center")
    statuses = [range_status(range_dir) for range_dir in ranges]
    for status in statuses:
        states = [f"[yellow]{state}[/yellow]" if state == STALE else state for state in status.states.values()]
        table.add_row(status.range_dir.name, *states)
    console.print(table)
    if not volume_pdf(volume_dir).exists():
        console.print(f"[yellow]PDF absent : {volume_pdf(volume_dir)} (nécessaire à numrev ocr).[/yellow]")
    if not ranges:
        console.print(f"Aucune plage. Pour en créer une : uv run numrev ocr {volume_dir}/<première page>-<dernière page> --apply")
    for status in statuses:
        console.print(
            f"[bold]{status.range_dir.name}[/bold] : {status.corrected_lines} lignes corrigées, {status.doubtful_lines} douteuses, "
            f"{status.corrected_entries} entrées NER corrigées."
        )
        console.print("  " + (f"Ensuite : {status.next_command}" if status.next_command else "[green]À jour.[/green]"), soft_wrap=True)
    console.print()


def run(args: argparse.Namespace) -> None:
    folder = require_dir(args.folder)
    if RANGE_PATTERN.match(folder.name):
        print_volume(folder.parent, [folder])
    elif range_dirs(folder) or volume_pdf(folder).exists():
        print_volume(folder, range_dirs(folder))
    else:
        volumes = volume_dirs(folder)
        if not volumes:
            raise CommandError(f"{folder} : aucun volume (dossier avec des plages ou son PDF <volume>/<volume>.pdf).")
        for volume_dir in volumes:
            print_volume(volume_dir, range_dirs(volume_dir))
