"""Conventions communes des commandes `numrev` (`numrev/cli.py`).

Chaque module de commande expose `add_arguments(parser)` et `run(args)`.
Une erreur attendue (fichier introuvable, entrée invalide) lève
`CommandError` : `numrev/cli.py` l'affiche et sort avec le code 1, comme
pour un conflit de curation (`numrev.curation.CurationConflict`).

Simulation par défaut : une commande qui écrit des fichiers de données
calcule et vérifie tout (conflits de curation et lignes orphelines compris)
puis affiche les fichiers qu'elle écrirait, sans rien écrire. `--apply`
écrit. `--force` est distinct : il passe outre un refus (conflit, orphelin,
fichier existant) mais n'écrit toujours qu'avec `--apply`.

    writes = Writes(args.apply)
    writes.add(output_path, lambda: write_csv(output_path, fields, rows))
    writes.finish(console)
"""

import argparse
from collections.abc import Callable
from pathlib import Path

from rich.console import Console

APPLY_FLAG = "--apply"

console = Console()


class CommandError(Exception):
    """Erreur attendue d'une commande : message affiché, code de sortie 1."""


def require_file(path: Path) -> Path:
    if not path.exists():
        raise CommandError(f"fichier '{path}' introuvable.")
    return path


def require_dir(path: Path) -> Path:
    if not path.is_dir():
        raise CommandError(f"dossier '{path}' introuvable.")
    return path


def default_output(input_path: Path, suffix: str) -> Path:
    """Sortie par défaut d'une étape : même document, suffixe de l'étape
    (`numrev/paths.py`)."""
    from numrev.paths import step_path

    try:
        return step_path(input_path, suffix)
    except ValueError as error:
        raise CommandError(f"{error} Précisez la sortie avec -o.") from error


def add_apply_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        APPLY_FLAG,
        action="store_true",
        help="Écrit les fichiers. Sans cette option : simulation, tout est calculé et vérifié mais rien n'est écrit.",
    )


def add_force_argument(parser: argparse.ArgumentParser, help: str) -> None:
    parser.add_argument("--force", action="store_true", help=help)


def add_capture_arguments(parser: argparse.ArgumentParser) -> None:
    """Options des étapes curées (`numrev.curation`)."""
    add_force_argument(
        parser,
        "Reprend la sortie machine pour les lignes modifiées sans « corrige = oui » et abandonne les corrections inapplicables.",
    )
    parser.add_argument("--no-capture", action="store_true", help="Ignore le CSV existant et repart du patch versionné.")


class Writes:
    """Écritures d'une commande : exécutées avec `--apply`, seulement
    listées sinon. L'état (nouveau / remplacé) est relevé avant l'écriture."""

    def __init__(self, apply: bool = True) -> None:
        self.apply = apply
        self.planned: list[tuple[Path, bool]] = []

    def add(self, path: Path, write: Callable[[], object]) -> None:
        path = Path(path)
        self.planned.append((path, path.exists()))
        if self.apply:
            path.parent.mkdir(parents=True, exist_ok=True)
            write()

    def finish(self, console: Console = console) -> None:
        """Bilan : fichiers écrits, ou qui l'auraient été."""
        if not self.planned:
            console.print("[dim]Aucun fichier à écrire.[/dim]")
            return
        lines = [f"  [yellow]{path}[/yellow] ({'remplacé' if existed else 'nouveau'})" for path, existed in self.planned]
        if self.apply:
            console.print("\n[bold green]✅ Fichiers écrits :[/bold green]", *lines, sep="\n")
        else:
            console.print(
                "\n[bold cyan]🔍 Simulation, rien n'a été écrit.[/bold cyan] Fichiers qui seraient écrits :",
                *lines,
                f"Relancez avec [bold]{APPLY_FLAG}[/bold] pour les écrire.",
                sep="\n",
            )
