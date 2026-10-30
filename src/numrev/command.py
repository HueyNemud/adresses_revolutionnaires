"""Conventions communes des commandes `numrev` (`numrev/cli.py`).

Chaque module de commande expose `add_arguments(parser)` et `run(args)`.
Une erreur attendue (fichier introuvable, entrée invalide) lève
`CommandError` : `numrev/cli.py` l'affiche et sort avec le code 1, comme
pour un conflit de curation (`numrev.curation.CurationConflict`).

Simulation par défaut : une commande qui écrit des fichiers de données
calcule et vérifie tout (conflits de curation et lignes orphelines compris)
puis affiche les fichiers qu'elle écrirait, sans rien écrire. `--apply`
écrit, et seulement une fois tout vérifié (`Writes.finish`) : une erreur sur
la dernière plage d'un volume n'en laisse aucune à moitié écrite. `--force`
est distinct : il passe outre un refus (conflit, orphelin, fichier existant)
mais n'écrit toujours qu'avec `--apply`.

    writes = Writes(args.apply)
    writes.add(output_path, lambda: write_csv(output_path, fields, rows))
    writes.finish(console)

Les étapes 1 à 5 désignent leurs documents par un dossier (`add_folder_argument`,
`require_ranges`) : une plage, ou un volume entier ; chaque étape trouve son
entrée (`require_input`) et nomme sa sortie d'après `numrev/paths.py`.
Les options d'expert vont dans le groupe `fine_tuning(parser)`, affiché après
l'essentiel dans l'aide.
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


FOLDER_HELP = "Une plage, annuaires/<volume>/<première page>-<dernière page>, ou un volume, annuaires/<volume> (toutes ses plages)."


def add_folder_argument(parser: argparse.ArgumentParser, help: str = FOLDER_HELP) -> None:
    parser.add_argument("folder", type=Path, metavar="dossier", help=help)


def require_ranges(folder: Path) -> list[Path]:
    """Dossiers de plages désignés par `folder` (une plage, ou toutes celles d'un volume)."""
    from numrev.paths import selected_ranges

    require_dir(folder)
    try:
        return selected_ranges(folder)
    except ValueError as error:
        raise CommandError(str(error)) from error


def require_volume(folder: Path, command: str) -> Path:
    """Dossier d'un volume entier ; CommandError s'il n'existe pas ou si c'est une plage."""
    from numrev.paths import RANGE_PATTERN

    require_dir(folder)
    if RANGE_PATTERN.match(folder.name):
        raise CommandError(f"{folder} est une plage : numrev {command} prend un volume entier ({folder.parent}).")
    return folder


def require_input(range_dir: Path, suffix: str, previous_command: str) -> Path:
    """Fichier d'entrée d'une étape pour une plage ; CommandError s'il n'existe pas."""
    from numrev.paths import range_document

    path = range_document(range_dir, suffix)
    if not path.exists():
        raise CommandError(f"entrée absente : {path} (lancez d'abord numrev {previous_command}).")
    return path


def fine_tuning(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Groupe « Réglages fins » de l'aide : les options d'expert, après l'essentiel. Un seul appel par commande."""
    return parser.add_argument_group("Réglages fins")


def add_apply_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        APPLY_FLAG,
        action="store_true",
        help="Écrit les fichiers. Sans cette option : simulation, tout est calculé et vérifié mais rien n'est écrit.",
    )


def add_force_argument(parser: argparse.ArgumentParser, help: str) -> None:
    parser.add_argument("--force", action="store_true", help=help)


def add_capture_arguments(parser: argparse.ArgumentParser, fine: argparse._ArgumentGroup) -> None:
    """Options des étapes curées (`numrev.curation`) : `--force` avec l'essentiel, `--no-capture` dans les réglages fins."""
    add_force_argument(
        parser,
        "Reprend la sortie machine pour les lignes modifiées sans « corrige = oui » et abandonne les corrections inapplicables.",
    )
    fine.add_argument("--no-capture", action="store_true", help="Ignore le CSV existant et repart du patch versionné.")


def write_or_plan(writes: "Writes | None", path: Path, write: Callable[[], object]) -> None:
    """Fonctions de bibliothèque : sans `writes` (tests, appel direct), écrit
    tout de suite ; sinon, ajoute l'écriture au plan de la commande."""
    if writes is None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        write()
    else:
        writes.add(path, write)


class Writes:
    """Écritures d'une commande : planifiées par `add`, exécutées par `finish`
    avec `--apply` (une fois tout calculé et vérifié), seulement listées sinon.
    L'état (nouveau / remplacé) est relevé au moment de `add`."""

    def __init__(self, apply: bool = True) -> None:
        self.apply = apply
        self.planned: list[tuple[Path, bool, Callable[[], object]]] = []

    def add(self, path: Path, write: Callable[[], object], *, now: bool = False) -> None:
        """`now` : écrire tout de suite (avec `--apply`), pour du travail humain
        qu'un échec de la suite de la commande ne doit pas faire perdre."""
        path = Path(path)
        if now and self.apply:
            self.planned.append((path, path.exists(), lambda: None))
            path.parent.mkdir(parents=True, exist_ok=True)
            write()
            return
        self.planned.append((path, path.exists(), write))

    def finish(self, console: Console = console) -> None:
        """Écrit (avec `--apply`) puis fait le bilan : fichiers écrits, ou qui l'auraient été."""
        if not self.planned:
            console.print("[dim]Aucun fichier à écrire.[/dim]")
            return
        if self.apply:
            for path, _, write in self.planned:
                path.parent.mkdir(parents=True, exist_ok=True)
                write()
        lines = [f"  [yellow]{path}[/yellow] ({'remplacé' if existed else 'nouveau'})" for path, existed, _ in self.planned]
        if self.apply:
            console.print("\n[bold green]✅ Fichiers écrits :[/bold green]", *lines, sep="\n")
        else:
            console.print(
                "\n[bold cyan]🔍 Simulation, rien n'a été écrit.[/bold cyan] Fichiers qui seraient écrits :",
                *lines,
                f"Relancez avec [bold]{APPLY_FLAG}[/bold] pour les écrire.",
                sep="\n",
            )
