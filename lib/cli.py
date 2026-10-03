"""Convention commune des commandes qui écrivent des fichiers de données.

Simulation par défaut : la commande calcule et vérifie tout (conflits de
curation et lignes orphelines compris) puis affiche les fichiers qu'elle
écrirait, sans rien écrire. `--apply` écrit. `--force` est distinct : il
passe outre un refus (conflit, orphelin, fichier existant) mais n'écrit
toujours qu'avec `--apply`.

    writes = Writes(args.apply)
    writes.add(output_path, lambda: write_csv(output_path, fields, rows))
    writes.finish(console)
"""

import argparse
from collections.abc import Callable
from pathlib import Path

APPLY_FLAG = "--apply"


def add_apply_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        APPLY_FLAG,
        action="store_true",
        help="Écrit les fichiers. Sans cette option : simulation, tout est calculé et vérifié mais rien n'est écrit.",
    )


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
            write()

    def finish(self, console) -> None:
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
