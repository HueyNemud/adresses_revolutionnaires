"""Chargement d'une paire d'annuaires pour les commandes qui l'alignent ou
la lisent (`numrev align`, `numrev join`, `numrev gold alignment`,
`numrev audit alignment`) : les deux volumes, la correspondance de leurs
rubriques et le patch des entrées ; et leurs bilans en console.

Une ligne orpheline d'un des deux patchs fait paniquer, sauf `--force`
(`numrev/curation.py`) ; un patch n'est réécrit (lignes réancrées) que si la
commande passe ses `writes`.
"""

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rich.table import Table

from numrev.alignment.patch import PatchEntry, Resolution, load_patch, orphan_descriptions, write_patch
from numrev.alignment.records import LINK_SOURCES, Link, Record, load_volume
from numrev.alignment.sections import DEFAULT_THRESHOLD, SOURCE_AUTO, SectionAlignment, load_section_alignment, section_orphans
from numrev.command import CommandError, Writes, console, require_dir, require_file
from numrev.curation import refuse_orphans
from numrev.paths import Pair


@dataclass
class LoadedPair:
    pair: Pair
    left_records: list[Record]  # ordre de l'annuaire
    right_records: list[Record]
    sections: SectionAlignment

    @property
    def left(self) -> dict[str, Record]:
        return {record.uuid: record for record in self.left_records}

    @property
    def right(self) -> dict[str, Record]:
        return {record.uuid: record for record in self.right_records}


def pair_of_dirs(left_dir: Path, right_dir: Path) -> Pair:
    try:
        return Pair.of_dirs(require_dir(left_dir), require_dir(right_dir))
    except ValueError as error:
        raise CommandError(str(error)) from error


def pair_of_file(path: Path) -> Pair:
    """Paire d'un fichier d'alignement ou de gold (`<gauche>__<droite>.….csv`)."""
    try:
        return Pair.of_file(require_file(path))
    except ValueError as error:
        raise CommandError(str(error)) from error


def load_pair(
    pair: Pair,
    *,
    force: bool = False,
    writes: Writes | None = None,
    section_threshold: float = DEFAULT_THRESHOLD,
    refuse: bool = True,
) -> LoadedPair:
    """Les deux volumes et la correspondance de leurs rubriques. `refuse=False`
    (gold, audit : ils ne produisent pas d'alignement) : orphelins laissés de côté sans paniquer."""
    try:
        left, right = load_volume(require_dir(pair.left_dir)), load_volume(require_dir(pair.right_dir))
    except (OSError, csv.Error) as error:
        raise CommandError(f"lecture des annuaires : {error}") from error
    console.print(f"{pair.left} : {len(left)} entrées ; {pair.right} : {len(right)} entrées")
    try:
        sections = load_section_alignment(left, right, pair.section_patch, section_threshold, writes)
    except (OSError, csv.Error, ValueError) as error:
        raise CommandError(f"patch des rubriques {pair.section_patch} : {error}") from error
    if refuse:
        refuse_orphans(section_orphans(sections), force, "ligne orpheline du patch des rubriques")
    return LoadedPair(pair, left, right, sections)


def load_entry_patch(
    loaded: LoadedPair, *, force: bool = False, writes: Writes | None = None
) -> tuple[list[PatchEntry], Resolution[PatchEntry]]:
    """Patch des entrées de la paire, résolu sur ses volumes : une ligne
    orpheline fait paniquer, sauf `force` (elle est alors signalée) ; des
    lignes réancrées réécrivent le patch via `writes` (sans `writes`, jamais :
    `numrev join`)."""
    path = loaded.pair.entry_patch
    try:
        entries, resolution = load_patch(path, loaded.left, loaded.right)
    except (OSError, csv.Error, ValueError) as error:
        raise CommandError(f"patch {path} : {error}") from error
    orphans = orphan_descriptions(resolution)
    refuse_orphans(orphans, force, "ligne orpheline du patch")
    if resolution.reanchored:
        rewrite = "le patch sera réécrit" if writes is not None else "numrev align nw (ou align dedupe) --apply réécrira le patch"
        console.print(
            f"[yellow]↻ {len(resolution.reanchored)} ligne(s) du patch des entrées réancrée(s) par le texte ; {rewrite}.[/yellow]"
        )
        if writes is not None:
            writes.add(path, lambda: write_patch(path, resolution.updated(entries)))
    if orphans:
        console.print(f"[bold red]⚠ {len(orphans)} ligne(s) orpheline(s) du patch, non appliquée(s) (--force) :[/bold red]")
        for description in orphans:
            console.print(f"  {description}", markup=False)
    return entries, resolution


# ----------------------------------------------------------------------
# Bilans en console
# ----------------------------------------------------------------------
def print_matched(left_name: str, right_name: str, n_left: int, n_right: int, matched: int) -> None:
    table = Table(title="Correspondances", show_header=True)
    table.add_column("Annuaire")
    table.add_column("Entrées", justify="right")
    table.add_column("Appariées", justify="right")
    table.add_column("Non appariées", justify="right")
    for name, total in ((left_name, n_left), (right_name, n_right)):
        table.add_row(name, str(total), f"{matched} ({matched / total:.1%})", str(total - matched))
    console.print(table)


def print_scores(links: list[Link], bounds: Sequence[float]) -> None:
    """Nombre de liens par tranche de score (bornes basses `bounds`), une colonne par source.
    Les paires saisies à la main, sans score, n'y figurent pas."""
    sources = [source for source in LINK_SOURCES if any(link.source == source and link.score is not None for link in links)]
    table = Table(title="Scores des correspondances", show_header=True)
    table.add_column("Score")
    for source in sources:
        table.add_column(source, justify="right")
    edges = [0.0, *bounds, float("inf")]
    for low, high in zip(edges, edges[1:]):
        label = f"[{low:.2f} ; {high:.2f}[" if high != float("inf") else f"≥ {low:.2f}"
        counts = [
            sum(link.source == source and link.score is not None and low <= link.score < high for link in links) for source in sources
        ]
        table.add_row(label, *map(str, counts))
    console.print(table)


def print_sections(alignment: SectionAlignment) -> None:
    auto = [group for group in alignment.groups if group.source == SOURCE_AUTO]
    manual = [group for group in alignment.groups if group.source != SOURCE_AUTO]
    renamed = sum(group.left[0].key != group.right[0].key for group in auto)
    console.print(
        f"Rubriques : {len(auto)} paire(s) alignée(s) automatiquement (dont {renamed} de clé différente), "
        f"{len(manual)} groupe(s) du patch, {len(alignment.unmatched_left)} seule(s) à gauche, "
        f"{len(alignment.unmatched_right)} seule(s) à droite."
    )
    for group in manual:
        left, right = (" + ".join(section.title for section in getattr(group, side)) for side in ("left", "right"))
        console.print(f"  patch · {left} ↔ {right}", markup=False)
    for side, label, unmatched in (("left", "gauche", alignment.unmatched_left), ("right", "droite", alignment.unmatched_right)):
        for section in unmatched:
            origin = "déclarée au patch" if (side, section.uuid) in alignment.declared else "non alignée"
            console.print(f"  {label} · {section.title} ({len(section.records)} entrées, {origin})", markup=False)
    resolution = alignment.resolution
    if resolution.reanchored:
        console.print(f"[yellow]↻ {len(resolution.reanchored)} ligne(s) du patch des rubriques réancrée(s) (patch mis à jour).[/yellow]")
    for entry in resolution.orphans:
        console.print(f"⚠ Ligne orpheline du patch des rubriques, non appliquée : {entry.left_title} ↔ {entry.right_title}", markup=False)
