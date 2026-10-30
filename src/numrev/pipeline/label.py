"""Étape 2 · annotation interactive des lignes par apprentissage actif (CRF)
(`<document>.lines.json` → `<document>.labeled.json`).

Propose des lignes à l'humain, réentraîne le CRF après chaque réponse et
exporte une copie du JSON où chaque ligne porte sa classe et sa provenance.
La session (`<document>.label-session.json`) est enregistrée après chaque
action et se reprend en relançant la commande ; le JSON annoté est exporté à
chaque arrêt (pas de `--apply` : annoter est déjà une action explicite).
Ce module ne contient que l'interface ; le moteur est dans `numrev.crf`.
"""

import argparse
import json
from enum import Enum, auto
from pathlib import Path
from typing import Any

from rich.columns import Columns
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from numrev.command import CommandError, Writes, add_folder_argument, console, fine_tuning, require_input, require_ranges
from numrev.crf.active_learning import (
    ActiveCRF,
    AnnotationHistory,
    LabelTimestamps,
    load_json_lines,
    parse_label_timestamps,
    validate_annotation_history,
    validate_labels,
)
from numrev.crf.labels import CLASSES, AnnotationLabel
from numrev.paths import LABEL_SESSION, LABELED, LINES, range_document

LABEL_KEYS = {
    "1": AnnotationLabel.BENTRY.value,
    "2": AnnotationLabel.IENTRY.value,
    "3": AnnotationLabel.SUBENTRY.value,
    "4": AnnotationLabel.BTITLE.value,
    "5": AnnotationLabel.ITITLE.value,
    "6": AnnotationLabel.UNK.value,
    "0": AnnotationLabel.OOS.value,
}
SESSION_VERSION = 2


# ----------------------------------------------------------------------
# Tableau de bord
# ----------------------------------------------------------------------
def _progress_panel(crf: ActiveCRF, target_idx: int) -> Panel:
    annotated_cnt = crf.annotated_count
    total_cnt = len(crf.lines)
    progress = annotated_cnt / total_cnt * 100 if total_cnt else 0.0

    metrics_table = Table(show_header=False, box=None)
    metrics_table.add_column("Key", style="bold cyan")
    metrics_table.add_column("Val", style="yellow")
    metrics_table.add_row(
        "Avancement :",
        f"{annotated_cnt}/{total_cnt} ({progress:.1f}%)",
    )
    metrics_table.add_row("Page courante :", crf.records[target_idx].page_index)
    return Panel(
        metrics_table,
        title="📈 [bold yellow]ÉTAT DE L'ANNOTATION CRF[/bold yellow]",
    )


def _transitions_line(crf: ActiveCRF, top_n: int = 3) -> str | None:
    """Les transitions entre classes les plus fortes du CRF, sur une ligne ; None avant le premier entraînement."""
    if crf.tagger is None:
        return None
    sorted_transitions = sorted(crf.tagger.info().transitions.items(), key=lambda item: item[1], reverse=True)
    parts = []
    for (from_label, to_label), weight in sorted_transitions[:top_n]:
        color = "green" if weight > 0 else "red"
        parts.append(f"{from_label}➔{to_label} [{color}]{weight:+.2f}[/{color}]")
    return "  ·  ".join(parts) if parts else None


def _legend_renderable() -> Columns:
    """Légende des touches, sur plusieurs colonnes."""
    items = [f"[bold cyan]{key}[/bold cyan] {label}" for key, label in LABEL_KEYS.items()]
    items.append("[bold cyan]u[/bold cyan] [dim]annuler[/dim]")
    items.append("[bold cyan]p[/bold cyan] [dim]page → OUT OF SCOPE[/dim]")
    items.append("[bold cyan]q[/bold cyan] [dim]arrêter[/dim]")
    return Columns(items, equal=True, expand=True, column_first=True)


def _candidates_marginals_table(crf: ActiveCRF, block_indices: list[int]) -> Table:
    """Probabilités de chaque classe, une colonne par ligne proposée."""
    table = Table(title="🎯 Probabilités", header_style="bold green")
    table.add_column("Classe")
    for index in block_indices:
        table.add_column(f"L.{crf.source_row_numbers[index]}", justify="right")

    all_probs = {index: crf.get_marginals(index) for index in block_indices}
    ordered_classes = sorted(
        CLASSES,
        key=lambda label: max(all_probs[index][label] for index in block_indices),
        reverse=True,
    )
    for label in ordered_classes:
        table.add_row(
            label,
            *(f"{all_probs[index][label] * 100:.1f}%" for index in block_indices),
        )
    return table


def _context_panel(crf: ActiveCRF, target_idx: int, block_indices: list[int], margin: int = 8) -> Panel:
    """Lignes voisines des lignes proposées, avec les changements de page."""
    start = max(0, target_idx - margin)
    end = min(len(crf.lines), target_idx + margin)
    text = Text()
    previous_page_pos: int | None = None
    for index in range(start, end):
        page_pos = crf.records[index].page_pos
        if previous_page_pos is not None and page_pos != previous_page_pos:
            page_index = crf.records[index].page_index
            text.append(f"── page {page_index} ──\n", style="bold blue")
        previous_page_pos = page_pos

        prefix = "▷ " if index in block_indices else "  "
        label = f"[{crf.records[index].data_block_label}]"
        style = "bold reverse green" if index in block_indices else "dim"
        text.append(f"{prefix}{label:<15} {crf.lines[index]}\n", style=style)
    return Panel(text, title="📄 Contexte du document", border_style="blue")


def display_dashboard(crf: ActiveCRF, target_idx: int, block_indices: list[int]) -> None:
    """Tableau de bord sur deux colonnes : contexte du document à gauche, progression et probabilités à droite."""
    console.clear()

    stats: list[Any] = [_progress_panel(crf, target_idx), _legend_renderable()]
    transitions_line = _transitions_line(crf)
    if transitions_line:
        stats.append(Panel(transitions_line, title="🔗 Transitions", border_style="magenta"))
    stats.append(_candidates_marginals_table(crf, block_indices))

    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(ratio=3)
    grid.add_column(ratio=2)
    grid.add_row(_context_panel(crf, target_idx, block_indices), Group(*stats))
    console.print(grid)

    if len(block_indices) == 1:
        console.print("[dim]Dernière ligne non annotée : bloc réduit à une ligne.[/dim]")


# ----------------------------------------------------------------------
# Session et commande
# ----------------------------------------------------------------------
DESCRIPTION = """Étape 2 · annotation interactive des lignes d'une plage (<document>.lines.json → <document>.labeled.json).

L'annotateur propose les lignes une à une ; on répond par une touche : 1 B-ENTRY, 2 I-ENTRY,
3 SUB-ENTRY, 4 B-TITLE, 5 I-TITLE, 0 OUT OF SCOPE, 6 incertain, u annuler, p toute la page en
OUT OF SCOPE, q arrêter. Un modèle (CRF) réapprend après chaque réponse et classe les autres lignes.
La session est enregistrée après chaque réponse et reprend où on s'est arrêté ; le fichier annoté est
écrit à chaque arrêt (pas d'option --apply). Conventions : docs/guide_annotation_lignes.md."""


def add_arguments(parser: argparse.ArgumentParser) -> None:
    add_folder_argument(
        parser, "La plage à annoter, annuaires/<volume>/<première page>-<dernière page> (ou un volume qui n'a qu'une plage)."
    )
    parser.add_argument("--reset-session", action="store_true", help="Ignore une session existante et démarre une nouvelle annotation.")
    fine_tuning(parser).add_argument(
        "--seed-size", type=int, default=12, help="Nombre d'annotations diversifiées avant l'échantillonnage par incertitude (défaut : 12)."
    )


def load_session_state(session_path: Path, document_hash: str) -> tuple[list[str | None], AnnotationHistory | None, LabelTimestamps] | None:
    """(classes, historique, horodatages) d'une session enregistrée, None sans session ;
    ValueError si elle porte sur un autre `.lines.json` ou si elle est mal formée."""
    if not session_path.exists():
        return None
    session = json.loads(session_path.read_text(encoding="utf-8"))
    if session.get("document_hash") != document_hash:
        raise ValueError(
            "elle porte sur une autre version du fichier de lignes (contenu du .lines.json modifié depuis) ; "
            "--reset-session recommence l'annotation, sans les annotations de cette session."
        )
    labels = session.get("labels")
    if not isinstance(labels, list):
        raise ValueError("La session ne contient pas de liste de labels valide.")
    annotated_indices = validate_labels(labels)
    annotation_history = session.get("annotation_history")
    if annotation_history is not None:
        if not isinstance(annotation_history, list):
            raise ValueError("La session ne contient pas d'historique d'annotation valide.")
        validate_annotation_history(annotation_history, annotated_indices)
    label_timestamps = parse_label_timestamps(session.get("label_timestamps") or {}, annotated_indices)
    return labels, annotation_history, label_timestamps


def save_session(session_path: Path, document_hash: str, crf: ActiveCRF) -> None:
    session_path.write_text(
        json.dumps(
            {
                "version": SESSION_VERSION,
                "document_hash": document_hash,
                "source_rows": crf.source_row_numbers,
                "labels": crf.labels,
                "annotation_history": crf.annotation_history,
                "label_timestamps": {str(index): timestamp for index, timestamp in crf.label_timestamps.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


class _Control(Enum):
    """Touches de contrôle (arrêter, annuler, page hors champ), par opposition aux classes."""

    STOP = auto()
    UNDO = auto()
    SKIP_PAGE = auto()


def _prompt_label(crf: ActiveCRF, index: int) -> str | _Control:
    """Classe donnée par l'humain pour une ligne, ou une touche de contrôle.
    Une touche inconnue repose la question : une faute de frappe ne perd rien."""
    while True:
        console.print(
            f"\n[bold]Label ligne {crf.source_row_numbers[index]} ?[/bold] ",
            end="",
        )
        choice = input().strip().lower()
        if choice == "q":
            return _Control.STOP
        if choice == "u":
            return _Control.UNDO
        if choice == "p":
            return _Control.SKIP_PAGE
        if choice in LABEL_KEYS:
            return LABEL_KEYS[choice]
        console.print("[yellow]Touche non reconnue — voir la légende ci-dessus.[/yellow]")


def _collect_block_labels(crf: ActiveCRF, block_indices: list[int]) -> dict[int, str] | _Control:
    """Classes données aux lignes d'un bloc, ou une touche de contrôle.

    `u` annule la dernière saisie du bloc et la redemande ; sur un bloc encore
    vide, il est rendu à l'appelant, qui annule la dernière ligne validée.
    `p` est rendu aussitôt (la page entière passe hors champ, saisies du bloc
    comprises).
    """
    pending_order: list[int] = []
    pending_labels: dict[int, str] = {}
    position = 0
    while position < len(block_indices):
        index = block_indices[position]
        label = _prompt_label(crf, index)

        if label is _Control.STOP or label is _Control.SKIP_PAGE:
            return label

        if label is _Control.UNDO:
            if not pending_order:
                # Rien à annuler dans ce bloc : on délègue à l'historique global.
                return _Control.UNDO
            undone_index = pending_order.pop()
            del pending_labels[undone_index]
            console.print(f"[green]Saisie annulée pour la ligne {crf.source_row_numbers[undone_index]} ; à ressaisir.[/green]")
            position -= 1
            continue

        pending_labels[index] = label
        pending_order.append(index)
        position += 1

    return pending_labels


def annotate_interactively(crf: ActiveCRF, session_path: Path, document_hash: str) -> None:
    """Boucle d'annotation humaine jusqu'à épuisement des lignes ou arrêt manuel."""
    reoffer_index: int | None = None
    while True:
        selection = crf.next_block(preferred_index=reoffer_index)
        if selection is None:
            console.print("\n[bold green]🎉 Annotation terminée pour tout le document ![/bold green]")
            return
        target_idx, block_indices = selection

        display_dashboard(crf, target_idx, block_indices)
        outcome = _collect_block_labels(crf, block_indices)

        if outcome is _Control.STOP:
            return
        if outcome is _Control.SKIP_PAGE:
            page_pos = crf.records[target_idx].page_pos
            page_index = crf.records[target_idx].page_index
            skipped = crf.mark_page_out_of_scope(page_pos)
            if skipped:
                save_session(session_path, document_hash, crf)
                console.print(f"[green]{len(skipped)} ligne(s) de la page {page_index} marquée(s) OUT OF SCOPE.[/green]")
            else:
                console.print("[yellow]Aucune ligne non annotée restante sur cette page.[/yellow]")
            reoffer_index = None
            continue
        if outcome is _Control.UNDO:
            reoffer_index = crf.undo_last_label()
            if reoffer_index is None:
                console.print("[yellow]Aucune classification humaine à annuler.[/yellow]")
            else:
                save_session(session_path, document_hash, crf)
                console.print("[green]Dernière classification annulée ; la ligne va être reproposée.[/green]")
            continue

        reoffer_index = None
        crf.set_labels(outcome)
        for index in sorted(outcome):
            console.print(f"[dim]✓ Ligne {crf.source_row_numbers[index]} → {outcome[index]}[/dim]")
        save_session(session_path, document_hash, crf)


def run(args: argparse.Namespace) -> None:
    ranges = require_ranges(args.folder)
    if len(ranges) > 1:
        raise CommandError(f"label annote une plage à la fois ; plages de ce volume : {', '.join(r.name for r in ranges)}.")
    range_dir = ranges[0]
    input_path = require_input(range_dir, LINES, "extract")
    if args.seed_size < 1:
        raise CommandError("--seed-size doit être supérieur à zéro.")
    try:
        records, raw_document, document_hash = load_json_lines(input_path)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise CommandError(f"JSON {input_path} : {error}") from error
    if not records:
        console.print("[yellow]Le JSON ne contient pas de lignes Markdown annotables.[/yellow]")
        return

    session_path = range_document(range_dir, LABEL_SESSION)
    output_path = range_document(range_dir, LABELED)
    writes = Writes(apply=True)  # export à la fin de la session, sans --apply (voir la docstring du module)
    crf = ActiveCRF(records, raw_document, seed_size=args.seed_size)
    if not args.reset_session:
        try:
            saved_session = load_session_state(session_path, document_hash)
        except (json.JSONDecodeError, ValueError) as error:
            raise CommandError(f"session {session_path} : {error}") from error
        if saved_session is not None:
            saved_labels, annotation_history, label_timestamps = saved_session
            crf.restore_labels(saved_labels, annotation_history, label_timestamps)
            console.print(f"[green]Session reprise : {crf.annotated_count} annotations chargées.[/green]")

    try:
        annotate_interactively(crf, session_path, document_hash)
        writes.add(output_path, lambda: crf.export_json(output_path))
        console.print(f"{len(crf.lines)} lignes, session : [yellow]{session_path}[/yellow]")
        writes.finish(console)
    finally:
        crf.close()
