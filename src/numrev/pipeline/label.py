"""Étape 2 · annotation interactive des lignes par apprentissage actif (CRF)
(`<document>.lines.json` → `<document>.labeled.json`).

Lit le JSON produit par `numrev extract`, propose des blocs de lignes à un
humain, réentraîne un CRF après chaque lot, et exporte une copie du JSON où chaque ligne est enrichie de sa
prédiction et de sa provenance.

La session (`<document>.label-session.json`) est enregistrée après chaque
action : c'est l'état de travail, qu'on reprend en relançant la commande.
L'export du JSON annoté, lui, suit la convention commune (numrev/command.py) :
simulation par défaut, écrit avec `--apply` — relancer avec `--apply` et
quitter aussitôt (`q`) exporte une session terminée.

Ce module ne contient que l'interface (dashboard, sessions, CLI) : le moteur
(features, entraînement, sélection des lignes) est dans `numrev.crf`.
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

from numrev.command import CommandError, Writes, add_apply_argument, console, default_output, require_file
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
from numrev.paths import LABEL_SESSION, LABELED

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
# 1. Dashboard Rich Debug
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
    """Résumé compact des transitions les plus fortes, sur une seule ligne."""
    if crf.tagger is None:
        return None
    sorted_transitions = sorted(
        crf.tagger.info().transitions.items(), key=lambda item: item[1], reverse=True
    )
    parts = []
    for (from_label, to_label), weight in sorted_transitions[:top_n]:
        color = "green" if weight > 0 else "red"
        parts.append(f"{from_label}➔{to_label} [{color}]{weight:+.2f}[/{color}]")
    return "  ·  ".join(parts) if parts else None


def _legend_renderable() -> Columns:
    """Légende compacte, répartie en plusieurs colonnes plutôt qu'une ligne par touche."""
    items = [
        f"[bold cyan]{key}[/bold cyan] {label}" for key, label in LABEL_KEYS.items()
    ]
    items.append("[bold cyan]u[/bold cyan] [dim]annuler[/dim]")
    items.append("[bold cyan]p[/bold cyan] [dim]page → OUT OF SCOPE[/dim]")
    items.append("[bold cyan]q[/bold cyan] [dim]arrêter[/dim]")
    return Columns(items, equal=True, expand=True, column_first=True)


def _candidates_marginals_table(crf: ActiveCRF, block_indices: list[int]) -> Table:
    """Une seule table de probabilités, une colonne par ligne candidate du bloc
    (au lieu d'une table complète par ligne : moins de bordures, moins de hauteur)."""
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


def _context_panel(
    crf: ActiveCRF, target_idx: int, block_indices: list[int], margin: int = 8
) -> Panel:
    """Contexte du document sous forme d'un seul renderable (pour tenir dans
    une colonne de la grille), au lieu d'une série de `console.print` séparés."""
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
        text.append(
            f"{prefix}"
            f"{label:<15} {crf.lines[index]}\n",
            style=style,
        )
    return Panel(text, title="📄 Contexte du document", border_style="blue")


def display_dashboard(
    crf: ActiveCRF, target_idx: int, block_indices: list[int]
) -> None:
    """Dashboard sur deux colonnes côte à côte (contexte / stats) plutôt qu'un
    empilement vertical : la hauteur totale est celle de la colonne la plus
    haute, pas la somme des deux."""
    console.clear()

    stats: list[Any] = [_progress_panel(crf, target_idx), _legend_renderable()]
    transitions_line = _transitions_line(crf)
    if transitions_line:
        stats.append(
            Panel(transitions_line, title="🔗 Transitions", border_style="magenta")
        )
    stats.append(_candidates_marginals_table(crf, block_indices))

    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(ratio=3)
    grid.add_column(ratio=2)
    grid.add_row(_context_panel(crf, target_idx, block_indices), Group(*stats))
    console.print(grid)

    if len(block_indices) == 1:
        console.print(
            "[dim]Dernière ligne non annotée : bloc réduit à une ligne.[/dim]"
        )


# ----------------------------------------------------------------------
# 2. Point d'entrée CLI
# ----------------------------------------------------------------------
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, help="Lignes extraites (<document>.lines.json, sortie de numrev extract).")
    parser.add_argument("-o", "--output", type=Path, default=None, help="JSON annoté de sortie (défaut : <document>.labeled.json).")
    parser.add_argument(
        "--seed-size", type=int, default=12, help="Nombre d'annotations diversifiées avant l'échantillonnage par incertitude (défaut : 12)."
    )
    parser.add_argument("--reset-session", action="store_true", help="Ignore une session existante et démarre une nouvelle annotation.")
    add_apply_argument(parser)


def load_session_state(
    session_path: Path, document_hash: str
) -> tuple[list[str | None], AnnotationHistory | None, LabelTimestamps] | None:
    if not session_path.exists():
        return None
    session = json.loads(session_path.read_text(encoding="utf-8"))
    if session.get("document_hash") != document_hash:
        raise ValueError("La session concerne une autre version du document source.")
    labels = session.get("labels")
    if not isinstance(labels, list):
        raise ValueError("La session ne contient pas de liste de labels valide.")
    annotated_indices = validate_labels(labels)
    annotation_history = session.get("annotation_history")
    if annotation_history is not None:
        if not isinstance(annotation_history, list):
            raise ValueError(
                "La session ne contient pas d'historique d'annotation valide."
            )
        validate_annotation_history(annotation_history, annotated_indices)
    label_timestamps = parse_label_timestamps(
        session.get("label_timestamps") or {}, annotated_indices
    )
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
                "label_timestamps": {
                    str(index): timestamp
                    for index, timestamp in crf.label_timestamps.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


class _Control(Enum):
    """Signaux de contrôle renvoyés par les prompts d'annotation interactive."""

    STOP = auto()
    UNDO = auto()
    SKIP_PAGE = auto()


def _prompt_label(crf: ActiveCRF, index: int) -> str | _Control:
    """Demande un label humain pour une ligne ; peut renvoyer un signal STOP/UNDO.

    Une touche non reconnue reprompte simplement au lieu d'arrêter toute la
    session : une faute de frappe ne doit jamais faire perdre le travail en
    cours.
    """
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
        console.print(
            "[yellow]Touche non reconnue — voir la légende ci-dessus.[/yellow]"
        )


def _collect_block_labels(
    crf: ActiveCRF, block_indices: list[int]
) -> dict[int, str] | _Control:
    """Recueille les labels humains d'un bloc, ou un signal STOP/UNDO/SKIP_PAGE.

    "u" a deux sens distincts selon le contexte, et ne perd jamais de saisie :
    - s'il reste au moins une ligne déjà étiquetée DANS CE BLOC, "u" annule
      uniquement la dernière saisie de ce bloc et la reprompte : rien n'est
      perdu et l'historique déjà validé (des blocs précédents) n'est pas
      touché ;
    - si le bloc est encore vide (tout juste commencé), "u" est renvoyé tel
      quel à l'appelant, qui annule alors la dernière ligne réellement
      validée (voir `annotate_interactively` / `ActiveCRF.undo_last_label`).

    "p" (SKIP_PAGE) est une action décisive et immédiate : elle est renvoyée
    telle quelle, sans attendre la fin du bloc, y compris si une saisie était
    déjà en attente pour ce bloc (perdue dans ce cas précis, car la page
    entière — dont la ligne visée par cette saisie — va de toute façon être
    classée OUT OF SCOPE par l'appelant).

    Avant le correctif du undo, un "u" pressé sur la deuxième ligne d'un
    bloc de deux faisait perdre silencieusement le label déjà saisi pour la
    première ligne, ET annulait une ligne totalement différente (la dernière
    du bloc précédent) au lieu de la ligne visée.
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
            console.print(
                f"[green]Saisie annulée pour la ligne "
                f"{crf.source_row_numbers[undone_index]} ; à ressaisir.[/green]"
            )
            position -= 1
            continue

        pending_labels[index] = label
        pending_order.append(index)
        position += 1

    return pending_labels


def annotate_interactively(
    crf: ActiveCRF, session_path: Path, document_hash: str
) -> None:
    """Boucle d'annotation humaine jusqu'à épuisement des lignes ou arrêt manuel."""
    reoffer_index: int | None = None
    while True:
        selection = crf.next_block(preferred_index=reoffer_index)
        if selection is None:
            console.print(
                "\n[bold green]🎉 Annotation terminée pour tout le document ![/bold green]"
            )
            return
        target_idx, block_indices = selection

        display_dashboard(crf, target_idx, block_indices)
        if len(block_indices) == 1:
            console.print(
                "\n[dim]Dernière ligne non annotée : bloc réduit à une ligne.[/dim]"
            )

        outcome = _collect_block_labels(crf, block_indices)

        if outcome is _Control.STOP:
            return
        if outcome is _Control.SKIP_PAGE:
            page_pos = crf.records[target_idx].page_pos
            page_index = crf.records[target_idx].page_index
            skipped = crf.mark_page_out_of_scope(page_pos)
            if skipped:
                save_session(session_path, document_hash, crf)
                console.print(
                    f"[green]{len(skipped)} ligne(s) de la page {page_index} "
                    "marquée(s) OUT OF SCOPE.[/green]"
                )
            else:
                console.print(
                    "[yellow]Aucune ligne non annotée restante sur cette page.[/yellow]"
                )
            reoffer_index = None
            continue
        if outcome is _Control.UNDO:
            reoffer_index = crf.undo_last_label()
            if reoffer_index is None:
                console.print(
                    "[yellow]Aucune classification humaine à annuler.[/yellow]"
                )
            else:
                save_session(session_path, document_hash, crf)
                console.print(
                    "[green]Dernière classification annulée ; "
                    "la ligne va être reproposée.[/green]"
                )
            continue

        reoffer_index = None
        crf.set_labels(outcome)
        for index in sorted(outcome):
            console.print(
                f"[dim]✓ Ligne {crf.source_row_numbers[index]} → {outcome[index]}[/dim]"
            )
        save_session(session_path, document_hash, crf)


def run(args: argparse.Namespace) -> None:
    require_file(args.input)
    if args.seed_size < 1:
        raise CommandError("--seed-size doit être supérieur à zéro.")
    try:
        records, raw_document, document_hash = load_json_lines(args.input)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise CommandError(f"JSON {args.input} : {error}") from error
    if not records:
        console.print("[yellow]Le JSON ne contient pas de lignes Markdown annotables.[/yellow]")
        return

    session_path = default_output(args.input, LABEL_SESSION)
    output_path = args.output or default_output(args.input, LABELED)
    writes = Writes(args.apply)
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
