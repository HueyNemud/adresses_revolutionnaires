"""Annotateur CRF par apprentissage actif pour des lignes Markdown OCRisées.

Lit le CSV produit par tabulate_chandra_output.py, propose des blocs de
lignes à un humain, réentraîne un CRF après chaque lot, et exporte les
prédictions avec leur provenance.
"""

import argparse
import csv
import enum
import hashlib
import io
import json
import os
import re
import tempfile
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pycrfsuite
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

INPUT_CSV_FIELDS = [
    "uid",
    "page_index",
    "chunk_index",
    "data_block_index",
    "line_index",
    "data_block_bbox",
    "data_block_label",
    "markdown",
]
OUTPUT_CSV_FIELDS = [*INPUT_CSV_FIELDS, "prediction", "provenance", "probability"]

CLASSES = ["ENTRY_BEGIN", "ENTRY_INSIDE", "TITLE", "OUT_OF_SCOPE"]


# ----------------------------------------------------------------------
# 1. Extraction des features & heuristique métier
# ----------------------------------------------------------------------
def get_shape(token: str) -> str:
    if token in (",", ".", "—", "-", "_"):
        return "PUNCT"
    if token.isdigit():
        return "NUM"
    if token.isupper():
        return "UPPER"
    if token.istitle():
        return "TITLE"
    if token.islower():
        return "LOWER"
    if token in ("*", "~_", "$", "#"):
        return "SYM"
    return "UNK"  # unknown / other


def get_heuristic_label(line: str, prev_line: str = "") -> str:
    """Retourne un indice de sélection, jamais une vérité de référence."""
    stripped = line.strip()
    if not stripped:
        return "OUT_OF_SCOPE"
    if stripped.startswith("#"):
        return "TITLE"
    if re.fullmatch(r"\{\d+\}[-—_]*", stripped) or re.fullmatch(r"[-—_]{3,}", stripped):
        return "OUT_OF_SCOPE"
    if stripped[0].islower() or stripped.startswith(("-", "—")):
        return "ENTRY_INSIDE"
    if prev_line and prev_line.rstrip().endswith(("-", "—")):
        return "ENTRY_INSIDE"
    return "ENTRY_BEGIN"


def normalize_ocr_label(label: str) -> str:
    """Normalize an OCR block class so equivalent spellings share one feature."""
    normalized = re.sub(r"[^\w]+", "_", label.strip().casefold())
    return normalized.strip("_") or "missing"


def extract_features(
    lines: list[str],
    source_line_numbers: list[int] | None = None,
    ocr_labels: list[str] | None = None,
) -> list[dict[str, str]]:
    if ocr_labels is not None and len(ocr_labels) != len(lines):
        raise ValueError("Chaque ligne doit avoir une classe de bloc OCR.")

    seq_features = []
    n = len(lines)

    for t, line in enumerate(lines):
        tokens = re.findall(r"\w+|[^\w\s]", line)
        shapes = [get_shape(tok) for tok in tokens]

        is_heading = line.startswith("#")
        heading_level = len(line) - len(line.lstrip("#")) if is_heading else 0

        feat = {
            "bias": "1.0",
            "is_heading": str(is_heading),
            "heading_level": str(heading_level),
            "starts_lower": str(shapes[0] == "LOWER") if shapes else "False",
            "ends_punct": str(shapes[-1] == "PUNCT") if shapes else "False",
            "token_count": str(min(len(tokens), 12)),
            # "has_markdown_bold": str("**" in line),
            "is_page_marker": str(bool(re.fullmatch(r"\{\d+\}[-—_]*", line))),
            "ocr_data_block_label": (
                normalize_ocr_label(ocr_labels[t])
                if ocr_labels is not None
                else "missing"
            ),
            "BOS": str(t == 0),
            "EOS": str(t == n - 1),
        }

        for i in range(min(4, len(shapes))):
            feat[f"shape_start_{i}"] = shapes[i]
            feat[f"shape_end_{i}"] = shapes[len(shapes) - 1 - i]

        if t > 0:
            feat["prev_is_heading"] = str(lines[t - 1].startswith("#"))
            feat["prev_ends_dash"] = str(lines[t - 1].rstrip().endswith(("-", "—")))
            if source_line_numbers is not None:
                feat["previous_source_gap"] = str(
                    source_line_numbers[t] - source_line_numbers[t - 1] > 1
                )

        seq_features.append(feat)

    return seq_features


@dataclass(frozen=True)
class SourceLine:
    """Une ligne Markdown annotable et sa provenance dans le CSV Chandra."""
    uid: str
    source_row: int
    text: str
    page_index: str = ""
    chunk_index: str = ""
    data_block_index: str = ""
    line_index: str = ""
    data_block_bbox: str = ""
    data_block_label: str = ""


def load_csv_lines(input_path: Path) -> tuple[list[SourceLine], str]:
    """Charge les lignes Markdown non vides et leur provenance Chandra."""
    raw_text = input_path.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(raw_text, newline=""))
    if reader.fieldnames is None:
        raise ValueError("Le CSV est vide ou ne contient pas d'en-tête.")
    missing_fields = [
        field for field in INPUT_CSV_FIELDS if field not in reader.fieldnames
    ]
    if missing_fields:
        raise ValueError(
            "Le CSV ne correspond pas à la sortie de tabulate_chandra_output : "
            f"colonnes manquantes {', '.join(missing_fields)}."
        )

    records = []
    for source_row, row in enumerate(reader, start=1):
        markdown = row["markdown"]
        if markdown is None:
            raise ValueError(f"La ligne CSV {source_row} ne contient pas de Markdown.")
        text = markdown.strip()
        if not text:
            continue
        records.append(
            SourceLine(
                source_row=source_row,
                uid=row["uid"],
                text=text,
                page_index=row["page_index"] or "",
                chunk_index=row["chunk_index"] or "",
                data_block_index=row["data_block_index"] or "",
                line_index=row["line_index"] or "",
                data_block_bbox=row["data_block_bbox"] or "",
                data_block_label=row["data_block_label"] or "",
            )
        )
    return records, hashlib.sha256(raw_text.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# 2. Moteur CRF Active Learning
# ----------------------------------------------------------------------
def _iter_labeled_segments(features: list[dict[str, str]], labels: list[str | None]):
    """Groupe les (feature, label) en segments contigus entièrement labellisés."""
    segment_features: list[dict[str, str]] = []
    segment_labels: list[str] = []
    for feature, label in zip(features, labels):
        if label is None:
            if segment_labels:
                yield segment_features, segment_labels
                segment_features, segment_labels = [], []
            continue
        segment_features.append(feature)
        segment_labels.append(label)
    if segment_labels:
        yield segment_features, segment_labels


def _train_tagger(
    features: list[dict[str, str]], labels: list[str | None], model_path: str
) -> Any | None:
    """Entraîne un tagger CRF sur les segments annotés, ou None si trop peu de données."""
    if len({label for label in labels if label is not None}) < 2:
        return None

    # python-crfsuite ne publie pas de stubs complets pour Pylance.
    trainer: Any = getattr(pycrfsuite, "Trainer")(verbose=False)
    trainer.set_params(
        {
            "c1": 0.1,
            "c2": 0.01,
            "max_iterations": 50,
            "feature.possible_transitions": True,
        }
    )
    for segment_features, segment_labels in _iter_labeled_segments(features, labels):
        trainer.append(segment_features, segment_labels)
    trainer.train(model_path)

    tagger: Any = getattr(pycrfsuite, "Tagger")()
    tagger.open(model_path)
    # `tag()` doit recevoir la séquence. Appeler `tag()` sans argument après
    # `set()` corrompt les marginales dans python-crfsuite.
    tagger.set(features)
    return tagger


class ActiveCRF:

    def __init__(
        self,
        records: list[SourceLine],
        seed_size: int = 12,
    ):
        self.records = records
        self.lines = [record.text for record in records]
        self.source_row_numbers = [record.source_row for record in records]
        self.features = extract_features(
            self.lines,
            self.source_row_numbers,
            [record.data_block_label for record in records],
        )
        self.labels: list[str | None] = [None] * len(records)
        self.annotation_history: list[int] = []
        self.heuristic_labels = [
            get_heuristic_label(line, self.lines[index - 1] if index else "")
            for index, line in enumerate(self.lines)
        ]
        self.seed_size = seed_size
        self.model_path = os.path.join(
            tempfile.gettempdir(), f"crf_{uuid.uuid4().hex}.crfsuite"
        )
        self.tagger: Any | None = None
        self.known_classes: set[str] = set()

    @property
    def annotated_count(self) -> int:
        return sum(label is not None for label in self.labels)

    def restore_labels(
        self,
        labels: list[str | None],
        annotation_history: list[int] | None = None,
    ) -> None:
        if len(labels) != len(self.labels):
            raise ValueError(
                "Le nombre de labels de session ne correspond pas au document."
            )
        invalid_labels = {
            label for label in labels if label is not None and label not in CLASSES
        }
        if invalid_labels:
            raise ValueError(f"Labels de session invalides : {sorted(invalid_labels)}")
        annotated_indices = {
            index for index, label in enumerate(labels) if label is not None
        }
        if annotation_history is None:
            annotation_history = sorted(annotated_indices)
        if (
            len(annotation_history) != len(set(annotation_history))
            or set(annotation_history) != annotated_indices
            or any(not isinstance(index, int) for index in annotation_history)
        ):
            raise ValueError("Historique d'annotation de session invalide.")
        self.labels = labels
        self.annotation_history = annotation_history
        self.retrain()

    def set_labels(self, annotations: dict[int, str]) -> None:
        """Enregistre atomiquement les labels humains d'un bloc d'annotation."""
        if not annotations:
            raise ValueError("Un bloc d'annotation ne peut pas être vide.")
        for index, label in annotations.items():
            if not 0 <= index < len(self.labels):
                raise IndexError(f"Indice de ligne invalide : {index}")
            if label not in CLASSES:
                raise ValueError(f"Classe inconnue : {label}")
        for index, label in annotations.items():
            self.labels[index] = label
            if index in self.annotation_history:
                self.annotation_history.remove(index)
            self.annotation_history.append(index)
        self.retrain()

    def undo_last_label(self) -> int | None:
        """Annule le dernier label humain et retourne l'indice à reproposer."""
        if not self.annotation_history:
            return None
        index = self.annotation_history.pop()
        self.labels[index] = None
        self.retrain()
        return index

    def retrain(self) -> None:
        """Entraîne le CRF sur les segments contigus validés par un humain."""
        self.known_classes = {label for label in self.labels if label is not None}
        self.tagger = _train_tagger(self.features, self.labels, self.model_path)

    def _next_exploration_index(self, unannotated: list[int]) -> int:
        """Échantillonne des strates heuristiques sous-représentées au démarrage."""
        observed = Counter(
            self.heuristic_labels[index]
            for index, label in enumerate(self.labels)
            if label is not None
        )
        return min(
            unannotated,
            key=lambda index: (
                observed[self.heuristic_labels[index]],
                self.heuristic_labels[index],
                index,
            ),
        )

    def _annotation_block(self, target_index: int, unannotated: list[int]) -> list[int]:
        """Retourne le candidat et un voisin non annoté, si disponible."""
        available = set(unannotated)
        neighbours = [
            index
            for index in (target_index - 1, target_index + 1)
            if index in available
        ]
        if not neighbours:
            return [target_index]
        # Préférer la ligne précédente : le candidat a ainsi un contexte CRF humain.
        neighbour = (
            target_index - 1 if target_index - 1 in neighbours else neighbours[0]
        )
        return sorted([target_index, neighbour])

    def next_block(
        self, preferred_index: int | None = None
    ) -> tuple[int, list[int]] | None:
        unannotated = [i for i, l in enumerate(self.labels) if l is None]
        if not unannotated:
            return None

        if preferred_index in unannotated:
            target_index = preferred_index
        elif self.annotated_count < self.seed_size or not self.tagger:
            target_index = self._next_exploration_index(unannotated)
        else:

            def margin(idx: int) -> float:
                probs = sorted(self.get_marginals(idx).values(), reverse=True)
                return probs[0] - probs[1]

            target_index = min(unannotated, key=margin)
        return target_index, self._annotation_block(target_index, unannotated)

    def get_marginals(self, idx: int) -> dict[str, float]:
        """Retourne des marginales normalisées, toujours comprises entre 0 et 1."""
        if not self.tagger:
            return {label: 0.0 for label in CLASSES}
        scores = {
            label: max(0.0, float(self.tagger.marginal(label, idx)))
            for label in self.known_classes
        }
        total = sum(scores.values())
        if total <= 0.0:
            return {label: 0.0 for label in CLASSES}
        return {label: scores.get(label, 0.0) / total for label in CLASSES}

    def _provenance_row(self, index: int) -> dict[str, str]:
        """Colonnes de provenance Chandra recopiées telles quelles dans l'export."""
        record = self.records[index]
        return {
            "uid": record.uid,
            "page_index": record.page_index,
            "chunk_index": record.chunk_index,
            "data_block_index": record.data_block_index,
            "line_index": record.line_index,
            "data_block_bbox": record.data_block_bbox,
            "data_block_label": record.data_block_label,
            "markdown": self.lines[index],
        }

    def export_csv(self, output_path: str) -> None:
        predictions = (
            self.tagger.tag(self.features) if self.tagger else [None] * len(self.lines)
        )

        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_CSV_FIELDS)
            writer.writeheader()
            for index in range(len(self.lines)):
                human_label = self.labels[index]
                model_label = predictions[index]
                prediction = human_label or model_label or ""
                provenance = (
                    "human"
                    if human_label
                    else ("model" if model_label else "unclassified")
                )
                probability = (
                    1.0
                    if human_label
                    else self.get_marginals(index).get(prediction, 0.0)
                )
                writer.writerow(
                    {
                        **self._provenance_row(index),
                        "prediction": prediction,
                        "provenance": provenance,
                        "probability": f"{probability:.4f}",
                    }
                )

        console.print(
            "\n[bold green]✅ Export CSV réussi :[/bold green] "
            f"[yellow]{output_path}[/yellow] ({len(self.lines)} lignes)"
        )


# ----------------------------------------------------------------------
# 3. Dashboard Rich Debug
# ----------------------------------------------------------------------
def _progress_panel(crf: ActiveCRF) -> Panel:
    annotated_cnt = crf.annotated_count
    total_cnt = len(crf.lines)

    metrics_table = Table(show_header=False, box=None)
    metrics_table.add_column("Key", style="bold cyan")
    metrics_table.add_column("Val", style="yellow")
    metrics_table.add_row(
        "Avancement :",
        f"{annotated_cnt}/{total_cnt} ({annotated_cnt/total_cnt*100:.1f}%)",
    )
    return Panel(
        metrics_table,
        title="📈 [bold yellow]ÉTAT DE L'ANNOTATION CRF[/bold yellow]",
    )


def _transitions_table(crf: ActiveCRF) -> Table | None:
    if crf.tagger is None:
        return None

    table = Table(
        title="🔗 Transitions apprises dans les blocs annotés",
        header_style="bold magenta",
    )
    table.add_column("Transition (Ligne N-1 ➔ Ligne N)")
    table.add_column("Poids CRF", justify="right")

    sorted_transitions = sorted(
        crf.tagger.info().transitions.items(), key=lambda item: item[1], reverse=True
    )
    for (from_label, to_label), weight in sorted_transitions[:5]:
        color = "green" if weight > 0 else "red"
        table.add_row(f"{from_label} ➔ {to_label}", f"[{color}]{weight:+.3f}[/]")
    return table


def _marginals_table(crf: ActiveCRF, target_idx: int) -> Table:
    target_record = crf.records[target_idx]
    table = Table(
        title=(
            # Print the text first 20 characters of the line tex for context
            f"{target_record.text[:20]}... "
            f"[p. {target_record.page_index} l. {target_record.line_index}]"
        ),
        header_style="bold green",
    )
    table.add_column("Classe")
    table.add_column("Probabilité", justify="right")

    probs = crf.get_marginals(target_idx)
    for label, proba in sorted(probs.items(), key=lambda item: item[1], reverse=True):
        table.add_row(label, f"{proba*100:.1f}%")
    return table


# def _block_table(crf: ActiveCRF, target_idx: int, block_indices: list[int]) -> Table:
#     table = Table(title="📝 Bloc à annoter", header_style="bold yellow")
#     table.add_column("Source", justify="right")
#     table.add_column("Texte")
#     for index in block_indices:
#         record = crf.records[index]
#         marker = " → cible" if index == target_idx else ""
#         table.add_row(
#             f"CSV {record.source_row} · p.{record.page_index} · "
#             f"b.{record.data_block_index} · l.{record.line_index} · "
#             f"OCR={normalize_ocr_label(record.data_block_label)}{marker}",
#             crf.lines[index],
#         )
#     return table


def _features_table(crf: ActiveCRF, target_idx: int) -> Table:
    target_record = crf.records[target_idx]
    table = Table(
        title=f"🔍 Features actives [CSV {target_record.source_row}]",
        header_style="bold cyan",
        padding=(0, 1),
    )
    table.add_column("Feature / Transition Interne", style="yellow")
    table.add_column("Valeur", style="green")
    for key, value in crf.features[target_idx].items():
        if value not in ("False", "0"):
            table.add_row(key, str(value))
    return table


def _print_document_context(
    crf: ActiveCRF, target_idx: int, block_indices: list[int]
) -> None:
    start = max(0, target_idx - 2)
    end = min(len(crf.lines), target_idx + 3)
    for index in range(start, end):
        prefix = "▷ " if index in block_indices else "  "
        # label = f"[{crf.labels[index]}]" if crf.labels[index] else "[NO_LABEL]"
        label = f"[{crf.records[index].data_block_label}]"
        style = "bold reverse green" if index in block_indices else "dim"
        console.print(
            f"[{prefix}{crf.source_row_numbers[index]:4d}] {label:15} "
            f"{crf.lines[index]}",
            style=style,
        )


def display_dashboard(
    crf: ActiveCRF, target_idx: int, block_indices: list[int]
) -> None:
    console.clear()
    console.print(_progress_panel(crf))

    transitions = _transitions_table(crf)
    if transitions is not None:
        console.print(transitions)

    for index in block_indices:
        console.print(_marginals_table(crf, index))

    # console.print(_marginals_table(crf, target_idx))
    # console.print(_block_table(crf, target_idx, block_indices))
    # console.print(_features_table(crf, target_idx))

    console.print("\n[bold yellow]📄 Contexte du document :[/bold yellow]")
    _print_document_context(crf, target_idx, block_indices)


# ----------------------------------------------------------------------
# 4. Point d'entrée CLI
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Active Learning CRF pour classification de lignes."
    )
    parser.add_argument(
        "csv_file",
        type=Path,
        help="CSV produit par tabulate_chandra_output.py",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="predictions_crf.csv",
        help="Nom du CSV de sortie",
    )
    parser.add_argument(
        "--seed-size",
        type=int,
        default=12,
        help="Nombre d'annotations diversifiées avant l'échantillonnage par incertitude.",
    )
    parser.add_argument(
        "--session",
        type=Path,
        help="Fichier JSON de session (défaut : suffixe .crf-session.json).",
    )
    parser.add_argument(
        "--reset-session",
        action="store_true",
        help="Ignore une session existante et démarre une nouvelle annotation.",
    )
    return parser.parse_args()


def load_session_state(
    session_path: Path, document_hash: str
) -> tuple[list[str | None], list[int] | None] | None:
    if not session_path.exists():
        return None
    session = json.loads(session_path.read_text(encoding="utf-8"))
    if session.get("document_hash") != document_hash:
        raise ValueError("La session concerne une autre version du document source.")
    labels = session.get("labels")
    if not isinstance(labels, list):
        raise ValueError("La session ne contient pas de liste de labels valide.")
    annotation_history = session.get("annotation_history")
    if annotation_history is not None and (
        not isinstance(annotation_history, list)
        or any(not isinstance(index, int) for index in annotation_history)
    ):
        raise ValueError("La session ne contient pas d'historique d'annotation valide.")
    return labels, annotation_history


def load_session(session_path: Path, document_hash: str) -> list[str | None] | None:
    """Charge les labels de session (compatibilité avec l'API existante)."""
    session_state = load_session_state(session_path, document_hash)
    return session_state[0] if session_state is not None else None


def save_session(session_path: Path, document_hash: str, crf: ActiveCRF) -> None:
    session_path.write_text(
        json.dumps(
            {
                "version": 2,
                "document_hash": document_hash,
                "source_rows": crf.source_row_numbers,
                "labels": crf.labels,
                "annotation_history": crf.annotation_history,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


LABEL_KEYS = {
    "1": "ENTRY_BEGIN",
    "2": "ENTRY_INSIDE",
    "3": "TITLE",
    "0": "OUT_OF_SCOPE",
}


class _Control(enum.Enum):
    """Signaux de contrôle renvoyés par les prompts d'annotation interactive."""

    STOP = enum.auto()
    UNDO = enum.auto()


def _prompt_label(crf: ActiveCRF, index: int) -> str | _Control:
    """Demande un label humain pour une ligne ; peut renvoyer un signal STOP/UNDO."""
    console.print(
        f"\n[bold]Label CSV {crf.source_row_numbers[index]} ?[/bold] "
        "(1=ENTRY_BEGIN, 2=ENTRY_INSIDE, 3=TITLE, 0=OUT_OF_SCOPE, "
        "u=undo, q=predict and stop)",
        end="",
    )
    choice = input().strip().lower()
    if choice == "q":
        return _Control.STOP
    if choice == "u":
        return _Control.UNDO
    if choice not in LABEL_KEYS:
        console.print("[yellow]Choix invalide : bloc non enregistré.[/yellow]")
        return _Control.STOP
    return LABEL_KEYS[choice]


def _collect_block_labels(
    crf: ActiveCRF, block_indices: list[int]
) -> dict[int, str] | _Control:
    """Recueille les labels humains d'un bloc, ou un signal STOP/UNDO de l'utilisateur."""
    pending_labels: dict[int, str] = {}
    for index in block_indices:
        label = _prompt_label(crf, index)
        if isinstance(label, _Control):
            return label
        pending_labels[index] = label
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
        save_session(session_path, document_hash, crf)


def main():
    args = parse_args()

    if not args.csv_file.exists():
        console.print(
            f"[bold red]Erreur :[/bold red] Fichier '{args.csv_file}' introuvable."
        )
        return

    if args.seed_size < 1:
        console.print(
            "[bold red]Erreur :[/bold red] --seed-size doit être supérieur à zéro."
        )
        return

    try:
        records, document_hash = load_csv_lines(args.csv_file)
    except (csv.Error, UnicodeDecodeError, ValueError) as error:
        console.print(f"[bold red]Erreur de CSV :[/bold red] {error}")
        return
    if not records:
        console.print(
            "[yellow]Le CSV ne contient pas de lignes Markdown annotables.[/yellow]"
        )
        return

    session_path = args.session or args.csv_file.with_suffix(".crf-session.json")
    crf = ActiveCRF(records, seed_size=args.seed_size)
    if not args.reset_session:
        try:
            saved_session = load_session_state(session_path, document_hash)
            if saved_session is not None:
                saved_labels, annotation_history = saved_session
                crf.restore_labels(saved_labels, annotation_history)
                console.print(
                    f"[green]Session reprise : {crf.annotated_count} annotations chargées.[/green]"
                )
        except (json.JSONDecodeError, ValueError) as error:
            console.print(f"[bold red]Erreur de session :[/bold red] {error}")
            return

    annotate_interactively(crf, session_path, document_hash)
    crf.export_csv(args.output)


if __name__ == "__main__":
    main()
