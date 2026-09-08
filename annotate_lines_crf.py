"""Annotateur CRF par apprentissage actif pour des lignes Markdown OCRisées.

Lit le JSON produit par extract_chandra_lines.py (ou par ce script
lui-même), propose des blocs de lignes à un humain, réentraîne un CRF après

chaque lot, et exporte une copie du JSON où chaque ligne est enrichie de sa
prédiction et de sa provenance.
"""

import argparse
import copy
import hashlib
import json
import re
import tempfile
import uuid
import weakref
from collections import Counter
from datetime import datetime
from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from pathlib import Path
from typing import Any, Iterator, TypeAlias

import pycrfsuite
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from chandra_document import LineLocation, iter_line_locations

console = Console()


class AnnotationLabel(StrEnum):
    """Labels supported by the sequence classifier and persisted sessions."""

    ENTRY_BEGIN = "ENTRY_BEGIN"
    ENTRY_INSIDE = "ENTRY_INSIDE"
    TITLE = "TITLE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


CLASSES: tuple[str, ...] = tuple(label.value for label in AnnotationLabel)
LABEL_KEYS = {
    "1": AnnotationLabel.ENTRY_BEGIN.value,
    "2": AnnotationLabel.ENTRY_INSIDE.value,
    "3": AnnotationLabel.TITLE.value,
    "0": AnnotationLabel.OUT_OF_SCOPE.value,
}
SESSION_VERSION = 2

Feature: TypeAlias = dict[str, str]
FeatureSequence: TypeAlias = list[Feature]
AnnotationHistory: TypeAlias = list[int]
LabelTimestamps: TypeAlias = dict[int, str]


def _current_timestamp() -> str:
    """Horodatage à la seconde, lisible et trié correctement (ISO 8601)."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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
        return AnnotationLabel.OUT_OF_SCOPE.value
    if stripped.startswith("#"):
        return AnnotationLabel.TITLE.value
    if re.fullmatch(r"\{\d+\}[-—_]*", stripped) or re.fullmatch(r"[-—_]{3,}", stripped):
        return AnnotationLabel.OUT_OF_SCOPE.value
    if stripped[0].islower() or stripped.startswith(("-", "—")):
        return AnnotationLabel.ENTRY_INSIDE.value
    if prev_line and prev_line.rstrip().endswith(("-", "—")):
        return AnnotationLabel.ENTRY_INSIDE.value
    return AnnotationLabel.ENTRY_BEGIN.value


def normalize_ocr_label(label: str) -> str:
    """Normalize an OCR block class so equivalent spellings share one feature."""
    normalized = re.sub(r"[^\w]+", "_", label.strip().casefold())
    return normalized.strip("_") or "missing"


def extract_features(
    lines: list[str],
    source_line_numbers: list[int] | None = None,
    ocr_labels: list[str] | None = None,
) -> FeatureSequence:
    if source_line_numbers is not None and len(source_line_numbers) != len(lines):
        raise ValueError("Chaque ligne doit avoir un numéro de ligne source.")
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
    """Une ligne Markdown annotable et sa provenance dans le JSON Chandra."""

    uid: str
    source_row: int
    text: str
    page_index: str = ""
    chunk_index: str = ""
    data_block_index: str = ""
    line_index: str = ""
    data_block_bbox: str = ""
    data_block_label: str = ""
    page_pos: int = 0
    block_pos: int = 0
    line_pos: int = 0

    @classmethod
    def from_location(
        cls, location: LineLocation, source_row: int, text: str
    ) -> "SourceLine":
        """Create an annotatable record from a validated Chandra line location."""
        return cls(
            uid=str(location.line.get("uid", "")),
            source_row=source_row,
            text=text,
            page_index=str(location.page.get("page_index", "")),
            chunk_index=str(location.block.get("chunk_index", "")),
            data_block_index=str(location.block.get("index", "")),
            line_index=str(location.line.get("line_index", "")),
            data_block_bbox=str(location.block.get("bbox", "")),
            data_block_label=str(location.block.get("label", "")),
            page_pos=location.page_pos,
            block_pos=location.block_pos,
            line_pos=location.line_pos,
        )


def load_json_lines(input_path: Path) -> tuple[list[SourceLine], Any, str]:
    """Charge les lignes Markdown non vides et leur provenance depuis un JSON
    produit par extract_chandra_lines.py (ou par ce script lui-même)."""
    raw_text = input_path.read_text(encoding="utf-8")
    document = json.loads(raw_text)
    if not isinstance(document, list):
        raise ValueError("Le JSON doit être une liste de pages.")

    records: list[SourceLine] = []
    source_row = 0
    for location in iter_line_locations(document):
        markdown = location.line.get("markdown")
        if markdown is None:
            raise ValueError(
                f"La ligne {location.line.get('uid', '?')} ne contient pas de Markdown."
            )
        text = markdown.strip()
        if not text:
            continue
        source_row += 1
        records.append(SourceLine.from_location(location, source_row, text))
    return records, document, hashlib.sha256(raw_text.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# 2. Moteur CRF Active Learning
# ----------------------------------------------------------------------
def _iter_labeled_segments(
    features: FeatureSequence, labels: list[str | None]
) -> Iterator[tuple[FeatureSequence, list[str]]]:
    """Groupe les (feature, label) en segments contigus entièrement labellisés."""
    segment_features: FeatureSequence = []
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
    features: FeatureSequence, labels: list[str | None], model_path: Path
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
    trainer.train(str(model_path))

    tagger: Any = getattr(pycrfsuite, "Tagger")()
    tagger.open(str(model_path))
    # `tag()` doit recevoir la séquence. Appeler `tag()` sans argument après
    # `set()` corrompt les marginales dans python-crfsuite.
    tagger.set(features)
    return tagger


def _annotated_indices(labels: list[str | None]) -> set[int]:
    return {index for index, label in enumerate(labels) if label is not None}


def _validate_labels(labels: list[str | None]) -> set[int]:
    if not all(label is None or isinstance(label, str) for label in labels):
        raise ValueError("La session ne contient pas de liste de labels valide.")
    invalid_labels = {
        label for label in labels if label is not None and label not in CLASSES
    }
    if invalid_labels:
        raise ValueError(f"Labels de session invalides : {sorted(invalid_labels)}")
    return _annotated_indices(labels)


def _validate_annotation_history(
    annotation_history: AnnotationHistory, annotated_indices: set[int]
) -> None:
    if (
        len(annotation_history) != len(set(annotation_history))
        or set(annotation_history) != annotated_indices
        or not all(isinstance(index, int) for index in annotation_history)
    ):
        raise ValueError("Historique d'annotation de session invalide.")


def _parse_label_timestamps(
    raw_timestamps: object, annotated_indices: set[int]
) -> LabelTimestamps:
    if not isinstance(raw_timestamps, dict):
        raise ValueError("La session ne contient pas d'horodatages valides.")
    try:
        timestamps = {int(index): value for index, value in raw_timestamps.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("La session ne contient pas d'horodatages valides.") from error
    if not set(timestamps).issubset(annotated_indices) or not all(
        isinstance(timestamp, str) for timestamp in timestamps.values()
    ):
        raise ValueError("La session ne contient pas d'horodatages valides.")
    return timestamps


@dataclass(frozen=True)
class LineAnnotation:
    """Prediction metadata written back to an OCR line."""

    prediction: str
    provenance: str
    probability: float
    timestamp: str


class ActiveCRF:
    def __init__(
        self,
        records: list[SourceLine],
        raw_document: list[Any] | None = None,
        seed_size: int = 12,
    ) -> None:
        self.records = records
        self.raw_document = raw_document if raw_document is not None else []
        self.lines = [record.text for record in records]
        self.source_row_numbers = [record.source_row for record in records]
        self.features = extract_features(
            self.lines,
            self.source_row_numbers,
            [record.data_block_label for record in records],
        )
        self.labels: list[str | None] = [None] * len(records)
        self.annotation_history: AnnotationHistory = []
        self.label_timestamps: LabelTimestamps = {}
        self.heuristic_labels = [
            get_heuristic_label(line, self.lines[index - 1] if index else "")
            for index, line in enumerate(self.lines)
        ]
        self.seed_size = seed_size
        self.model_path = (
            Path(tempfile.gettempdir()) / f"crf_{uuid.uuid4().hex}.crfsuite"
        )
        self._model_file_finalizer = weakref.finalize(
            self, self.model_path.unlink, missing_ok=True
        )
        self.tagger: Any | None = None
        self.known_classes: set[str] = set()

    def close(self) -> None:
        """Remove the temporary CRF model file when it is no longer needed."""
        self._model_file_finalizer()

    @property
    def annotated_count(self) -> int:
        return sum(label is not None for label in self.labels)

    def restore_labels(
        self,
        labels: list[str | None],
        annotation_history: list[int] | None = None,
        label_timestamps: dict[int, str] | None = None,
    ) -> None:
        if len(labels) != len(self.labels):
            raise ValueError(
                "Le nombre de labels de session ne correspond pas au document."
            )
        annotated_indices = _validate_labels(labels)
        if annotation_history is None:
            annotation_history = sorted(annotated_indices)
        _validate_annotation_history(annotation_history, annotated_indices)
        label_timestamps = label_timestamps or {}
        if not set(label_timestamps).issubset(annotated_indices):
            raise ValueError("Horodatages de session invalides.")
        self.labels = labels
        self.annotation_history = annotation_history
        self.label_timestamps = dict(label_timestamps)
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
        timestamp = _current_timestamp()
        for index, label in annotations.items():
            self.labels[index] = label
            self.label_timestamps[index] = timestamp
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
        self.label_timestamps.pop(index, None)
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
            # Une ligne reproposée après annulation est traitée seule : on évite
            # d'y associer une voisine, pour la reproposer sans ambiguïté.
            return preferred_index, [preferred_index]
        elif self.annotated_count < self.seed_size or not self.tagger:
            target_index = self._next_exploration_index(unannotated)
        else:

            target_index = min(unannotated, key=self._prediction_margin)
        return target_index, self._annotation_block(target_index, unannotated)

    def _prediction_margin(self, index: int) -> float:
        """Return the gap between the two most likely labels for one line."""
        probabilities = sorted(self.get_marginals(index).values(), reverse=True)
        return probabilities[0] - probabilities[1]

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

    def _line_annotation(
        self, index: int, model_label: str | None, export_timestamp: str
    ) -> LineAnnotation:
        human_label = self.labels[index]
        prediction = human_label or model_label or ""
        if human_label:
            return LineAnnotation(
                prediction, "human", 1.0, self.label_timestamps[index]
            )
        if model_label:
            return LineAnnotation(
                prediction,
                "model",
                self.get_marginals(index).get(prediction, 0.0),
                export_timestamp,
            )
        return LineAnnotation("", "unclassified", 0.0, "")

    def export_json(self, output_path: str | Path) -> None:
        predictions = (
            self.tagger.tag(self.features) if self.tagger else [None] * len(self.lines)
        )
        export_timestamp = _current_timestamp()

        output_document = copy.deepcopy(self.raw_document)
        output_lines = {
            (location.page_pos, location.block_pos, location.line_pos): location.line
            for location in iter_line_locations(output_document)
        }
        for index, record in enumerate(self.records):
            position = (record.page_pos, record.block_pos, record.line_pos)
            try:
                line = output_lines[position]
            except KeyError as error:
                raise ValueError(
                    f"La ligne source {record.uid!r} est introuvable dans le document."
                ) from error
            annotation = self._line_annotation(
                index, predictions[index], export_timestamp
            )
            line.update(
                prediction=annotation.prediction,
                provenance=annotation.provenance,
                probability=round(annotation.probability, 4),
                timestamp=annotation.timestamp,
            )

        Path(output_path).write_text(
            json.dumps(output_document, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        console.print(
            "\n[bold green]✅ Export JSON réussi :[/bold green] "
            f"[yellow]{output_path}[/yellow] ({len(self.lines)} lignes)"
        )


# ----------------------------------------------------------------------
# 3. Dashboard Rich Debug
# ----------------------------------------------------------------------
def _progress_panel(crf: ActiveCRF) -> Panel:
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


def _print_document_context(
    crf: ActiveCRF, target_idx: int, block_indices: list[int]
) -> None:
    start = max(0, target_idx - 2)
    end = min(len(crf.lines), target_idx + 3)
    for index in range(start, end):
        prefix = "▷ " if index in block_indices else "  "
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
        "json_file",
        type=Path,
        help="JSON produit par extract_chandra_lines.py (ou par ce script).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="predictions_crf.json",
        help="Nom du JSON de sortie",
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
) -> tuple[list[str | None], AnnotationHistory | None, LabelTimestamps] | None:
    if not session_path.exists():
        return None
    session = json.loads(session_path.read_text(encoding="utf-8"))
    if session.get("document_hash") != document_hash:
        raise ValueError("La session concerne une autre version du document source.")
    labels = session.get("labels")
    if not isinstance(labels, list):
        raise ValueError("La session ne contient pas de liste de labels valide.")
    annotated_indices = _validate_labels(labels)
    annotation_history = session.get("annotation_history")
    if annotation_history is not None:
        if not isinstance(annotation_history, list):
            raise ValueError(
                "La session ne contient pas d'historique d'annotation valide."
            )
        _validate_annotation_history(annotation_history, annotated_indices)
    label_timestamps = _parse_label_timestamps(
        session.get("label_timestamps") or {}, annotated_indices
    )
    return labels, annotation_history, label_timestamps


def load_session(session_path: Path, document_hash: str) -> list[str | None] | None:
    """Charge les labels de session (compatibilité avec l'API existante)."""
    session_state = load_session_state(session_path, document_hash)
    return session_state[0] if session_state is not None else None


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


def _prompt_label(crf: ActiveCRF, index: int) -> str | _Control:
    """Demande un label humain pour une ligne ; peut renvoyer un signal STOP/UNDO."""
    console.print(
        f"\n[bold]Label ligne {crf.source_row_numbers[index]} ?[/bold] "
        f"({', '.join(f'{key}={label}' for key, label in LABEL_KEYS.items())}, "
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

    if not args.json_file.exists():
        console.print(
            f"[bold red]Erreur :[/bold red] Fichier '{args.json_file}' introuvable."
        )
        return

    if args.seed_size < 1:
        console.print(
            "[bold red]Erreur :[/bold red] --seed-size doit être supérieur à zéro."
        )
        return

    try:
        records, raw_document, document_hash = load_json_lines(args.json_file)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        console.print(f"[bold red]Erreur de JSON :[/bold red] {error}")
        return
    if not records:
        console.print(
            "[yellow]Le JSON ne contient pas de lignes Markdown annotables.[/yellow]"
        )
        return

    session_path = args.session or args.json_file.with_suffix(".crf-session.json")
    crf = ActiveCRF(records, raw_document, seed_size=args.seed_size)
    if not args.reset_session:
        try:
            saved_session = load_session_state(session_path, document_hash)
            if saved_session is not None:
                saved_labels, annotation_history, label_timestamps = saved_session
                crf.restore_labels(saved_labels, annotation_history, label_timestamps)
                console.print(
                    f"[green]Session reprise : {crf.annotated_count} annotations chargées.[/green]"
                )
        except (json.JSONDecodeError, ValueError) as error:
            console.print(f"[bold red]Erreur de session :[/bold red] {error}")
            return

    try:
        annotate_interactively(crf, session_path, document_hash)
        crf.export_json(args.output)
    finally:
        crf.close()


if __name__ == "__main__":
    main()
