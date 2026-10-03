"""Moteur d'apprentissage actif : lignes sources, sélection des lignes à
annoter, réentraînement et export des prédictions dans le JSON Chandra."""

import copy
import hashlib
import json
import tempfile
import uuid
import weakref
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeAlias

from lib.chandra_document import LineLocation, iter_line_locations
from lib.crf.features import (
    PRODUCTION_GROUPS,
    SequenceContext,
    extract_features_from_context,
    get_heuristic_label,
    parse_bbox,
)
from lib.crf.labels import CLASSES, AnnotationLabel
from lib.crf.model import (
    DEFAULT_CONFIG,
    TrainingConfig,
    normalized_marginals,
    posterior_decode,
    train_tagger,
)

AnnotationHistory: TypeAlias = list[int]
LabelTimestamps: TypeAlias = dict[int, str]


def current_timestamp() -> str:
    """Horodatage à la seconde, lisible et trié correctement (ISO 8601)."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------------------------------------------------
# Lignes sources
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SourceLine:
    """Une ligne Markdown annotable et sa provenance dans le JSON Chandra."""

    uid: str
    source_row: int
    text: str
    cle: str = ""  # clé stable de la ligne (lib/curation.py)
    page_index: str = ""
    chunk_index: str = ""
    data_block_index: str = ""
    line_index: str = ""
    data_block_bbox: str = ""
    data_block_label: str = ""
    page_pos: int = 0
    block_pos: int = 0
    line_pos: int = 0
    # La ligne qui la précède dans l'ordre du document (ignorée par le CRF
    # car vide) était blanche.
    follows_blank: bool = False

    @classmethod
    def from_location(
        cls, location: LineLocation, source_row: int, text: str, follows_blank: bool = False
    ) -> "SourceLine":
        """Create an annotatable record from a validated Chandra line location."""
        return cls(
            uid=str(location.line.get("uid", "")),
            source_row=source_row,
            text=text,
            cle=str(location.line.get("cle", "")),
            page_index=str(location.page.get("page_index", "")),
            chunk_index=str(location.block.get("chunk_index", "")),
            data_block_index=str(location.block.get("index", "")),
            line_index=str(location.line.get("line_index", "")),
            data_block_bbox=str(location.block.get("bbox", "")),
            data_block_label=str(location.block.get("label", "")),
            page_pos=location.page_pos,
            block_pos=location.block_pos,
            line_pos=location.line_pos,
            follows_blank=follows_blank,
        )


def load_json_lines(input_path: Path) -> tuple[list[SourceLine], Any, str]:
    """Charge les lignes Markdown non vides et leur provenance depuis un JSON
    produit par extract_chandra_lines.py (ou par annotate_lines_crf.py)."""
    raw_text = input_path.read_text(encoding="utf-8")
    document = json.loads(raw_text)
    if not isinstance(document, list):
        raise ValueError("Le JSON doit être une liste de pages.")

    records: list[SourceLine] = []
    source_row = 0
    previous_blank = False
    for location in iter_line_locations(document):
        markdown = location.line.get("markdown")
        if markdown is None:
            raise ValueError(
                f"La ligne {location.line.get('uid', '?')} ne contient pas de Markdown."
            )
        text = markdown.strip()
        if not text:
            previous_blank = True
            continue
        source_row += 1
        records.append(SourceLine.from_location(location, source_row, text, previous_blank))
        previous_blank = False
    return records, document, hashlib.sha256(raw_text.encode("utf-8")).hexdigest()


def context_from_records(records: Sequence[SourceLine]) -> SequenceContext:
    """Contexte de séquence complet (production + groupes candidats)."""
    return SequenceContext(
        lines=[record.text for record in records],
        source_line_numbers=[record.source_row for record in records],
        ocr_labels=[record.data_block_label for record in records],
        page_positions=[record.page_pos for record in records],
        block_keys=[(record.page_pos, record.block_pos) for record in records],
        block_bboxes=[parse_bbox(record.data_block_bbox) for record in records],
        follows_blank=[record.follows_blank for record in records],
    )


# ----------------------------------------------------------------------
# Validation des labels / historiques (sessions)
# ----------------------------------------------------------------------
def annotated_indices(labels: list[str | None]) -> set[int]:
    return {index for index, label in enumerate(labels) if label is not None}


def validate_labels(labels: list[str | None]) -> set[int]:
    if not all(label is None or isinstance(label, str) for label in labels):
        raise ValueError("La session ne contient pas de liste de labels valide.")
    invalid_labels = {
        label for label in labels if label is not None and label not in CLASSES
    }
    if invalid_labels:
        raise ValueError(f"Labels de session invalides : {sorted(invalid_labels)}")
    return annotated_indices(labels)


def validate_annotation_history(
    annotation_history: AnnotationHistory, annotated: set[int]
) -> None:
    if (
        len(annotation_history) != len(set(annotation_history))
        or set(annotation_history) != annotated
        or not all(isinstance(index, int) for index in annotation_history)
    ):
        raise ValueError("Historique d'annotation de session invalide.")


def parse_label_timestamps(raw_timestamps: object, annotated: set[int]) -> LabelTimestamps:
    if not isinstance(raw_timestamps, dict):
        raise ValueError("La session ne contient pas d'horodatages valides.")
    try:
        timestamps = {int(index): value for index, value in raw_timestamps.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("La session ne contient pas d'horodatages valides.") from error
    if not set(timestamps).issubset(annotated) or not all(
        isinstance(timestamp, str) for timestamp in timestamps.values()
    ):
        raise ValueError("La session ne contient pas d'horodatages valides.")
    return timestamps


# ----------------------------------------------------------------------
# Moteur CRF Active Learning
# ----------------------------------------------------------------------
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
        feature_groups: Sequence[str] = PRODUCTION_GROUPS,
        training_config: TrainingConfig = DEFAULT_CONFIG,
    ) -> None:
        self.records = records
        self.raw_document = raw_document if raw_document is not None else []
        self.lines = [record.text for record in records]
        self.source_row_numbers = [record.source_row for record in records]
        self.page_positions = [record.page_pos for record in records]
        self.feature_groups = tuple(feature_groups)
        self.training_config = training_config
        self.features = extract_features_from_context(
            context_from_records(records), self.feature_groups
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
        annotated = validate_labels(labels)
        if annotation_history is None:
            annotation_history = sorted(annotated)
        validate_annotation_history(annotation_history, annotated)
        label_timestamps = label_timestamps or {}
        if not set(label_timestamps).issubset(annotated):
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
        timestamp = current_timestamp()
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

    def mark_page_out_of_scope(self, page_pos: int) -> list[int]:
        """Marque en une fois toutes les lignes non annotées d'une page en OUT OF SCOPE.

        Pratique pour écarter rapidement des pages entières non pertinentes
        (ex. une table des matières, une page de publicité) sans repasser
        ligne par ligne. Chaque ligne est ajoutée individuellement à
        l'historique (via `set_labels`) : `undo_last_label` peut donc
        toujours défaire ce geste, une ligne à la fois.
        """
        indices = [
            index
            for index, position in enumerate(self.page_positions)
            if position == page_pos and self.labels[index] is None
        ]
        if indices:
            self.set_labels({index: AnnotationLabel.OOS.value for index in indices})
        return indices

    def retrain(self) -> None:
        """Entraîne le CRF sur les segments contigus validés par un humain."""
        self.known_classes = {label for label in self.labels if label is not None}
        self.tagger = train_tagger(
            self.features, self.labels, self.model_path, self.training_config
        )

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
        return normalized_marginals(self.tagger, idx, self.known_classes, CLASSES)

    def _model_prediction(self, index: int) -> tuple[str | None, float]:
        """Décodage postérieur pour une ligne : renvoie l'étiquette de plus
        forte probabilité marginale ainsi que cette probabilité elle-même.

        On utilise volontairement l'argmax des marginales plutôt que
        `tagger.tag()` (décodage de Viterbi). Le Viterbi optimise la
        séquence globale et peut donc choisir, à une position donnée, une
        étiquette différente de celle qui a la plus forte probabilité
        marginale à cette même position : le libellé exporté et la
        confiance affichée pouvaient alors ne pas correspondre (une
        confiance basse pour une étiquette qui n'était pas la plus
        probable). Avec l'argmax des marginales, la probabilité exportée
        est toujours exactement celle de l'étiquette exportée. Cela évite
        aussi tout appel à `tagger.tag()` après `tagger.set()`, qui corrompt
        les marginales dans python-crfsuite (cf. `train_tagger`).
        """
        if not self.tagger:
            return None, 0.0
        return posterior_decode(self.get_marginals(index))

    def _line_annotation(
        self,
        index: int,
        model_label: str | None,
        model_probability: float,
        export_timestamp: str,
    ) -> LineAnnotation:
        human_label = self.labels[index]
        if human_label:
            return LineAnnotation(
                human_label, "human", 1.0, self.label_timestamps[index]
            )
        if model_label:
            return LineAnnotation(
                model_label, "model", model_probability, export_timestamp
            )
        return LineAnnotation("", "unclassified", 0.0, "")

    def export_json(self, output_path: str | Path) -> None:
        export_timestamp = current_timestamp()

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
            if self.labels[index] is not None:
                model_label, model_probability = None, 0.0
            else:
                model_label, model_probability = self._model_prediction(index)
            annotation = self._line_annotation(
                index, model_label, model_probability, export_timestamp
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
