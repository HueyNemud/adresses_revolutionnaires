"""Entraînement et inférence du CRF (python-crfsuite)."""

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pycrfsuite

from numrev.crf.features import FeatureSequence


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparamètres de l'entraînement L-BFGS (valeurs de production par défaut)."""

    c1: float = 0.1
    c2: float = 0.01
    max_iterations: int = 50
    possible_transitions: bool = True

    def as_params(self) -> dict[str, object]:
        return {
            "c1": self.c1,
            "c2": self.c2,
            "max_iterations": self.max_iterations,
            "feature.possible_transitions": self.possible_transitions,
        }


DEFAULT_CONFIG = TrainingConfig()


def iter_labeled_segments(
    features: FeatureSequence, labels: Sequence[str | None]
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


def train_model(
    sequences: Iterable[tuple[FeatureSequence, Sequence[str]]],
    model_path: Path,
    config: TrainingConfig = DEFAULT_CONFIG,
) -> None:
    """Entraîne un CRF sur des séquences entièrement labellisées."""
    # python-crfsuite ne publie pas de stubs complets pour Pylance.
    trainer: Any = getattr(pycrfsuite, "Trainer")(verbose=False)
    trainer.set_params(config.as_params())
    for features, labels in sequences:
        trainer.append(features, list(labels))
    trainer.train(str(model_path))


def open_tagger(model_path: Path) -> Any:
    tagger: Any = getattr(pycrfsuite, "Tagger")()
    tagger.open(str(model_path))
    return tagger


def train_tagger(
    features: FeatureSequence,
    labels: Sequence[str | None],
    model_path: Path,
    config: TrainingConfig = DEFAULT_CONFIG,
) -> Any | None:
    """Entraîne un tagger CRF sur les segments annotés, ou None si trop peu de données.

    Le tagger renvoyé est déjà positionné (`set`) sur toute la séquence
    `features`, prêt pour `tagger.marginal(label, index)`.
    """
    if len({label for label in labels if label is not None}) < 2:
        return None
    train_model(iter_labeled_segments(features, labels), model_path, config)
    tagger = open_tagger(model_path)
    # `tag()` doit recevoir la séquence. Appeler `tag()` sans argument après
    # `set()` corrompt les marginales dans python-crfsuite.
    tagger.set(features)
    return tagger


def normalized_marginals(
    tagger: Any, index: int, known_classes: Iterable[str], all_classes: Sequence[str]
) -> dict[str, float]:
    """Marginales de `known_classes` renormalisées, étendues à `all_classes`.

    Suppose que `tagger.set(...)` a déjà été appelé sur la séquence voulue.
    """
    scores = {label: max(0.0, float(tagger.marginal(label, index))) for label in known_classes}
    total = sum(scores.values())
    if total <= 0.0:
        return {label: 0.0 for label in all_classes}
    return {label: scores.get(label, 0.0) / total for label in all_classes}


def sequence_marginals(
    tagger: Any, features: FeatureSequence, all_classes: Sequence[str]
) -> list[dict[str, float]]:
    """Marginales normalisées de chaque position d'une séquence."""
    tagger.set(features)
    known = tagger.labels()
    return [
        normalized_marginals(tagger, index, known, all_classes)
        for index in range(len(features))
    ]


def posterior_decode(marginals: dict[str, float]) -> tuple[str | None, float]:
    """Argmax des marginales (et non Viterbi) : la probabilité renvoyée est
    toujours exactement celle de l'étiquette renvoyée."""
    label, probability = max(marginals.items(), key=lambda item: item[1])
    if probability <= 0.0:
        return None, 0.0
    return label, probability
