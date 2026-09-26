"""Évaluation du CRF sur les silver datasets : protocoles de validation
croisée, exécution parallèle des entraînements, métriques et intervalles de
confiance.

Choix méthodologiques
---------------------
- **Unité statistique = la page.** Les lignes d'une même page ne sont pas
  indépendantes (même mise en page, même rubrique, contexte CRF partagé) :
  les plis de validation croisée sont des blocs de pages contiguës et les
  intervalles de confiance sont obtenus par bootstrap *par page* (cluster
  bootstrap), pas par ligne — un bootstrap par ligne serait trop optimiste.
- **Comparaisons appariées.** Deux configurations sont comparées sur
  exactement les mêmes plis et, pour l'intervalle de confiance, sur les
  mêmes rééchantillonnages de pages (bootstrap apparié de la différence).
- **Deux protocoles complémentaires** :
  * `intra-document` : K plis de pages contiguës par document, modèle
    entraîné sur les autres plis du même document (régime de production :
    un modèle par document) ;
  * `inter-volumes` : chaque volume est prédit par un modèle entraîné sur
    les autres volumes (généralisation à un annuaire inédit).
- **Décodage postérieur** (argmax des marginales), comme en production.
"""

import hashlib
import math
import os
import pickle
import tempfile
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from lib.crf.features import FeatureSequence, extract_features_from_context, get_heuristic_label
from lib.crf.labels import CLASSES, AnnotationLabel
from lib.crf.model import (
    DEFAULT_CONFIG,
    TrainingConfig,
    iter_labeled_segments,
    open_tagger,
    sequence_marginals,
    train_model,
)
from lib.crf.silver import SilverDocument

LABEL_INDEX = {label: index for index, label in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)
LineRange = tuple[int, int, int]  # (document, début, fin exclue)


# ----------------------------------------------------------------------
# Expériences et plis
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Experiment:
    """Une configuration de features + hyperparamètres à évaluer."""

    name: str
    groups: tuple[str, ...]
    config: TrainingConfig = DEFAULT_CONFIG
    description: str = ""


@dataclass(frozen=True)
class Split:
    """Un pli : plages de lignes d'entraînement et de test."""

    protocol: str
    name: str
    train: tuple[LineRange, ...]
    test: tuple[LineRange, ...]


def page_boundaries(document: SilverDocument) -> list[int]:
    """Indices de début de chaque page (plus la fin du document)."""
    starts = [
        index
        for index, record in enumerate(document.records)
        if index == 0 or record.page_pos != document.records[index - 1].page_pos
    ]
    return starts + [len(document)]


def within_document_splits(
    documents: Sequence[SilverDocument], folds: int = 5
) -> list[Split]:
    """K plis de pages contiguës par document."""
    splits: list[Split] = []
    for doc_index, document in enumerate(documents):
        bounds = page_boundaries(document)
        n_pages = len(bounds) - 1
        k = min(folds, n_pages)
        cuts = [bounds[round(i * n_pages / k)] for i in range(k + 1)]
        for fold in range(k):
            start, end = cuts[fold], cuts[fold + 1]
            train = tuple(
                (doc_index, a, b) for a, b in ((0, start), (end, len(document))) if b > a
            )
            splits.append(
                Split("intra-document", f"{document.name}/pli{fold + 1}", train, ((doc_index, start, end),))
            )
    return splits


def cross_volume_splits(documents: Sequence[SilverDocument]) -> list[Split]:
    """Un pli par volume : entraîné sur tous les autres volumes."""
    volumes = sorted({document.volume for document in documents})
    splits: list[Split] = []
    if len(volumes) < 2:
        return splits
    for volume in volumes:
        train = tuple(
            (i, 0, len(d)) for i, d in enumerate(documents) if d.volume != volume
        )
        test = tuple((i, 0, len(d)) for i, d in enumerate(documents) if d.volume == volume)
        splits.append(Split("inter-volumes", volume, train, test))
    return splits


def subsample_training_pages(
    split: Split, documents: Sequence[SilverDocument], n_pages: int, seed: int
) -> Split:
    """Réduit l'entraînement d'un pli à `n_pages` pages tirées au hasard
    (chaque page devient une séquence d'entraînement indépendante)."""
    pages: list[LineRange] = []
    for doc_index, start, end in split.train:
        gold = documents[doc_index].gold
        bounds = page_boundaries(documents[doc_index])
        pages.extend(
            (doc_index, a, b)
            for a, b in zip(bounds, bounds[1:])
            if a >= start and b <= end and any(label is not None for label in gold[a:b])
        )
    rng = np.random.default_rng(seed)
    if n_pages < len(pages):
        chosen = sorted(rng.choice(len(pages), size=n_pages, replace=False))
        pages = [pages[i] for i in chosen]
    return Split(split.protocol, f"{split.name}/{n_pages}p/s{seed}", tuple(pages), split.test)


# ----------------------------------------------------------------------
# Exécution (parallèle) des entraînements
# ----------------------------------------------------------------------
_WORKER_DOCUMENTS: Sequence[SilverDocument] = ()


@lru_cache(maxsize=6)
def _document_features(doc_index: int, groups: tuple[str, ...]) -> FeatureSequence:
    return extract_features_from_context(_WORKER_DOCUMENTS[doc_index].context, groups)


@dataclass(frozen=True)
class _Task:
    experiment: Experiment
    split: Split


def _run_task(task: _Task) -> list[tuple[int, int, np.ndarray]]:
    """Entraîne sur `split.train`, renvoie les marginales sur `split.test`."""
    groups = task.experiment.groups
    sequences: list[tuple[FeatureSequence, list[str]]] = []
    for doc_index, start, end in task.split.train:
        features = _document_features(doc_index, groups)[start:end]
        gold = _WORKER_DOCUMENTS[doc_index].gold[start:end]
        sequences.extend(iter_labeled_segments(features, gold))
    if not sequences:
        raise ValueError(f"Pli {task.split.name} sans données d'entraînement.")
    with tempfile.TemporaryDirectory() as tmp:
        model_path = Path(tmp) / "model.crfsuite"
        train_model(sequences, model_path, task.experiment.config)
        tagger = open_tagger(model_path)
        results = []
        for doc_index, start, end in task.split.test:
            features = _document_features(doc_index, groups)[start:end]
            marginals = sequence_marginals(tagger, features, CLASSES)
            matrix = np.array(
                [[line[label] for label in CLASSES] for line in marginals], dtype=np.float32
            )
            results.append((doc_index, start, matrix))
        tagger.close()
    return results


@dataclass
class Predictions:
    """Marginales prédites (NaN si non prédite) pour chaque ligne du corpus."""

    experiment: Experiment
    protocol: str
    marginals: list[np.ndarray]

    @classmethod
    def empty(cls, experiment: Experiment, protocol: str, documents: Sequence[SilverDocument]):
        return cls(
            experiment,
            protocol,
            [np.full((len(d), N_CLASSES), np.nan, dtype=np.float32) for d in documents],
        )


def corpus_fingerprint(documents: Sequence[SilverDocument]) -> str:
    """Empreinte des observations et de la vérité du corpus (clé de cache)."""
    digest = hashlib.sha256()
    for document in documents:
        digest.update(document.name.encode())
        for record, label in zip(document.records, document.gold):
            digest.update(f"{record.uid}\t{record.text}\t{record.data_block_label}\t{label}\n".encode())
    return digest.hexdigest()[:16]


def _task_key(task: _Task, fingerprint: str) -> str:
    """Clé de cache : corpus, groupes, hyperparamètres, plis, et code des
    features (une modification de features.py invalide le cache)."""
    features_source = (Path(__file__).parent / "features.py").read_bytes()
    payload = repr((fingerprint, task.experiment.groups, task.experiment.config, task.split.train, task.split.test))
    return hashlib.sha256(payload.encode() + features_source).hexdigest()[:24]


def run_experiments(
    documents: Sequence[SilverDocument],
    experiments: Sequence[Experiment],
    splits: Sequence[Split],
    workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    cache_dir: Path | None = None,
) -> dict[tuple[str, str], Predictions]:
    """Évalue chaque expérience sur chaque pli ; résultats indexés par
    (nom d'expérience, nom de protocole).

    Avec `cache_dir`, le résultat de chaque (expérience, pli) est conservé
    sur disque : une relance ne réentraîne que ce qui manque.
    """
    global _WORKER_DOCUMENTS
    _WORKER_DOCUMENTS = documents
    _document_features.cache_clear()
    tasks = [_Task(experiment, split) for experiment in experiments for split in splits]
    fingerprint = corpus_fingerprint(documents) if cache_dir else ""
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[tuple[str, str], Predictions] = {}
    for task in tasks:
        key = (task.experiment.name, task.split.protocol)
        if key not in outputs:
            outputs[key] = Predictions.empty(task.experiment, task.split.protocol, documents)

    def store(task: _Task, results: list[tuple[int, int, np.ndarray]]) -> None:
        predictions = outputs[(task.experiment.name, task.split.protocol)]
        for doc_index, start, matrix in results:
            predictions.marginals[doc_index][start : start + len(matrix)] = matrix

    done = 0
    pending: list[tuple[_Task, Path | None]] = []
    for task in tasks:
        path = cache_dir / f"{_task_key(task, fingerprint)}.pkl" if cache_dir else None
        if path is not None and path.exists():
            store(task, pickle.loads(path.read_bytes()))
            done += 1
        else:
            pending.append((task, path))
    if progress:
        progress(done, len(tasks))

    workers = workers or min(8, os.cpu_count() or 1)
    pending_tasks = [task for task, _ in pending]
    if workers <= 1:
        results_iter = map(_run_task, pending_tasks)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=get_context("fork"))
        results_iter = pool.map(_run_task, pending_tasks, chunksize=1)
    try:
        for (task, path), results in zip(pending, results_iter):
            store(task, results)
            if path is not None:
                path.write_bytes(pickle.dumps(results))
            done += 1
            if progress:
                progress(done, len(tasks))
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
    return outputs


def rule_predictions(
    name: str,
    documents: Sequence[SilverDocument],
    rule: Callable[[SilverDocument], list[str]],
    protocol: str,
) -> Predictions:
    """Prédictions « one-hot » d'une règle (baselines sans apprentissage)."""
    experiment = Experiment(name, ())
    predictions = Predictions.empty(experiment, protocol, documents)
    for doc_index, document in enumerate(documents):
        for index, label in enumerate(rule(document)):
            row = np.zeros(N_CLASSES, dtype=np.float32)
            row[LABEL_INDEX[label]] = 1.0
            predictions.marginals[doc_index][index] = row
    return predictions


def heuristic_rule(document: SilverDocument) -> list[str]:
    lines = [record.text for record in document.records]
    return [get_heuristic_label(line, lines[i - 1] if i else "") for i, line in enumerate(lines)]


def majority_rule(document: SilverDocument) -> list[str]:
    return [AnnotationLabel.BENTRY.value] * len(document)


# ----------------------------------------------------------------------
# Lignes évaluées
# ----------------------------------------------------------------------
@dataclass
class ScoredLines:
    """Lignes à la fois labellisées et prédites, à plat sur tout le corpus."""

    doc_index: np.ndarray
    line_index: np.ndarray
    gold: np.ndarray
    pred: np.ndarray
    confidence: np.ndarray
    margin: np.ndarray
    page_code: np.ndarray
    page_names: list[str]
    marginals: np.ndarray = field(repr=False)

    @property
    def correct(self) -> np.ndarray:
        return self.gold == self.pred

    def __len__(self) -> int:
        return len(self.gold)


def score_lines(predictions: Predictions, documents: Sequence[SilverDocument]) -> ScoredLines:
    doc_idx, line_idx, gold, pages, rows = [], [], [], [], []
    for d, document in enumerate(documents):
        matrix = predictions.marginals[d]
        page_ids = document.page_ids
        for i, label in enumerate(document.gold):
            if label is None or np.isnan(matrix[i, 0]):
                continue
            doc_idx.append(d)
            line_idx.append(i)
            gold.append(LABEL_INDEX[label])
            pages.append(page_ids[i])
            rows.append(matrix[i])
    marginals = np.array(rows, dtype=np.float64).reshape(-1, N_CLASSES)
    ordered = np.sort(marginals, axis=1)
    page_names, page_code = np.unique(np.array(pages, dtype=object), return_inverse=True)
    return ScoredLines(
        doc_index=np.array(doc_idx, dtype=int),
        line_index=np.array(line_idx, dtype=int),
        gold=np.array(gold, dtype=int),
        pred=marginals.argmax(axis=1) if len(marginals) else np.array([], dtype=int),
        confidence=ordered[:, -1] if len(marginals) else np.array([]),
        margin=(ordered[:, -1] - ordered[:, -2]) if len(marginals) else np.array([]),
        page_code=page_code.astype(int),
        page_names=list(page_names),
        marginals=marginals,
    )


# ----------------------------------------------------------------------
# Métriques par ligne (vectorisées, pour le bootstrap)
# ----------------------------------------------------------------------
def page_confusions(scored: ScoredLines) -> np.ndarray:
    """Tenseur (pages, K, K) des matrices de confusion (lignes = vérité)."""
    n_pages = len(scored.page_names)
    flat = (scored.page_code * N_CLASSES + scored.gold) * N_CLASSES + scored.pred
    counts = np.bincount(flat, minlength=n_pages * N_CLASSES * N_CLASSES)
    return counts.reshape(n_pages, N_CLASSES, N_CLASSES).astype(np.float64)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, numerator / np.where(denominator > 0, denominator, 1), 0.0)


def prf(confusion: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(précision, rappel, F1, support) par classe ; accepte (..., K, K)."""
    tp = np.diagonal(confusion, axis1=-2, axis2=-1)
    predicted = confusion.sum(axis=-2)
    support = confusion.sum(axis=-1)
    precision = _safe_divide(tp, predicted)
    recall = _safe_divide(tp, support)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1, support


def macro_f1(confusion: np.ndarray, classes_mask: np.ndarray) -> np.ndarray:
    return prf(confusion)[2][..., classes_mask].mean(axis=-1)


def accuracy(confusion: np.ndarray) -> np.ndarray:
    total = confusion.sum(axis=(-2, -1))
    return _safe_divide(np.diagonal(confusion, axis1=-2, axis2=-1).sum(axis=-1), total)


# ----------------------------------------------------------------------
# Métriques par entité (règles de merge_annotated_lines.py)
# ----------------------------------------------------------------------
ENTITY_TYPES = ("ENTRY", "TITLE")
_ROOT = {AnnotationLabel.BENTRY.value: "ENTRY", AnnotationLabel.BTITLE.value: "TITLE"}
_CONTINUATION = {
    AnnotationLabel.IENTRY.value: "ENTRY",
    AnnotationLabel.SUBENTRY.value: "ENTRY",
    AnnotationLabel.ITITLE.value: "TITLE",
}


def group_entities(labels: Sequence[str | None]) -> list[tuple[str, tuple[int, ...]]]:
    """Reconstitue les entités comme merge_annotated_lines.py : deux pistes
    indépendantes (ENTRY, TITLE) ouvertes jusqu'à la racine suivante de leur
    type ; une continuation orpheline devient une nouvelle racine ; les
    autres classes (et les lignes None) sont ignorées."""
    entities: list[tuple[str, list[int]]] = []
    open_tracks: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        if label in _ROOT:
            kind = _ROOT[label]
            open_tracks[kind] = [index]
            entities.append((kind, open_tracks[kind]))
        elif label in _CONTINUATION:
            kind = _CONTINUATION[label]
            if kind in open_tracks:
                open_tracks[kind].append(index)
            else:
                open_tracks[kind] = [index]
                entities.append((kind, open_tracks[kind]))
    return [(kind, tuple(indices)) for kind, indices in entities]


def page_entity_counts(scored: ScoredLines, documents: Sequence[SilverDocument]) -> np.ndarray:
    """Tenseur (pages, types, 3) : [vrais positifs, prédites, attendues],
    chaque entité étant rattachée à la page de sa première ligne.

    Les entités sont reconstituées sur les seules lignes évaluées (labellisées
    et prédites), identiquement côté vérité et côté prédiction.
    """
    counts = np.zeros((len(scored.page_names), len(ENTITY_TYPES), 3))
    for d in range(len(documents)):
        mask = scored.doc_index == d
        if not mask.any():
            continue
        pages = scored.page_code[mask]
        gold = [CLASSES[i] for i in scored.gold[mask]]
        pred = [CLASSES[i] for i in scored.pred[mask]]
        gold_entities = set(group_entities(gold))
        pred_entities = set(group_entities(pred))
        for entities, column in ((gold_entities, 2), (pred_entities, 1)):
            for kind, indices in entities:
                counts[pages[indices[0]], ENTITY_TYPES.index(kind), column] += 1
        for kind, indices in gold_entities & pred_entities:
            counts[pages[indices[0]], ENTITY_TYPES.index(kind), 0] += 1
    return counts


def entity_f1(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(précision, rappel, F1) par type d'entité ; accepte (..., types, 3)."""
    precision = _safe_divide(counts[..., 0], counts[..., 1])
    recall = _safe_divide(counts[..., 0], counts[..., 2])
    return precision, recall, _safe_divide(2 * precision * recall, precision + recall)


# ----------------------------------------------------------------------
# Bootstrap par page
# ----------------------------------------------------------------------
def bootstrap_weights(n_pages: int, n_samples: int, seed: int = 0) -> np.ndarray:
    """Poids multinomiaux (échantillons, pages) : rééchantillonnage des pages."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_pages, np.full(n_pages, 1 / n_pages), size=n_samples).astype(np.float64)


def resample(page_tensor: np.ndarray, weights: np.ndarray) -> np.ndarray:
    shape = page_tensor.shape[1:]
    flat = page_tensor.reshape(page_tensor.shape[0], -1)
    return (weights @ flat).reshape((weights.shape[0], *shape))


def interval(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    alpha = (1 - level) / 2
    low, high = np.quantile(samples, [alpha, 1 - alpha])
    return float(low), float(high)


# ----------------------------------------------------------------------
# Calibration et détection d'erreurs
# ----------------------------------------------------------------------
def calibration_table(confidence: np.ndarray, correct: np.ndarray, bins: int = 10):
    """Lignes (borne basse, borne haute, n, confiance moyenne, exactitude)."""
    edges = np.linspace(0, 1, bins + 1)
    indices = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = indices == b
        if mask.any():
            rows.append((edges[b], edges[b + 1], int(mask.sum()), float(confidence[mask].mean()), float(correct[mask].mean())))
    return rows


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    total = len(confidence)
    if not total:
        return math.nan
    return sum(n / total * abs(conf - acc) for _, _, n, conf, acc in calibration_table(confidence, correct, bins))


def roc_auc(scores: np.ndarray, positives: np.ndarray) -> float:
    """AUROC (Mann-Whitney) : probabilité qu'une erreur ait un score plus
    élevé qu'une ligne correcte."""
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if not n_pos or not n_neg:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    # Rangs moyens pour les ex aequo.
    _, first, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    for start, count in zip(first, counts):
        ranks[order[start : start + count]] = start + (count + 1) / 2
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def review_capture(uncertainty: np.ndarray, errors: np.ndarray, budgets: Sequence[float]) -> list[tuple[float, int, float]]:
    """Part des erreurs trouvées en relisant les `budget` % de lignes les plus incertaines."""
    order = np.argsort(-uncertainty, kind="mergesort")
    total_errors = int(errors.sum())
    rows = []
    for budget in budgets:
        n = int(math.ceil(budget * len(uncertainty)))
        found = int(errors[order[:n]].sum())
        rows.append((budget, n, found / total_errors if total_errors else math.nan))
    return rows


# ----------------------------------------------------------------------
# Statistiques de features sans modèle (information mutuelle)
# ----------------------------------------------------------------------
def entropy_bits(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def mutual_information_bits(x_codes: np.ndarray, y_codes: np.ndarray) -> float:
    """I(X;Y) en bits, estimateur plug-in."""
    nx, ny = x_codes.max() + 1, y_codes.max() + 1
    joint = np.bincount(x_codes * ny + y_codes, minlength=nx * ny).reshape(nx, ny).astype(float)
    return entropy_bits(joint.sum(1)) + entropy_bits(joint.sum(0)) - entropy_bits(joint.ravel())


def miller_madow_bias_bits(x_codes: np.ndarray, y_codes: np.ndarray) -> float:
    """Biais positif attendu de l'estimateur plug-in de I(X;Y) sous
    indépendance, (|X|-1)(|Y|-1) / (2 N ln 2) : à retrancher avant de
    comparer des features de cardinalités différentes."""
    kx, ky = len(np.unique(x_codes)), len(np.unique(y_codes))
    return (kx - 1) * (ky - 1) / (2 * len(x_codes) * math.log(2))


def factorize(values: Sequence[Any]) -> np.ndarray:
    _, codes = np.unique(np.array(values, dtype=object).astype(str), return_inverse=True)
    return codes.astype(int)
