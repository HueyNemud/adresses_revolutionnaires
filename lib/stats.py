"""Outils statistiques communs aux audits CRF et NER : bootstrap par
grappe (page), intervalles, calibration, qualité d'un score comme outil de
tri des erreurs."""

import math
from collections.abc import Sequence

import numpy as np


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
