"""Évaluation d'une segmentation NER contre le gold.

Métrique principale : **exactitude par entrée**, la part des entrées dont la
segmentation prédite ne demande aucune correction (même forme canonique que
le gold, voir `canonical_spans` : la ponctuation de liaison en bord d'empan
ne compte pas). C'est le proxy le plus direct de l'effort de reprise.

Métriques secondaires :
- P/R/F1 par classe sur les empans canoniques exacts (bornes + classe) ;
- exactitude par token (découpage par espaces, tokens de pure ponctuation
  exclus) : distingue une frontière décalée d'un mot (peu d'impact) d'une
  classe fausse sur tout un segment ;
- type d'erreur : `signature` (suite des classes différente : un segment en
  trop, manquant ou mal classé) ou `frontière` (mêmes classes dans le même
  ordre, bornes différentes).

Agrégation : chaque entrée porte un poids (inverse de sa probabilité
d'inclusion dans l'échantillon gold stratifié) et une grappe (la page).
Les intervalles de confiance sont obtenus par bootstrap des pages, comme
pour l'audit du CRF (`numrev.stats`), et les comparaisons entre
systèmes sont appariées (mêmes rééchantillonnages).
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from numrev.ner.spans import LABELS, Span, canonical_spans, signature, tokenize_with_offsets
from numrev.stats import bootstrap_weights, interval, resample

ALNUM = re.compile(r"\w")

# Colonnes du vecteur par entrée : exact, (tp, fp, fn) × classe, tokens corrects, tokens.
VECTOR_COLUMNS = ("exact", *(f"{kind}_{label}" for label in LABELS for kind in ("tp", "fp", "fn")), "tok_ok", "tok_n")
METRICS = ("exactitude", "F1 SUBJ", "F1 DESC", "F1 ADDR", "F1 micro", "exactitude tokens")


@dataclass(frozen=True)
class Comparison:
    exact: bool
    error_kind: str  # "ok", "signature" ou "frontière"
    vector: np.ndarray  # colonnes VECTOR_COLUMNS


def token_labels(text: str, spans: Sequence[Span]) -> list[str | None]:
    """Classe de chaque token (None : pure ponctuation, non évalué). Un
    token prend la classe de l'empan qui recouvre son premier caractère
    alphanumérique."""
    labels: list[str | None] = []
    for token in tokenize_with_offsets(text):
        match = ALNUM.search(token.text)
        if match is None:
            labels.append(None)
            continue
        position = token.start_char + match.start()
        label = next((span.label for span in spans if span.start <= position < span.end), "O")
        labels.append(label)
    return labels


def compare(text: str, gold: Sequence[Span], predicted: Sequence[Span]) -> Comparison:
    gold_canonical = set(canonical_spans(text, gold))
    predicted_canonical = set(canonical_spans(text, predicted))
    exact = gold_canonical == predicted_canonical
    if exact:
        kind = "ok"
    elif signature(gold) != signature(predicted):
        kind = "signature"
    else:
        kind = "frontière"

    vector = np.zeros(len(VECTOR_COLUMNS))
    vector[0] = exact
    for index, label in enumerate(LABELS):
        gold_label = {span for span in gold_canonical if span[0] == label}
        predicted_label = {span for span in predicted_canonical if span[0] == label}
        vector[1 + 3 * index] = len(gold_label & predicted_label)
        vector[2 + 3 * index] = len(predicted_label - gold_label)
        vector[3 + 3 * index] = len(gold_label - predicted_label)
    pairs = [(g, p) for g, p in zip(token_labels(text, gold), token_labels(text, predicted)) if g is not None]
    vector[-2] = sum(g == p for g, p in pairs)
    vector[-1] = len(pairs)
    return Comparison(exact, kind, vector)


def _divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, numerator / np.where(denominator > 0, denominator, 1), np.nan)


def _f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    return _divide(2 * tp, 2 * tp + fp + fn)


def metrics_from_sums(sums: np.ndarray, total_weight: np.ndarray) -> np.ndarray:
    """(…, VECTOR_COLUMNS) sommes pondérées → (…, METRICS)."""
    columns = {name: sums[..., i] for i, name in enumerate(VECTOR_COLUMNS)}
    per_label = [_f1(columns[f"tp_{l}"], columns[f"fp_{l}"], columns[f"fn_{l}"]) for l in LABELS]
    micro = _f1(*(sum(columns[f"{kind}_{l}"] for l in LABELS) for kind in ("tp", "fp", "fn")))
    return np.stack(
        [_divide(columns["exact"], total_weight), *per_label, micro, _divide(columns["tok_ok"], columns["tok_n"])],
        axis=-1,
    )


@dataclass
class Scored:
    """Comparaisons d'un système sur les entrées gold (mêmes entrées, même
    ordre pour tous les systèmes comparés)."""

    vectors: np.ndarray  # (entrées, VECTOR_COLUMNS)
    weights: np.ndarray  # (entrées,)
    clusters: np.ndarray  # (entrées,) indices de page 0..n-1

    def cluster_tensor(self) -> tuple[np.ndarray, np.ndarray]:
        n = int(self.clusters.max()) + 1
        sums = np.zeros((n, self.vectors.shape[1]))
        np.add.at(sums, self.clusters, self.vectors * self.weights[:, None])
        weight = np.bincount(self.clusters, weights=self.weights, minlength=n)
        return sums, weight

    def point(self) -> np.ndarray:
        sums, weight = self.cluster_tensor()
        return metrics_from_sums(sums.sum(0), weight.sum())

    def samples(self, bootstrap: np.ndarray) -> np.ndarray:
        sums, weight = self.cluster_tensor()
        return metrics_from_sums(resample(sums, bootstrap), bootstrap @ weight)


def page_bootstrap(n_clusters: int, n_samples: int, seed: int = 0) -> np.ndarray:
    return bootstrap_weights(n_clusters, n_samples, seed)


def summarize(scored: Scored, bootstrap: np.ndarray) -> list[tuple[float, tuple[float, float]]]:
    """[(valeur, IC 95 %)] pour chaque métrique de METRICS."""
    point, samples = scored.point(), scored.samples(bootstrap)
    return [(float(point[m]), _nan_interval(samples[:, m])) for m in range(len(METRICS))]


def paired_delta(
    reference: Scored, candidate: Scored, bootstrap: np.ndarray
) -> list[tuple[float, tuple[float, float]]]:
    """Δ (candidat − référence) et IC 95 % sur les mêmes rééchantillonnages."""
    point = candidate.point() - reference.point()
    samples = candidate.samples(bootstrap) - reference.samples(bootstrap)
    return [(float(point[m]), _nan_interval(samples[:, m])) for m in range(len(METRICS))]


def _nan_interval(values: np.ndarray) -> tuple[float, float]:
    values = values[~np.isnan(values)]
    return interval(values) if len(values) else (float("nan"), float("nan"))
