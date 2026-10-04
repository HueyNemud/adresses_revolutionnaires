"""Paires à vérifier après l'alignement automatique : motifs et niveau
d'incertitude (`numrev view alignment`, `numrev join`,
`numrev audit alignment`).

L'alignement ne change pas : on signale seulement, avec un motif explicite,
les décisions qu'une relecture humaine peut corriger. Sur le gold des
inversions 1807/1808, les cas douteux demandent un savoir que les données
n'ont pas (homonymes, père / fils, coquilles des éditeurs, rues renommées) :
mieux vaut les montrer que les trancher automatiquement. Motifs, calculés
dans chaque segment (`numrev.alignment.sections.segments`, même similarité que
`numrev align nw`) :

- `déduite des voisines (p < 0,9)` : paire retenue par le pair-HMM
  (`nw-contexte`) parce que ses voisines sont appariées, mais de
  probabilité a posteriori < `CONTEXT_REVIEW` ;
- `homonyme proche` : paire décidée sur sa seule similarité (`nw`,
  `nw-residuel`, `dedupe`) alors qu'une autre entrée du segment, de l'un ou
  l'autre côté, est presque aussi proche (écart < `margin`) ; ces
  concurrentes sont gardées (`Review.rivals`) pour que le relecteur les voie ;
- `candidate non appariée` : deux entrées restées sans correspondance,
  chacune la plus proche de l'autre dans le segment, de similarité dans la
  zone grise [`low` ; `high`[ — par défaut entre le seuil de
  Needleman-Wunsch et celui de la passe résiduelle
  d'`numrev align nw`. Ce n'est pas un lien : seulement une paire
  soumise au relecteur.

Les paires du patch (relues) n'ont pas de motif. Niveau d'incertitude
(ordinal, pour trier la relecture ; ce n'est pas une probabilité) :
`faible` (0) sans motif ; `moyenne` (1) un motif ; `forte` (2) une
candidate non appariée, plusieurs motifs ou une paire déduite des voisines
de probabilité < `CONTEXT_DOUBT`. Aucun paramètre n'est appris sur un volume :
`numrev audit alignment` vérifie, sur le gold d'une nouvelle paire
d'annuaires, que les motifs attrapent les erreurs et que le niveau reste
ordonné.
"""

import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from numrev.alignment.records import (
    SOURCE_CANDIDATE,
    SOURCE_DEDUPE,
    SOURCE_NW,
    SOURCE_NW_CONTEXT,
    SOURCE_NW_RESIDUAL,
    Link,
    Record,
    similarity_matrix,
)
from numrev.alignment.sections import SectionAlignment, segment_entries

REASON_CONTEXT = "déduite des voisines (p < 0,9)"
REASON_HOMONYM = "homonyme proche"
REASON_CANDIDATE = "candidate non appariée"
REASONS = (REASON_CONTEXT, REASON_HOMONYM, REASON_CANDIDATE)
SEPARATOR = " | "
CONTEXT_REVIEW = 0.9  # probabilité a posteriori sous laquelle une paire du pair-HMM est à vérifier (REASON_CONTEXT)
CONTEXT_DOUBT = 0.7  # … et sous laquelle son incertitude est forte (niveau 2)
LEVEL_LABELS = {0: "faible", 1: "moyenne", 2: "forte"}  # niveau d'incertitude
DEFAULT_MARGIN = 0.05  # écart de similarité avec la concurrente (observé sur 1807/1808)
SIMILARITY_SOURCES = {SOURCE_NW, SOURCE_NW_RESIDUAL, SOURCE_DEDUPE}  # paires décidées sur leur similarité


@dataclass(frozen=True)
class Review:
    reasons: tuple[str, ...]
    level: int  # clé de LEVEL_LABELS : 0 faible, 1 moyenne, 2 forte
    rivals: tuple[tuple[str, str, float], ...] = ()  # homonyme proche : (côté, uuid, similarité) des concurrentes


@dataclass
class ReviewResult:
    reviews: dict[tuple[str, str], Review]  # (uuid gauche, uuid droit) → motifs, pour les paires qui en ont
    candidates: list[Link]  # source SOURCE_CANDIDATE, score = similarité


def level(reasons: tuple[str, ...], link: Link) -> int:
    if not reasons:
        return 0
    doubtful_context = REASON_CONTEXT in reasons and link.score is not None and link.score < CONTEXT_DOUBT
    return 2 if REASON_CANDIDATE in reasons or len(reasons) > 1 or doubtful_context else 1


def runner_up(similarity: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """(meilleure, deuxième meilleure) similarité de chaque ligne (`axis` 1)
    ou colonne (`axis` 0) ; 0 en l'absence de deuxième."""
    if similarity.shape[axis] < 2:
        return similarity.max(axis=axis), np.zeros(similarity.shape[1 - axis], dtype=similarity.dtype)
    top = np.partition(similarity, -2, axis=axis)
    top = top[:, -2:] if axis == 1 else top[-2:, :].T
    return top[:, 1], top[:, 0]


def review_segment(
    left_records: list[Record],
    right_records: list[Record],
    similarity: np.ndarray,
    links: list[Link | None],
    busy_left: list[bool],
    busy_right: list[bool],
    low: float,
    high: float,
    margin: float,
) -> tuple[dict[tuple[str, str], list[str]], dict[tuple[str, str], list[tuple[str, str, float]]], list[Link]]:
    """Motifs, concurrentes et candidates d'un segment. `links[i]` : lien de
    l'entrée de gauche `i` (None sans lien), `busy_*` : entrées liées ou
    déclarées sans correspondance (jamais candidates)."""
    reasons: dict[tuple[str, str], list[str]] = {}
    rivals: dict[tuple[str, str], list[tuple[str, str, float]]] = {}
    candidates = []
    right_index = {record.uuid: j for j, record in enumerate(right_records)}
    row_best, row_second = runner_up(similarity, 1)
    column_best, column_second = runner_up(similarity, 0)

    # Liens dont les deux entrées sont dans le segment
    for i, link in enumerate(links):
        if link is None or link.right_uuid not in right_index:
            continue
        found = []
        if link.source == SOURCE_NW_CONTEXT and link.score is not None and link.score < CONTEXT_REVIEW:
            found.append(REASON_CONTEXT)
        if link.source in SIMILARITY_SOURCES:
            j = right_index[link.right_uuid]
            value = similarity[i, j]
            # Meilleure concurrente : sur la ligne hors j, sur la colonne hors i.
            rival = max(
                row_second[i] if value >= row_best[i] else row_best[i], column_second[j] if value >= column_best[j] else column_best[j]
            )
            if value - rival < margin:
                found.append(REASON_HOMONYM)
                floor = value - margin
                close = [("right", right_records[k].uuid, float(similarity[i, k])) for k in np.flatnonzero(similarity[i] > floor) if k != j]
                close += [
                    ("left", left_records[k].uuid, float(similarity[k, j])) for k in np.flatnonzero(similarity[:, j] > floor) if k != i
                ]
                rivals[link.left_uuid, link.right_uuid] = sorted(close, key=lambda c: -c[2])
        if found:
            reasons[link.left_uuid, link.right_uuid] = found

    # Candidates non appariées : meilleures partenaires mutuelles, toutes deux libres
    best_right, best_left = similarity.argmax(axis=1), similarity.argmax(axis=0)
    for i, record in enumerate(left_records):
        j = int(best_right[i])
        if best_left[j] == i and low <= similarity[i, j] < high and not busy_left[i] and not busy_right[j]:
            partner = right_records[j]
            candidates.append(Link(record.uuid, partner.uuid, float(similarity[i, j]), SOURCE_CANDIDATE))
            reasons[record.uuid, partner.uuid] = [REASON_CANDIDATE]
    return reasons, rivals, candidates


class Reviewer:
    """`review` pour des liens qui changent peu d'un appel à l'autre (le
    viewer, après chaque décision) : le résultat de chaque segment est gardé
    avec ses entrées (liens et entrées occupées du segment) et réutilisé
    tant qu'elles ne changent pas ; les matrices de similarité des derniers
    segments recalculés sont gardées (`MATRICES`). Même résultat que `review`."""

    MATRICES = 8

    def __init__(self, sections: SectionAlignment, low: float, high: float, subj_weight: float, margin: float = DEFAULT_MARGIN):
        self.segments = [(left, right) for left, right in segment_entries(sections) if left and right]
        self.low, self.high, self.subj_weight, self.margin = low, high, subj_weight, margin
        self._results: dict[int, tuple[tuple, tuple]] = {}  # segment → (entrées, résultat)
        self._matrices: OrderedDict[int, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def similarity(self, number: int) -> np.ndarray:
        if number in self._matrices:
            self._matrices.move_to_end(number)
        else:
            self._matrices[number] = similarity_matrix(*self.segments[number], self.subj_weight)
            if len(self._matrices) > self.MATRICES:
                self._matrices.popitem(last=False)
        return self._matrices[number]

    def __call__(self, links: list[Link], declared: set[str] = frozenset()) -> ReviewResult:
        """Motifs des liens `links` (après patch) et candidates non appariées.
        `declared` : uuid déclarés sans correspondance par le patch, jamais
        proposés."""
        by_left = {link.left_uuid: link for link in links}
        busy_left = {link.left_uuid for link in links} | set(declared)
        busy_right = {link.right_uuid for link in links} | set(declared)
        reasons: dict[tuple[str, str], list[str]] = {}
        rivals: dict[tuple[str, str], list[tuple[str, str, float]]] = {}
        candidates = []
        with self._lock:
            for number, (left_records, right_records) in enumerate(self.segments):
                inputs = (
                    tuple(by_left.get(record.uuid) for record in left_records),
                    tuple(record.uuid in busy_left for record in left_records),
                    tuple(record.uuid in busy_right for record in right_records),
                )
                kept = self._results.get(number)
                if kept is None or kept[0] != inputs:
                    found = review_segment(left_records, right_records, self.similarity(number), *inputs, self.low, self.high, self.margin)
                    kept = self._results[number] = (inputs, found)
                segment_reasons, segment_rivals, segment_candidates = kept[1]
                reasons |= segment_reasons
                rivals |= segment_rivals
                candidates += segment_candidates
        by_pair = {(link.left_uuid, link.right_uuid): link for link in links}
        every = {**by_pair, **{(link.left_uuid, link.right_uuid): link for link in candidates}}
        reviews = {key: Review(tuple(found), level(tuple(found), every[key]), tuple(rivals.get(key, ()))) for key, found in reasons.items()}
        return ReviewResult(reviews, candidates)


def review(
    links: list[Link],
    left: list[Record],
    right: list[Record],
    sections: SectionAlignment,
    low: float,
    high: float,
    subj_weight: float,
    margin: float = DEFAULT_MARGIN,
    declared: set[str] = frozenset(),
) -> ReviewResult:
    """Motifs des liens `links` (après patch) et candidates non appariées. `low`, `high`,
    `subj_weight` : ceux de `numrev.alignment.nw.Params` (seuils de
    Needleman-Wunsch et de la passe résiduelle, poids du SUBJ). `declared` :
    uuid déclarés sans correspondance par le patch, jamais proposés.
    `left`, `right` : les entrées des deux annuaires (les segments viennent
    de `sections`)."""
    return Reviewer(sections, low, high, subj_weight, margin)(links, declared)
