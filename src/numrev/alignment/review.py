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
    uuid déclarés sans correspondance par le patch, jamais proposés."""
    by_left = {link.left_uuid: link for link in links}
    busy_left = {link.left_uuid for link in links} | set(declared)
    busy_right = {link.right_uuid for link in links} | set(declared)
    reasons: dict[tuple[str, str], list[str]] = {}
    rivals: dict[tuple[str, str], list[tuple[str, str, float]]] = {}
    candidates = []
    for left_records, right_records in segment_entries(sections):
        if not left_records or not right_records:
            continue
        similarity = similarity_matrix(left_records, right_records, subj_weight)
        right_index = {record.uuid: j for j, record in enumerate(right_records)}

        # Liens dont les deux entrées sont dans le segment
        for i, record in enumerate(left_records):
            link = by_left.get(record.uuid)
            if link is None or link.right_uuid not in right_index:
                continue
            found = []
            if link.source == SOURCE_NW_CONTEXT and link.score is not None and link.score < CONTEXT_REVIEW:
                found.append(REASON_CONTEXT)
            if link.source in SIMILARITY_SOURCES:
                j = right_index[link.right_uuid]
                rival = max(np.delete(similarity[i], j).max(initial=0.0), np.delete(similarity[:, j], i).max(initial=0.0))
                if similarity[i, j] - rival < margin:
                    found.append(REASON_HOMONYM)
                    floor = similarity[i, j] - margin
                    close = [("right", right_records[k].uuid, float(similarity[i, k])) for k in range(len(right_records)) if k != j]
                    close += [("left", left_records[k].uuid, float(similarity[k, j])) for k in range(len(left_records)) if k != i]
                    rivals[link.left_uuid, link.right_uuid] = sorted((c for c in close if c[2] > floor), key=lambda c: -c[2])
            if found:
                reasons[link.left_uuid, link.right_uuid] = found

        # Candidates non appariées : meilleures partenaires mutuelles, toutes deux libres
        best_right, best_left = similarity.argmax(axis=1), similarity.argmax(axis=0)
        for i, record in enumerate(left_records):
            j = int(best_right[i])
            partner = right_records[j]
            if best_left[j] == i and low <= similarity[i, j] < high and record.uuid not in busy_left and partner.uuid not in busy_right:
                candidates.append(Link(record.uuid, partner.uuid, float(similarity[i, j]), SOURCE_CANDIDATE))
                reasons[record.uuid, partner.uuid] = [REASON_CANDIDATE]
    by_pair = {(link.left_uuid, link.right_uuid): link for link in links}
    every = {**by_pair, **{(link.left_uuid, link.right_uuid): link for link in candidates}}
    reviews = {key: Review(tuple(found), level(tuple(found), every[key]), tuple(rivals.get(key, ()))) for key, found in reasons.items()}
    return ReviewResult(reviews, candidates)
