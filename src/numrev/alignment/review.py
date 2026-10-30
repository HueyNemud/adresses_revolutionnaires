"""Paires à vérifier après l'alignement : motifs de relecture et niveau
d'incertitude (`numrev view alignment`, `numrev join`, `numrev audit alignment`).

L'alignement ne change pas : on montre au relecteur, avec un motif, les
décisions qu'il peut corriger (homonymes, père / fils, rues renommées : un
savoir que les données n'ont pas). Calculés par segment, avec la similarité
de `numrev align nw` :

- `déduite des voisines (p < 0,9)` : paire `nw-contexte` de probabilité a
  posteriori < `CONTEXT_REVIEW` ;
- `homonyme proche` : paire décidée sur sa similarité (`nw`, `nw-residuel`,
  `dedupe`) alors qu'une autre entrée du segment est à moins de `margin` ;
- `candidate non appariée` : deux entrées sans correspondance, chacune la
  plus proche de l'autre, de similarité entre les seuils de Needleman-Wunsch
  et de la passe résiduelle. Ce n'est pas un lien.

Les paires du patch (relues) n'ont pas de motif. Niveau d'incertitude
(ordinal, pas une probabilité) : `faible` sans motif, `moyenne` avec un
motif, `forte` pour une candidate, plusieurs motifs ou une paire déduite des
voisines de probabilité < `CONTEXT_DOUBT`. Rien n'est appris sur un volume.
"""

from dataclasses import dataclass

import numpy as np

from numrev.alignment.nw import Params
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
    params: Params = Params(),
    margin: float = DEFAULT_MARGIN,
    declared: set[str] = frozenset(),
) -> ReviewResult:
    """Motifs des liens `links` (après patch) et candidates non appariées :
    similarité (poids du SUBJ de `params`) dans [`params.threshold` ;
    `params.residual_threshold`[. `declared` : uuid déclarés sans
    correspondance par le patch, jamais proposés."""
    low, high = params.threshold, params.residual_threshold
    by_left = {link.left_uuid: link for link in links}
    busy_left = {link.left_uuid for link in links} | set(declared)
    busy_right = {link.right_uuid for link in links} | set(declared)
    reasons: dict[tuple[str, str], list[str]] = {}
    candidates = []
    for left_records, right_records in segment_entries(sections):
        if not left_records or not right_records:
            continue
        similarity = similarity_matrix(left_records, right_records, params.subj_weight)
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
    reviews = {key: Review(tuple(found), level(tuple(found), every[key])) for key, found in reasons.items()}
    return ReviewResult(reviews, candidates)
