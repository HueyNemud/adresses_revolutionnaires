"""Paires à relire après l'alignement automatique : motifs et niveau
d'incertitude (`tools/display_alignment.py`, `tools/export_alignment.py`,
`tools/audit_alignment_review.py`).

L'alignement ne change pas : on signale seulement, avec un motif explicite,
les décisions qu'une relecture humaine peut corriger. Sur le gold des
inversions 1807/1808, les cas douteux demandent un savoir que les données
n'ont pas (homonymes, père / fils, coquilles des éditeurs, rues renommées) :
mieux vaut les montrer que les trancher automatiquement. Motifs, calculés
dans chaque segment (`lib.section_alignment.segments`, même similarité que
`align_directories_nw.py`) :

- `contexte incertain` : paire décidée par le pair-HMM (`nw-contexte`) de
  probabilité a posteriori < `CONTEXT_REVIEW` ;
- `homonyme proche` : paire décidée sur sa seule similarité (`nw`,
  `nw-residuel`, `dedupe`) alors qu'une autre entrée du segment, de l'un ou
  l'autre côté, est presque aussi proche (écart < `margin`) ;
- `proposition` : deux entrées restées sans correspondance, chacune la plus
  proche de l'autre dans le segment, de similarité dans la zone grise
  [`low` ; `high`[ — par défaut entre le seuil de Needleman-Wunsch et celui
  de la passe résiduelle d'`align_directories_nw.py`. Ce n'est pas un
  lien : seulement une suggestion pour le relecteur.

Les paires du patch (relues) n'ont pas de motif. Niveau d'incertitude
(ordinal, pour trier la relecture ; ce n'est pas une probabilité) : 0 sans
motif ; 1 un motif ; 2 une proposition, plusieurs motifs ou un contexte de
probabilité < `CONTEXT_DOUBT`. Aucun paramètre n'est appris sur un volume :
`tools/audit_alignment_review.py` vérifie, sur le gold d'une nouvelle paire
d'annuaires, que les motifs attrapent les erreurs et que le niveau reste
ordonné.
"""

from dataclasses import dataclass

import numpy as np

from lib.alignment import (
    SOURCE_DEDUPE,
    SOURCE_NW,
    SOURCE_NW_CONTEXT,
    SOURCE_NW_RESIDUAL,
    SOURCE_PROPOSAL,
    Link,
    Record,
    similarity_matrix,
)
from lib.section_alignment import SectionAlignment, segments

REASON_CONTEXT = "contexte incertain"
REASON_HOMONYM = "homonyme proche"
REASON_PROPOSAL = "proposition"
REASONS = (REASON_CONTEXT, REASON_HOMONYM, REASON_PROPOSAL)
SEPARATOR = " | "
CONTEXT_REVIEW = 0.9  # probabilité a posteriori sous laquelle une paire du pair-HMM est à relire
CONTEXT_DOUBT = 0.7  # … et sous laquelle elle est très incertaine (niveau 2)
DEFAULT_MARGIN = 0.05  # écart de similarité avec la concurrente (observé sur 1807/1808)
SIMILARITY_SOURCES = {SOURCE_NW, SOURCE_NW_RESIDUAL, SOURCE_DEDUPE}  # paires décidées sur leur similarité


@dataclass(frozen=True)
class Review:
    reasons: tuple[str, ...]
    level: int  # 0 sûre, 1 à relire, 2 très incertaine


@dataclass
class ReviewResult:
    reviews: dict[tuple[str, str], Review]  # (uuid gauche, uuid droit) → motifs, pour les paires qui en ont
    proposals: list[Link]  # source SOURCE_PROPOSAL, score = similarité


def level(reasons: tuple[str, ...], link: Link) -> int:
    if not reasons:
        return 0
    doubtful_context = REASON_CONTEXT in reasons and link.score is not None and link.score < CONTEXT_DOUBT
    return 2 if REASON_PROPOSAL in reasons or len(reasons) > 1 or doubtful_context else 1


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
    """Motifs des liens `links` (après patch) et propositions. `low`, `high`,
    `subj_weight` : ceux d'`align_directories_nw.Params` (seuils de
    Needleman-Wunsch et de la passe résiduelle, poids du SUBJ). `declared` :
    uuid déclarés sans correspondance par le patch, jamais proposés."""
    by_left = {link.left_uuid: link for link in links}
    busy_left = {link.left_uuid for link in links} | set(declared)
    busy_right = {link.right_uuid for link in links} | set(declared)
    reasons: dict[tuple[str, str], list[str]] = {}
    proposals = []
    for segment_left, segment_right in segments(sections):
        left_records = [record for section in segment_left for record in section.records]
        right_records = [record for section in segment_right for record in section.records]
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
            if found:
                reasons[link.left_uuid, link.right_uuid] = found

        # Propositions : meilleures partenaires mutuelles, toutes deux libres
        best_right, best_left = similarity.argmax(axis=1), similarity.argmax(axis=0)
        for i, record in enumerate(left_records):
            j = int(best_right[i])
            partner = right_records[j]
            if best_left[j] == i and low <= similarity[i, j] < high and record.uuid not in busy_left and partner.uuid not in busy_right:
                proposals.append(Link(record.uuid, partner.uuid, float(similarity[i, j]), SOURCE_PROPOSAL))
                reasons[record.uuid, partner.uuid] = [REASON_PROPOSAL]
    by_pair = {(link.left_uuid, link.right_uuid): link for link in links}
    every = {**by_pair, **{(link.left_uuid, link.right_uuid): link for link in proposals}}
    reviews = {key: Review(tuple(found), level(tuple(found), every[key])) for key, found in reasons.items()}
    return ReviewResult(reviews, proposals)
