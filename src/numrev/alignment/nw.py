"""Alignement ordonné des entrées de deux éditions d'un annuaire
(Needleman-Wunsch + pair-HMM), sans apprentissage : algorithme de
`numrev align nw`, alternative à `numrev align dedupe`.

D'une édition à l'autre, l'ordre des rubriques est stable et, dans une
rubrique, celui des entrées l'est presque (ajouts, suppressions, quelques
inversions locales du tri alphabétique). L'alignement se fait donc en trois
temps :

1. **rubriques** : `numrev/alignment/sections.py` (partagé avec Dedupe et le
   viewer) aligne les deux suites de rubriques par Needleman-Wunsch sur la
   similarité Jaro-Winkler de leur clé (« liste » / « listes de
   non-commerçans », « sellieres » / « selliers »), après application du
   patch des rubriques (`data/alignment/<gauche>__<droite>.sections.csv`) :
   groupes imposés à la main, éventuellement 1-N ou N-1, et rubriques
   déclarées sans correspondance ;
2. **entrées** : dans chaque groupe de rubriques appariées — entrées des N
   rubriques d'un groupe manuel concaténées dans l'ordre de chaque annuaire
   — les deux suites d'entrées sont alignées par Needleman-Wunsch, dont les paires très
   sûres servent d'**ancres** ; entre deux ancres, un **pair-HMM**
   (`numrev/alignment/pair_hmm.py`) décide des autres paires selon leur probabilité a
   posteriori, qui tient compte du contexte : une paire encadrée par deux
   paires est plus probable qu'une paire isolée au milieu d'ajouts et de
   suppressions (voir `docs/alignement_ordonne.md`) ;
3. **passe résiduelle** : dans chaque segment, les entrées restées seules
   sont appariées sans contrainte d'ordre (affectation optimale), à un seuil
   plus strict, pour récupérer les inversions locales.

On n'apparie jamais deux entrées de rubriques qui ne se correspondent pas :
une rubrique sans correspondance (renommée au-delà du seuil, scindée…) ne
s'aligne qu'une fois liée dans le patch des rubriques.

Similarité de deux entrées, sur les mêmes champs que Dedupe
(`dedupe_records`, en minuscules) : `w · JaroWinkler(subj) + (1 − w) ·
Indel(text)` (distance d'édition normalisée), le texte seul si l'une des deux
n'a pas de SUBJ. Jaro-Winkler convient au SUBJ (court, le nom en tête) mais
sature sur le texte complet, long et dont l'adresse varie.

Needleman-Wunsch maximise Σ (similarité − seuil) sur les appariements qui
respectent l'ordre, sans pénalité de trou : une paire sous le seuil n'est
jamais retenue, et ajouts ou suppressions ne coûtent rien. Les paramètres du
pair-HMM sont estimés sans étiquettes (EM) sur l'ensemble des fenêtres entre
ancres. `--no-context` s'en tient au Needleman-Wunsch seul.

`source` et sens de `score` des liens : `nw` (ancre, ou toute paire
Needleman-Wunsch avec `--no-context`) : similarité ; `nw-contexte` (décidée
par le pair-HMM) : probabilité a posteriori ; `nw-residuel` : similarité.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from numrev.alignment import pair_hmm
from numrev.alignment.records import SOURCE_NW, SOURCE_NW_CONTEXT, SOURCE_NW_RESIDUAL, Link, Record, similarity_matrix
from numrev.alignment.sections import DEFAULT_THRESHOLD, SectionAlignment, align_sections, segment_entries
from numrev.alignment.sequence import needleman_wunsch


@dataclass(frozen=True)
class Params:
    threshold: float = 0.75  # similarité minimale d'une paire (Needleman-Wunsch)
    residual_threshold: float = 0.85  # idem, passe résiduelle
    section_threshold: float = DEFAULT_THRESHOLD  # similarité minimale de deux clés de rubrique
    subj_weight: float = 0.5  # poids du SUBJ dans la similarité
    anchor_threshold: float = 0.9  # similarité minimale d'une ancre (pair-HMM entre les ancres)
    context: bool = True  # False : Needleman-Wunsch seul


@dataclass
class Result:
    links: list[Link]
    sections: SectionAlignment
    fit: pair_hmm.Fit | None = None  # None avec --no-context


@dataclass
class Segment:
    left: list[Record]
    right: list[Record]
    similarity: np.ndarray
    ordered: list[tuple[int, int]]  # paires Needleman-Wunsch
    residual: list[tuple[int, int]]  # passe résiduelle sur les entrées hors `ordered`


def residual_pairs(similarity: np.ndarray, pairs: list[tuple[int, int]], threshold: float) -> list[tuple[int, int]]:
    """Affectation optimale, sans contrainte d'ordre, des lignes et colonnes
    absentes de `pairs` ; seules les paires de similarité ≥ seuil sont
    gardées."""
    rows = sorted(set(range(similarity.shape[0])) - {i for i, _ in pairs})
    cols = sorted(set(range(similarity.shape[1])) - {j for _, j in pairs})
    if not rows or not cols:
        return []
    sub = similarity[np.ix_(rows, cols)]
    sub = np.where(sub >= threshold, sub, 0.0)
    return sorted((rows[i], cols[j]) for i, j in zip(*linear_sum_assignment(sub, maximize=True)) if sub[i, j] > 0)


def windows(
    anchors: list[tuple[int, int]], n: int, m: int, excluded: list[tuple[int, int]] = ()
) -> list[tuple[list[int], list[int]]]:
    """Fenêtres (lignes, colonnes) entre deux ancres consécutives, avec des
    ancres fictives avant et après le segment, y compris les fenêtres vides
    ou d'un seul côté ; sans les entrées des paires `excluded` (inversions
    de la passe résiduelle, tenues pour acquises)."""
    left_out, right_out = {i for i, _ in excluded}, {j for _, j in excluded}
    bounds = [(-1, -1), *anchors, (n, m)]
    return [
        ([i for i in range(i1 + 1, i2) if i not in left_out], [j for j in range(j1 + 1, j2) if j not in right_out])
        for (i1, j1), (i2, j2) in zip(bounds, bounds[1:])
    ]


NEIGHBOUR_OFFSETS = (1, 2, 3)  # voisins d'une ancre pris comme paires « différentes »


def neighbour_similarities(similarity: np.ndarray, anchors: list[tuple[int, int]]) -> np.ndarray:
    """Similarités d'entrées différentes mais **proches dans l'ordre** :
    chaque ancre (i, j) comparée aux voisines de son partenaire, (i, j ± d)
    et (i ± d, j). C'est le bon modèle nul pour une paire candidate entre
    deux ancres : dans une liste alphabétique, deux voisines partagent
    souvent leurs initiales et se ressemblent bien plus que deux entrées
    prises au hasard."""
    n, m = similarity.shape
    values = []
    for i, j in anchors:
        for d in NEIGHBOUR_OFFSETS:
            for a, b in ((i, j - d), (i, j + d), (i - d, j), (i + d, j)):
                if 0 <= a < n and 0 <= b < m:
                    values.append(similarity[a, b])
    return np.array(values)


def unique_subject_similarities(segment: "Segment") -> np.ndarray:
    """Similarités de paires sûrement identiques, choisies **sans regarder
    l'ordre ni la similarité** : même SUBJ (en minuscules), présent une seule
    fois de chaque côté du segment. Échantillon biaisé vers le haut (un SUBJ
    identique garantit sim ≥ 0,5), donc prudent : il sous-estime les paires
    identiques peu similaires (nom mal lu et adresse changée)."""
    left = Counter(record.subj.lower() for record in segment.left)
    right = Counter(record.subj.lower() for record in segment.right)
    right_index = {record.subj.lower(): j for j, record in enumerate(segment.right)}
    return np.array(
        [
            segment.similarity[i, right_index[key]]
            for i, record in enumerate(segment.left)
            if (key := record.subj.lower()) and left[key] == 1 and right[key] == 1
        ]
    )


def contextual_pairs(segments: list[Segment], anchor_threshold: float) -> tuple[list[list[tuple[int, int, float, str]]], pair_hmm.Fit]:
    """Par segment : ancres (similarité ≥ seuil parmi les paires
    Needleman-Wunsch) et paires de probabilité a posteriori > 0,5 dans les
    fenêtres entre ancres, sous un pair-HMM commun au volume : émissions
    tirées de paires sûrement identiques (`unique_subject_similarities`) et
    sûrement différentes (`neighbour_similarities`), transitions par EM.
    Les paires de la passe résiduelle sont retirées des fenêtres : le
    pair-HMM, qui ne voit que l'ordre, ne doit pas prendre une entrée à une
    inversion évidente. Chaque paire : (i, j, score, source)."""
    anchors = [[(i, j) for i, j in segment.ordered if segment.similarity[i, j] >= anchor_threshold] for segment in segments]
    boxes = [windows(found, *segment.similarity.shape, segment.residual) for segment, found in zip(segments, anchors)]
    matrices = [segment.similarity[np.ix_(rows, cols)] for segment, found in zip(segments, boxes) for rows, cols in found]
    same = sum(pair_hmm.histogram(unique_subject_similarities(segment)) for segment in segments)
    different = sum(pair_hmm.histogram(neighbour_similarities(segment.similarity, found)) for segment, found in zip(segments, anchors))
    fitted = pair_hmm.fit(matrices, same, different)

    result = []
    for segment, found, segment_boxes in zip(segments, anchors, boxes):
        pairs = [(i, j, float(segment.similarity[i, j]), SOURCE_NW) for i, j in found]
        for rows, cols in segment_boxes:
            if rows and cols:
                posterior = pair_hmm.window_posteriors(segment.similarity[np.ix_(rows, cols)], fitted.model).posterior
                pairs += [(rows[i], cols[j], float(posterior[i, j]), SOURCE_NW_CONTEXT) for i, j in pair_hmm.decode(posterior)]
        result.append(pairs)
    return result, fitted


def align(
    left_records: list[Record], right_records: list[Record], params: Params = Params(), sections: SectionAlignment | None = None
) -> Result:
    """`sections` : alignement des rubriques déjà calculé (avec le patch) ;
    à défaut, alignement automatique sans patch."""
    if sections is None:
        sections = align_sections(left_records, right_records, threshold=params.section_threshold)
    segment_list = []
    for left, right in segment_entries(sections):
        similarity = similarity_matrix(left, right, params.subj_weight)
        ordered = needleman_wunsch(similarity, params.threshold)
        residual = residual_pairs(similarity, ordered, params.residual_threshold)
        segment_list.append(Segment(left, right, similarity, ordered, residual))

    fitted = None
    if params.context and segment_list:
        chosen, fitted = contextual_pairs(segment_list, params.anchor_threshold)
    else:
        chosen = [[(i, j, float(segment.similarity[i, j]), SOURCE_NW) for i, j in segment.ordered] for segment in segment_list]

    links = []
    for segment, pairs in zip(segment_list, chosen):
        pairs = pairs + [(i, j, float(segment.similarity[i, j]), SOURCE_NW_RESIDUAL) for i, j in segment.residual]
        links += [Link(segment.left[i].uuid, segment.right[j].uuid, score, source) for i, j, score, source in sorted(pairs)]
    return Result(links=links, sections=sections, fit=fitted)


