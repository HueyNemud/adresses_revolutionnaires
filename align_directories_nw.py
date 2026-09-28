"""Aligne les entrées de deux éditions d'un annuaire en suivant leur ordre
(Needleman-Wunsch), sans apprentissage : alternative à `align_directories.py`.

    uv run align_directories_nw.py annuaires/1807_AD75-PER292 annuaires/1808_AD75-PER292

D'une édition à l'autre, l'ordre des rubriques est stable et, dans une
rubrique, celui des entrées l'est presque (ajouts, suppressions, quelques
inversions locales du tri alphabétique). L'alignement se fait donc en trois
temps :

1. **rubriques** : `lib/section_alignment.py` (partagé avec Dedupe et le
   viewer) aligne les deux suites de rubriques par Needleman-Wunsch sur la
   similarité Jaro-Winkler de leur clé (« liste » / « listes de
   non-commerçans », « sellieres » / « selliers »), après application du
   patch des rubriques (`data/alignement/<gauche>__<droite>.sections.csv`) :
   groupes imposés à la main, éventuellement 1-N ou N-1, et rubriques
   déclarées sans correspondance ;
2. **entrées** : dans chaque groupe de rubriques — entrées des N rubriques
   d'un groupe manuel concaténées dans l'ordre de chaque annuaire — et dans
   chaque « trou » entre deux paires automatiques, qui regroupe les
   rubriques restées seules de part et d'autre (rubrique renommée au-delà du
   seuil, scindée…, mais pas celles déclarées seules au patch) — les deux
   suites d'entrées sont alignées par Needleman-Wunsch, dont les paires très
   sûres servent d'**ancres** ; entre deux ancres, un **pair-HMM**
   (`lib/pair_hmm.py`) décide des autres paires selon leur probabilité a
   posteriori, qui tient compte du contexte : une paire encadrée par deux
   paires est plus probable qu'une paire isolée au milieu d'ajouts et de
   suppressions (voir `docs/alignement_ordonne.md`) ;
3. **passe résiduelle** : dans chaque segment, les entrées restées seules
   sont appariées sans contrainte d'ordre (affectation optimale), à un seuil
   plus strict, pour récupérer les inversions locales.

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

Sortie : `annuaires/alignements/<gauche>__<droite>.nw.csv`, au format de
`align_directories.py` (`lib/alignment.py`). `source` et sens de `score` :
`nw` (ancre, ou toute paire Needleman-Wunsch avec `--no-context`) :
similarité ; `nw-contexte` (décidée par le pair-HMM) : probabilité a
posteriori ; `nw-residuel` : similarité. Le patch de corrections manuelles
des entrées n'est pas appliqué : il sert de référence pour comparer les deux
méthodes.
"""

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Indel, JaroWinkler
from rapidfuzz.process import cdist
from rich.console import Console
from rich.table import Table
from scipy.optimize import linear_sum_assignment

from lib import pair_hmm
from lib.alignment import SOURCE_NW, SOURCE_NW_CONTEXT, SOURCE_NW_RESIDUAL, Link, Record, dedupe_records, load_volume, read_links, write_links
from lib.section_alignment import (
    DEFAULT_THRESHOLD,
    SECTION_PATCH_SUFFIX,
    SOURCE_AUTO,
    Section,
    SectionAlignment,
    align_sections,
    load_section_alignment,
)
from lib.sequence import needleman_wunsch

console = Console()

DEFAULT_OUTPUT_DIR = Path("annuaires/alignements")
DEFAULT_PATCH_DIR = Path("data/alignement")
NW_SUFFIX = ".nw.csv"
SCORE_BINS = (0.75, 0.8, 0.85, 0.9, 0.95, 0.99)


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


def parse_args() -> argparse.Namespace:
    defaults = Params()
    parser = argparse.ArgumentParser(
        description=(
            "Met en correspondance (un-à-un) les ENTRY de deux annuaires complets en suivant leur ordre "
            "(Needleman-Wunsch par rubrique), sans apprentissage."
        )
    )
    parser.add_argument("left", type=Path, help="Dossier de l'annuaire de gauche (ex. annuaires/1807_AD75-PER292).")
    parser.add_argument("right", type=Path, help="Dossier de l'annuaire de droite (ex. annuaires/1808_AD75-PER292).")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"CSV des correspondances (défaut : {DEFAULT_OUTPUT_DIR}/<gauche>__<droite>{NW_SUFFIX}).",
    )
    parser.add_argument(
        "--section-patch",
        type=Path,
        default=None,
        help=f"Patch des rubriques (défaut : {DEFAULT_PATCH_DIR}/<gauche>__<droite>{SECTION_PATCH_SUFFIX}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=defaults.threshold,
        help=f"Similarité minimale d'une paire alignée (défaut : {defaults.threshold}).",
    )
    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=defaults.residual_threshold,
        help=f"Similarité minimale d'une paire de la passe résiduelle, hors ordre (défaut : {defaults.residual_threshold}).",
    )
    parser.add_argument(
        "--section-threshold",
        type=float,
        default=defaults.section_threshold,
        help=f"Similarité (Jaro-Winkler) minimale de deux rubriques alignées (défaut : {defaults.section_threshold}).",
    )
    parser.add_argument(
        "--anchor-threshold",
        type=float,
        default=defaults.anchor_threshold,
        help=f"Similarité minimale d'une ancre ; entre deux ancres, le pair-HMM décide (défaut : {defaults.anchor_threshold}).",
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Needleman-Wunsch seul, sans pair-HMM entre les ancres.",
    )
    parser.add_argument(
        "--subj-weight",
        type=float,
        default=defaults.subj_weight,
        help=f"Poids du SUBJ (Jaro-Winkler) face au texte complet (Indel) (défaut : {defaults.subj_weight}).",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
# Alignement
# ----------------------------------------------------------------------
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


def similarity_matrix(left: list[Record], right: list[Record], subj_weight: float) -> np.ndarray:
    """Similarité de chaque entrée de gauche à chaque entrée de droite, sur
    les champs de Dedupe."""
    left_fields, right_fields = dedupe_records(left), dedupe_records(right)
    left_values = [left_fields[record.uuid] for record in left]
    right_values = [right_fields[record.uuid] for record in right]

    def matrix(name: str, scorer) -> np.ndarray:
        return cdist(
            [values[name] or "" for values in left_values],
            [values[name] or "" for values in right_values],
            scorer=scorer.normalized_similarity,
            dtype=np.float32,
            workers=-1,
        )

    text = matrix("text", Indel)
    subj = matrix("subj", JaroWinkler)
    both = np.outer([bool(values["subj"]) for values in left_values], [bool(values["subj"]) for values in right_values])
    return np.where(both, subj_weight * subj + (1 - subj_weight) * text, text)


def segments(alignment: SectionAlignment) -> list[tuple[list[Section], list[Section]]]:
    """Segments à aligner : chaque groupe manuel de rubriques (entrées de ses
    N rubriques concaténées) ; chaque paire automatique et, entre deux paires
    (ou avant la première, après la dernière), le « trou » des rubriques
    restées seules des deux côtés, s'il en a des deux côtés (hors rubriques
    déclarées seules au patch)."""
    result = [(group.left, group.right) for group in alignment.groups if group.source != SOURCE_AUTO]
    left, right = alignment.auto_left, alignment.auto_right
    previous_i, previous_j = -1, -1
    for i, j in [*alignment.auto_pairs, (len(left), len(right))]:
        gap_left, gap_right = left[previous_i + 1 : i], right[previous_j + 1 : j]
        if gap_left and gap_right:
            result.append((gap_left, gap_right))
        if i < len(left):
            result.append(([left[i]], [right[j]]))
        previous_i, previous_j = i, j
    return result


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
    for segment_left, segment_right in segments(sections):
        left = [record for section in segment_left for record in section.records]
        right = [record for section in segment_right for record in section.records]
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


# ----------------------------------------------------------------------
# Résumé
# ----------------------------------------------------------------------
def print_summary(left_name: str, right_name: str, n_left: int, n_right: int, result: Result) -> None:
    links = result.links
    matched = len(links)
    by_source = {source: sum(link.source == source for link in links) for source in (SOURCE_NW, SOURCE_NW_CONTEXT, SOURCE_NW_RESIDUAL)}
    table = Table(title="Correspondances", show_header=True)
    table.add_column("Annuaire")
    table.add_column("Entrées", justify="right")
    table.add_column("Appariées", justify="right")
    table.add_column("Non appariées", justify="right")
    for name, total in ((left_name, n_left), (right_name, n_right)):
        table.add_row(name, str(total), f"{matched} ({matched / total:.1%})", str(total - matched))
    console.print(table)
    console.print(
        f"Dont {by_source[SOURCE_NW]} {SOURCE_NW}, {by_source[SOURCE_NW_CONTEXT]} {SOURCE_NW_CONTEXT} "
        f"(pair-HMM), {by_source[SOURCE_NW_RESIDUAL]} {SOURCE_NW_RESIDUAL} (hors ordre)."
    )

    bins = Table(title="Scores des correspondances", show_header=True)
    bins.add_column("Score")
    sources = [source for source, count in by_source.items() if count]
    for source in sources:
        bins.add_column(source, justify="right")
    edges = [0.0, 0.5, *SCORE_BINS, float("inf")]
    for low, high in zip(edges, edges[1:]):
        label = f"[{low:.2f} ; {high:.2f}[" if high != float("inf") else f"≥ {low:.2f}"
        bins.add_row(label, *(str(sum(link.source == source and low <= link.score < high for link in links)) for source in sources))
    console.print(bins)
    if result.fit:
        print_model(result.fit)

    print_sections(result.sections)


def print_sections(alignment: SectionAlignment) -> None:
    auto = [group for group in alignment.groups if group.source == SOURCE_AUTO]
    manual = [group for group in alignment.groups if group.source != SOURCE_AUTO]
    renamed = sum(group.left[0].key != group.right[0].key for group in auto)
    console.print(
        f"Rubriques : {len(auto)} paire(s) alignée(s) automatiquement (dont {renamed} de clé différente), "
        f"{len(manual)} groupe(s) du patch, {len(alignment.unmatched_left)} seule(s) à gauche, "
        f"{len(alignment.unmatched_right)} seule(s) à droite."
    )
    for group in manual:
        left, right = (" + ".join(section.title for section in getattr(group, side)) for side in ("left", "right"))
        console.print(f"  patch · {left} ↔ {right}", markup=False)
    for side, label, unmatched in (("left", "gauche", alignment.unmatched_left), ("right", "droite", alignment.unmatched_right)):
        for section in unmatched:
            origin = "déclarée au patch" if (side, section.uuid) in alignment.declared else "non alignée"
            console.print(f"  {label} · {section.title} ({len(section.records)} entrées, {origin})", markup=False)
    resolution = alignment.resolution
    if resolution.reanchored:
        console.print(f"[yellow]↻ {len(resolution.reanchored)} ligne(s) du patch des rubriques réancrée(s) (patch mis à jour).[/yellow]")
    for entry in resolution.orphans:
        console.print(f"⚠ Ligne orpheline du patch des rubriques, non appliquée : {entry.left_title} ↔ {entry.right_title}", markup=False)


def print_model(fit: pair_hmm.Fit) -> None:
    """Paramètres du pair-HMM estimés par EM."""
    model = fit.model
    table = Table(title=f"Pair-HMM : transitions estimées ({len(fit.log_likelihoods)} itération(s) d'EM)", show_header=True)
    table.add_column("de \\ vers")
    for state in pair_hmm.STATES:
        table.add_column(state, justify="right")
    for row, state in zip(model.transitions, pair_hmm.STATES):
        table.add_row(state, *(f"{value:.3f}" for value in row))
    console.print(table)
    console.print(
        "M : paire, X : seulement à gauche, Y : seulement à droite. "
        f"Une paire devient plus vraisemblable que deux entrées différentes à partir de sim ≈ {model.break_even():.2f}."
    )


def print_dedupe_comparison(path: Path, links: list[Link], left: dict[str, Record], right: dict[str, Record]) -> None:
    """Accord avec la sortie brute de Dedupe (première comparaison, en
    attendant des alignements curés)."""
    dedupe_links = [link for link in read_links(path) if link.left_uuid in left and link.right_uuid in right]
    ours = {(link.left_uuid, link.right_uuid) for link in links}
    theirs = {(link.left_uuid, link.right_uuid) for link in dedupe_links}
    our_left = {left_uuid: right_uuid for left_uuid, right_uuid in ours}
    conflicting = sum(left_uuid in our_left for left_uuid, _ in theirs - ours)
    console.print(
        f"\nComparaison avec [yellow]{path}[/yellow] : {len(ours & theirs)} paire(s) communes, "
        f"{len(ours - theirs)} seulement ici, {len(theirs - ours)} seulement chez Dedupe "
        f"(pour {conflicting} d'entre elles, l'entrée de gauche est appariée ailleurs ici)."
    )


def main() -> None:
    args = parse_args()
    for directory in (args.left, args.right):
        if not directory.is_dir():
            console.print(f"[bold red]Erreur :[/bold red] dossier '{directory}' introuvable.")
            sys.exit(1)

    pair_name = f"{args.left.name}__{args.right.name}"
    output_path = args.output or DEFAULT_OUTPUT_DIR / f"{pair_name}{NW_SUFFIX}"
    section_patch = args.section_patch or DEFAULT_PATCH_DIR / f"{pair_name}{SECTION_PATCH_SUFFIX}"
    params = Params(
        args.threshold, args.residual_threshold, args.section_threshold, args.subj_weight, args.anchor_threshold, not args.no_context
    )

    try:
        left_list, right_list = load_volume(args.left), load_volume(args.right)
    except (OSError, csv.Error) as error:
        console.print(f"[bold red]Erreur de lecture :[/bold red] {error}")
        sys.exit(1)
    console.print(f"{args.left.name} : {len(left_list)} entrées ; {args.right.name} : {len(right_list)} entrées")

    try:
        sections = load_section_alignment(left_list, right_list, section_patch, params.section_threshold)
    except (OSError, csv.Error, ValueError) as error:
        console.print(f"[bold red]Erreur dans le patch des rubriques :[/bold red] {error}")
        sys.exit(1)
    console.print("Alignement…")
    result = align(left_list, right_list, params, sections)
    left_records = {record.uuid: record for record in left_list}
    right_records = {record.uuid: record for record in right_list}
    write_links(output_path, result.links, left_records, right_records)

    console.print(f"\n[bold green]✅ Correspondances :[/bold green] [yellow]{output_path}[/yellow]")
    print_summary(args.left.name, args.right.name, len(left_list), len(right_list), result)
    dedupe_path = DEFAULT_OUTPUT_DIR / f"{pair_name}.dedupe.csv"
    if dedupe_path.exists():
        print_dedupe_comparison(dedupe_path, result.links, left_records, right_records)


if __name__ == "__main__":
    main()
