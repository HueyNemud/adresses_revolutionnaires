"""Étape 6 · alignement ordonné de deux annuaires complets, sans
apprentissage (méthode : `numrev/alignment/nw.py`, `docs/alignement_ordonne.md`).

    uv run numrev align nw annuaires/1807_AD75-PER292 annuaires/1808_AD75-PER292 --apply

Sortie : `annuaires/alignments/<gauche>__<droite>.nw.csv`, au format de
`numrev align dedupe` (`numrev/alignment/records.py`). Le patch des
rubriques s'applique (`data/alignment/<paire>.sections.csv`) ; celui des
entrées non : il sert de référence pour comparer les méthodes. Si la sortie
de Dedupe existe, l'accord des deux méthodes est affiché.
"""

import argparse
from pathlib import Path

from rich.table import Table

from numrev.alignment import pair_hmm
from numrev.alignment.nw import Params, Result, align
from numrev.alignment.pair import load_pair, pair_of_dirs, print_matched, print_sections
from numrev.alignment.records import SOURCE_NW, SOURCE_NW_CONTEXT, SOURCE_NW_RESIDUAL, Link, Record, read_links, write_links
from numrev.command import Writes, add_apply_argument, add_force_argument, console
from numrev.paths import DEDUPE_SUFFIX, NW_SUFFIX

DESCRIPTION = (
    "Met en correspondance (un-à-un) les ENTRY de deux annuaires complets en suivant leur ordre "
    "(Needleman-Wunsch par rubrique, pair-HMM entre les ancres), sans apprentissage."
)
SCORE_BINS = (0.75, 0.8, 0.85, 0.9, 0.95, 0.99)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = Params()
    parser.add_argument("left", type=Path, help="Dossier de l'annuaire de gauche (ex. annuaires/1807_AD75-PER292).")
    parser.add_argument("right", type=Path, help="Dossier de l'annuaire de droite (ex. annuaires/1808_AD75-PER292).")
    parser.add_argument("-o", "--output", type=Path, default=None, help=f"CSV des correspondances (défaut : annuaires/alignments/<gauche>__<droite>{NW_SUFFIX}).")
    parser.add_argument("--threshold", type=float, default=defaults.threshold, help=f"Similarité minimale d'une paire alignée (défaut : {defaults.threshold}).")
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
    parser.add_argument("--no-context", action="store_true", help="Needleman-Wunsch seul, sans pair-HMM entre les ancres.")
    parser.add_argument(
        "--subj-weight",
        type=float,
        default=defaults.subj_weight,
        help=f"Poids du SUBJ (Jaro-Winkler) face au texte complet (Indel) (défaut : {defaults.subj_weight}).",
    )
    add_force_argument(parser, "Ignore les lignes orphelines du patch des rubriques (sinon : arrêt sans rien écrire).")
    add_apply_argument(parser)


# ----------------------------------------------------------------------
# Résumé
# ----------------------------------------------------------------------
def print_summary(left_name: str, right_name: str, n_left: int, n_right: int, result: Result) -> None:
    links = result.links
    matched = len(links)
    by_source = {source: sum(link.source == source for link in links) for source in (SOURCE_NW, SOURCE_NW_CONTEXT, SOURCE_NW_RESIDUAL)}
    print_matched(left_name, right_name, n_left, n_right, matched)
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


def run(args: argparse.Namespace) -> None:
    pair = pair_of_dirs(args.left, args.right)
    output_path = args.output or pair.output(NW_SUFFIX)
    writes = Writes(args.apply)
    params = Params(
        args.threshold, args.residual_threshold, args.section_threshold, args.subj_weight, args.anchor_threshold, not args.no_context
    )
    loaded = load_pair(pair, force=args.force, writes=writes, section_threshold=params.section_threshold)
    console.print("Alignement…")
    result = align(loaded.left_records, loaded.right_records, params, loaded.sections)
    left_records, right_records = loaded.left, loaded.right
    writes.add(output_path, lambda: write_links(output_path, result.links, left_records, right_records))

    print_summary(pair.left, pair.right, len(left_records), len(right_records), result)
    dedupe_path = pair.output(DEDUPE_SUFFIX)
    if dedupe_path.exists():
        print_dedupe_comparison(dedupe_path, result.links, left_records, right_records)
    writes.finish(console)
