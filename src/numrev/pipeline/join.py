"""Étape 7 · jointure lisible de deux annuaires alignés, en CSV, pour les
utilisateurs des données (historiens).

    uv run numrev join annuaires/alignments/<gauche>__<droite>.nw.csv [--excel] --apply

L'entrée est une sortie d'alignement (`*.dedupe.csv`, `*.nw.csv`, ou le CSV
final `<gauche>__<droite>.csv`) ; comme dans `numrev view alignment`, la
paire de volumes se déduit du nom de fichier, les deux annuaires sont relus
en entier (`annuaires/<gauche>/`, `annuaires/<droite>/`, leurs
`*.ner.csv`) et le patch des corrections manuelles
`data/alignment/<gauche>__<droite>.patch.csv` est appliqué en mémoire (il
n'est jamais réécrit ici ; `--no-patch` l'ignore).

Une ligne par correspondance ou entrée sans correspondance, dans l'ordre
naturel des listes (`numrev/alignment/export.py`) : `statut` (apparié / gauche
seulement / droite seulement), `score`, `methode` (source du lien),
`certitude` (relue / incertaine à la relecture / automatique),
`niveau_incertitude` (faible, moyenne, forte) et
`motifs_relecture` (`numrev/alignment/review.py`), puis pour chaque
côté (`gauche_…`, `droite_…`) : volume, page, rubrique, texte sans Markdown,
empans `sujet` / `description` / `adresse` (plusieurs empans de même classe
séparés par « | »), texte balisé et uuid.

`--candidates` ajoute les candidates non appariées à vérifier (deux entrées
sans correspondance, statut `candidate`) ; par défaut, l'export ne contient
que les liens retenus.

On n'apparie qu'entre rubriques appariées (`numrev/alignment/sections.py`) : un
lien entre rubriques qui ne se correspondent pas (Dedupe en produit) est
écarté, et compté en console.

Sortie par défaut : `<entrée sans .csv>.join.csv` à côté de l'entrée.
`--excel` : séparateur `;` et UTF-8 avec BOM, pour un tableur en français.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

from numrev.alignment.export import STATUS_LABELS, export_csv, export_encoding, natural_rows
from numrev.alignment.nw import Params
from numrev.alignment.pair import load_pair, pair_of_file
from numrev.alignment.patch import apply_patch, declared_unmatched, load_patch, orphan_descriptions
from numrev.alignment.records import read_links
from numrev.alignment.review import DEFAULT_MARGIN, review
from numrev.alignment.sections import restrict_to_corresponding
from numrev.command import CommandError, Writes, add_apply_argument, add_force_argument, console
from numrev.curation import refuse_orphans
from numrev.paths import join_output

DESCRIPTION = "Exporte en CSV la jointure lisible de deux annuaires alignés (une ligne par paire ou entrée seule)."


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("alignment", type=Path, help="Sortie d'alignement (ex. annuaires/alignments/<gauche>__<droite>.nw.csv).")
    parser.add_argument("-o", "--output", type=Path, help="CSV de sortie (défaut : <entrée>.join.csv).")
    parser.add_argument("--excel", action="store_true", help="Séparateur `;` et UTF-8 avec BOM (tableur en français).")
    parser.add_argument("--no-patch", action="store_true", help="Ne pas appliquer le patch des corrections manuelles.")
    parser.add_argument("--candidates", action="store_true", help="Ajouter les candidates non appariées à vérifier (statut `candidate`).")
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help=f"Écart de similarité sous lequel une concurrente fait signaler `homonyme proche` (défaut : {DEFAULT_MARGIN}).",
    )
    add_force_argument(parser, "Ignorer les lignes orphelines des patchs (sinon : arrêt sans rien écrire).")
    add_apply_argument(parser)


def run(args: argparse.Namespace) -> None:
    writes = Writes(args.apply)
    pair = pair_of_file(args.alignment)
    loaded = load_pair(pair, force=args.force)
    left, right = loaded.left, loaded.right
    links = read_links(args.alignment)

    declared: set[str] = set()
    if not args.no_patch:
        try:
            entries, resolution = load_patch(pair.entry_patch, left, right)
        except (OSError, csv.Error, ValueError) as error:
            raise CommandError(f"patch {pair.entry_patch} : {error}") from error
        refuse_orphans(orphan_descriptions(resolution), args.force, "ligne orpheline du patch")
        links, stats = apply_patch(links, resolution.entries)
        declared = declared_unmatched(resolution)
        console.print(f"Patch [yellow]{pair.entry_patch}[/yellow] : {len(entries)} ligne(s), {stats.manual_pairs} paire(s) manuelle(s).")
        if resolution.reanchored:
            console.print(f"[yellow]↻ {len(resolution.reanchored)} ligne(s) réancrée(s) par le texte (patch non réécrit).[/yellow]")

    links, dropped = restrict_to_corresponding(links, left, right, loaded.sections)
    if dropped:
        console.print(f"[yellow]{len(dropped)} lien(s) entre rubriques non appariées écarté(s).[/yellow]")
    params = Params()
    found = review(
        links, loaded.left_records, loaded.right_records, loaded.sections, params.threshold, params.residual_threshold,
        params.subj_weight, args.margin, declared,
    )
    rows, n_missing = natural_rows(links + (found.candidates if args.candidates else []), left, right)
    if n_missing:
        console.print(f"[bold red]⚠ {n_missing} lien(s) vers des entrées disparues, ignoré(s) : relancer l'alignement.[/bold red]")

    output = args.output or join_output(args.alignment)
    writes.add(output, lambda: output.write_text(export_csv(rows, args.excel, found.reviews), encoding=export_encoding(args.excel), newline=""))
    counts = Counter(row.kind for row in rows)
    summary = ", ".join(f"{counts[kind]} {label}" for kind, label in STATUS_LABELS.items())
    console.print(f"{len(rows)} ligne(s) ({summary}).")
    levels = Counter(
        found.reviews[key].level for row in rows if row.link and (key := (row.link.left_uuid, row.link.right_uuid)) in found.reviews
    )
    console.print(
        f"À vérifier : {levels[1]} paire(s) d'incertitude moyenne, {levels[2]} d'incertitude forte"
        + ("" if args.candidates else f" ; {len(found.candidates)} candidate(s) non appariée(s) non exportée(s) (--candidates)")
        + "."
    )
    writes.finish(console)
