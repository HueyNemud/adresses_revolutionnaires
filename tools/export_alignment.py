"""Exporte en CSV la jointure de deux annuaires alignés, pour les
utilisateurs des données (historiens).

    uv run tools/export_alignment.py annuaires/alignements/<gauche>__<droite>.nw.csv [--excel]

L'entrée est une sortie d'alignement (`*.dedupe.csv`, `*.nw.csv`, ou le CSV
final `<gauche>__<droite>.csv`) ; comme dans `tools/display_alignment.py`, la
paire de volumes se déduit du nom de fichier, les deux annuaires sont relus
en entier (`annuaires/<gauche>/`, `annuaires/<droite>/`, leurs
`*.merged.ner.csv`) et le patch des corrections manuelles
`data/alignement/<gauche>__<droite>.patch.csv` est appliqué en mémoire (il
n'est jamais réécrit ici ; `--sans-patch` l'ignore).

Une ligne par correspondance ou entrée sans correspondance, dans l'ordre
naturel des listes (`lib/alignment_export.py`) : `statut` (apparié / gauche
seulement / droite seulement), `score`, `methode` (source du lien),
`certitude` (relue / incertaine à la relecture / automatique),
`niveau_incertitude` (faible, moyenne, forte) et
`motifs_relecture` (`lib/alignment_review.py`), puis pour chaque
côté (`gauche_…`, `droite_…`) : volume, page, rubrique, texte sans Markdown,
empans `sujet` / `description` / `adresse` (plusieurs empans de même classe
séparés par « | »), texte balisé et uuid.

`--candidates` ajoute les candidates non appariées à vérifier (deux entrées
sans correspondance, statut `candidate`) ; par défaut, l'export ne contient
que les liens retenus.

On n'apparie qu'entre rubriques appariées (`lib/section_alignment.py`) : un
lien entre rubriques qui ne se correspondent pas (Dedupe en produit) est
écarté, et compté en console.

Sortie par défaut : `<entrée sans .csv>.jointure.csv` à côté de l'entrée.
`--excel` : séparateur `;` et UTF-8 avec BOM, pour un tableur en français.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # accès à lib/ depuis tools/

from rich.console import Console

from align_directories_nw import Params
from lib.alignment import load_volume, read_links
from lib.alignment_review import DEFAULT_MARGIN, review
from lib.alignment_export import STATUS_LABELS, export_csv, export_encoding, natural_rows
from lib.alignment_patch import apply_patch, read_patch, resolve, validate
from lib.cli import Writes, add_apply_argument
from lib.curation import CurationConflict, print_conflict, refuse_orphans
from lib.section_alignment import SECTION_PATCH_SUFFIX, load_section_alignment, restrict_to_corresponding, section_orphans

ANNUAIRES_DIR = Path("annuaires")
PATCH_DIR = Path("data/alignement")

console = Console()


def pair_of(path: Path) -> str:
    """`<gauche>__<droite>` d'une sortie d'alignement (les noms de volumes n'ont pas de point)."""
    return path.name.split(".")[0]


def default_output(path: Path) -> Path:
    return path.with_name(f"{path.name.removesuffix('.csv')}.jointure.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporte en CSV la jointure lisible de deux annuaires alignés.")
    parser.add_argument("alignment", type=Path, help="Sortie d'alignement (ex. annuaires/alignements/<gauche>__<droite>.nw.csv).")
    parser.add_argument("-o", "--output", type=Path, help="CSV de sortie (défaut : <entrée>.jointure.csv).")
    parser.add_argument("--excel", action="store_true", help="Séparateur `;` et UTF-8 avec BOM (tableur en français).")
    parser.add_argument("--sans-patch", action="store_true", help="Ne pas appliquer le patch des corrections manuelles.")
    parser.add_argument("--candidates", action="store_true", help="Ajouter les candidates non appariées à vérifier (statut `candidate`).")
    parser.add_argument(
        "--ecart",
        type=float,
        default=DEFAULT_MARGIN,
        help=f"Écart de similarité sous lequel une concurrente fait signaler `homonyme proche` (défaut : {DEFAULT_MARGIN}).",
    )
    parser.add_argument("--force", action="store_true", help="Ignorer les lignes orphelines des patchs (sinon : arrêt sans rien écrire).")
    add_apply_argument(parser)
    args = parser.parse_args()
    writes = Writes(args.apply)

    pair_name = pair_of(args.alignment)
    left_name, separator, right_name = pair_name.partition("__")
    if not separator:
        parser.error(f"nom de fichier sans paire `<gauche>__<droite>` : {args.alignment.name}")
    left = {record.uuid: record for record in load_volume(ANNUAIRES_DIR / left_name)}
    right = {record.uuid: record for record in load_volume(ANNUAIRES_DIR / right_name)}
    links = read_links(args.alignment)

    declared: set[str] = set()
    if not args.sans_patch:
        patch_path = PATCH_DIR / f"{pair_name}.patch.csv"
        entries = read_patch(patch_path)
        validate(entries)
        resolution = resolve(entries, left, right)
        links, stats = apply_patch(links, resolution.entries)
        declared = {uuid for entry in resolution.entries if not entry.is_pair for _, uuid in entry.uuids()}
        console.print(f"Patch [yellow]{patch_path}[/yellow] : {len(entries)} ligne(s), {stats.manual_pairs} paire(s) manuelle(s).")
        if resolution.reanchored:
            console.print(f"[yellow]↻ {len(resolution.reanchored)} ligne(s) réancrée(s) par le texte (patch non réécrit).[/yellow]")
        if resolution.orphans:
            console.print(f"[bold red]⚠ {len(resolution.orphans)} ligne(s) orpheline(s) du patch, non appliquée(s).[/bold red]")
            if not args.force:
                print_conflict(console, CurationConflict(
                    [f"{entry.left_uuid or '—'} ↔ {entry.right_uuid or '—'}" for entry in resolution.orphans],
                    "corrigez le patch à la main, ou relancez avec --force pour ignorer ces lignes.",
                ))
                sys.exit(1)

    sections = load_section_alignment(
        list(left.values()), list(right.values()), PATCH_DIR / f"{pair_name}{SECTION_PATCH_SUFFIX}"
    )
    try:
        refuse_orphans(section_orphans(sections), args.force, "ligne orpheline du patch des rubriques")
    except CurationConflict as conflict:
        print_conflict(console, conflict)
        sys.exit(1)
    links, dropped = restrict_to_corresponding(links, left, right, sections)
    if dropped:
        console.print(f"[yellow]{len(dropped)} lien(s) entre rubriques non appariées écarté(s).[/yellow]")
    params = Params()
    found = review(
        links, list(left.values()), list(right.values()), sections, params.threshold, params.residual_threshold, params.subj_weight,
        args.ecart, declared,
    )
    rows, n_missing = natural_rows(links + (found.candidates if args.candidates else []), left, right)
    if n_missing:
        console.print(f"[bold red]⚠ {n_missing} lien(s) vers des entrées disparues, ignoré(s) : relancer l'alignement.[/bold red]")

    output = args.output or default_output(args.alignment)

    def write() -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(export_csv(rows, args.excel, found.reviews), encoding=export_encoding(args.excel), newline="")

    writes.add(output, write)
    counts = Counter(row.kind for row in rows)
    summary = ", ".join(f"{counts[kind]} {label}" for kind, label in STATUS_LABELS.items())
    console.print(f"{len(rows)} ligne(s) ({summary}).")
    levels = Counter(found.reviews[row.link.left_uuid, row.link.right_uuid].level for row in rows if row.link and (row.link.left_uuid, row.link.right_uuid) in found.reviews)
    console.print(
        f"À vérifier : {levels[1]} paire(s) d'incertitude moyenne, {levels[2]} d'incertitude forte"
        + ("" if args.candidates else f" ; {len(found.candidates)} candidate(s) non appariée(s) non exportée(s) (--candidates)")
        + "."
    )
    writes.finish(console)


if __name__ == "__main__":
    main()
