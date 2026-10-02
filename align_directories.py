"""Aligne les entrées de deux éditions d'un annuaire avec Dedupe.

    uv run align_directories.py annuaires/1807_AD75-PER292 annuaires/1808_AD75-PER292

Chaque annuaire est lu en entier (`lib/alignment.py` : toutes ses plages de
pages, dans l'ordre, depuis leurs `*.merged.ner.curated.csv`). Les ENTRY sont
comparées sur trois champs :

- `section` : la rubrique (titre `##`, à défaut `#`) ;
- `subj` : le texte des empans SUBJ, sans Markdown ;
- `text` : le texte complet de l'entrée, sans Markdown.

La rubrique est comparée par sa **clé canonique** : les rubriques des deux
annuaires qui se correspondent (`lib/section_alignment.py`, alignement
partagé avec `align_directories_nw.py` et corrigé par le patch des rubriques
`data/alignement/<gauche>__<droite>.sections.csv`) reçoivent la même clé, y
compris dans les paires du fichier d'entraînement, réécrites à la lecture.
Variante d'origine, gardée pour comparaison sur un gold : `--raw-sections`
compare la clé propre à chaque annuaire (sorties par défaut
`<gauche>__<droite>.rubriques-brutes.csv` et `….rubriques-brutes.dedupe.csv`,
pour ne pas écraser la variante par défaut).

Dedupe ne compare la rubrique que comme un champ parmi d'autres : il peut
donc lier deux entrées de rubriques qui ne se correspondent pas. Ces liens
sont **toujours écartés** (`restrict_to_corresponding`), dans la sortie
brute comme dans la sortie finale, y compris les paires du patch des
entrées (lier d'abord les rubriques dans le patch des rubriques).

DESC et ADDR ne sont pas comparés séparément : ils varient d'une édition à
l'autre, même pour une même personne, et restent présents dans `text`.

Dedupe apprend ses poids sur des paires étiquetées. À la première exécution
(ou avec `--label`), l'étiquetage se fait en console (`y` : même entrée,
`n` : entrées différentes, `u` : incertain, `f` : terminer) ; les paires
sont enregistrées dans un fichier d'entraînement JSON (par défaut
`data/alignement/<gauche>__<droite>.training.json`, à versionner) que les
exécutions suivantes réutilisent sans interaction.

L'appariement est un-à-un : chaque entrée a au plus une correspondance dans
l'autre annuaire. Deux sorties, triées dans l'ordre de l'annuaire de gauche :

- `<gauche>__<droite>.dedupe.csv` : les correspondances brutes de Dedupe ;
- `<gauche>__<droite>.csv` : les mêmes après application du **patch** de
  corrections manuelles (`data/alignement/<gauche>__<droite>.patch.csv`,
  voir `lib/alignment_patch.py`), le résultat final.

Colonnes : `left_file`, `left_uuid`, `right_uuid`, `right_file` (identification
des deux entrées), `score` (probabilité estimée par Dedupe, vide pour une paire
saisie à la main), `source` (`dedupe` / `manuel`), puis un instantané pour la
relecture : `left_section`, `right_section`, `left_tagged_text`,
`right_tagged_text`.

Le patch s'édite à la main ; `tools/display_alignment.py` affiche le
résultat et copie uuid ou lignes de patch prêtes à coller. `--apply-only`
réapplique le patch à la sortie Dedupe existante, sans relancer Dedupe. Les
entrées absentes du CSV final n'ont pas de correspondance.
"""

import argparse
import csv
import io
import sys
from pathlib import Path

import dedupe
from rich.console import Console
from rich.table import Table

from lib.alignment import Link, Record, dedupe_records, load_volume, read_links, write_links
from lib.alignment_patch import PatchStats, Resolution, apply_patch, read_patch, resolve, updated_patch, validate, write_patch
from lib.section_alignment import (
    SECTION_PATCH_SUFFIX,
    SOURCE_AUTO,
    SectionAlignment,
    canonical_keys,
    canonical_training_text,
    load_section_alignment,
    restrict_to_corresponding,
)

console = Console()

DEFAULT_OUTPUT_DIR = Path("annuaires/alignements")
DEFAULT_TRAINING_DIR = Path("data/alignement")
RAW_SECTIONS_SUFFIX = ".rubriques-brutes.csv"  # sortie de la variante --raw-sections
SCORE_BINS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Met en correspondance (un-à-un) les ENTRY de deux annuaires complets "
            "avec Dedupe, sur la rubrique, le SUBJ et le texte de l'entrée."
        )
    )
    parser.add_argument("left", type=Path, help="Dossier de l'annuaire de gauche (ex. annuaires/1807_AD75-PER292).")
    parser.add_argument("right", type=Path, help="Dossier de l'annuaire de droite (ex. annuaires/1808_AD75-PER292).")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            f"CSV final des correspondances (défaut : {DEFAULT_OUTPUT_DIR}/<gauche>__<droite>.csv) ; "
            "la sortie brute de Dedupe est écrite à côté, en .dedupe.csv."
        ),
    )
    parser.add_argument(
        "--patch",
        type=Path,
        default=None,
        help=f"Corrections manuelles (défaut : {DEFAULT_TRAINING_DIR}/<gauche>__<droite>.patch.csv).",
    )
    parser.add_argument(
        "--section-patch",
        type=Path,
        default=None,
        help=f"Patch des rubriques (défaut : {DEFAULT_TRAINING_DIR}/<gauche>__<droite>{SECTION_PATCH_SUFFIX}).",
    )
    parser.add_argument(
        "--apply-only",
        action="store_true",
        help="Ne pas relancer Dedupe : réappliquer le patch à la sortie .dedupe.csv existante.",
    )
    parser.add_argument(
        "--training",
        type=Path,
        default=None,
        help=f"Fichier d'entraînement Dedupe (défaut : {DEFAULT_TRAINING_DIR}/<gauche>__<droite>.training.json).",
    )
    parser.add_argument(
        "--label",
        action="store_true",
        help="Étiqueter (ou compléter) les paires en console même si le fichier d'entraînement existe.",
    )
    parser.add_argument(
        "--raw-sections",
        action="store_true",
        help=(
            "Comparer la rubrique par sa clé propre à chaque annuaire plutôt que par la clé commune de son groupe "
            f"(variante d'origine, pour comparaison ; défaut de sortie : {DEFAULT_OUTPUT_DIR}/<gauche>__<droite>"
            f"{RAW_SECTIONS_SUFFIX})."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score minimal d'une correspondance retenue (défaut : 0.5).",
    )
    return parser.parse_args()


def build_linker(left: dict, right: dict) -> dedupe.RecordLink:
    corpus = [record["text"] for data in (left, right) for record in data.values()]
    variables = [
        dedupe.variables.String("section", has_missing=True),
        dedupe.variables.String("subj", has_missing=True),
        dedupe.variables.Text("text", corpus=corpus),
    ]
    return dedupe.RecordLink(variables)


def train(
    linker: dedupe.RecordLink, left: dict, right: dict, training_path: Path, label: bool, section_keys: tuple[dict, dict]
) -> None:
    if training_path.exists():
        console.print(f"Paires d'entraînement lues dans [yellow]{training_path}[/yellow]")
        # Rubriques des paires enregistrées ramenées à la clé canonique, comme les données.
        text = canonical_training_text(training_path.read_text(encoding="utf-8"), *section_keys)
        linker.prepare_training(left, right, training_file=io.StringIO(text))
    else:
        linker.prepare_training(left, right)

    if label or not training_path.exists():
        console.print(
            "[bold]Étiquetage :[/bold] y = même entrée, n = différentes, u = incertain, "
            "f = terminer, p = annuler la précédente."
        )
        dedupe.console_label(linker)
        training_path.parent.mkdir(parents=True, exist_ok=True)
        with training_path.open("w", encoding="utf-8") as handle:
            linker.write_training(handle)
        console.print(f"[bold green]💾 Paires d'entraînement :[/bold green] [yellow]{training_path}[/yellow]")

    console.print("Entraînement…")
    linker.train()


def dedupe_path(output_path: Path) -> Path:
    return output_path.with_suffix(".dedupe.csv")


def infer_links(
    left_records: dict[str, Record], right_records: dict[str, Record], args, training_path: Path, sections: SectionAlignment
) -> list[Link]:
    """Dedupe compare la rubrique par sa clé canonique : celle du groupe de
    rubriques (`lib/section_alignment.py`), commune aux deux annuaires ; avec
    `--raw-sections`, par sa clé propre à chaque annuaire (variante
    d'origine, gardée pour comparaison)."""
    left_keys, right_keys = ({}, {}) if args.raw_sections else canonical_keys(sections)
    left = dedupe_records(list(left_records.values()), left_keys)
    right = dedupe_records(list(right_records.values()), right_keys)
    linker = build_linker(left, right)
    train(linker, left, right, training_path, args.label, (left_keys, right_keys))
    console.print("Appariement…")
    links = linker.join(left, right, threshold=args.threshold, constraint="one-to-one")
    return [Link(left_id, right_id, float(score)) for (left_id, right_id), score in links]


def patch_links(
    links: list[Link], patch_path: Path, left_records: dict[str, Record], right_records: dict[str, Record]
) -> tuple[list[Link], Resolution, PatchStats]:
    """Applique le patch ; le réécrit si des lignes ont été réancrées."""
    entries = read_patch(patch_path)
    validate(entries)
    resolution = resolve(entries, left_records, right_records)
    if resolution.reanchored:
        write_patch(patch_path, updated_patch(entries, resolution))
    patched, stats = apply_patch(links, resolution.entries)
    return patched, resolution, stats


def print_patch_summary(patch_path: Path, resolution: Resolution, stats: PatchStats) -> None:
    n_entries = len(resolution.entries) + len(resolution.orphans)
    if not n_entries:
        console.print(f"Aucune correction manuelle ([yellow]{patch_path}[/yellow] absent ou vide).")
        return
    console.print(
        f"Patch [yellow]{patch_path}[/yellow] : {n_entries} ligne(s) — {stats.manual_pairs} paire(s) manuelle(s), "
        f"{stats.unmatched_left} gauche et {stats.unmatched_right} droite sans correspondance, "
        f"{stats.overridden} lien(s) Dedupe écarté(s)."
    )
    if resolution.reanchored:
        console.print(f"[yellow]↻ {len(resolution.reanchored)} ligne(s) réancrée(s) par le texte (patch mis à jour).[/yellow]")
    if resolution.orphans:
        console.print(f"[bold red]⚠ {len(resolution.orphans)} ligne(s) orpheline(s), non appliquée(s) :[/bold red]")
        for entry in resolution.orphans:
            for side in ("left", "right"):
                if getattr(entry, f"{side}_uuid"):
                    console.print(
                        f"  {side} {getattr(entry, f'{side}_uuid')} · {getattr(entry, f'{side}_section')} · "
                        f"{getattr(entry, f'{side}_tagged_text')}",
                        markup=False,
                    )


def print_sections(sections: SectionAlignment, path: Path) -> None:
    renamed = sum(
        group.source != SOURCE_AUTO or group.left[0].key != group.right[0].key or len(group.left) + len(group.right) > 2
        for group in sections.groups
    )
    manual = sum(group.source != SOURCE_AUTO for group in sections.groups)
    console.print(
        f"Rubriques : {len(sections.groups)} groupe(s) (dont {manual} du patch [yellow]{path}[/yellow]), "
        f"{renamed} ramené(s) à une clé commune, {len(sections.unmatched_left)} seule(s) à gauche, "
        f"{len(sections.unmatched_right)} seule(s) à droite."
    )
    if sections.resolution.reanchored:
        console.print(f"[yellow]↻ {len(sections.resolution.reanchored)} ligne(s) du patch des rubriques réancrée(s).[/yellow]")
    for entry in sections.resolution.orphans:
        console.print(f"⚠ Ligne orpheline du patch des rubriques, non appliquée : {entry.left_title} ↔ {entry.right_title}", markup=False)


def print_summary(left_name: str, right_name: str, n_left: int, n_right: int, links: list[Link]) -> None:
    matched = len(links)
    scores = [link.score for link in links if link.score is not None]
    table = Table(title="Correspondances", show_header=True)
    table.add_column("Annuaire")
    table.add_column("Entrées", justify="right")
    table.add_column("Appariées", justify="right")
    table.add_column("Non appariées", justify="right")
    for name, total in ((left_name, n_left), (right_name, n_right)):
        table.add_row(name, str(total), f"{matched} ({matched / total:.1%})", str(total - matched))
    console.print(table)

    bins = Table(title="Scores des correspondances", show_header=True)
    bins.add_column("Score")
    bins.add_column("Paires", justify="right")
    edges = [*SCORE_BINS, float("inf")]
    for low, high in zip(edges, edges[1:]):
        label = f"[{low:.2f} ; {high:.2f}[" if high != float("inf") else f"≥ {low:.2f}"
        bins.add_row(label, str(sum(low <= score < high for score in scores)))
    below = sum(score < SCORE_BINS[0] for score in scores)
    if below:
        bins.add_row(f"< {SCORE_BINS[0]:.2f}", str(below))
    console.print(bins)


def main() -> None:
    args = parse_args()
    for directory in (args.left, args.right):
        if not directory.is_dir():
            console.print(f"[bold red]Erreur :[/bold red] dossier '{directory}' introuvable.")
            sys.exit(1)

    pair_name = f"{args.left.name}__{args.right.name}"
    output_path = args.output or DEFAULT_OUTPUT_DIR / f"{pair_name}{RAW_SECTIONS_SUFFIX if args.raw_sections else '.csv'}"
    training_path = args.training or DEFAULT_TRAINING_DIR / f"{pair_name}.training.json"
    patch_path = args.patch or DEFAULT_TRAINING_DIR / f"{pair_name}.patch.csv"
    section_patch_path = args.section_patch or DEFAULT_TRAINING_DIR / f"{pair_name}{SECTION_PATCH_SUFFIX}"

    try:
        left_records = {record.uuid: record for record in load_volume(args.left)}
        right_records = {record.uuid: record for record in load_volume(args.right)}
    except (OSError, csv.Error) as error:
        console.print(f"[bold red]Erreur de lecture :[/bold red] {error}")
        sys.exit(1)
    console.print(f"{args.left.name} : {len(left_records)} entrées ; {args.right.name} : {len(right_records)} entrées")

    try:
        sections = load_section_alignment(list(left_records.values()), list(right_records.values()), section_patch_path)
    except (OSError, csv.Error, ValueError) as error:
        console.print(f"[bold red]Erreur dans le patch des rubriques :[/bold red] {error}")
        sys.exit(1)

    raw_path = dedupe_path(output_path)
    if args.apply_only:
        if not raw_path.exists():
            console.print(f"[bold red]Erreur :[/bold red] '{raw_path}' introuvable : lancer d'abord sans --apply-only.")
            sys.exit(1)
        links = read_links(raw_path)
        unknown = [link for link in links if link.left_uuid not in left_records or link.right_uuid not in right_records]
        if unknown:
            console.print(
                f"[bold red]Erreur :[/bold red] {len(unknown)} lien(s) de '{raw_path}' désignent des entrées disparues "
                "(étapes amont modifiées) : relancer Dedupe sans --apply-only."
            )
            sys.exit(1)
        links, dropped = restrict_to_corresponding(links, left_records, right_records, sections)
    else:
        print_sections(sections, section_patch_path)
        links, dropped = restrict_to_corresponding(
            infer_links(left_records, right_records, args, training_path, sections), left_records, right_records, sections
        )
        write_links(raw_path, links, left_records, right_records)
        console.print(f"[bold green]✅ Sortie Dedupe :[/bold green] [yellow]{raw_path}[/yellow]")

    try:
        links, resolution, stats = patch_links(links, patch_path, left_records, right_records)
    except (OSError, csv.Error, ValueError) as error:
        console.print(f"[bold red]Erreur dans le patch :[/bold red] {error}")
        sys.exit(1)
    links, manual_dropped = restrict_to_corresponding(links, left_records, right_records, sections)
    write_links(output_path, links, left_records, right_records)

    if dropped:
        console.print(f"[yellow]{len(dropped)} lien(s) Dedupe entre rubriques non appariées écarté(s).[/yellow]")
    for link in manual_dropped:
        left_record, right_record = left_records[link.left_uuid], right_records[link.right_uuid]
        console.print(
            f"⚠ Paire du patch entre rubriques non appariées, non appliquée (lier les rubriques dans {section_patch_path}) : "
            f"{left_record.section_title} / {left_record.text} ↔ {right_record.section_title} / {right_record.text}",
            markup=False,
        )
    console.print(f"\n[bold green]✅ Correspondances :[/bold green] [yellow]{output_path}[/yellow]")
    print_patch_summary(patch_path, resolution, stats)
    print_summary(args.left.name, args.right.name, len(left_records), len(right_records), links)

if __name__ == "__main__":
    main()
