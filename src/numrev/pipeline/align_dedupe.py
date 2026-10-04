"""Étape 6 · alignement de deux éditions d'un annuaire avec Dedupe.

    uv run numrev align dedupe annuaires/1807_AD75-PER292 annuaires/1808_AD75-PER292 --apply

Chaque annuaire est lu en entier (`numrev/alignment/records.py` : toutes ses plages
de pages, dans l'ordre, depuis leurs `*.ner.csv`). Les ENTRY sont
comparées sur trois champs :

- `section` : la rubrique (titre `##`, à défaut `#`) ;
- `subj` : le texte des empans SUBJ, sans Markdown ;
- `text` : le texte complet de l'entrée, sans Markdown.

La rubrique est comparée par sa **clé canonique** : les rubriques des deux
annuaires qui se correspondent (`numrev/alignment/sections.py`, alignement
partagé avec `numrev align nw` et corrigé par le patch des rubriques
`data/alignment/<gauche>__<droite>.sections.csv`) reçoivent la même clé, y
compris dans les paires du fichier d'entraînement, réécrites à la lecture.
Variante d'origine, gardée pour comparaison sur un gold : `--raw-sections`
compare la clé propre à chaque annuaire (sorties par défaut
`<gauche>__<droite>.raw-sections.csv` et `….raw-sections.dedupe.csv`,
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
`data/alignment/<gauche>__<droite>.training.json`, à versionner) que les
exécutions suivantes réutilisent sans interaction.

L'appariement est un-à-un : chaque entrée a au plus une correspondance dans
l'autre annuaire. Deux sorties, triées dans l'ordre de l'annuaire de gauche :

- `<gauche>__<droite>.dedupe.csv` : les correspondances brutes de Dedupe ;
- `<gauche>__<droite>.csv` : les mêmes après application du **patch** de
  corrections manuelles (`data/alignment/<gauche>__<droite>.patch.csv`,
  voir `numrev/alignment/patch.py`), le résultat final.

Colonnes : `left_file`, `left_uuid`, `right_uuid`, `right_file` (identification
des deux entrées), `score` (probabilité estimée par Dedupe, vide pour une paire
saisie à la main), `source` (`dedupe` / `manuel`), puis un instantané pour la
relecture : `left_section`, `right_section`, `left_tagged_text`,
`right_tagged_text`.

Le patch s'édite à la main ; `numrev view alignment` affiche le
résultat et copie uuid ou lignes de patch prêtes à coller. `--patch-only`
réapplique le patch à la sortie Dedupe existante, sans relancer Dedupe.

Simulation par défaut (`numrev/command.py`) : `--apply` écrit les sorties, le
fichier d'entraînement et les patchs réancrés. L'étiquetage en console
exige `--apply`, pour ne pas perdre les paires étiquetées. Les
entrées absentes du CSV final n'ont pas de correspondance.
"""

import argparse
import csv
import io
from pathlib import Path

import dedupe
from rich.table import Table

from numrev.alignment.pair import load_pair, pair_of_dirs, print_matched, print_sections
from numrev.alignment.patch import PatchStats, Resolution, apply_patch, load_patch, orphan_descriptions, updated_patch, write_patch
from numrev.alignment.records import Link, Record, dedupe_records, read_links, write_links
from numrev.alignment.sections import SectionAlignment, canonical_keys, canonical_training_text, restrict_to_corresponding
from numrev.command import APPLY_FLAG, CommandError, Writes, add_apply_argument, add_force_argument, console
from numrev.curation import refuse_orphans
from numrev.paths import RAW_SECTIONS, raw_alignment

DESCRIPTION = (
    "Met en correspondance (un-à-un) les ENTRY de deux annuaires complets avec Dedupe, "
    "sur la rubrique, le SUBJ et le texte de l'entrée, puis applique le patch des corrections manuelles."
)
SCORE_BINS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("left", type=Path, help="Dossier de l'annuaire de gauche (ex. annuaires/1807_AD75-PER292).")
    parser.add_argument("right", type=Path, help="Dossier de l'annuaire de droite (ex. annuaires/1808_AD75-PER292).")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "CSV final des correspondances (défaut : annuaires/alignments/<gauche>__<droite>.csv) ; "
            "la sortie brute de Dedupe est écrite à côté, en .dedupe.csv."
        ),
    )
    parser.add_argument(
        "--patch-only",
        action="store_true",
        help="Ne pas relancer Dedupe : réappliquer le patch à la sortie .dedupe.csv existante.",
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
            f"(variante d'origine, pour comparaison ; défaut de sortie : annuaires/alignments/<gauche>__<droite>{RAW_SECTIONS}.csv)."
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Score minimal d'une correspondance retenue (défaut : 0.5).")
    add_force_argument(parser, "Ignore les lignes orphelines des patchs (sinon : arrêt sans rien écrire).")
    add_apply_argument(parser)


def build_linker(left: dict, right: dict) -> dedupe.RecordLink:
    corpus = [record["text"] for data in (left, right) for record in data.values()]
    variables = [
        dedupe.variables.String("section", has_missing=True),
        dedupe.variables.String("subj", has_missing=True),
        dedupe.variables.Text("text", corpus=corpus),
    ]
    return dedupe.RecordLink(variables)


def needs_labelling(training_path: Path, label: bool) -> bool:
    return label or not training_path.exists()


def train(
    linker: dedupe.RecordLink,
    left: dict,
    right: dict,
    training_path: Path,
    label: bool,
    section_keys: tuple[dict, dict],
    writes: Writes,
) -> None:
    if training_path.exists():
        console.print(f"Paires d'entraînement lues dans [yellow]{training_path}[/yellow]")
        # Rubriques des paires enregistrées ramenées à la clé canonique, comme les données.
        text = canonical_training_text(training_path.read_text(encoding="utf-8"), *section_keys)
        linker.prepare_training(left, right, training_file=io.StringIO(text))
    else:
        linker.prepare_training(left, right)

    if needs_labelling(training_path, label):
        console.print(
            "[bold]Étiquetage :[/bold] y = même entrée, n = différentes, u = incertain, " "f = terminer, p = annuler la précédente."
        )
        dedupe.console_label(linker)

        def write() -> None:
            with training_path.open("w", encoding="utf-8") as handle:
                linker.write_training(handle)

        writes.add(training_path, write)

    console.print("Entraînement…")
    linker.train()


def infer_links(
    left_records: dict[str, Record],
    right_records: dict[str, Record],
    args: argparse.Namespace,
    training_path: Path,
    sections: SectionAlignment,
    writes: Writes,
) -> list[Link]:
    """Dedupe compare la rubrique par sa clé canonique : celle du groupe de
    rubriques (`numrev/alignment/sections.py`), commune aux deux annuaires ;
    avec `--raw-sections`, par sa clé propre à chaque annuaire (variante
    d'origine, gardée pour comparaison)."""
    left_keys, right_keys = ({}, {}) if args.raw_sections else canonical_keys(sections)
    left = dedupe_records(list(left_records.values()), left_keys)
    right = dedupe_records(list(right_records.values()), right_keys)
    linker = build_linker(left, right)
    train(linker, left, right, training_path, args.label, (left_keys, right_keys), writes)
    console.print("Appariement…")
    links = linker.join(left, right, threshold=args.threshold, constraint="one-to-one")
    return [Link(left_id, right_id, float(score)) for (left_id, right_id), score in links]


def read_raw_links(raw_path: Path, left_records: dict[str, Record], right_records: dict[str, Record]) -> list[Link]:
    """Sortie brute existante (`--patch-only`), dont toutes les entrées doivent exister encore."""
    if not raw_path.exists():
        raise CommandError(f"'{raw_path}' introuvable : lancer d'abord sans --patch-only.")
    links = read_links(raw_path)
    unknown = [link for link in links if link.left_uuid not in left_records or link.right_uuid not in right_records]
    if unknown:
        raise CommandError(
            f"{len(unknown)} lien(s) de '{raw_path}' désignent des entrées disparues "
            "(étapes amont modifiées) : relancer Dedupe sans --patch-only."
        )
    return links


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
        console.print(f"[yellow]↻ {len(resolution.reanchored)} ligne(s) réancrée(s) par le texte (patch à réécrire).[/yellow]")
    if resolution.orphans:
        console.print(f"[bold red]⚠ {len(resolution.orphans)} ligne(s) orpheline(s), non appliquée(s) :[/bold red]")
        for description in orphan_descriptions(resolution):
            console.print(f"  {description}", markup=False)


def print_scores(links: list[Link]) -> None:
    scores = [link.score for link in links if link.score is not None]
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


def run(args: argparse.Namespace) -> None:
    pair = pair_of_dirs(args.left, args.right)
    output_path = args.output or pair.output(f"{RAW_SECTIONS}.csv" if args.raw_sections else ".csv")
    raw_path = raw_alignment(output_path)
    writes = Writes(args.apply)
    if not args.patch_only and needs_labelling(pair.training, args.label) and not args.apply:
        raise CommandError(
            f"l'étiquetage en console écrit {pair.training} : relancez avec {APPLY_FLAG} " "(sinon les paires étiquetées seraient perdues)."
        )

    loaded = load_pair(pair, force=args.force, writes=writes)
    left_records, right_records, sections = loaded.left, loaded.right, loaded.sections
    try:
        entries, resolution = load_patch(pair.entry_patch, left_records, right_records)
    except (OSError, csv.Error, ValueError) as error:
        raise CommandError(f"patch {pair.entry_patch} : {error}") from error
    refuse_orphans(orphan_descriptions(resolution), args.force, "ligne orpheline du patch")

    if args.patch_only:
        links, dropped = restrict_to_corresponding(
            read_raw_links(raw_path, left_records, right_records), left_records, right_records, sections
        )
    else:
        print_sections(sections)
        links, dropped = restrict_to_corresponding(
            infer_links(left_records, right_records, args, pair.training, sections, writes), left_records, right_records, sections
        )
        raw_links = links
        writes.add(raw_path, lambda: write_links(raw_path, raw_links, left_records, right_records))

    if resolution.reanchored:
        writes.add(pair.entry_patch, lambda: write_patch(pair.entry_patch, updated_patch(entries, resolution)))
    links, stats = apply_patch(links, resolution.entries)
    links, manual_dropped = restrict_to_corresponding(links, left_records, right_records, sections)
    final_links = links
    writes.add(output_path, lambda: write_links(output_path, final_links, left_records, right_records))

    if dropped:
        console.print(f"[yellow]{len(dropped)} lien(s) Dedupe entre rubriques non appariées écarté(s).[/yellow]")
    for link in manual_dropped:
        left_record, right_record = left_records[link.left_uuid], right_records[link.right_uuid]
        console.print(
            f"⚠ Paire du patch entre rubriques non appariées, non appliquée (lier les rubriques dans {pair.section_patch}) : "
            f"{left_record.section_title} / {left_record.text} ↔ {right_record.section_title} / {right_record.text}",
            markup=False,
        )
    print_patch_summary(pair.entry_patch, resolution, stats)
    print_matched(pair.left, pair.right, len(left_records), len(right_records), len(links))
    print_scores(links)
    writes.finish(console)
