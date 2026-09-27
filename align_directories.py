"""Aligne les entrées de deux éditions d'un annuaire avec Dedupe.

    uv run align_directories.py annuaires/1807_AD75-PER292 annuaires/1808_AD75-PER292

Chaque annuaire est lu en entier (`lib/alignment.py` : toutes ses plages de
pages, dans l'ordre, depuis leurs `*.merged.ner.curated.csv`). Les ENTRY sont
comparées sur trois champs :

- `section` : la rubrique (titre `##`, à défaut `#`) ;
- `subj` : le texte des empans SUBJ, sans Markdown ;
- `text` : le texte complet de l'entrée, sans Markdown.

DESC et ADDR ne sont pas comparés séparément : ils varient d'une édition à
l'autre, même pour une même personne, et restent présents dans `text`.

Dedupe apprend ses poids sur des paires étiquetées. À la première exécution
(ou avec `--label`), l'étiquetage se fait en console (`y` : même entrée,
`n` : entrées différentes, `u` : incertain, `f` : terminer) ; les paires
sont enregistrées dans un fichier d'entraînement JSON (par défaut
`data/alignement/<gauche>__<droite>.training.json`, à versionner) que les
exécutions suivantes réutilisent sans interaction.

L'appariement est un-à-un : chaque entrée a au plus une correspondance dans
l'autre annuaire. Sortie : un CSV trié dans l'ordre de l'annuaire de gauche,
une ligne par correspondance :

- `left_file`, `left_uuid`, `right_uuid`, `right_file` : l'identification
  des deux entrées ;
- `score` : probabilité de correspondance estimée par Dedupe ;
- `left_section`, `right_section`, `left_tagged_text`, `right_tagged_text` :
  titre de la rubrique et texte balisé de chaque entrée, pour relire les
  paires à la main et dériver un `.curated.csv` validé (seules les colonnes
  d'identification font foi ; les autres sont un instantané pour la
  relecture).

Les entrées absentes du CSV n'ont pas de correspondance ;
`tools/display_alignment.py` les affiche.
"""

import argparse
import csv
import sys
from pathlib import Path

import dedupe
from rich.console import Console
from rich.table import Table

from lib.alignment import Record, dedupe_records, load_volume

console = Console()

DEFAULT_OUTPUT_DIR = Path("annuaires/alignements")
DEFAULT_TRAINING_DIR = Path("data/alignement")
OUTPUT_FIELDS = [
    "left_file",
    "left_uuid",
    "right_uuid",
    "right_file",
    "score",
    "left_section",
    "right_section",
    "left_tagged_text",
    "right_tagged_text",
]
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
        help=f"CSV des correspondances (défaut : {DEFAULT_OUTPUT_DIR}/<gauche>__<droite>.csv).",
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


def train(linker: dedupe.RecordLink, left: dict, right: dict, training_path: Path, label: bool) -> None:
    if training_path.exists():
        console.print(f"Paires d'entraînement lues dans [yellow]{training_path}[/yellow]")
        with training_path.open(encoding="utf-8") as handle:
            linker.prepare_training(left, right, training_file=handle)
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


def write_links(path: Path, links, left: dict[str, Record], right: dict[str, Record]) -> list[tuple[Record, Record, float]]:
    pairs = sorted(
        ((left[left_id], right[right_id], float(score)) for (left_id, right_id), score in links),
        key=lambda pair: pair[0].order,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(OUTPUT_FIELDS)
        for left_record, right_record, score in pairs:
            writer.writerow(
                [
                    left_record.document,
                    left_record.uuid,
                    right_record.uuid,
                    right_record.document,
                    f"{score:.4f}",
                    left_record.section_title,
                    right_record.section_title,
                    left_record.tagged_text or left_record.markdown.strip(),
                    right_record.tagged_text or right_record.markdown.strip(),
                ]
            )
    return pairs


def print_summary(left_name: str, right_name: str, n_left: int, n_right: int, scores: list[float]) -> None:
    matched = len(scores)
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
    output_path = args.output or DEFAULT_OUTPUT_DIR / f"{pair_name}.csv"
    training_path = args.training or DEFAULT_TRAINING_DIR / f"{pair_name}.training.json"

    try:
        left_records = {record.uuid: record for record in load_volume(args.left)}
        right_records = {record.uuid: record for record in load_volume(args.right)}
    except (OSError, csv.Error) as error:
        console.print(f"[bold red]Erreur de lecture :[/bold red] {error}")
        sys.exit(1)
    console.print(f"{args.left.name} : {len(left_records)} entrées ; {args.right.name} : {len(right_records)} entrées")

    left = dedupe_records(list(left_records.values()))
    right = dedupe_records(list(right_records.values()))
    linker = build_linker(left, right)
    train(linker, left, right, training_path, args.label)

    console.print("Appariement…")
    links = linker.join(left, right, threshold=args.threshold, constraint="one-to-one")
    pairs = write_links(output_path, links, left_records, right_records)

    console.print(f"\n[bold green]✅ Correspondances :[/bold green] [yellow]{output_path}[/yellow]")
    print_summary(args.left.name, args.right.name, len(left_records), len(right_records), [score for *_, score in pairs])


if __name__ == "__main__":
    main()
