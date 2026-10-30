"""Étape 7 · alignement de deux éditions d'un annuaire avec Dedupe (`RecordLink`, un-à-un).

    uv run numrev align dedupe annuaires/1807_AD75-PER292 annuaires/1808_AD75-PER292 --apply

Compare les ENTRY des deux annuaires (`numrev/alignment/records.py`) sur la
rubrique, le SUBJ et le texte complet (DESC et ADDR, trop variables d'une
édition à l'autre, n'y entrent que par le texte). La rubrique est comparée
par sa clé canonique : deux rubriques appariées (`numrev/alignment/sections.py`)
ont la même clé, y compris dans les paires d'entraînement, réécrites à la
lecture. Un lien entre rubriques qui ne se correspondent pas est toujours
écarté, qu'il vienne de Dedupe ou du patch des entrées.

Dedupe apprend sur des paires étiquetées en console (première exécution ou
`--label` ; exige `--apply`), enregistrées aussitôt dans
`data/alignment/<gauche>__<droite>.training.json` et réutilisées ensuite.
Sorties, dans l'ordre de l'annuaire de gauche : `<gauche>__<droite>.dedupe.csv`
(liens bruts) et `<gauche>__<droite>.csv` (liens + patch des entrées,
`numrev/alignment/patch.py`). `--patch-only` réapplique le patch à la sortie
brute existante, sans relancer Dedupe.
"""

import argparse
import io
from pathlib import Path

import dedupe

from numrev.alignment.pair import load_entry_patch, load_pair, pair_of_dirs, print_matched, print_scores, print_sections
from numrev.alignment.patch import PatchStats, apply_patch
from numrev.alignment.records import Link, Record, dedupe_records, read_links, write_links
from numrev.alignment.sections import SectionAlignment, canonical_keys, canonical_training_text, restrict_to_corresponding
from numrev.command import APPLY_FLAG, CommandError, Writes, add_apply_argument, add_force_argument, console, fine_tuning
from numrev.paths import DEDUPE_SUFFIX

DESCRIPTION = (
    "Met en correspondance (un-à-un) les ENTRY de deux annuaires complets avec Dedupe, "
    "sur la rubrique, le SUBJ et le texte de l'entrée, puis applique le patch des corrections manuelles."
)
SCORE_BOUNDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("left", type=Path, help="Dossier de l'annuaire de gauche (ex. annuaires/1807_AD75-PER292).")
    parser.add_argument("right", type=Path, help="Dossier de l'annuaire de droite (ex. annuaires/1808_AD75-PER292).")
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
    add_force_argument(parser, "Ignore les lignes orphelines des patchs (sinon : arrêt sans rien écrire).")
    fine_tuning(parser).add_argument(
        "--threshold", type=float, default=0.5, help="Score minimal d'une correspondance retenue (défaut : 0.5)."
    )
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
        console.print("[bold]Étiquetage :[/bold] y = même entrée, n = différentes, u = incertain, f = terminer, p = annuler la précédente.")
        dedupe.console_label(linker)

        def write() -> None:
            with training_path.open("w", encoding="utf-8") as handle:
                linker.write_training(handle)

        writes.add(training_path, write, now=True)  # travail humain : enregistré même si Dedupe échoue ensuite

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
    rubriques (`numrev/alignment/sections.py`), commune aux deux annuaires."""
    left_keys, right_keys = canonical_keys(sections)
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


def print_patch_summary(patch_path: Path, lines: int, stats: PatchStats) -> None:
    if not lines:
        console.print(f"Aucune correction manuelle ([yellow]{patch_path}[/yellow] absent ou vide).")
        return
    console.print(
        f"Patch [yellow]{patch_path}[/yellow] : {lines} ligne(s) — {stats.manual_pairs} paire(s) manuelle(s), "
        f"{stats.unmatched_left} gauche et {stats.unmatched_right} droite sans correspondance, "
        f"{stats.overridden} lien(s) Dedupe écarté(s)."
    )


def run(args: argparse.Namespace) -> None:
    pair = pair_of_dirs(args.left, args.right)
    output_path = pair.output()
    raw_path = pair.output(DEDUPE_SUFFIX)
    writes = Writes(args.apply)
    if not args.patch_only and needs_labelling(pair.training, args.label) and not args.apply:
        raise CommandError(
            f"l'étiquetage en console écrit {pair.training} : relancez avec {APPLY_FLAG} (sinon les paires étiquetées seraient perdues)."
        )

    loaded = load_pair(pair, force=args.force, writes=writes)
    left_records, right_records, sections = loaded.left, loaded.right, loaded.sections
    entries, resolution = load_entry_patch(loaded, force=args.force, writes=writes)

    if args.patch_only:
        raw_links = read_raw_links(raw_path, left_records, right_records)
    else:
        print_sections(sections)
        raw_links = infer_links(left_records, right_records, args, pair.training, sections, writes)
    raw_links, dropped = restrict_to_corresponding(raw_links, left_records, right_records, sections)
    if not args.patch_only:
        writes.add(raw_path, lambda: write_links(raw_path, raw_links, left_records, right_records))
    links, stats = apply_patch(raw_links, resolution.entries)
    links, manual_dropped = restrict_to_corresponding(links, left_records, right_records, sections)
    writes.add(output_path, lambda: write_links(output_path, links, left_records, right_records))

    if dropped:
        console.print(f"[yellow]{len(dropped)} lien(s) Dedupe entre rubriques non appariées écarté(s).[/yellow]")
    for link in manual_dropped:
        left_record, right_record = left_records[link.left_uuid], right_records[link.right_uuid]
        console.print(
            f"⚠ Paire du patch entre rubriques non appariées, non appliquée (lier les rubriques dans {pair.section_patch}) : "
            f"{left_record.section_title} / {left_record.text} ↔ {right_record.section_title} / {right_record.text}",
            markup=False,
        )
    print_patch_summary(pair.entry_patch, len(entries), stats)
    print_matched(pair.left, pair.right, len(left_records), len(right_records), len(links))
    print_scores(links, SCORE_BOUNDS)
    writes.finish(console)
