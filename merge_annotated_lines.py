"""Fusionne les prédictions CRF par entités logiques depuis un CSV.

Script conçu pour traiter les sorties de annotate_lines_crf.py (par ex.
bpt6k62915570-114_326-ocr.lines.annotated.csv), en reconstituant trois types
d'entités finales à partir des classes ligne par ligne :

- ENTRY  : une B-ENTRY, suivie de ses éventuelles I-ENTRY et SUB-ENTRY.
- TITLE  : une B-TITLE, suivie de ses éventuelles I-TITLE.
- OUT OF SCOPE (et toute classe non reconnue) : jamais fusionnée, recopiée
  telle quelle.

Règles de fusion :
- I-ENTRY se fusionne (avec un espace) à la ligne ENTRY/SUB-ENTRY la plus
  proche qui la précède, quitte à « remonter » au-delà de lignes
  OUT OF SCOPE ou [B|I]-TITLE intercalées.
- SUB-ENTRY se fusionne (avec un saut de ligne) à la même ligne ENTRY la
  plus proche, avec la même tolérance de remontée.
- I-TITLE se fusionne (avec un espace) à la ligne TITLE la plus proche, avec
  la même tolérance de remontée (généralisation symétrique à la règle
  ENTRY : non explicitement spécifiée à l'origine, mais plus robuste).

En pratique, ceci est implémenté par deux « pistes » indépendantes (un
groupe ENTRY actif, un groupe TITLE actif) qui restent ouvertes tant
qu'aucune nouvelle racine (B-ENTRY / B-TITLE) n'apparaît : les lignes
OUT OF SCOPE ou de l'autre famille ne les referment pas, ce qui reproduit
exactement la « remontée » sans avoir à rescanner l'historique.

Un rapport d'analyse (.txt, à côté du CSV de sortie) recense les comptages
et les deux familles de problèmes détectés : les lignes de continuation
sans ancre valable, et les entrées qui rompent l'ordre alphabétique (réinit-
ialisé à chaque nouveau TITLE).
"""

import argparse
import csv
import re
import unicodedata
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

console = Console()

# Valeurs exactes d'`annotate_lines_crf.AnnotationLabel` : elles doivent
# rester synchronisées avec ce script si la convention change de ce côté-là.
LABEL_BEGIN_ENTRY = "B-ENTRY"
LABEL_INSIDE_ENTRY = "I-ENTRY"
LABEL_SUB_ENTRY = "SUB-ENTRY"
LABEL_BEGIN_TITLE = "B-TITLE"
LABEL_INSIDE_TITLE = "I-TITLE"
LABEL_OOS = (
    "OUT OF SCOPE"  # Note : PAS "OUT_OF_SCOPE" (avec espaces, pas d'underscores).
)

KNOWN_LABELS = {
    LABEL_BEGIN_ENTRY,
    LABEL_INSIDE_ENTRY,
    LABEL_SUB_ENTRY,
    LABEL_BEGIN_TITLE,
    LABEL_INSIDE_TITLE,
    LABEL_OOS,
}

NORMALIZED_ENTRY = "ENTRY"
NORMALIZED_TITLE = "TITLE"


@dataclass
class MergeReport:
    """Compteurs et listes de problèmes accumulés pendant la fusion."""

    rows_read: int = 0
    rows_written: int = 0
    final_entries: int = 0
    final_titles: int = 0
    final_oos: int = 0
    entry_lines_merged: int = 0
    subentry_lines_merged: int = 0
    title_lines_merged: int = 0
    orphan_subentries: list[str] = field(default_factory=list)
    orphan_entry_continuations: list[str] = field(default_factory=list)
    orphan_title_continuations: list[str] = field(default_factory=list)
    unknown_labels: list[tuple[str, str]] = field(default_factory=list)
    # (identifiant, clé fautive, clé précédente, extrait du texte)
    alpha_violations: list[tuple[str, str, str, str]] = field(default_factory=list)


def alpha_sort_key(text: str) -> str:
    """Clé de tri : mots jusqu'à la première virgule ou au premier point,
    concaténés en majuscules, sans accents, espaces ni ponctuation.
    """
    # Isole le segment situé avant la première virgule ou le premier point
    segment = re.split(r"[,.]", text, maxsplit=1)[0]

    normalized = unicodedata.normalize("NFKD", segment)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return "".join(char for char in ascii_text if char.isalnum()).upper()


def _new_root(row: dict[str, str], normalized_entity: str) -> dict[str, str]:
    root = row.copy()
    root["entity"] = normalized_entity
    return root


def _merge_into(
    group: dict[str, str],
    row: dict[str, str],
    separator: str,
    fieldnames: list[str],
) -> None:
    for col in fieldnames:
        if col == "markdown":
            group[col] = f"{group[col]}{separator}{row.get(col, '')}"
        else:
            group[col] = f"{group.get(col, '')},{row.get(col, '')}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fusionne les lignes ENTRY/TITLE en entités logiques, selon la "
            "convention BIO d'annotate_lines_crf.py, et produit un rapport "
            "d'analyse .txt."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Fichier CSV d'entrée (par ex. bpt6k62915570-114_326-ocr.lines.annotated.csv).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Chemin du CSV de sortie (défaut : <entrée>.merged.csv).",
    )
    parser.add_argument(
        "-r",
        "--report",
        type=Path,
        default=None,
        help="Chemin du rapport .txt (défaut : <sortie>.report.txt, dans le même dossier).",
    )
    parser.add_argument(
        "--class-column",
        type=str,
        default="prediction_curated",
        help="Nom de la colonne contenant la classe prédite (défaut : 'prediction_curated').",
    )
    parser.add_argument(
        "--uid-column",
        type=str,
        default="uid",
        help="Nom de la colonne identifiant chaque ligne (défaut : 'uid').",
    )
    return parser.parse_args()


def process_csv(
    input_path: Path,
    output_path: Path,
    class_col: str,
    uid_col: str,
) -> MergeReport:
    """Lit le CSV, fusionne les entités ENTRY/TITLE, exporte le résultat et
    retourne le rapport d'analyse correspondant."""
    report = MergeReport()

    with input_path.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            raise ValueError("Le fichier CSV d'entrée est vide ou invalide.")

        input_fieldnames = list(reader.fieldnames)

        if class_col not in input_fieldnames:
            raise ValueError(
                f"La colonne de classe '{class_col}' est introuvable dans le CSV."
            )
        if "markdown" not in input_fieldnames:
            console.print(
                "[yellow]Avertissement : la colonne 'markdown' n'a pas été trouvée.[/yellow]"
            )
        if uid_col not in input_fieldnames:
            console.print(
                f"[yellow]Avertissement : la colonne '{uid_col}' est introuvable ; "
                "les identifiants du rapport utiliseront le numéro de ligne lue.[/yellow]"
            )

        output_fieldnames = ["uuid"]
        for col in input_fieldnames:
            if col == "uuid":
                continue
            output_fieldnames.append(col)
            if col == "markdown":
                output_fieldnames.append("entity")

        if "entity" not in output_fieldnames:
            output_fieldnames.append("entity")

        with output_path.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=output_fieldnames)
            writer.writeheader()

            active_entry: dict[str, str] | None = None
            active_title: dict[str, str] | None = None
            last_alpha_key: str | None = None

            def identifier(row: dict[str, str]) -> str:
                return row.get(uid_col) or f"<ligne {report.rows_read}>"

            def register_entry_root(row: dict[str, str]) -> None:
                """Vérifie l'ordre alphabétique au moment où une entrée démarre
                (pas à son flush, différé jusqu'au B-ENTRY suivant) : c'est ce
                moment qui reflète l'ordre chronologique réel du document et
                respecte correctement la réinitialisation à chaque TITLE.
                """
                nonlocal last_alpha_key
                key = alpha_sort_key(row.get("markdown", ""))
                if not key:
                    return
                if last_alpha_key is not None and key < last_alpha_key:
                    report.alpha_violations.append(
                        (
                            identifier(row),
                            key,
                            last_alpha_key,
                            row.get("markdown", "")[:60],
                        )
                    )
                last_alpha_key = key

            def flush_entry() -> None:
                nonlocal active_entry
                if active_entry is None:
                    return
                active_entry["uuid"] = str(uuid.uuid4())
                writer.writerow(active_entry)
                report.rows_written += 1
                report.final_entries += 1
                active_entry = None

            def flush_title() -> None:
                nonlocal active_title
                if active_title is None:
                    return
                active_title["uuid"] = str(uuid.uuid4())
                writer.writerow(active_title)
                report.rows_written += 1
                report.final_titles += 1
                active_title = None

            for row in reader:
                report.rows_read += 1
                pred = row.get(class_col, "").strip()
                row_id = identifier(row)

                if pred == LABEL_BEGIN_ENTRY:
                    flush_entry()
                    register_entry_root(row)
                    active_entry = _new_root(row, NORMALIZED_ENTRY)

                elif pred == LABEL_INSIDE_ENTRY:
                    if active_entry is not None:
                        _merge_into(active_entry, row, " ", input_fieldnames)
                        report.entry_lines_merged += 1
                    else:
                        report.orphan_entry_continuations.append(row_id)
                        flush_entry()
                        register_entry_root(row)
                        active_entry = _new_root(row, NORMALIZED_ENTRY)

                elif pred == LABEL_SUB_ENTRY:
                    if active_entry is not None:
                        _merge_into(active_entry, row, "\n", input_fieldnames)
                        report.subentry_lines_merged += 1
                    else:
                        report.orphan_subentries.append(row_id)
                        flush_entry()
                        register_entry_root(row)
                        active_entry = _new_root(row, NORMALIZED_ENTRY)

                elif pred == LABEL_BEGIN_TITLE:
                    flush_title()
                    active_title = _new_root(row, NORMALIZED_TITLE)
                    last_alpha_key = (
                        None  # Nouvelle section : on réinitialise l'ordre alphabétique.
                    )

                elif pred == LABEL_INSIDE_TITLE:
                    if active_title is not None:
                        _merge_into(active_title, row, " ", input_fieldnames)
                        report.title_lines_merged += 1
                    else:
                        report.orphan_title_continuations.append(row_id)
                        flush_title()
                        active_title = _new_root(row, NORMALIZED_TITLE)

                else:
                    # OUT OF SCOPE, ou toute classe non reconnue : jamais fusionnée.
                    if pred != LABEL_OOS and pred != "":
                        report.unknown_labels.append((row_id, pred))
                    out_row = row.copy()
                    out_row["uuid"] = str(uuid.uuid4())
                    out_row["entity"] = LABEL_OOS
                    writer.writerow(out_row)
                    report.rows_written += 1
                    report.final_oos += 1

            flush_entry()
            flush_title()

    return report


def format_report(report: MergeReport, input_path: Path, output_path: Path) -> str:
    lines = [
        "RAPPORT DE FUSION — merge_annotated_lines.py",
        f"Entrée  : {input_path}",
        f"Sortie  : {output_path}",
        "",
        "== Comptages ==",
        f"Lignes lues                         : {report.rows_read}",
        f"Lignes écrites                      : {report.rows_written}",
        f"Entités finales ENTRY                : {report.final_entries}",
        f"Entités finales TITLE                : {report.final_titles}",
        f"Lignes OUT OF SCOPE (ou inconnues)   : {report.final_oos}",
        f"Lignes I-ENTRY fusionnées            : {report.entry_lines_merged}",
        f"Lignes SUB-ENTRY fusionnées          : {report.subentry_lines_merged}",
        f"Lignes I-TITLE fusionnées            : {report.title_lines_merged}",
        "",
        "== Cas problématiques ==",
    ]

    if report.orphan_subentries:
        lines.append(
            f"SUB-ENTRY sans [I|B]-ENTRY précédente ({len(report.orphan_subentries)}) "
            "— traitées comme une nouvelle ENTRY :"
        )
        lines.extend(f"  - {uid}" for uid in report.orphan_subentries)
    else:
        lines.append("SUB-ENTRY sans ancre précédente : aucune.")

    if report.orphan_entry_continuations:
        lines.append(
            f"I-ENTRY sans [B|I|SUB]-ENTRY précédente ({len(report.orphan_entry_continuations)}) "
            "— traitées comme une nouvelle ENTRY :"
        )
        lines.extend(f"  - {uid}" for uid in report.orphan_entry_continuations)
    else:
        lines.append("I-ENTRY sans ancre précédente : aucune.")

    if report.orphan_title_continuations:
        lines.append(
            f"I-TITLE sans [B|I]-TITLE précédente ({len(report.orphan_title_continuations)}) "
            "— traitées comme un nouveau TITLE :"
        )
        lines.extend(f"  - {uid}" for uid in report.orphan_title_continuations)
    else:
        lines.append("I-TITLE sans ancre précédente : aucune.")

    if report.unknown_labels:
        lines.append(f"Classes non reconnues ({len(report.unknown_labels)}) :")
        lines.extend(
            f"  - {uid} : classe '{label}'" for uid, label in report.unknown_labels
        )

    lines.append("")
    lines.append("== Ordre alphabétique (réinitialisé à chaque nouveau TITLE) ==")
    if report.alpha_violations:
        lines.append(
            f"Entrées rompant l'ordre alphabétique ({len(report.alpha_violations)}) :"
        )
        for uid, key, previous_key, excerpt in report.alpha_violations:
            lines.append(
                f'  - {uid} : "{key}" ("{excerpt}...") suit "{previous_key}", '
                "ordre alphabétique rompu."
            )
    else:
        lines.append("Aucune rupture d'ordre alphabétique détectée.")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    if not args.input_csv.exists():
        console.print(
            f"[bold red]Erreur :[/bold red] Fichier '{args.input_csv}' introuvable."
        )
        return

    output_path = args.output or args.input_csv.with_suffix(".merged.csv")
    report_path = args.report or output_path.with_suffix(".report.txt")

    try:
        report = process_csv(
            args.input_csv, output_path, args.class_column, args.uid_column
        )
    except (OSError, csv.Error, ValueError) as error:
        console.print(f"[bold red]Erreur lors du traitement :[/bold red] {error}")
        return

    report_path.write_text(
        format_report(report, args.input_csv, output_path), encoding="utf-8"
    )

    problem_count = (
        len(report.orphan_subentries)
        + len(report.orphan_entry_continuations)
        + len(report.orphan_title_continuations)
        + len(report.alpha_violations)
    )
    console.print(
        "\n[bold green]✅ Fusion CSV réussie :[/bold green] "
        f"[yellow]{output_path}[/yellow] "
        f"({report.rows_read} lignes lues → {report.rows_written} lignes écrites)"
    )
    console.print(
        f"[bold green]📄 Rapport :[/bold green] [yellow]{report_path}[/yellow]"
    )
    if problem_count:
        console.print(
            f"[yellow]⚠ {problem_count} cas à vérifier (voir le rapport pour le détail).[/yellow]"
        )


if __name__ == "__main__":
    main()
