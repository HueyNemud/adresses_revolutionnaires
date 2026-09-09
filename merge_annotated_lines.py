"""Fusionne les prédictions CRF par blocs logiques depuis un CSV.

Script conçu pour traiter les sorties de annotate_lines_crf.py (par ex.
bpt6k62915570-114_326-ocr.lines.annotated.csv) en fusionnant les lignes
se terminant par _INSIDE avec leur _BEGIN parent pour former une ligne de
classe unifiée.
"""

import argparse
import csv
from pathlib import Path
from typing import Any, Iterator

from rich.console import Console

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fusionne les lignes de classe X_INSIDE avec la ligne X_BEGIN précédente."
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
        default=Path("merged_predictions.csv"),
        help="Chemin du fichier CSV de sortie.",
    )
    parser.add_argument(
        "-s",
        "--separator",
        type=str,
        default=" ",
        help="Séparateur utilisé pour concaténer la colonne 'markdown' (défaut : espace).",
    )
    parser.add_argument(
        "--class-column",
        type=str,
        default="prediction",
        help="Nom de la colonne contenant la classe prédite (défaut : 'prediction').",
    )
    return parser.parse_args()


def process_csv(
    input_path: Path, output_path: Path, separator: str, class_col: str
) -> None:
    """Lit le CSV, fusionne les blocs _INSIDE dans _BEGIN, et exporte le résultat."""
    with open(input_path, "r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            raise ValueError("Le fichier CSV d'entrée est vide ou invalide.")

        fieldnames = list(reader.fieldnames)

        if class_col not in fieldnames:
            raise ValueError(
                f"La colonne de classe '{class_col}' est introuvable dans le CSV."
            )
        if "markdown" not in fieldnames:
            console.print(
                "[yellow]Avertissement : la colonne 'markdown' n'a pas été trouvée.[/yellow]"
            )

        with open(output_path, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()

            active_merge: dict[str, str] | None = None

            def flush_merge() -> None:
                if active_merge is not None:
                    writer.writerow(active_merge)

            for row in reader:
                pred = row.get(class_col, "").strip()

                if pred.endswith("_BEGIN"):
                    flush_merge()
                    active_merge = row.copy()
                    active_merge[class_col] = pred[: -len("_BEGIN")]

                elif pred.endswith("_INSIDE"):
                    base_class = pred[: -len("_INSIDE")]

                    if (
                        active_merge is not None
                        and active_merge.get(class_col) == base_class
                    ):
                        # Fusion avec la ligne parente existante
                        for col in fieldnames:
                            if col == class_col:
                                continue
                            elif col == "markdown":
                                active_merge[col] = (
                                    f"{active_merge[col]}{separator}{row.get(col, '')}"
                                )
                            else:
                                active_merge[col] = (
                                    f"{active_merge.get(col, '')},{row.get(col, '')}"
                                )
                    else:
                        # Fallback de sécurité : traité comme un début de bloc s'il n'y a pas de parent
                        flush_merge()
                        active_merge = row.copy()
                        active_merge[class_col] = base_class

                else:
                    # Lignes OUT_OF_SCOPE, TITLE ou autres classes singulières
                    flush_merge()
                    active_merge = None
                    writer.writerow(row)

            flush_merge()


def main() -> None:
    args = parse_args()

    if not args.input_csv.exists():
        console.print(
            f"[bold red]Erreur :[/bold red] Fichier '{args.input_csv}' introuvable."
        )
        return

    try:
        process_csv(args.input_csv, args.output, args.separator, args.class_column)
        console.print(
            "\n[bold green]✅ Fusion CSV réussie :[/bold green] "
            f"[yellow]{args.output}[/yellow]"
        )
    except Exception as error:
        console.print(f"[bold red]Erreur lors du traitement :[/bold red] {error}")


if __name__ == "__main__":
    main()
