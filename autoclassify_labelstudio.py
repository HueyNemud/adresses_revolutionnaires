"""Pré-annote des entrées d'annuaire depuis un CSV via Ollama et exporte un JSON
de prédictions NER importable dans Label Studio.

Chaque ligne du fichier d'entrée est segmentée et classée par le modèle en :
  - "SUBJ" : entité désignée (unique, obligatoire, premier segment) ;
  - "DESC" : descriptif d'activité (optionnel, récurrence libre) ;
  - "ADDR" : descriptif d'adresse (en fin d'entrée, optionnel).
"""

import argparse
import csv
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import ollama
from pydantic import BaseModel
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console()

DEFAULT_MODEL = "qwen3.8:27b"
DEFAULT_FROM_NAME = "label"
DEFAULT_TO_NAME = "text"
DEFAULT_TEXT_COLUMN = "markdown"

SEGMENT_COLORS = {"SUBJ": "cyan", "DESC": "yellow", "ADDR": "magenta"}

SYSTEM_PROMPT = """Tu es un système spécialisé en extraction d'information et NER sur des annuaires commerciaux anciens.

À partir de l'entrée d'annuaire fournie, sépare et classifie les segments de texte contigus :
1. Entité désignée : classe "SUBJ" (unique, obligatoire, premier segment)
2. Descriptif d'activité : classe "DESC" (optionnel, récurrence libre)
3. Descriptif d'adresse : classe "ADDR" (en fin d'entrée, optionnel, récurrence libre)

Les segments concaténés dans l'ordre doivent reconstituer exactement le texte d'entrée, sans altérer l'orthographe ni les espaces.

Exemples :
    'Sibire ( lomb. Serilly ), C. Batave, R. S. Denis, 63. — Lomb' ---> 'Sibire ( lomb. Serilly )' = SUBJ, 'C. Batave, R. S. Denis, 63. — Lomb' =  ADDR
    'Galois-le-Gendre, *fabr. et émailleur*, rue du Petit-Lion-Saint-Sauveur, 20, et cour St.-Martin.' ---> 'Galois-le-Gendre' = SUBJ, '*fabr. et émailleur*' = DESC, 'rue du Petit-Lion-Saint-Sauveur, et cour St.-Martin.' = ADDR
    '*Léonard*, (Mad.) place du Carrousel, 12. Voyez aussi Baboulinet.' ---> '*Léonard*, (Mad.)' = SUBJ, 'place du Carrousel, 12' = ADDR, 'Voyez aussi Baboulinet.' = DESC
    '**journal des Savoir**, distribué sous les colonnes du Palais Royal tous les jeudis.' --> '**journal des Savoir**' = SUBJ, ', distribué sous les colonnes du Palais Royal tous les jeudis.' = DESC
"""


class Segment(BaseModel):
    """Un empan de texte contigu et sa classe, tel que renvoyé par le modèle."""

    text: str
    label: Literal["SUBJ", "DESC", "ADDR"]


class Annotation(BaseModel):
    """Schéma de sortie imposé au modèle via le mode structured output d'Ollama."""

    segments: list[Segment]


def annotate_entry(entry_text: str, model: str, temperature: float) -> list[Segment]:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": entry_text},
        ],
        format=Annotation.model_json_schema(),
        options={"temperature": temperature},
    )
    annotation = Annotation.model_validate_json(response.message.content)
    return annotation.segments


def locate_segment(haystack: str, needle: str, start: int) -> tuple[int, int]:
    index = haystack.find(needle, start)
    if index == -1:
        raise ValueError(
            f"Segment {needle!r} introuvable dans le texte à partir de la "
            f"position {start} (le modèle a peut-être altéré le texte)."
        )
    return index, index + len(needle)


def segments_to_label_studio_task(
    entry_text: str,
    segments: list[Segment],
    model: str,
    from_name: str,
    to_name: str,
    metadata: dict[str, str] | None = None,
) -> dict[str, object]:
    results = []
    cursor = 0
    for segment in segments:
        start, end = locate_segment(entry_text, segment.text, cursor)
        results.append(
            {
                "from_name": from_name,
                "to_name": to_name,
                "type": "labels",
                "value": {
                    "start": start,
                    "end": end,
                    "text": segment.text,
                    "labels": [segment.label],
                },
            }
        )
        cursor = end

    data = dict(metadata) if metadata else {}
    data[to_name] = entry_text

    return {
        "data": data,
        "predictions": [{"model_version": model, "result": results}],
    }


def format_segments(task: dict[str, object]) -> str:
    """Représentation colorée et compacte des segments réellement parsés,
    pour affichage console — pas juste un aperçu du texte d'entrée."""
    parts = []
    for item in task["predictions"][0]["result"]:
        label = item["value"]["labels"][0]
        text = item["value"]["text"].strip()
        preview = text if len(text) <= 30 else f"{text[:27]}..."
        color = SEGMENT_COLORS.get(label, "white")
        parts.append(f"[{color}]{label}[/{color}] {preview!r}")
    return "  ".join(parts) if parts else "[dim](aucun segment)[/dim]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pré-annote des entrées d'annuaire à partir d'un CSV via Ollama "
            "et exporte des prédictions NER au format Label Studio."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Fichier CSV d'entrée contenant les lignes à annoter.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Chemin du JSON de sortie (défaut : <entrée>.ls-annotations.json).",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default=DEFAULT_TEXT_COLUMN,
        help=f"Nom de la colonne CSV contenant le texte (défaut : '{DEFAULT_TEXT_COLUMN}').",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Modèle Ollama à utiliser (défaut : {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Température d'échantillonnage (défaut : 0.0, déterministe).",
    )
    parser.add_argument(
        "--from-name",
        type=str,
        default=DEFAULT_FROM_NAME,
        help=(
            "Nom du tag <Labels> de la config Label Studio "
            f"(défaut : '{DEFAULT_FROM_NAME}')."
        ),
    )
    parser.add_argument(
        "--to-name",
        type=str,
        default=DEFAULT_TO_NAME,
        help=(
            "Nom du tag <Text> de la config Label Studio "
            f"(défaut : '{DEFAULT_TO_NAME}')."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help=(
            "Nombre de requêtes Ollama envoyées en parallèle (défaut : 4). "
            "Sans effet si le serveur Ollama n'autorise pas ce parallélisme "
            "(variable d'environnement OLLAMA_NUM_PARALLEL côté serveur) : "
            "les requêtes s'y mettront simplement en file d'attente."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_path.exists():
        console.print(
            f"[bold red]Erreur :[/bold red] Fichier '{args.input_path}' introuvable."
        )
        return

    rows: list[dict[str, str]] = []
    with open(args.input_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or args.text_column not in reader.fieldnames:
            console.print(
                f"[bold red]Erreur :[/bold red] Colonne '{args.text_column}' introuvable dans le CSV."
            )
            return
        for row in reader:
            if row.get(args.text_column, "").strip():
                rows.append(row)

    if not rows:
        console.print(
            "[yellow]Le fichier CSV ne contient aucune ligne exploitable.[/yellow]"
        )
        return

    output_path = args.output or args.input_path.with_suffix(".ls-annotations.json")

    console.print(
        f"Début du traitement de {len(rows)} entrées issues de [yellow]{args.input_path.name}[/yellow] "
        f"avec [cyan]{args.model}[/cyan] (concurrence : {args.concurrency})...\n"
    )

    def process_row(row: dict[str, str]) -> dict[str, object]:
        """Traite une ligne ; exécuté dans un thread du pool.

        Peut lever n'importe quelle exception (réseau, JSON hors schéma,
        texte altéré...) : elle est récupérée par l'appelant via
        `future.result()`, dans le thread principal, et n'interrompt donc
        jamais les autres requêtes en cours.
        """
        text = row[args.text_column].strip()
        segments = annotate_entry(text, args.model, args.temperature)
        return segments_to_label_studio_task(
            text, segments, args.model, args.from_name, args.to_name, metadata=row
        )

    # `results` est indexé comme `rows` : l'ordre du fichier de sortie ne
    # dépend donc pas de l'ordre d'achèvement des requêtes, même si les
    # lignes de progression, elles, s'affichent dans l'ordre où les threads
    # terminent (normal et attendu en exécution parallèle).
    results: list[dict[str, object] | None] = [None] * len(rows)
    failures = 0
    print_lock = threading.Lock()

    progress_columns = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("restant :"),
        TimeRemainingColumn(),
    )

    with Progress(*progress_columns, console=console) as progress:
        # `Progress.update` est explicitement conçu pour être appelé depuis
        # plusieurs threads (cf. les exemples officiels de rich combinant
        # ThreadPoolExecutor et Progress) ; le print_lock protège seulement
        # nos propres `console.print` du dessous, pour un affichage net.
        task_id = progress.add_task("Annotation", total=len(rows))

        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            future_to_index = {
                executor.submit(process_row, row): idx for idx, row in enumerate(rows)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                preview = rows[idx][args.text_column].strip()[:50]
                try:
                    task = future.result()
                except Exception as error:
                    failures += 1
                    with print_lock:
                        console.print(
                            f"[red]✗[/red] #{idx + 1:>3} {preview!r} : {error}"
                        )
                else:
                    results[idx] = task
                    with print_lock:
                        console.print(f"[green]✓[/green] #{idx + 1:>3} {format_segments(task)}")
                progress.update(task_id, advance=1)

    tasks = [task for task in results if task is not None]

    output_path.write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary_style = "bold green" if failures == 0 else "bold yellow"
    console.print(
        f"\n[{summary_style}]Terminé :[/{summary_style}] "
        f"[yellow]{output_path}[/yellow] "
        f"({len(tasks)}/{len(rows)} entrées annotées"
        + (f", {failures} échec(s)" if failures else "")
        + ")"
    )


if __name__ == "__main__":
    main()