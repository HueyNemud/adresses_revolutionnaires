"""Pré-annote des entrées d'annuaire via Ollama et exporte un JSON de
prédictions NER importable dans Label Studio.

Chaque ligne du fichier d'entrée (par ex. la colonne 'markdown' d'un CSV
fusionné par merge_annotated_lines.py, exportée une entrée par ligne) est
segmentée et classée par le modèle en :
  - "SUBJ" : entité désignée (unique, obligatoire, premier segment) ;
  - "DESC" : descriptif d'activité (optionnel, récurrence libre) ;
  - "ADDR" : descriptif d'adresse (en fin d'entrée, optionnel).

Le modèle est interrogé en mode "structured output" d'Ollama (un schéma
JSON contraint, généré depuis un modèle Pydantic), pas seulement en mode
JSON libre : la forme de la réponse est garantie, pas seulement sa validité
JSON.

Ces classes (SUBJ/DESC/ADDR) sont indépendantes des classes BIO de
annotate_lines_crf.py (B-ENTRY, I-ENTRY, ...) : ce script classe des empans
de texte à l'intérieur d'une entrée déjà reconstituée, pas des lignes.
"""

import argparse
import json
from pathlib import Path
from typing import Literal

import ollama
from pydantic import BaseModel, ValidationError
from rich.console import Console

console = Console()

DEFAULT_MODEL = "qwen3.8:27b"
DEFAULT_FROM_NAME = "label"
DEFAULT_TO_NAME = "text"

SYSTEM_PROMPT = """Tu es un système spécialisé en extraction d'information et NER sur des annuaires commerciaux anciens.

À partir de l'entrée d'annuaire fournie, sépare et classifie les segments de texte contigus :
1. Entité désignée : classe "SUBJ" (unique, obligatoire, premier segment)
2. Descriptif d'activité : classe "DESC" (optionnel, récurrence libre)
3. Descriptif d'adresse : classe "ADDR" (en fin d'entrée, optionnel, récurrence libre)

Les segments concaténés dans l'ordre doivent reconstituer exactement le texte d'entrée, sans altérer l'orthographe ni les espaces.
"""


class Segment(BaseModel):
    """Un empan de texte contigu et sa classe, tel que renvoyé par le modèle."""

    text: str
    label: Literal["SUBJ", "DESC", "ADDR"]


class Annotation(BaseModel):
    """Schéma de sortie imposé au modèle via le mode structured output d'Ollama."""

    segments: list[Segment]


def annotate_entry(entry_text: str, model: str, temperature: float) -> list[Segment]:
    """Interroge Ollama en mode structured output (schéma JSON contraint).

    Contrairement à `format="json"` (qui garantit seulement du JSON
    syntaxiquement valide, de forme libre), passer le schéma Pydantic
    contraint la sortie du modèle à respecter exactement la structure
    attendue (clé "segments", "text"/"label" par élément, classes limitées
    à SUBJ/DESC/ADDR).
    """
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
    """Localise un segment dans le texte source à partir d'une position donnée.

    La recherche démarre au curseur courant (pas depuis le début du texte),
    pour placer correctement des segments dont le texte se répète (ex. deux
    "16" à des positions différentes). Lève une ValueError explicite si le
    modèle a altéré le texte : mieux vaut faire échouer proprement cette
    entrée que produire un empan Label Studio mal positionné en silence.
    """
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
) -> dict[str, object]:
    """Construit une tâche Label Studio avec ses prédictions NER par empans."""
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
    return {
        "data": {"text": entry_text},
        "predictions": [{"model_version": model, "result": results}],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pré-annote des entrées d'annuaire via Ollama (structured output) "
            "et exporte des prédictions NER au format Label Studio."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Fichier texte, une entrée d'annuaire par ligne.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Chemin du JSON de sortie (défaut : <entrée>.ls-annotations.json).",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_path.exists():
        console.print(
            f"[bold red]Erreur :[/bold red] Fichier '{args.input_path}' introuvable."
        )
        return

    entries = [
        line.strip()
        for line in args.input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not entries:
        console.print(
            "[yellow]Le fichier d'entrée ne contient aucune ligne exploitable.[/yellow]"
        )
        return

    output_path = args.output or args.input_path.with_suffix(".ls-annotations.json")

    console.print(
        f"Début du traitement de {len(entries)} entrées avec "
        f"[cyan]{args.model}[/cyan]...\n"
    )

    tasks: list[dict[str, object]] = []
    failures = 0
    for idx, text in enumerate(entries, start=1):
        try:
            segments = annotate_entry(text, args.model, args.temperature)
            task = segments_to_label_studio_task(
                text, segments, args.model, args.from_name, args.to_name
            )
        except Exception as error:
            # Catch-all volontairement large : un lot d'appels LLM peut échouer
            # de mille façons (réseau, modèle indisponible, JSON hors schéma,
            # texte altéré par le modèle...). On isole l'échec à cette seule
            # entrée et on poursuit le lot plutôt que d'interrompre tout le
            # traitement pour un incident ponctuel.
            failures += 1
            console.print(
                f"[red][{idx}/{len(entries)}] ERREUR[/red] sur {text!r} : {error}"
            )
            continue
        tasks.append(task)
        console.print(f"[green][{idx}/{len(entries)}] OK[/green] — {text[:50]}...")

    output_path.write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary_style = "bold green" if failures == 0 else "bold yellow"
    console.print(
        f"\n[{summary_style}]✅ Terminé :[/{summary_style}] "
        f"[yellow]{output_path}[/yellow] "
        f"({len(tasks)}/{len(entries)} entrées annotées"
        + (f", {failures} échec(s)" if failures else "")
        + ")"
    )


if __name__ == "__main__":
    main()
