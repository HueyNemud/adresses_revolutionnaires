"""Lance un pipeline Datalab/Chandra sur un PDF et enregistre son résultat JSON.

Lance l'exécution en mode non bloquant, puis interroge périodiquement son
statut jusqu'à ce qu'elle se termine (succès, échec, timeout), avant
d'enregistrer le résultat de l'étape 0 dans un fichier JSON.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from datalab_sdk import DatalabClient
from datalab_sdk.exceptions import DatalabAPIError
from rich.console import Console

console = Console()

DEFAULT_PIPELINE_ID = "pl_tdR-RmvNf0VV"
DEFAULT_POLL_INTERVAL = 10  # secondes entre deux vérifications de statut
DEFAULT_TIMEOUT = 1800  # secondes avant abandon si le pipeline ne se termine pas
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "error"}


def log(message: str) -> None:
    """Affiche un message horodaté."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.print(f"[dim][{timestamp}][/dim] {message}")


def _progress_suffix(execution: object) -> str:
    """Résumé optionnel de l'avancement (pages traitées, étape en cours)."""
    details = []
    if getattr(execution, "pages_processed", None) is not None and getattr(
        execution, "total_pages", None
    ):
        details.append(f"Pages: {execution.pages_processed}/{execution.total_pages}")
    if getattr(execution, "current_step", None):
        details.append(f"Étape: {execution.current_step}")
    return f" ({', '.join(details)})" if details else ""


def run_pipeline_and_wait(
    client: DatalabClient,
    pipeline_id: str,
    pdf_path: Path,
    poll_interval: int,
    timeout: int,
) -> tuple[str, dict, int]:
    """Lance le pipeline et attend sa fin.

    Retourne (execution_id, résultat de l'étape 0, durée en secondes).
    Lève TimeoutError si le délai est dépassé, RuntimeError si le pipeline
    se termine sur un statut autre que "completed".
    """
    log(f"Lancement du pipeline {pipeline_id} pour {pdf_path}...")
    execution = client.run_pipeline(
        pipeline_id,
        file_path=str(pdf_path),
        output_format="json",
        run_evals=False,
        max_polls=0,
    )
    execution_id = execution.execution_id
    log(f"Démarrage réussi. Execution ID : {execution_id}")

    start_time = time.time()
    poll_count = 0
    status = "unknown"

    while True:
        poll_count += 1
        execution = client.get_pipeline_execution(execution_id)
        elapsed_sec = int(time.time() - start_time)
        status = getattr(execution, "status", "unknown").lower()

        log(
            f"[Poll #{poll_count} | +{elapsed_sec}s] Statut : "
            f"{status.upper()}{_progress_suffix(execution)}"
        )

        if status in TERMINAL_STATUSES:
            break
        if elapsed_sec >= timeout:
            raise TimeoutError(
                f"Délai d'attente dépassé ({timeout}s) : le pipeline est resté "
                f"en statut '{status}' sans se terminer. Execution ID : {execution_id}"
            )
        time.sleep(poll_interval)

    if status != "completed":
        raise RuntimeError(f"Traitement interrompu avec le statut : {status}")

    log("Traitement terminé ! Récupération du résultat de l'étape 0...")
    step_0_result = client.get_step_result(execution_id, step_index=0)
    return execution_id, step_0_result, int(time.time() - start_time)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lance un pipeline Datalab/Chandra sur un PDF et enregistre son résultat JSON."
    )
    parser.add_argument("pdf_path", type=Path, help="PDF à envoyer au pipeline.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Chemin du JSON de sortie (défaut : <entrée>.datalab.json).",
    )
    parser.add_argument(
        "--pipeline-id",
        type=str,
        default=DEFAULT_PIPELINE_ID,
        help=f"Identifiant du pipeline Datalab (défaut : {DEFAULT_PIPELINE_ID}).",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=DEFAULT_POLL_INTERVAL,
        help=f"Secondes entre deux vérifications de statut (défaut : {DEFAULT_POLL_INTERVAL}).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=(
            "Secondes avant abandon si le pipeline ne se termine pas "
            f"(défaut : {DEFAULT_TIMEOUT})."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.pdf_path.exists():
        console.print(
            f"[bold red]Erreur :[/bold red] Fichier '{args.pdf_path}' introuvable."
        )
        sys.exit(1)

    api_key = os.getenv("DATALAB_API_KEY")
    if not api_key:
        console.print(
            "[bold red]Erreur :[/bold red] La variable d'environnement "
            "DATALAB_API_KEY n'est pas définie."
        )
        sys.exit(1)

    output_path = args.output or args.pdf_path.with_suffix(".datalab.json")
    client = DatalabClient(api_key=api_key, timeout=30_000)

    try:
        execution_id, step_0_result, duration_seconds = run_pipeline_and_wait(
            client,
            args.pipeline_id,
            args.pdf_path,
            args.poll_interval,
            args.timeout,
        )
    except DatalabAPIError as error:
        console.print(
            f"[bold red]Erreur API Datalab ({error.status_code}) :[/bold red] "
            f"{error.response_data}"
        )
        sys.exit(1)
    except (RuntimeError, TimeoutError) as error:
        console.print(f"[bold red]Erreur :[/bold red] {error}")
        sys.exit(1)

    final_output = {
        "execution_id": execution_id,
        "pipeline_id": args.pipeline_id,
        "status": "completed",
        "duration_seconds": duration_seconds,
        "data": {"step_0": step_0_result},
    }
    output_path.write_text(
        json.dumps(final_output, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    console.print(
        "\n[bold green]✅ Résultat enregistré avec succès :[/bold green] "
        f"[yellow]{output_path}[/yellow] ({duration_seconds}s)"
    )


if __name__ == "__main__":
    main()
