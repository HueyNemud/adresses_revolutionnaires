import json
import os
import time
from datetime import datetime
from datalab_sdk import DatalabClient
from datalab_sdk.exceptions import DatalabAPIError

# Configuration
API_KEY = os.getenv("DATALAB_API_KEY", "VYuwjVr0_nlbMHjWOJOJQX6AeQKsL-v_lm4DqgZQIms")
PIPELINE_ID = "pl_tdR-RmvNf0VV"
PDF_PATH = "scans/bpt6k62915570-5_454-ocr.pdf"
OUTPUT_JSON = "bpt6k62915570-5_454-ocr.pdf.json"

POLL_INTERVAL = 10  # Intervalle de vérification en secondes

client = DatalabClient(
    api_key=API_KEY,
    timeout=30_000,
)


def log(message: str):
    """Affiche un message horodaté."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


try:
    log(f"1. Lancement du pipeline {PIPELINE_ID} pour {PDF_PATH}...")

    # 1. Lancement non-bloquant (max_polls=0) pour récupérer l'ID d'exécution immédiatement
    execution = client.run_pipeline(
        PIPELINE_ID,
        file_path=PDF_PATH,
        output_format="json",
        run_evals=False,
        max_polls=0,
    )

    execution_id = execution.execution_id
    log(f"   Démarrage réussi. Execution ID : {execution_id}")

    # 2. Boucle de suivi / polling manuel avec logging dynamique
    start_time = time.time()
    poll_count = 0

    while True:
        poll_count += 1
        execution = client.get_pipeline_execution(execution_id)
        elapsed_sec = int(time.time() - start_time)
        status = getattr(execution, "status", "unknown").lower()

        # Extraire d'éventuels détails d'avancement (pages, étape en cours)
        progress_details = []
        if getattr(execution, "pages_processed", None) is not None and getattr(
            execution, "total_pages", None
        ):
            progress_details.append(
                f"Pages: {execution.pages_processed}/{execution.total_pages}"
            )
        if getattr(execution, "current_step", None):
            progress_details.append(f"Étape: {execution.current_step}")

        extra_info = f" ({', '.join(progress_details)})" if progress_details else ""
        log(
            f"   [Poll #{poll_count} | +{elapsed_sec}s] Statut : {execution.status.upper()}{extra_info}"
        )

        # Arrêt de la boucle si l'exécution s'est achevée ou a échoué
        if status in ["completed", "failed", "cancelled", "error"]:
            break

        time.sleep(POLL_INTERVAL)

    if status != "completed":
        raise Exception(f"Traitement interrompu avec le statut : {execution.status}")

    log("2. Traitement terminé ! Récupération des résultats de chaque étape...")

    # 3. Récupération du résultat de l'étape 0
    results = {}
    step_0_result = client.get_step_result(execution_id, step_index=0)
    results["step_0"] = step_0_result

    # 4. Enregistrement du fichier complet
    final_output = {
        "execution_id": execution_id,
        "pipeline_id": PIPELINE_ID,
        "status": execution.status,
        "duration_seconds": int(time.time() - start_time),
        "data": results,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False, default=str)

    log(f"✅ Résultat complet enregistré avec succès dans : {OUTPUT_JSON}")

except DatalabAPIError as e:
    log(f"❌ Erreur API Datalab ({e.status_code}) : {e.response_data}")
except Exception as e:
    log(f"❌ Une erreur est survenue : {e}")
