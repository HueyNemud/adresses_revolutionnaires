#!/usr/bin/env bash
set -euo pipefail

# Dossier cible passé en paramètre (ou le dossier courant par défaut)
TARGET_DIR="${1:-.}"

# Recherche du fichier *.ocr.json dans le dossier
OCR_JSON=$(find "$TARGET_DIR" -maxdepth 1 -type f -name "*.ocr.json" | head -n 1)

if [[ -z "$OCR_JSON" ]]; then
    echo "Erreur : aucun fichier *.ocr.json trouvé dans '$TARGET_DIR'." >&2
    exit 1
fi

# Déduction du préfixe (ex: AD75_PER292-1807)
DIRNAME=$(dirname "$OCR_JSON")
FILENAME=$(basename "$OCR_JSON")
BASE_NAME="${FILENAME%.ocr.json}"
PREFIX="${DIRNAME}/${BASE_NAME}"

echo "=== Traitement de : ${BASE_NAME} ==="

# 1. Extraction des lignes
echo "--> [1/4] Extraction des lignes..."
uv run extract_chandra_lines.py "${PREFIX}.ocr.json" --notables \
    -o "${PREFIX}.ocr.lines.json"

# 2. Annotation CRF (interactive)
echo "--> [2/4] Annotation CRF..."
uv run annotate_lines_crf.py "${PREFIX}.ocr.lines.json" \
    -o "${PREFIX}.ocr.lines.annotated.json" \
    --session "${PREFIX}.ocr.lines.crf-session.json"

# 3. Export CSV
echo "--> [3/4] Export en CSV..."
uv run export_lines_csv.py "${PREFIX}.ocr.lines.annotated.json" \
    -o "${PREFIX}.ocr.lines.annotated.csv"

# 4. Fusion des blocs
echo "--> [4/4] Entités et arbre des titres..."
uv run build_entity_tree.py "${PREFIX}.ocr.lines.annotated.csv" \
    -o "${PREFIX}.ocr.lines.annotated.merged.csv"

echo "=== Pipeline terminé pour ${BASE_NAME} ==="
