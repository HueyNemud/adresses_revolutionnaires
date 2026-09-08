# Numérotation révolutionnaire

Outils pour extraire les lignes d'une sortie OCR Chandra/Datalab, les annoter
interactivement avec un modèle CRF, puis les exporter en CSV.

Le pipeline conserve le JSON comme format principal : les annotations restent
attachées à chaque ligne OCR et le CSV sert uniquement à la consultation ou à
l'édition dans un tableur.

## Prérequis

- Python 3.12 ou plus récent
- [uv](https://docs.astral.sh/uv/)
- Un fichier JSON produit par Chandra/Datalab

## Installation

Depuis la racine du dépôt :

```bash
uv sync
```

`uv` crée et maintient automatiquement l'environnement virtuel `.venv`.

## Utilisation

### 1. Extraire les blocs et lignes OCR

`extract_chandra_lines.py` copie le JSON Chandra et ajoute `data_blocks` à
chaque page. Chaque bloc contient ses lignes Markdown et leur provenance.

```bash
uv run python extract_chandra_lines.py \
	annuaires/bpt6k62915570/bpt6k62915570-5_835-ocr.json \
	annuaires/bpt6k62915570/extracted-lines.json
```

Par défaut, les tableaux restent au format Markdown. Pour exporter chaque
cellule de tableau comme une ligne distincte :

```bash
uv run python extract_chandra_lines.py input.json extracted-lines.json --notables
```

### 2. Annoter les lignes avec le CRF

`annotate_lines_crf.py` charge le JSON extrait, propose des lignes à annoter,
réentraîne le modèle progressivement, puis produit une copie JSON enrichie.

```bash
uv run python annotate_lines_crf.py \
	annuaires/bpt6k62915570/extracted-lines.json \
	--output annuaires/bpt6k62915570/annotated-lines.json
```

Pendant l'annotation, utilisez :

| Touche | Annotation / action                      |
| ------ | ---------------------------------------- |
| `1`    | Début d'une entrée (`ENTRY_BEGIN`)       |
| `2`    | Suite d'une entrée (`ENTRY_INSIDE`)      |
| `3`    | Titre (`TITLE`)                          |
| `0`    | Hors périmètre (`OUT_OF_SCOPE`)          |
| `u`    | Annuler la dernière annotation humaine   |
| `q`    | Prédire les lignes restantes et terminer |

Une session est enregistrée à côté du JSON d'entrée avec le suffixe
`.crf-session.json`. Relancez la même commande pour la reprendre. Utilisez
`--reset-session` pour l'ignorer, ou `--session mon-fichier.json` pour choisir
son emplacement. `--seed-size N` contrôle le nombre d'annotations initiales
avant l'échantillonnage par incertitude.

Les lignes du JSON de sortie reçoivent les champs `prediction`, `provenance`,
`probability` et `timestamp`.

### 3. Exporter un CSV

`export_lines_csv.py` transforme un JSON extrait ou annoté en une ligne CSV par
ligne Markdown. Il ne modifie jamais le JSON.

```bash
uv run python export_lines_csv.py \
	annuaires/bpt6k62915570/annotated-lines.json \
	annuaires/bpt6k62915570/annotated-lines.csv
```

Le CSV inclut la provenance (`page_index`, `chunk_index`, bloc, ligne), le
contenu Markdown, et les champs d'annotation lorsqu'ils sont disponibles.

## Sélectionner une plage de pages (optionnel)

Avec `jq`, créez un JSON ne contenant qu'une plage de pages avant de lancer le
pipeline :

```bash
jq 'map(select(.page_index >= 114 and .page_index <= 326))' \
	annuaires/bpt6k62915570/bpt6k62915570-5_835-ocr.json \
	> annuaires/bpt6k62915570/bpt6k62915570-114_326-ocr.json
```

## Tests

```bash
uv run python -m unittest discover -s tests -v
```