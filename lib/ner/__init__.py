"""Segmentation NER des entrées d'annuaire en SUBJ / DESC / ADDR.

Conventions : `docs/guide_annotation_ner.md`.

- `spans` : empans en caractères, rendu / lecture de `tagged_text`, format
  Label Studio, normalisation Markdown, tokens GLiNER ;
- `shapes` : « forme » typographique d'une entrée (sans étiquette), pour
  l'échantillonnage et la ventilation des résultats ;
- `rules` : règles de convention appliquées aux annotations (silver) ou en
  post-traitement d'inférence ;
- `corpus` : lecture des CSV `*.merged.ner*.csv` du dossier `annuaires/` ;
- `metrics` : comparaison entrée par entrée, agrégats pondérés, bootstrap
  par page.
"""
