"""NER des entrées : découpage en SUBJ (qui ?), DESC (quoi ?) et ADDR (où ?)
(conventions : `docs/guide_annotation_ner.md`).

- `spans` : empans, `tagged_text`, format Label Studio, texte sans Markdown, tokens GLiNER ;
- `gliner` : chargement d'un modèle et prédiction ;
- `suspicion` : motifs de relecture prioritaire (`ner_suspect`) ;
- `corpus` : lecture des CSV NER de `annuaires/` ;
- `shapes` : forme typographique d'une entrée (tirages, ventilation de l'audit) ;
- `metrics` : comparaison au gold, agrégats pondérés, bootstrap par page ;
- `html` : rendu des empans pour le viewer d'alignement.
"""
