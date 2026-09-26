"""Code partagé par les scripts de la chaîne de traitement.

- `lib.chandra_document` : schéma du JSON produit par extract_chandra_lines.py ;
- `lib.crf` : cœur du classifieur de lignes CRF (labels, features, modèle,
  apprentissage actif, évaluation sur les silver datasets).
"""
