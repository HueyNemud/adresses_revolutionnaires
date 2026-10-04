"""Classifieur de lignes CRF.

- `labels` : classes de lignes ;
- `features` : groupes de features (production et candidats) ;
- `model` : entraînement / inférence python-crfsuite ;
- `active_learning` : lignes sources et moteur d'apprentissage actif ;
- `silver` : chargement des CSV curés servant de référence ;
- `evaluation` : métriques, validation croisée, bootstrap par page.
"""
