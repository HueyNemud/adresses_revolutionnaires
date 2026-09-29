# Explorations

Notebooks exploratoires sur les sorties de la chaîne. Leurs dépendances
(Jupyter, matplotlib) sont dans le groupe `explorations` de `pyproject.toml` :

```bash
uv sync --group explorations
uv run --group explorations jupyter nbconvert --to notebook --execute --inplace explorations/<notebook>.ipynb
```

(ou ouvrir le notebook dans un éditeur avec le noyau de `.venv`). Les
notebooks s'exécutent depuis `explorations/` ou depuis la racine du dépôt,
et lisent les données locales `annuaires/` (hors git).

| Notebook | Sujet |
|---|---|
| `consolidation_adresses_1807_1808.ipynb` | Classe les adresses des paires alignées 1807/1808 (équivalence, changement de numéro, déménagement, adresse complexe, sans numéro, absente), visualise la situation et exporte la jointure avec la classe : `annuaires/alignements/<paire>.nw.jointure.consolidee.csv` |
