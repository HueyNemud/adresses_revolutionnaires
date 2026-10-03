# Expérience : GLiNER-relex à la place de GLiNER-bi (octobre 2026)

**Verdict : on garde GLiNER-bi (`models/latynna.gliner-model`).** Fine-tuné
dans les mêmes conditions, `knowledgator/gliner-relex-large-v0.5` fait
autant d'erreurs sur le dev du gold, pas moins, pour un modèle trois fois
plus gros et plus lent. Le modèle relex entraîné a été supprimé ; ce
document garde la méthode et les résultats.

## Modèles comparés

| | `latynna` (en production) | `latynna-relex` (essai) |
| :-- | :-- | :-- |
| modèle de base | `knowledgator/gliner-bi-base-v2.0` | `knowledgator/gliner-relex-large-v0.5` |
| architecture | bi-encodeur (texte : `jhu-clsp/ettin-encoder-150m`, libellés : `BAAI/bge-small-en-v1.5`) | encodeur unique `microsoft/deberta-v3-large` (0,5 G paramètres), tête relations inutilisée |
| `max_width` | 44 mots | 44 mots (12 par défaut, élargi) |

Mêmes conditions d'entraînement (`tools/train_gliner.py`) : mêmes données
(`data/ner/train.ls.json`, 15 002 exemples, 11 977 en entraînement et 3 025
en validation, split par page), mêmes libellés (`person or business name`,
`activity description`, `postal address`), 3 époques ≈ 4 494 pas, batch 8,
taux d'apprentissage 5e-5, bf16, texte Markdown normalisé. Entraînement
relex : 13 min sur une RTX 5000 Ada (32 Go).

Pour faire tourner un modèle relex, `lib/ner/gliner.py` (`predict_entities`)
lui demande les seules entités (`return_relations=False`) : par défaut il
renvoie un couple (entités, relations), qu'on aurait sinon associé aux
mauvais textes sans erreur.

## Résultats

Mesure de référence : `audit_ner.py` sur le **dev** du gold (159 entrées
relues, pondérées par le plan de sondage, intervalles bootstrap par page à
95 %). Le split test n'a pas servi : la décision ne change rien, il reste
intact.

```bash
uv run audit_ner.py --split dev --model models/latynna.gliner-model \
    --model models/latynna-relex.gliner-model --sweep
```

| modèle | exactitude | F1 SUBJ | F1 DESC | F1 ADDR | F1 micro | entrées en erreur |
| :-- | --: | --: | --: | --: | --: | --: |
| `latynna` | 0,974 [0,958 ; 0,985] | 0,983 | 0,931 | 0,993 | 0,982 | 27 |
| `latynna-relex` | 0,961 [0,922 ; 0,983] | 0,983 | 0,934 | 0,985 | 0,979 | 27 |
| relex sans fine-tuning | 0,002 | 0,576 | 0,167 | 0,018 | 0,290 | — |

- **Différence appariée** (relex − bi) : Δ exactitude −0,013 [−0,050 ;
  +0,006], Δ F1 micro −0,003 [−0,012 ; +0,003] : non significative.
- **Erreurs brutes** : 27 de chaque côté, dont 23 communes ; 4 propres à
  chacun. Le −1,3 point d'exactitude vient surtout d'une erreur de relex
  dans la strate `courant`, la plus pondérée (« Passard, sous les Colonades
  du Louvre, passage de la R. du Coq. » : adresse non reconnue) ; ses trois
  autres erreurs propres sont dans la strate `désaccord`.
- **Seuil** : de 0,2 à 0,7, aucun seuil ne change le classement (meilleur
  relex : 0,962 à 0,2–0,3).
- **Sans fine-tuning**, relex repère les morceaux mais pas nos conventions
  (prénom entre parenthèses séparé du nom, adresse sans ponctuation finale) :
  inutilisable tel quel.
- **Validation interne** (split par page des données d'entraînement, empans
  exacts) : F1 micro 96,4 % pour relex contre 96,2 % pour bi — même
  constat d'égalité.
- **Coût** : environ ×3 en taille et en temps d'inférence sur CPU (11 s
  contre 3 s pour les 159 entrées du dev).

Seul avantage observé : la **confiance** de relex classe mieux les erreurs
(AUC 0,66 contre 0,49 pour bi ; relire les 10 % d'entrées les plus
incertaines retrouve 19 % des erreurs contre 15 %). C'est sur un petit
échantillon (27 erreurs) et cela ne compense pas l'absence de gain
d'exactitude.

## Ce qu'aucun des deux ne résout

Les erreurs communes sont des **conventions** du guide
(`docs/guide_annotation_ner.md`) plutôt que des limites de modèle :
« (veuve) », « (de) », « (aîné) », « (dame) » laissés hors de SUBJ (classés
DESC), titres de noblesse entre parenthèses, adresses sans numéro ou
descriptives (« Palais du Tribunal », « Marché Boulainvilliers ») et renvois
(« voyez … »). La marge de progrès est du côté des données d'entraînement
(exemples de ces formes, respect du guide), pas du changement de modèle.

## Refaire l'essai

```bash
uv run tools/train_gliner.py data/ner/train.ls.json \
    --model knowledgator/gliner-relex-large-v0.5 \
    -o models/latynna-relex.gliner-model --apply
uv run audit_ner.py --split dev --model models/latynna.gliner-model \
    --model models/latynna-relex.gliner-model --sweep
```

Il faut environ 30 Go de mémoire GPU libre (un serveur vLLM résident suffit
à faire échouer l'entraînement faute de mémoire).
