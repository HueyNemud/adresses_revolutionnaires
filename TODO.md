# À faire

## Recalibrer le seuil `douteuse` pour latynna-v4

Une ligne est marquée `douteuse` quand la confiance OCR de son mot le moins
sûr est sous `OCR_DOUBTFUL_THRESHOLD` (`src/numrev/pipeline/tabulate.py`).
Ce seuil vaut 0,9 : il a été mesuré sur latynna-v1, pas sur latynna-v4 (en
production depuis le 2026-10-24).

**Règle retenue : un budget de relecture d'environ 100 lignes par volume.**
Pour 1808 (environ 16 400 lignes), cela revient à signaler environ 0,6 % des
lignes. On choisit le seuil qui signale cette part, et on mesure le
**rappel** obtenu : la part des lignes fausses qui sont signalées (les
autres échappent à la relecture). Si le rappel est trop faible, on revoit le
budget à l'aide de la courbe (voir « Comment choisir » plus bas).

Deux réserves :

- les pages de validation (« val » : une vingtaine de pages, quelques
  milliers de lignes) ne contiendront à ce niveau que quelques dizaines de
  lignes signalées : l'intervalle de confiance du rappel sera large ;
- on peut se passer de seuil : trier la table des lignes sur `confiance` et
  relire les 100 premières.

Étapes :

1. **Écrire le script de calibration** dans
   `../finetuning_chandra_for_directories`, là où se trouvent les pages
   relues qui servent de référence. Il doit :
   - lire les pages **val** (jamais le test) avec latynna-v4, depuis le PDF en
     pleine résolution ;
   - apparier les lignes et les mots prédits avec ceux de la référence ;
   - pour chaque seuil de 0,50 à 0,99, donner la part des lignes signalées, le
     rappel et la précision, avec un bootstrap par page ;
   - proposer le seuil qui respecte le budget (option du type
     `--budget 0.006`, la part des lignes à signaler) ;
   - montrer quelques appariements à vérifier ;
   - compter à part les lignes sans scores.
2. **Lancer** le script, serveur démarré :
   `uv run latynna serve --detach && uv run latynna serve --wait`.
3. **Relire le tableau** la première fois, pour vérifier les appariements.
4. **Reporter le seuil** dans `OCR_DOUBTFUL_THRESHOLD`, avec en commentaire le
   modèle, les pages, la part des lignes signalées et le rappel. Mettre aussi à
   jour le « Repère » du README de latynna, et la mention « à recalibrer » de
   `docs/pipeline.md` (étape 3).

Limites à garder en tête : une erreur lue avec assurance (par exemple 5 lu 3
avec un score élevé) n'est jamais signalée, et une ligne sans scores non plus.
Si possible, ajouter des pages relues d'autres volumes (1808, ans
républicains), car le seuil peut varier selon la police.

À refaire à chaque nouveau modèle OCR, avec la même règle.

### Comment choisir, une fois la courbe sous les yeux

- **Convertir en heures** : (lignes signalées par volume) × (temps de
  relecture d'une ligne). Par exemple, 1808 compte environ 16 400 lignes : à
  10 % et 10 s par ligne, cela fait environ 4 h 30 par volume. Le seuil le
  plus haut qui tient dans le temps qu'on veut y passer est un bon candidat.
- **Chercher le coude** : le seuil au-delà duquel relire beaucoup plus de
  lignes n'attrape presque plus d'erreurs. Monter plus haut coûte cher pour
  peu de gain.
- **Regarder la précision** : si moins d'une ligne signalée sur trois ou
  quatre est vraiment fausse, on finit par ne plus tenir compte du
  signalement.
- **Penser aux numéros** : pour ce projet, une erreur sur un numéro
  d'adresse compte plus qu'une coquille dans une description. Demander au
  script le rappel sur les seules lignes dont un **chiffre** est faux. Si un
  seuil propre aux chiffres fait mieux, on pourra signaler une ligne dès
  qu'un de ses nombres a un score bas.

## Accolades : ce qui reste

Le livrable d'un volume (étape 6) déplie les groupes à accolade : chaque
personne y hérite du texte commun (`docs/format_export.md`). Restent à
traiter :

- le `.ner.csv`, l'alignement de deux éditions et la jointure, qui voient
  encore un groupe comme une seule entrée ;
- l'annotation des lignes : le modèle (CRF) ne tient pas compte des signes
  `{` et `}`, et les lignes d'un groupe se classent à la main (`B-ENTRY`
  puis `SUB-ENTRY`).

La forme des accolades écrites par l'OCR est décrite dans `docs/numrev.md`
du projet `../finetuning_chandra_for_directories`.

## Enregistrer le nom des modèles

Les sorties de l'OCR (étape 0) et du NER (étape 5) ne notent pas le modèle
qui les a produites ; la provenance du livrable ne peut donc pas le donner
(`docs/format_export.md`, « Limites connues »).
