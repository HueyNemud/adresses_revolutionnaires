# À faire

## Recalibrer le seuil `douteuse` pour latynna-v4

`OCR_DOUBTFUL_THRESHOLD` (`src/numrev/pipeline/tabulate.py`) vaut 0,9 : c'est le
repère mesuré sur latynna-v1, pas sur v4 (en production depuis le 2026-10-24).

1. **Budget retenu : environ 100 lignes à relire par volume.** Pour 1808
   (environ 16 400 lignes), cela fait environ 0,6 % des lignes. À ce niveau,
   le val (une vingtaine de pages, quelques milliers de lignes) ne contient
   que quelques dizaines de lignes signalées : l'intervalle du bootstrap sera
   large. Une variante sans seuil : trier sur `confiance` et relire les 100
   premières lignes. Le script
   propose le seuil qui signale cette part sur le val et donne le rappel
   obtenu, c'est-à-dire la part des erreurs qui échappe à la relecture. Si le
   rappel est trop faible, revoir le budget en regardant la courbe (voir
   « Comment choisir » plus bas).
2. **Faire écrire le script de calibration** dans
   `../finetuning_chandra_for_directories`, là où se trouvent les cibles
   relues. Il doit :
   - lire les pages **val** (jamais le test) avec latynna-v4, depuis le PDF en
     pleine résolution ;
   - apparier lignes et mots prédits avec la cible relue ;
   - pour chaque seuil de 0,50 à 0,99, donner la part des lignes signalées, le
     rappel et la précision, avec un bootstrap par page ;
   - proposer le seuil selon la règle (option du type `--budget 0.10`) ;
   - montrer quelques appariements à vérifier ;
   - compter à part les lignes sans scores.
3. **Lancer** le script, serveur démarré :
   `uv run latynna serve --detach && uv run latynna serve --wait`.
4. **Relire le tableau** la première fois, pour vérifier les appariements.
5. **Reporter le seuil** dans `OCR_DOUBTFUL_THRESHOLD`, avec en commentaire le
   modèle, les pages, la part des lignes signalées et le rappel. Mettre aussi à
   jour le « Repère » du README de latynna.

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
  quatre est vraiment fausse, le drapeau lasse et on cesse de le suivre.
- **Penser aux numéros** : pour ce projet, une erreur sur un numéro
  d'adresse compte plus qu'une coquille dans une description. Demander au
  script le rappel sur les seules lignes dont un **chiffre** est faux. Si un
  seuil propre aux chiffres fait mieux, on pourra signaler une ligne dès
  qu'un de ses nombres a un score bas.

## Interpréter les accolades (session dédiée)

Depuis latynna-v4 (2026-10-24), une ligne groupée par une accolade porte son
signe : `… }` en fin de ligne (le texte commun est une précision, à droite),
`{ …` en début (le texte commun est un titre, à gauche). Le texte commun n'est
écrit que sur la première ligne du groupe. `extract` garde ces lignes telles
quelles ; rien ne les interprète encore. Forme, exemples, règles de lecture,
fiabilité et questions à trancher (trait du CRF, clé `cle`, propagation du
texte commun aux entités et au NER) : `docs/numrev.md` du projet
`../finetuning_chandra_for_directories`.
