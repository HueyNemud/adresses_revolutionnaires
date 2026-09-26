# Guide d'annotation NER des entrées d'annuaire (SUBJ / DESC / ADDR)

> **Statut : validé (2026-09-26).** Les points marqués **[Arbitré]** sont des choix de convention que les données v1 appliquaient de façon incohérente. Ce guide est la **source unique** des conventions : prompt LLM, consignes Label Studio, règles automatiques (`lib/ner/rules.py`) et contrôles de cohérence en dérivent.

## Principe

Chaque ENTRY est découpée en empans **contigus**, dans l'ordre du texte, selon **la question à laquelle le segment répond** :

| Classe | Question | Contenu |
|---|---|---|
| `SUBJ` | **Qui ?** | Désignation de la personne, de la raison sociale ou de l'objet (journal, établissement) listé, avec tout ce qui **identifie** ce sujet. |
| `DESC` | **Quoi ?** | Activité, profession, spécialité, fonction, produits, informations commerciales, renvois. |
| `ADDR` | **Où ?** | Toute information de localisation : adresse(s), numéro, section/quartier, repères. |

Test de décision : *si on retire le segment, perd-on l'identité du sujet (SUBJ), ce qu'il fait (DESC) ou l'endroit où le trouver (ADDR) ?*

## Règles générales

1. **Ponctuation de liaison hors empans** : la virgule, le point, le point-virgule ou le tiret qui *séparent* deux empans ne sont annotés dans aucun des deux. La ponctuation *interne* à un empan (`R. S. Denis, 25.`) en fait partie, **y compris le point final** de l'adresse.
2. **Emphase Markdown ignorée** : `*`, `**` ne sont que de la typographie OCR. Les frontières se placent comme si elles n'existaient pas ; le texte est normalisé (emphase retirée) pour l'annotation et l'évaluation. Pour le modèle, garder ou non l'emphase est une option mesurée par `audit_ner.py` (l'italique signale souvent une description dans les volumes 1807-1808). *(Actuellement : `<SUBJ>**Cetto</SUBJ> <DESC>(…)**</DESC>`, frontière placée à l'intérieur du gras.)*
3. **Pas de texte orphelin** : tout mot (hors ponctuation de liaison) appartient à un empan. *(143 entrées actuelles laissent du texte non annoté, ex. `<SUBJ>Villard</SUBJ>, R. de Lille, 539, <DESC>F. de Gr.</DESC>`.)*
4. **Un SUBJ par entrée, en tête**, sauf entrée qui en fusionne réellement deux (erreur de fusion de lignes : chaque sujet reçoit son SUBJ).

## SUBJ : qui ?

Inclure dans le SUBJ tout ce qui **distingue cette personne ou raison sociale d'une autre de même nom** :

- Nom, prénoms, initiales, **y compris entre parenthèses** :
  `<SUBJ>Andry ( Ch. L. Fr. )</SUBJ>, <ADDR>R. des Ecouffes, 8. — Dr. de l'H.</ADDR>`
  *(Dans les données v1, ≈ 600 entrées (1 %) placent ainsi prénoms, civilité ou rang familial en DESC ; les ≈ 5 800 autres parenthèses en DESC après le nom sont de vraies descriptions.)*
- **[Arbitré] Toujours SUBJ** : civilité, état civil, rang familial : `(Mad.)`, `(Me.)`, `(Ve.)`, `veuve`, `fils`, `aîné`, `jeune`, `le jeune`, `père et fils`, `frères`, avec ou sans parenthèses :
  `<SUBJ>Lafitte (le jeune), ( J. Bapt. )</SUBJ>, <ADDR>R. Favart, 425. — Lepelletier.</ADDR>`
  `<SUBJ>Rasmann (frères)</SUBJ>, <ADDR>Place des Vosges, 297. — Indivisibilité.</ADDR>`
  *(Aujourd'hui incohérent : `Coustellier aîné` en SUBJ mais `Berthé` + `<DESC>(aîné)</DESC>`.)*
- Titres de noblesse : `(le baron d')`, `(Ve. de)`, `comtesse`. En revanche, les grades et fonctions (`général`, `sénateur`, `notaire`) vont dans DESC.
- Associés et raison sociale : `et comp.`, `et Cie`, `Robert frères et Paradis`, `Barbereux (M.e Ad.) et Boubée aîné`.
- Titre d'un journal ou d'un établissement listé comme sujet : `<SUBJ>Journal de Paris</SUBJ>`.
- **[Arbitré] Précision d'homonymie ou raison sociale entre parenthèses → SUBJ** : `Sibire ( lomb. Serilly )`, `Gerboin (Lomb. Lussan)`, `Delavéronnière (Moysse et Sollivet)`.
  La parenthèse sert à distinguer le sujet ou à nommer une raison sociale ; ce n'est pas une adresse de contact. C'était déjà l'exemple du prompt LLM v1, alors que la règle 2 de `correct_annotations.py` la déplaçait en DESC (minuscule initiale).

## DESC : quoi ?

- Profession, commerce, spécialité, y compris **entre parenthèses** et en minuscule :
  `<SUBJ>Bresler</SUBJ> <DESC>(piano)</DESC>, <ADDR>R. Ste. Avoie, 155. — Réunion.</ADDR>`
  `<SUBJ>Rabillon</SUBJ> <DESC>( histoire )</DESC>, …` (spécialité sous un titre « Peintres »)
  `<SUBJ>Warnier</SUBJ> <DESC>(en cuivre)</DESC>, …`
- Fonction, qualité, titre honorifique ou de charge : `( médecin du Gouvernement ) , professeur de l'école de méd.`, `(de l'Impératrice)`, `(de Malte)`.
- Informations commerciales (prix, périodicité, rédacteurs d'un journal) : `12 f. pour 3 mois, …`, `par Sedillot jeune`.
- **Renvois** : `<SUBJ>Marotte</SUBJ>. <DESC>Voyez Carlier et Marotte.</DESC>`. Pas d'étiquette dédiée : le renvoi dit où trouver l'information, pas qui est le sujet. *(252 entrées, cohérent aujourd'hui.)*
- Parenthèses successives de natures différentes : couper selon la question. `<SUBJ>Aubert (veuve)</SUBJ> <DESC>(fourbiss.)</DESC>`.
- **[Arbitré] Mention mixte identité / activité** : `(Mad.) (de Malte)`, `(Me.) (sage-femme)` : la civilité va dans SUBJ, la qualité ou la fonction dans DESC.

## ADDR : où ?

- Adresse complète, **y compris la section ou le quartier après le tiret** : `<ADDR>R. Vivienne, 44. — Mail.</ADDR>`, `<ADDR>R. de Varennes, 464.—O.</ADDR>` (≈ 23 000 entrées, cohérent).
- Adresses multiples et « et » : un **seul** ADDR couvrant l'ensemble : `<ADDR>rue de Richelieu, 10; et de Quiberon, 5.</ADDR>`, `<ADDR>Palais du Tribunal, galerie de bois, 260, et R. de l'Arbre Sec, 249. — G. Françaises.</ADDR>`.
- Repères de localisation : `près l'opéra`, `en face le corps de garde`, `près la municipalité` → dans l'ADDR.
- **[Arbitré] Section ou quartier placé après l'adresse sans tiret → ADDR** : `R. S. Denis, 19. Amis de la Patrie.`, `rue du Harlay, 5. (Indivisib.)`, `rue des Lavandières, 29, G. F.`, `R. de Boulogne, 65. F. G.`
  Ce sont des noms de sections ou de faubourgs. Dans les données v1, ils sont annotés DESC (signature SUBJ,ADDR,DESC, 53 cas).
- **[Arbitré] Enseigne → DESC** : `( *Café du Caveau* )`, `Au Grand Monarque` (dénomination commerciale, pas localisation), sauf si elle sert manifestement à localiser (« près du Café … »).

## Cas particuliers

- **Entrée tronquée ou sans adresse** : on n'invente rien. `<SUBJ>Pajot</SUBJ>` seul est valide.
- **Césure OCR** (`har-nois`, `Impéra- trice`) : le mot coupé reste dans un seul empan.
- **Erreur de fusion** (deux entrées dans une ligne) : annoter chacune (`SUBJ,ADDR,SUBJ,ADDR`) ; ces cas sont aussi à signaler pour l'étape de fusion.

## Correspondance avec les règles automatiques

| Règle (`lib/ner/rules.py`) | Convention appliquée |
|---|---|
| Parenthèse de prénoms/initiales ou de civilité après SUBJ → fusionnée dans SUBJ | SUBJ : prénoms, civilité |
| Qualificatif familial (`aîné`, `jeune`, `fils`, `frères`, `veuve`…) seul dans un DESC collé au SUBJ → SUBJ | SUBJ : rang familial |
| DESC commençant par un tiret juste après ADDR → fusionné dans ADDR | ADDR : section après tiret (règle 1 de `correct_annotations.py`) |
| Section connue sans tiret après ADDR → ADDR | ADDR : section/quartier |
| Ponctuation en début/fin d'empan retirée | Règle générale 1 (règle 3 de `correct_annotations.py`) |
| ~~Parenthèse minuscule en fin de SUBJ → DESC~~ | **Supprimée** (règle 2 de `correct_annotations.py`) : cause de `Lafitte <DESC>(le jeune)</DESC>` ; remplacée par les deux premières lignes |
