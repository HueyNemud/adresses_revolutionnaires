# Guide d'annotation NER des entrées d'annuaire (SUBJ / DESC / ADDR)

L'étape 5 découpe le texte de chaque entrée en **empans** : des morceaux de texte, chacun d'une classe `SUBJ`, `DESC` ou `ADDR`. Ce guide fixe les conventions de ce découpage. Elles valent partout où l'on annote : quand on corrige la colonne `tagged_text` d'un `….ner.csv` (voir le [guide de la curation](guide_curation.md#corriger-le-ner-et-les-rubriques-étape-5)), quand on relit l'échantillon d'évaluation dans Label Studio (`data/ner/gold.ls.json`), et dans les jeux d'entraînement : le modèle réapprend les écarts de convention de ce qu'on lui montre.

Dans `tagged_text`, un empan s'écrit entre deux balises : `<SUBJ>Dupont</SUBJ>, <DESC>boulanger</DESC>, <ADDR>rue de la Paix, 12.</ADDR>`. On ne change que les balises, jamais le texte entre elles.

## Principe

Chaque entrée est découpée en empans **contigus**, dans l'ordre du texte, selon **la question à laquelle le segment répond** :

| Classe | Question | Contenu |
|---|---|---|
| `SUBJ` | **Qui ?** | Désignation de la personne, de la raison sociale ou de l'objet (journal, établissement) listé, avec tout ce qui **identifie** ce sujet. |
| `DESC` | **Quoi ?** | Activité, profession, spécialité, fonction, produits, informations commerciales, renvois. |
| `ADDR` | **Où ?** | Toute information de localisation : adresse(s), numéro, section/quartier, repères. |

Test de décision : *si on retire le segment, perd-on l'identité du sujet (SUBJ), ce qu'il fait (DESC) ou l'endroit où le trouver (ADDR) ?*

## Règles générales

1. **Ponctuation de liaison hors empans** : la virgule, le point, le point-virgule ou le tiret qui *séparent* deux empans ne sont annotés dans aucun des deux. La ponctuation *interne* à un empan (`R. S. Denis, 25.`) en fait partie, **y compris le point final** de l'adresse. L'évaluation ignore la ponctuation en bord d'empan.
2. **Italique et gras ignorés** : `*…*` (italique) et `**…**` (gras) ne sont que de la typographie relevée par l'OCR. Le modèle reçoit le texte sans eux et l'évaluation les ignore ; aucune frontière d'empan ne tombe donc entre les deux marqueurs d'une même paire.
3. **Pas de texte orphelin** : tout mot (hors ponctuation de liaison) appartient à un empan.
4. **Un SUBJ par entrée, en tête**, sauf entrée qui en fusionne réellement deux (erreur de fusion de lignes : chaque sujet reçoit son SUBJ).

## SUBJ : qui ?

Inclure dans le SUBJ tout ce qui **distingue cette personne ou raison sociale d'une autre de même nom** :

- Nom, prénoms, initiales, **y compris entre parenthèses** :
  `<SUBJ>Andry ( Ch. L. Fr. )</SUBJ>, <ADDR>R. des Ecouffes, 8. — Dr. de l'H.</ADDR>`
- Civilité, état civil, rang familial, avec ou sans parenthèses : `(Mad.)`, `(Me.)`, `(Ve.)`, `veuve`, `fils`, `aîné`, `jeune`, `le jeune`, `père et fils`, `frères` :
  `<SUBJ>Lafitte (le jeune), ( J. Bapt. )</SUBJ>, <ADDR>R. Favart, 425. — Lepelletier.</ADDR>`
  `<SUBJ>Rasmann (frères)</SUBJ>, <ADDR>Place des Vosges, 297. — Indivisibilité.</ADDR>`
- Titres de noblesse : `(le baron d')`, `(Ve. de)`, `comtesse`. En revanche, les grades et fonctions (`général`, `sénateur`, `notaire`, `(D. R.)`) vont dans DESC.
- Associés et raison sociale : `et comp.`, `et Cie`, `Robert frères et Paradis`, `Barbereux (M.e Ad.) et Boubée aîné`.
- Titre d'un journal ou d'un établissement listé comme sujet : `<SUBJ>Journal de Paris</SUBJ>`.
- Précision d'homonymie ou raison sociale entre parenthèses : `Sibire ( lomb. Serilly )`, `Gerboin (Lomb. Lussan)`, `Delavéronnière (Moysse et Sollivet)`. La parenthèse sert à distinguer le sujet ; ce n'est pas une adresse de contact.

## DESC : quoi ?

- Profession, commerce, spécialité, y compris **entre parenthèses** et en minuscule :
  `<SUBJ>Bresler</SUBJ> <DESC>(piano)</DESC>, <ADDR>R. Ste. Avoie, 155. — Réunion.</ADDR>`
  `<SUBJ>Rabillon</SUBJ> <DESC>( histoire )</DESC>, …` (spécialité sous un titre « Peintres »)
  `<SUBJ>Warnier</SUBJ> <DESC>(en cuivre)</DESC>, …`
- Produits ou articles, même introduits par « et » : `<SUBJ>Duval</SUBJ>, <DESC>et madras</DESC>, <ADDR>rue N.-St.-Denis, 13.</ADDR>`.
- Fonction, qualité, titre honorifique ou de charge : `( médecin du Gouvernement ) , professeur de l'école de méd.`, `(de l'Impératrice)`, `(de Malte)`, `(D. R.)`.
- Informations commerciales (prix, périodicité, rédacteurs d'un journal) : `12 f. pour 3 mois, …`, `par Sedillot jeune`.
- Enseigne : `( Café du Caveau )`, `Au Grand Monarque` (dénomination commerciale), sauf si elle sert manifestement à localiser (« près du Café … »).
- **Renvois** : `<SUBJ>Marotte</SUBJ>. <DESC>Voyez Carlier et Marotte.</DESC>`. Pas d'étiquette dédiée : le renvoi dit où trouver l'information, pas qui est le sujet.
- Parenthèses successives de natures différentes : couper selon la question. `<SUBJ>Aubert (veuve)</SUBJ> <DESC>(fourbiss.)</DESC>`, `<SUBJ>Lolive (Mad.)</SUBJ> <DESC>(de Malte)</DESC>`.

## ADDR : où ?

- Adresse complète, **y compris la section ou le quartier après le tiret** : `<ADDR>R. Vivienne, 44. — Mail.</ADDR>`, `<ADDR>R. de Varennes, 464.—O.</ADDR>`.
- Section ou quartier placé après l'adresse sans tiret : `R. S. Denis, 19. Amis de la Patrie.`, `rue du Harlay, 5. (Indivisib.)`, `R. de Boulogne, 65. F. G.`
- Adresses multiples et « et » : un **seul** ADDR couvrant l'ensemble : `<ADDR>rue de Richelieu, 10; et de Quiberon, 5.</ADDR>`, `<ADDR>Palais du Tribunal, galerie de bois, 260, et R. de l'Arbre Sec, 249. — G. Françaises.</ADDR>`.
- Repères de localisation : `près l'opéra`, `en face le corps de garde`, `près la municipalité`.
- Lieux sans nom de rue : `marché Boulainvilliers, 13.`, `allée d'Antin`, `pal. du Trib., passage du Perron, 94.`

## Cas particuliers

- **Entrée tronquée ou sans adresse** : on n'invente rien. `<SUBJ>Pajot</SUBJ>` seul est valide.
- **Césure OCR** (`har-nois`, `Impéra- trice`) : le mot coupé reste dans un seul empan.
- **Erreur de fusion** (deux entrées dans une ligne) : annoter chacune (`SUBJ,ADDR,SUBJ,ADDR`) ; la ligne est aussi à couper en deux dans la table des lignes (étape 3, [guide de la curation](guide_curation.md#couper-une-ligne-en-deux-dupliquer)).
