# Guide d'annotation des lignes (classes et niveaux de titres)

Ce guide fixe les conventions de l'annotation des lignes (`numrev label`,
étape 2) et de leur correction dans la table des lignes (`….lines.csv`,
étape 3, voir le [guide de la curation](guide_curation.md)). Chaque ligne
de l'OCR reçoit une **classe** ; la chaîne s'en sert ensuite pour
reconstituer les entrées et la hiérarchie des titres (étape 4), puis pour
comparer les rubriques de deux éditions (étape 7). Les exemples sont tirés
ou inspirés des corrections faites sur 1807 et 1808.

## Les classes

| Classe | Touche (`numrev label`) | Quand l'utiliser | Ce qu'en fait la chaîne |
|---|---|---|---|
| `B-ENTRY` | `1` | Première ligne d'une **entrée** : une personne, une maison de commerce, un établissement. | Ouvre une nouvelle entrée. |
| `I-ENTRY` | `2` | **Suite** de l'entrée précédente : son texte continue à la ligne (retour à la ligne de l'impression, adresse coupée). | Collée à l'entrée ouverte, avec une espace. |
| `SUB-ENTRY` | `3` | Ligne **rattachée** à l'entrée précédente sans en être la suite directe : une seconde personne à la même adresse, un complément, une liste sous l'entrée. | Ajoutée à l'entrée ouverte, après un saut de ligne. |
| `B-TITLE` | `4` | Première ligne d'un **titre** : grande partie, rubrique, sous-rubrique. Son niveau est donné par les `#` (voir plus bas). | Ouvre un titre dans l'arbre des titres. |
| `I-TITLE` | `5` | Suite d'un titre écrit sur plusieurs lignes. | Collée au titre ouvert, avec une espace. |
| `OUT OF SCOPE` | `0` | Ligne qui n'est ni une entrée ni un titre de la hiérarchie : titre courant répété en haut de page, numéro de page, mention de l'éditeur, en-tête de colonne. | Gardée telle quelle, jamais fusionnée ; elle n'interrompt pas une entrée ou un titre qui continue après elle. |
| `¯\_(ツ)_/¯` | `6` | Pendant l'annotation seulement : on ne sait pas trancher tout de suite. | **À remplacer** par une vraie classe pendant la curation : l'étape 4 la traite comme `OUT OF SCOPE` et la signale dans son rapport. |
| `SUPPRIMÉE` | — | Pendant la curation seulement : la ligne ne devrait pas exister (page scannée deux fois, ligne en double). | Ignorée par l'étape 4 : elle disparaît des données. |

Dans la table des lignes, la classe s'écrit **exactement** comme dans ce
tableau (majuscules, tirets, accent de `SUPPRIMÉE`). Une classe vide ou mal
écrite (`b-entry`, `B-ENTRY.`) sur une ligne corrigée arrête
`numrev tabulate` sans rien écrire, pour ne pas perdre l'entrée en silence.

### Exemples

Une entrée sur deux lignes :

| markdown | classe |
|---|---|
| `Lagrange, rue des` | `B-ENTRY` |
| `Saints-Pères , 47.` | `I-ENTRY` |

Une personne ajoutée sous l'entrée, à la même adresse :

| markdown | classe |
|---|---|
| `Nogues, rue …` | `B-ENTRY` |
| `Nogues (Ve.), même demeure` | `SUB-ENTRY` |

Un titre sur plusieurs lignes, puis les rubriques :

| markdown | classe |
|---|---|
| `# LISTES PARTICULIÈRES,` | `B-TITLE` |
| `PAR ORDRE ALPHABÉTIQUE,` | `I-TITLE` |
| `## AGENS DE CHANGE.` | `B-TITLE` |
| `### Membres composant le syndicat.` | `B-TITLE` |

Une entrée coupée par un changement de page : le titre courant et le
numéro de la nouvelle page sont `OUT OF SCOPE`, et la suite de l'entrée
reste `I-ENTRY` : elle est recollée par-dessus ces lignes.

| markdown | classe |
|---|---|
| `Martin, rue S. Honoré,` | `B-ENTRY` |
| `Banquiers.` | `OUT OF SCOPE` |
| `213` | `OUT OF SCOPE` |
| `124.` | `I-ENTRY` |

## Les niveaux de titres

Le nombre de `#` en tête d'un titre donne son niveau. L'OCR le choisit
(balises `<h1>`…`<h6>`) et se trompe souvent : c'est la correction la plus
importante de l'étape 3.

| Niveau | Rôle | Exemples |
|---|---|---|
| `#` | Grande partie de l'annuaire | `# LISTES PARTICULIÈRES,` |
| `##` | **Rubrique** : profession, liste | `## BANQUIERS.`, `## BOIS (MARCHANDS DE).` |
| `###` | Sous-rubrique | `### Membres composant le syndicat.`, `### *Anciens greffiers des bâtimens.*` |

Pourquoi c'est important :

- l'étape 4 range chaque entrée sous le dernier titre qui la précède, et
  chaque titre sous le dernier titre de niveau plus élevé : de mauvais
  niveaux donnent un mauvais arbre ;
- l'alignement de deux éditions (étape 7) compare les entrées **rubrique par
  rubrique**, la rubrique d'une entrée étant son titre de niveau `##` (à
  défaut `#`). Une entrée dont la rubrique est mal rattachée n'est comparée
  qu'aux entrées d'une autre rubrique ;
- **un titre sans `#`** est placé au niveau le plus profond et signalé dans le
  rapport de l'étape 4 ; il ne compte jamais comme rubrique pour
  l'alignement.

Le titre se corrige dans la colonne `markdown` : `#### PAPETIERS.` →
`## PAPETIERS.`, avec `corrige = oui`.

## Vérifier avec le rapport de l'étape 4

`numrev assemble` écrit `<document>.entities.report.txt`, qui montre l'arbre
des titres (avec le nombre d'entrées sous chacun) et les cas à vérifier :

- **titres sans `#`** : leur donner un niveau ;
- **continuations sans ancre** (`I-ENTRY`, `SUB-ENTRY` ou `I-TITLE` sans ligne
  qui les précède) : souvent une `B-ENTRY` prise pour une suite, ou l'inverse ;
- **ruptures de l'ordre alphabétique** dans une rubrique : une entrée mal
  coupée (deux entrées collées, ou une suite prise pour une entrée), un nom
  mal lu, ou un vrai désordre de l'annuaire ;
- **classes incertaines ou inconnues** : des `¯\_(ツ)_/¯` à trancher.

On corrige la table des lignes, on relance `numrev tabulate` puis
`numrev assemble`, et on relit le nouveau rapport.

## Une plage qui commence au milieu d'une rubrique

L'arbre des titres est construit plage par plage. Les entrées placées avant
le premier titre d'une plage n'ont donc pas de rubrique : l'étape 4 les
signale (rapport et console), et l'alignement de deux éditions ne les
apparie jamais. La chaîne ne les rattache jamais d'elle-même à la dernière
rubrique de la plage précédente : deux plages successives ne sont souvent
pas contiguës, et ces entrées appartiennent alors à une autre rubrique.

Pour les rattacher, on **ajoute la ligne de titre de leur rubrique tout en
haut de la table des lignes**, avant la première ligne, en laissant sa clé
vide (comme toute ligne ajoutée, voir le
[guide de la curation](guide_curation.md#ajouter-une-ligne)) :

| cle | markdown | classe | corrige |
|---|---|---|---|
| *(vide)* → `debut+1` | **`## BIJOUTIERS ET JOUAILLIERS.`** | **`B-TITLE`** | *(mis à `oui` automatiquement)* |
| `478…` | `*Bijoutiers et Jouailliers.*` | `OUT OF SCOPE` | |
| `869…` | `**Camus**, R. S. Denis, 175. — Amis de la Patrie.` | `B-ENTRY` | |

- **La rubrique continue celle de la fin de la plage précédente** : recopier
  ce titre tel qu'il est dans la table des lignes de la plage précédente
  (mêmes mots ; la casse, les accents et la ponctuation peuvent varier).
  L'alignement réunit alors les deux morceaux en une seule rubrique.
- **C'est une autre rubrique** : recopier son titre tel qu'il est imprimé
  (sur une page laissée hors des plages, par exemple).
- **Le titre n'est imprimé sur aucune page du PDF** (page manquante) et on le
  déduit des entrées : l'écrire comme un titre imprimé, sans marque
  particulière.

Pour que le chemin complet des titres soit juste, on peut ajouter aussi,
au-dessus, le titre de partie (`#`) dont dépend la rubrique.

Le titre restitué ne vient pas de l'OCR : sa clé est `debut+1` (`debut+2`…
pour les lignes suivantes), il n'a ni `confiance` ni annotation du modèle,
et il reste reconnaissable dans les données (colonne `cle` du `….ner.csv`).

## Groupes à accolade

Une accolade imprimée relie plusieurs lignes à un texte commun, écrit une
seule fois (« Fieffé. } Membres du conseil… », puis « Pastoret. } »…).
Depuis latynna-v4, l'OCR écrit le signe sur chaque ligne du groupe (`… }` en
fin de ligne, `{ …` en début), et le texte commun sur la première ligne
seulement. On annote le groupe comme **une seule entrée** :

- la première ligne en `B-ENTRY`, les suivantes en `SUB-ENTRY` (une ligne qui
  continue la précédente reste `I-ENTRY`) ;
- **chaque ligne garde son signe**, et le texte commun reste sur la première
  ligne du groupe, à droite de `}` ou à gauche de `{` ;
- deux accolades emboîtées (un arrondissement `{` qui regroupe des quartiers,
  eux-mêmes reliés par `}` à un juge) s'écrivent dans l'ordre de la page :
  `I.er... { Tuileries..... } Cuynet, place des Italiens, 1.`

Le livrable du volume (étape 6) donne alors chaque membre du groupe avec le
texte commun dont il hérite. Son rapport, `<volume>.export.report.txt`,
signale les groupes que la règle ne résout pas (une ligne sans signe, un
texte commun absent de la première ligne, une accolade à trois niveaux) :
on les corrige ici, dans la table des lignes. La règle complète est dans
[`format_export.md`](format_export.md#groupes-à-accolade).

## Ce que la chaîne ne traite pas encore

- **Accolades dans les CSV** : le `….ner.csv` et l'alignement de deux
  éditions traitent un groupe à accolade comme une seule entrée ; seul le
  livrable du volume (étape 6) en déplie les membres.
