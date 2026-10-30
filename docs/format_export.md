# Format du livrable d'un volume (`numrev export`)

Ce document décrit les deux fichiers JSON que produit l'étape 6,
`numrev export`. Ils forment la version structurée d'un annuaire, et chaque
information y garde son lien avec l'endroit du document imprimé d'où elle
vient. Il s'adresse aux personnes qui utilisent ces fichiers (visualisation,
analyse, géocodage) et à celles qui les font évoluer.

Les schémas JSON qui valident ces fichiers sont dans
`src/numrev/export/schemas/` (`document.schema.json`,
`structure.schema.json`, format JSON Schema 2020-12).

## Sommaire

- [Vue d'ensemble](#vue-densemble)
- [Produire le livrable](#produire-le-livrable)
- [En-tête commun](#en-tête-commun)
- [Couche documentaire : `<volume>.document.json`](#couche-documentaire--volumedocumentjson)
- [Couche structurée : `<volume>.structure.json`](#couche-structurée--volumestructurejson)
- [Groupes à accolade](#groupes-à-accolade)
- [Identifiants et stabilité](#identifiants-et-stabilité)
- [Exemples d'utilisation](#exemples-dutilisation)
- [Limites connues](#limites-connues)
- [Faire évoluer le format](#faire-évoluer-le-format)

## Vue d'ensemble

Un volume (un annuaire, `annuaires/<volume>/`) donne deux fichiers :

| Fichier | Contenu | Pour |
|---|---|---|
| `<volume>.document.json` | la **couche documentaire** : pages, blocs et lignes, chaque ligne avec sa boîte sur la page, son texte lu par l'OCR, son texte retenu après correction, sa classe et ses confiances | afficher le document, surligner une information sur la page |
| `<volume>.structure.json` | la **couche structurée** : titres (rubriques) et entrées, avec leurs empans `SUBJ` (qui), `DESC` (quoi) et `ADDR` (où), et les membres des groupes à accolade | interroger l'annuaire comme une base de données, géocoder les adresses |

Les deux couches sont reliées par les **identifiants de lignes**. Une entrée
liste ses lignes, et chacun de ses empans liste les morceaux de lignes qu'il
couvre, avec des positions en caractères. On passe ainsi d'une adresse à la
boîte de sa ligne sur la page, et d'une ligne de la page à l'entrée qui la
contient.

```text
structure.json                               document.json
  entrée ─ lines[] ─────────────────────────▶ ligne (page, bloc, boîte, texte)
     └─ spans[] ─ pieces[] ─ (ligne, début, fin) ─▶ caractères du texte de la ligne
```

Le volume est l'unité livrée : les plages de pages traitées séparément par
la chaîne (`7-177`, `178-182`…) y sont réunies dans l'ordre de leur première
page. Un volume dont seule une partie des pages est traitée est exporté
quand même, et l'en-tête liste les plages couvertes : les pages absentes ne
sont simplement pas dans les fichiers.

Les lignes hors champ (titres courants, numéros de page, texte libre :
classe `OUT OF SCOPE`) ne figurent que dans la couche documentaire, avec
leur classe.

## Produire le livrable

```bash
uv run numrev export annuaires/<volume>           # simulation : calcule, vérifie, liste les fichiers
uv run numrev export annuaires/<volume> --apply   # écrit les deux JSON et le rapport
```

L'export est une étape **purement dérivée** : rien ne s'y corrige à la main.
Il se calcule à partir des sorties des étapes 1 à 5 de chaque plage
(`<document>.lines.json`, `<document>.lines.csv`, `<document>.ner.csv`),
corrections comprises, et se recalcule à l'identique (seule la date d'export
change). Pour corriger le livrable, on corrige la table des lignes ou le CSV
NER de la plage (voir `guide_curation.md`), on relance les étapes en aval
(`numrev status` dit lesquelles), puis l'export.

La commande refuse un volume dont une plage a une étape absente ou périmée
(sortie plus ancienne que son entrée). Elle écrit aussi
`<volume>.export.report.txt`, qui donne les comptes, les groupes à accolade
que la règle ne résout pas (voir [Groupes à accolade](#groupes-à-accolade))
et les lignes OCR sans boîte. `numrev status annuaires/<volume>` indique si
le livrable est absent, à jour ou périmé.

Pour le consulter, `uv run numrev view directory annuaires/<volume>` ouvre
dans le navigateur une page qui pose les deux fichiers sur le PDF du volume
(`src/numrev/viewers/directory.html`, voir `pipeline.md`). Cette page lit
le format décrit ici, et rien d'autre : elle montre ce qu'en tire un
utilisateur des fichiers.

## En-tête commun

Les deux fichiers commencent par les mêmes champs :

| Champ | Contenu |
|---|---|
| `schema` | `numrev-document/1` ou `numrev-structure/1` : nom et version du format |
| `volume` | nom du dossier du volume, ex. `1807_AD75-PER292` |
| `source` | `pdf` : nom du PDF du volume ; `sha256` : son empreinte (`null` s'il est absent) |
| `ranges` | plages exportées : `name` (`7-177`), `first_page`, `last_page` (pages du PDF, comptées à partir de 1) |
| `coordinates` | repère de toutes les boîtes : `{"width": 1000, "height": 1000, "origin": "top-left"}` |
| `provenance` | `numrev.commit` (commit du code qui a exporté) et `numrev.uncommitted_changes` (code modifié non commité), `exported_at` (date UTC), `inputs` (empreinte sha256 de chaque fichier lu) |

**Repère des boîtes.** Une boîte est `[x0, y0, x1, y1]` dans le repère de
latynna : de 0 à 1000 sur la largeur et sur la hauteur de l'image de la
page, origine en haut à gauche. Pour la placer sur une image de `W × H`
pixels : `x × W / 1000`, `y × H / 1000`. Le champ `size` d'une page donne la
taille, en pixels, de l'image qu'a lue l'OCR.

Les fichiers sont écrits sans indentation (ils font une dizaine de mégaoctets
pour un gros volume). Pour les lire : `jq . fichier.json | less`, ou
`json.load` en Python.

## Couche documentaire : `<volume>.document.json`

`pages` : la liste des pages, dans l'ordre des plages puis des pages.

| Champ d'une page | Contenu |
|---|---|
| `id` | `<plage>:<page>`, ex. `7-177:10` |
| `page` | numéro de la page dans le PDF (à partir de 1) |
| `range` | plage d'où vient la page |
| `size` | `[largeur, hauteur]` de l'image de la page lue par l'OCR, en pixels |
| `blocks` | blocs de l'OCR : `id` (`<plage>:<page>.<bloc>`), `label` (`List-Group`, `Section-Header`, `Page-Header`…), `bbox` |
| `lines` | lignes de la page, dans l'ordre de lecture retenu (celui de la table des lignes) |

Une page partagée par deux plages (la fin d'une plage et le début de la
suivante, ex. p. 192 en 1807) apparaît **deux fois**, une fois par plage :
ce sont deux lectures OCR distinctes. La partie de la page qui appartient à
l'autre plage y est classée `OUT OF SCOPE` (convention de curation), si bien
qu'une ligne utile n'est classée dans le champ qu'une seule fois.

| Champ d'une ligne | Contenu |
|---|---|
| `id` | `<plage>:<clé>`, ex. `192-210:43a0785fd10e` ; c'est l'identifiant auquel renvoie la couche structurée |
| `key` | clé stable de la ligne dans sa plage (`cle` de la table des lignes : empreinte du texte OCR ; `debut+1`, `<clé>+1`… pour une ligne ajoutée à la main) |
| `uid` | position dans la sortie OCR, `page.bloc.ligne` (à partir de 1) ; `null` pour une ligne ajoutée à la main |
| `block` | identifiant du bloc qui contient la ligne (pour une ligne ajoutée : celui de sa voisine) |
| `origin` | `ocr` (lue par l'OCR) ou `added` (ajoutée à la main pendant la curation, par exemple un titre de rubrique restitué en tête de plage) |
| `class` | classe de la ligne : `B-ENTRY`, `I-ENTRY`, `SUB-ENTRY`, `B-TITLE`, `I-TITLE`, `OUT OF SCOPE`, ou `SUPPRIMÉE` (ligne écartée à la main) ; voir `guide_annotation_lignes.md` |
| `text` | texte **retenu**, après correction (niveaux de titre `#`, signes d'accolade, lignes fusionnées) ; c'est à ce texte que renvoient toutes les positions de la couche structurée |
| `ocr_text` | texte lu par l'OCR, avant correction ; `null` pour une ligne ajoutée |
| `bbox_estimate` | boîte de la ligne, **estimée** par latynna après la lecture (elle peut être décalée d'une ligne dans une colonne serrée) ; `null` si latynna n'a pas pu placer la ligne ou si elle est ajoutée |
| `ocr` | `confidence` (plus faible score de mot), `doubtful` (vrai sous le seuil 0,9 de `numrev tabulate`), `words` (`[[mot, score], …]`) ; `null` sans scores |
| `labeling` | classe proposée par le modèle d'annotation : `probability`, `provenance` ; `null` si la ligne n'a pas été annotée par lui |
| `corrected` | `true` si la ligne a été corrigée ou validée à la main (`corrige = oui`) |

## Couche structurée : `<volume>.structure.json`

`root` : l'identifiant de la racine de l'arbre des titres
(`00000000-0000-0000-0000-000000000000`), parent des titres de premier
niveau et des entrées qui précèdent tout titre de leur plage.

`titles` : les titres (rubriques), dans l'ordre du document.

| Champ d'un titre | Contenu |
|---|---|
| `id` | uuid du titre (celui de `numrev assemble`) |
| `parent` | uuid du titre parent, ou `root` |
| `range`, `pages` | plage et pages (PDF) de ses lignes |
| `level` | niveau (nombre de `#` en tête) ; `null` pour un titre sans `#` |
| `heading` | texte lisible, sans `#` ni marques d'emphase |
| `text` | texte complet, lignes jointes par une espace |
| `lines` | ses lignes : `{line, start, end}` = identifiant de la ligne et place de son texte dans `text` |

`entries` : les entrées, dans l'ordre du document.

| Champ d'une entrée | Contenu |
|---|---|
| `id` | uuid de l'entrée (celui de `numrev assemble` et du `.ner.csv` : les alignements entre éditions y renvoient) |
| `parent` | uuid de sa rubrique (titre), ou `root` si elle précède tout titre de sa plage |
| `range`, `pages` | plage et pages (PDF) de ses lignes |
| `kind` | `single` (entrée simple) ou `group` (groupe à accolade, voir plus bas) |
| `text` | texte de l'entrée : ses lignes jointes par une espace, ou par un saut de ligne devant une ligne `SUB-ENTRY` |
| `lines` | ses lignes : `{line, start, end}`, place du texte de chaque ligne dans `text` |
| `spans` | ses empans (voir ci-dessous) |
| `members` | membres d'un groupe à accolade ; `null` pour une entrée simple |
| `quality` | indicateurs de qualité (voir ci-dessous) |

Un **empan** est un morceau du texte de l'entrée étiqueté `SUBJ` (qui :
nom, raison sociale), `DESC` (quoi : profession, qualité) ou `ADDR` (où :
adresse) ; conventions dans `guide_annotation_ner.md`.

| Champ d'un empan | Contenu |
|---|---|
| `id` | identifiant stable de l'empan (voir [Identifiants](#identifiants-et-stabilité)) |
| `label` | `SUBJ`, `DESC` ou `ADDR` |
| `start`, `end` | position dans `text` de l'entrée (`end` exclu, caractères Unicode) |
| `text` | texte tel qu'il est écrit, marques d'emphase Markdown comprises (`*boulanger*`) |
| `normalized` | texte sans marques d'emphase, blancs réduits à une espace |
| `pieces` | un morceau par ligne couverte : `{line, start, end}` en positions dans le `text` de cette ligne (couche documentaire) ; une adresse coupée en fin de ligne a deux morceaux |

Les **adresses** sont les empans `ADDR`. Chacune a son identifiant : une
future étape d'analyse (voie, numéro) ou de géocodage y rattachera ses
résultats sans modifier ces fichiers.

| Champ de `quality` | Contenu |
|---|---|
| `ocr_confidence` | plus faible confiance OCR des lignes de l'entrée ; `null` sans scores |
| `ocr_doubtful` | vrai si l'une de ses lignes est douteuse |
| `ner_confidence` | plus faible score du modèle NER parmi ses empans ; `null` si l'entrée n'a pas été traitée |
| `ner_suspect` | motifs de relecture prioritaire de `numrev tag` (`score bas`, `SUBJ absent en tête`…) ; liste vide si rien à signaler |
| `corrected_lines` | vrai si l'une de ses lignes a été corrigée ou validée à la main |
| `corrected_ner` | vrai si ses empans ou sa rubrique ont été corrigés ou validés à la main |

## Groupes à accolade

Une accolade imprimée relie plusieurs lignes à un texte commun écrit une
seule fois :

```text
Fieffé.            }
Pastoret.          }  Membres du conseil d'administration
Richard d'Aubigny. }  des hospices et secours.
Thouret.           }
```

La curation des lignes en fait **une seule entrée** (première ligne
`B-ENTRY`, suivantes `SUB-ENTRY`, voir `guide_annotation_lignes.md`) : c'est
ce que l'on voit sur la page. Pour la vue « base de données » et le
géocodage, l'export donne en plus les **membres** du groupe. Chaque membre
correspond à une ligne `B-ENTRY` ou `SUB-ENTRY`, avec ses éventuelles
lignes `I-ENTRY`, et hérite du texte commun. Ici, quatre membres ont chacun
leur `SUBJ`, et tous héritent du même `DESC`.

| Champ d'un membre | Contenu |
|---|---|
| `id` | identifiant stable du membre |
| `lines` | ses lignes (identifiants de la couche documentaire) |
| `own` | `{start, end}` : son texte propre dans `text` de l'entrée |
| `inherited` | `[{start, end}, …]` : les textes communs dont il hérite |
| `spans` | identifiants de ses empans propres |
| `inherited_spans` | identifiants des empans hérités ; un texte commun n'a qu'un seul empan, partagé par tous les membres (une adresse commune n'est géocodée qu'une fois) |

**Règle.** Le texte de chaque ligne porte le signe de l'accolade : `}` à la
fin de la partie propre, `{` au début. Le texte commun n'est écrit que sur
la première ligne de sa suite :

- **`}`** : le texte à **droite** du signe est commun à la suite des lignes
  qui portent `}` ;
- **`{`** : le texte à **gauche** du signe est commun à la suite des lignes
  qui portent `{` ;
- un signe suivi (ou précédé) d'un texte commun non vide ouvre une nouvelle
  suite, un signe seul prolonge la suite en cours, une ligne sans le signe
  la termine ;
- les deux sortes s'emboîtent : un membre hérite de la suite `{` et de la
  suite `}` qui le contiennent.

Exemple emboîté (juges de paix par arrondissement et quartier, 1807, p. 197) :

```text
I.er... { Tuileries..... }  Cuynet, place des Italiens, 1.
        { Champs-Élysées }
        { Place Vendôme. }  Besnard, rue des Lions-St.-Paul, 8.
        { Roule..... }
```

Le membre « Champs-Élysées » a pour texte propre `Champs-Élysées` et
hérite de `I.er...` (suite `{`) et de `Cuynet, place des Italiens, 1.`
(suite `}`) ; « Roule » hérite de `Besnard…`.

Une entrée est un groupe quand elle a au moins deux lignes `B-ENTRY` /
`SUB-ENTRY` et au moins un signe d'accolade. Une entrée de plusieurs lignes
`SUB-ENTRY` sans aucun signe reste une entrée simple.

**Cas non résolus.** La règle ne devine rien. Le rapport
`<volume>.export.report.txt` liste, avec la page, la ligne et un extrait :

| Message | Ce qu'on fait |
|---|---|
| `ligne sans accolade dans un groupe` | ajouter le signe manquant à la ligne dans la table des lignes (`… }`) |
| `« } » sans texte commun à sa droite` / `« { » sans texte commun à sa gauche` | placer le texte commun sur la première ligne de la suite |
| `signes d'accolade inattendus` | accolade à trois niveaux ou signes en double : simplifier le texte des lignes, ou accepter le cas tel quel (le membre garde tout son texte, sans héritage). Exemple accepté tel quel : 1807, p. 210, « Places à Charbon { d'Aval / { Cisalpine », une accolade qui donne deux valeurs possibles sans dire laquelle va avec quelle halle ; elle reste listée dans le rapport à chaque export |
| `empan … à cheval sur une accolade` | redécouper l'empan dans le CSV NER pour qu'il ne franchisse pas le signe |
| `membre sans SUBJ, ni propre ni hérité` | vérifier le balisage NER du groupe |

Après la correction, on relance les étapes en aval (`numrev status`), puis
l'export. Les membres d'un groupe signalé sont tout de même exportés,
avec ce que la règle a pu établir.

## Identifiants et stabilité

| Objet | Identifiant | Ce dont il dépend |
|---|---|---|
| ligne | `<plage>:<clé>` | plage et texte OCR de la ligne (rang d'occurrence pour un texte répété) |
| page, bloc | `<plage>:<page>`, `<plage>:<page>.<bloc>` | plage et position dans la sortie OCR |
| titre, entrée | uuid de `numrev assemble` | nom de la plage et clé de sa première ligne : ni le texte corrigé, ni les lignes de continuation |
| empan | uuid | uuid de l'entrée, classe, texte normalisé, rang parmi les empans de même classe et de même texte dans l'entrée |
| membre | uuid | uuid de l'entrée et identifiant de sa première ligne |

Ces identifiants sont **stables tant que l'OCR ne change pas**. On peut
relancer la chaîne, corriger des lignes ou des empans : une entrée garde
son identifiant, et un empan garde le sien tant que son propre texte ne
change pas. La correction d'un autre empan de la même entrée ne le change
pas. En revanche, un nouvel OCR d'une plage change les clés de ses lignes,
et donc tous ses identifiants : les résultats rattachés à ces identifiants
(alignements, géocodage) sont alors à refaire ou à réancrer.

## Exemples d'utilisation

Les adresses d'un volume, avec leur rubrique et la boîte de leur première
ligne :

```python
import json

structure = json.load(open("annuaires/1807_AD75-PER292/1807_AD75-PER292.structure.json"))
document = json.load(open("annuaires/1807_AD75-PER292/1807_AD75-PER292.document.json"))
lines = {line["id"]: (page, line) for page in document["pages"] for line in page["lines"]}
titles = {title["id"]: title["heading"] for title in structure["titles"]}

for entry in structure["entries"]:
    for span in entry["spans"]:
        if span["label"] == "ADDR":
            page, line = lines[span["pieces"][0]["line"]]
            print(span["id"], titles.get(entry["parent"], ""), span["normalized"], page["page"], line["bbox_estimate"])
```

Une ligne par personne, groupes à accolade dépliés :

```python
def people(entry):
    spans = {span["id"]: span for span in entry["spans"]}
    if entry["kind"] == "single":
        yield entry["id"], list(spans.values())
    else:
        for member in entry["members"]:
            yield member["id"], [spans[i] for i in member["spans"] + member["inherited_spans"]]
```

## Limites connues

- **Boîtes de lignes estimées.** latynna ne donne qu'une boîte par ligne,
  estimée après la lecture, et aucune boîte par mot. Un empan se situe donc
  par sa ligne et ses positions en caractères ; on peut surligner la ligne
  entière, pas exactement les mots de l'adresse.
- **Lignes fusionnées à la main.** Quand la curation fusionne deux lignes OCR
  (une ligne de tableau lue en deux, par exemple), la ligne retenue garde la
  boîte de sa seule ligne d'origine, et l'autre est `SUPPRIMÉE`.
- **Lignes ajoutées à la main** : elles n'ont ni boîte, ni texte OCR, ni
  confiance.
- **Modèles non tracés (à ajouter).** La sortie OCR ne note pas le modèle
  latynna qui l'a produite, et le CSV NER ne note pas le modèle NER : la
  provenance donne le commit de numrev et l'empreinte des fichiers lus, pas
  le nom des modèles. Pour les ajouter, il faudra d'abord que `numrev ocr`
  et `numrev tag` enregistrent le modèle qu'ils utilisent (ce qui change le
  format de leurs sorties), puis reporter ces noms dans `provenance`.
- **Accolades** : voir les cas non résolus ci-dessus. Les accolades
  qu'écrit latynna-v4 dans le texte OCR ne sont interprétées qu'à travers
  la curation des lignes (classe `SUB-ENTRY`, texte commun sur la première
  ligne).

## Faire évoluer le format

Le code de l'export est dans `src/numrev/pipeline/export.py` (commande) et
`src/numrev/export/` (`layers.py` : construction des deux couches ;
`braces.py` : règle des accolades ; `schemas/`). Toute modification de la
forme des fichiers passe par :

1. ce document et les schémas ;
2. le changement de version dans `schema` (`numrev-document/2`…) si la
   modification n'est pas un simple ajout de champ facultatif ;
3. les tests (`tests/test_export.py`, `tests/test_export_braces.py`), qui
   valident la sortie contre les schémas.
4. le viewer `src/numrev/viewers/directory.html`, qui lit ce format et
   refuse une autre version de `schema`.

Hors du périmètre de ce format, et laissés à des étapes futures : un
outil web publié (le viewer local `numrev view directory` en est une
première forme), l'analyse des adresses (voie, numéro, complément) et leur
géocodage, des identifiants qui survivent à un nouvel OCR, une conversion
vers IIIF / Web Annotation pour les visionneuses patrimoniales, et
l'alignement des membres de groupes entre éditions.
