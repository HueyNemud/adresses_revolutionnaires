# Documentation technique de la chaîne

Référence pour faire évoluer la chaîne : fichiers, étapes, règles de
conception et code. L'installation et la vue d'ensemble sont dans le
[README](../README.md) ; le pas-à-pas d'un annuaire, côté humain, dans
[`guide_traitement.md`](guide_traitement.md). Chaque commande détaille ses
options : `uv run numrev <commande> --help`.

## Sommaire

- [Fichiers et conventions](#fichiers-et-conventions)
- [Corrections humaines rejouables](#corrections-humaines-rejouables--numrevcurationpy)
- [Étape 0 — OCR](#étape-0--ocr--numrev-ocr)
- [Étape 1 — Lignes](#étape-1--lignes--numrev-extract)
- [Étape 2 — Annotation des lignes](#étape-2--annotation-des-lignes--numrev-label)
- [Étape 3 — Table des lignes](#étape-3--table-des-lignes--numrev-tabulate)
- [Étape 4 — Entités et arbre des titres](#étape-4--entités-et-arbre-des-titres--numrev-assemble)
- [Étape 5 — Empans SUBJ / DESC / ADDR](#étape-5--empans-subj--desc--addr)
- [Étape 6 — Livrable du volume](#étape-6--livrable-du-volume--numrev-export)
- [Étapes 7 et 8 — Alignement de deux éditions](#étapes-7-et-8--alignement-de-deux-éditions)
- [Audit du CRF](#audit-du-crf--numrev-audit-crf)
- [Organisation du code](#organisation-du-code)

## Fichiers et conventions

Un **volume** (un annuaire) a son dossier `annuaires/<volume>/` (hors git),
avec son PDF complet. Le PDF est traité par **plages de pages**, chacune
dans un sous-dossier `<première page>-<dernière page>` (pages du PDF,
comptées à partir de 1). Une plage est un **document**, `<volume>.<plage>` ;
chaque étape écrit à côté de son entrée `<document><suffixe>`. Ces noms sont
définis en un seul endroit, `src/numrev/paths.py`.

```text
annuaires/
├── 1808_AD75-PER292/
│   ├── 1808_AD75-PER292.pdf                  PDF du volume
│   ├── 7-186/                                une plage (pages 7 à 186 du PDF)
│   │   ├── 1808_AD75-PER292.7-186.ocr.json   0 · ocr
│   │   ├── ….lines.json                      1 · extract
│   │   ├── ….labeled.json                    2 · label (+ ….label-session.json)
│   │   ├── ….lines.csv                       3 · tabulate (corrigé à la main)
│   │   ├── ….entities.csv                    4 · assemble (+ ….entities.report.txt)
│   │   └── ….ner.csv                         5 · tag (corrigé à la main)
│   ├── 1808_AD75-PER292.document.json        6 · export (+ .structure.json, .export.report.txt)
└── alignments/                               7 · align, 8 · join : <A>__<B>.nw.csv, .dedupe.csv, .csv, ….join.csv
data/                                         décisions humaines, versionnées
├── curation/<document>.lines.patch.csv       3 · corrections des lignes
├── curation/<document>.ner.patch.csv         5 · corrections NER
├── alignment/<A>__<B>.sections.csv           7 · patch des rubriques
├── alignment/<A>__<B>.patch.csv              7 · patch des entrées
├── alignment/<A>__<B>.training.json          7 · paires étiquetées pour Dedupe
├── alignment/<A>__<B>.gold-inversions.csv    gold de l'alignement
└── ner/gold.ls.json, ner/train.ls.json       gold et jeu d'entraînement NER
reports/crf/, reports/ner/, reports/alignment/   rapports d'audit
models/ner.gliner-model/                      modèle NER de production (hors git)
```

**Une commande, des dossiers.** Tout passe par `numrev` (`uv run numrev`
liste les commandes). Les étapes 1 à 5 prennent une plage ou un volume
entier (toutes ses plages) ; `ocr` et `label`, une plage ; `export`, un
volume ; `align`, deux volumes ; `join`, une sortie d'alignement. Chaque
étape trouve son entrée et nomme sa sortie d'après `paths.py` : pas d'option
de chemin. `uv run numrev status [dossier]` montre, par plage, les étapes
faites, absentes ou **périmées** (sortie plus ancienne que son entrée), les
lignes corrigées, les lignes douteuses à relire et les entrées NER
corrigées, la commande à lancer ensuite, puis l'état du livrable du volume.

**Simulation par défaut.** Une commande qui écrit des données calcule et
vérifie tout (conflits de curation, lignes orphelines), liste les fichiers
qu'elle écrirait, et n'écrit qu'avec `--apply`, une fois tout vérifié : une
erreur sur la dernière plage d'un volume n'en laisse aucune à moitié écrite.
`--force` passe outre un refus (conflit, orphelin, fichier existant) et
n'écrit lui aussi qu'avec `--apply`. Exceptions : `label` enregistre sa
session à chaque réponse et exporte à chaque arrêt ; `ocr` sans `--apply`
ne lance pas l'OCR ; `train` sans `--apply` convertit les données puis
s'arrête. Audits et viewers n'ont pas `--apply`.

**Messages.** Une erreur dit ce qui a été constaté (où, quoi), jamais une
cause supposée ; les numéros montrés (pages, blocs, lignes, `uid`)
commencent à 1.

## Corrections humaines rejouables : `numrev/curation.py`

Mode d'emploi, avec exemples et messages d'arrêt :
[`guide_curation.md`](guide_curation.md). Le but : relancer la chaîne depuis
un même OCR (meilleur CRF, nouveau modèle NER…) sans perdre une correction.

| Étape | Clé stable | Champs éditables | Patch versionné |
|---|---|---|---|
| 3 · lignes | `cle` | `classe`, `markdown` (+ lignes ajoutées) | `data/curation/<document>.lines.patch.csv` |
| 5 · NER | `uuid` d'entité | `tagged_text`, `parent_uuid` | `data/curation/<document>.ner.patch.csv` |
| 7 · alignement | uuid gauche / droit | paire, entrée seule, `certitude` | `data/alignment/<A>__<B>.patch.csv`, `.sections.csv` |

Aux étapes 3 et 5, le CSV de sortie est à la fois la sortie machine et le
fichier corrigé (tableur, OpenRefine). `corrige = oui` marque une ligne
corrigée ou validée ; `empreinte` est le hash des champs éditables tels que
la machine les a écrits. Chaque exécution, entièrement ou pas du tout :

1. **capture** : les lignes `corrige = oui` du fichier existant réécrivent le
   patch (git en garde l'historique) ;
2. **génération** de la sortie machine ;
3. **application** du patch, qui gagne.

La commande **panique** (`CurationConflict` : elle liste les lignes en cause
et n'écrit rien) sur une ligne modifiée sans `corrige = oui` (empreinte qui
ne correspond plus), une correction qui ne s'applique plus (clé disparue,
texte d'entité changé en amont, titre parent disparu, classe vide ou
inconnue), une ligne machine effacée ou déplacée. `--force` prend la sortie
machine pour ces lignes et retire du patch les corrections inapplicables
(c'est aussi le moyen d'annuler une correction : vider `corrige`, relancer
avec `--force`) ; `--no-capture` repart du patch sans lire le fichier
existant (après un `git pull`).

Étape 3, opérations sur les lignes :

- **supprimer** : classe `SUPPRIMÉE`, que l'étape 4 ignore ;
- **ajouter** ou **dupliquer** : la ligne reçoit la clé `<clé précédente>+n`
  (`debut+n` avant la première ligne, pour restituer un titre en tête de
  plage) et le patch retient sa place (`apres`) ; elle reprend les colonnes
  de sa voisine, sauf celles que la machine calcule pour chaque ligne
  (`confiance`, `douteuse`, `provenance`, `probability`, `timestamp`) ;
- **déplacer** : `SUPPRIMÉE` sur l'originale, copie sans clé à la bonne
  place (les lignes machine gardent l'ordre de l'OCR).

La **clé** `cle` d'une ligne, calculée à l'étape 1, est le sha1 (12
caractères) de son texte OCR normalisé (NFC, espaces : normalisation figée),
suffixé `~n` à la n-ième occurrence d'un même texte. Elle ne dépend ni de la
page ni du découpage en blocs, au contraire de l'`uid` (`page.bloc.ligne`) ;
elle change quand un nouvel OCR lit la ligne autrement. L'`uuid` d'une
entité (étape 4) dérive du nom du document et de la clé de sa ligne racine :
stable dans les mêmes limites, il change aussi si les plages sont
redécoupées.

Les patchs d'alignement suivent les mêmes règles (le patch gagne, une ligne
orpheline fait paniquer, `--force` l'ignore) mais s'éditent à la main depuis
le viewer, sans capture. Les commits de données sont séparés de ceux du code
(préfixe `données:`) ; les commandes ne commitent jamais.

## Étape 0 — OCR : `numrev ocr`

```bash
uv sync --group ocr
uv run latynna serve --detach && uv run latynna serve --wait   # serveur vLLM (machine GPU)
uv run numrev ocr annuaires/<volume>/<plage> --apply
```

L'OCR est fait par **latynna** (Chandra 2 affiné sur ces annuaires, modèle
latynna-v4, servi par vLLM), paquet du projet voisin
`finetuning_chandra_for_directories` (son README décrit le format de
sortie). `<document>.ocr.json` est le JSON de Chandra, une entrée par page,
où **chaque ligne est un `<span>`** portant `data-word-confidence` (un score
par mot) et `data-line-bbox-estimate` (boîte estimée, repère 0–1000).
`page_index` est le numéro de la page dans le PDF, compté à partir de 1. Un
OCR existant n'est remplacé qu'avec `--force`.

`uv run numrev view ocr annuaires/<volume>/<plage>` pose l'OCR sur le scan
(`src/numrev/viewers/ocr.html`, page autonome que la commande sert en local
avec le JSON et le PDF ; on peut aussi l'ouvrir seule et y déposer les
fichiers).

## Étape 1 — Lignes : `numrev extract`

Copie du JSON OCR où chaque page reçoit `data_blocks` (un par `div` de
premier niveau du `raw`), chacun avec ses `lines`. **Un `<span>` = une
ligne** :

- `uid` (`page.bloc.ligne`, tout compté à partir de 1), `line_index`,
  `markdown` (Markdown du seul contenu du `<span>`, précédé de `#` × niveau
  dans un `<h1>`…`<h6>`), `cle`, et si latynna les donne `word_confidence`
  (`[[mot, score], …]`, abandonné plutôt que décalé si le nombre de scores
  n'est pas celui des mots) et `line_bbox_estimate` ;
- les blocs `Image` / `Figure` et les `<span>` vides ne donnent pas de ligne ;
  du texte hors de tout `<span>` arrête la commande (page, bloc, extrait) ;
- les accolades de latynna-v4 (`… }`, `{ …`) restent dans le texte : seul
  l'export (étape 6) les interprète.

Ce schéma pages → `data_blocks` → `lines` est lu et validé en un seul
endroit, `numrev/document.py` : c'est là qu'on le change.

## Étape 2 — Annotation des lignes : `numrev label`

Annotateur interactif (une plage) : un CRF (`python-crfsuite`) réentraîné à
chaque réponse propose d'abord une graine diversifiée (`--seed-size`, 12),
puis les lignes dont il est le moins sûr, et prédit les autres. Classes
(`numrev/crf/labels.py`) : `B-ENTRY`, `I-ENTRY`, `SUB-ENTRY`, `B-TITLE`,
`I-TITLE`, `OUT OF SCOPE`, `¯\_(ツ)_/¯` (incertain) ; conventions :
[`guide_annotation_lignes.md`](guide_annotation_lignes.md).

- Sortie : `<document>.labeled.json`, où chaque ligne porte `prediction`,
  `provenance` (`human` / `model` / `unclassified`), `probability` et
  `timestamp`, exporté à chaque arrêt.
- Session : `<document>.label-session.json`, réécrite après chaque action,
  liée au sha256 du `.lines.json` (refusée si celui-ci a changé ;
  `--reset-session` recommence).
- La probabilité est l'argmax des **marginales**, pas un Viterbi : elle est
  toujours celle de la classe exportée.
- Les features (`numrev/crf/features.py`, `PRODUCTION_GROUPS`) sont
  calculées sur le texte sans emphase, `#` de tête ni espaces finaux ;
  l'italique reste une feature. Les modifier change l'annotateur : pour
  essayer une feature, l'ajouter aux groupes candidats et lancer
  [l'audit du CRF](#audit-du-crf--numrev-audit-crf).

## Étape 3 — Table des lignes : `numrev tabulate`

Une ligne CSV par ligne du `<document>.labeled.json` (à défaut du
`.lines.json` brut : classes vides et pas de patch, pour inspecter
l'extraction). Colonnes (`CSV_FIELDS`) : `cle`, `uid`, `page_index`,
`chunk_index`, `data_block_index`, `line_index`, `data_block_bbox`,
`data_block_label`, `markdown`, `confiance`, `douteuse`, `classe`,
`corrige`, `provenance`, `probability`, `timestamp`, `empreinte`.

- `confiance` = plus faible score des mots de la ligne (`0,44`) ;
  `douteuse = oui` sous `OCR_DOUBTFUL_THRESHOLD` (0,9, repère de latynna-v1,
  à recalibrer). Non éditables.
- Curation (étape `lines`) : classe et texte (dont les `#`), lignes
  supprimées, ajoutées, déplacées. Une correction dont la classe n'est ni une
  classe connue ni `SUPPRIMÉE` fait paniquer : l'étape 4 en ferait une ligne
  hors champ sans le dire.

## Étape 4 — Entités et arbre des titres : `numrev assemble`

Regroupe les lignes en entités (`<document>.entities.csv`) et écrit
`<document>.entities.report.txt` (comptes, arbre des titres avec ses
entrées directes et totales, cas à vérifier).

- **ENTRY** = `B-ENTRY` + ses `I-ENTRY` (jointes par une espace) et
  `SUB-ENTRY` (saut de ligne) ; **TITLE** = `B-TITLE` + ses `I-TITLE` ;
  **OUT OF SCOPE** (et toute classe inconnue, signalée) : une ligne, jamais
  fusionnée. Une entrée et un titre restent ouverts, chacun sur sa piste,
  jusqu'à la prochaine `B-ENTRY` / `B-TITLE` : une ligne hors champ ou un
  titre courant ne coupe pas une entrée. Une suite sans entité ouverte en
  ouvre une, signalée.
- Colonnes fusionnées : valeurs jointes par des virgules (`uid`, `cle`,
  `page_index`…), sauf `confiance` (la plus faible) et `douteuse` (`oui` si
  l'une l'est). Les lignes `SUPPRIMÉE` et les colonnes `corrige` /
  `empreinte` disparaissent.
- **`uuid`** : uuid5 du nom du document et de la `cle` de la ligne racine,
  jamais du texte ni de la page. La colonne `cle` d'une entité garde la clé
  de chacune de ses lignes.
- **`parent_uuid`** : niveau d'un titre = nombre de `#` (sans `#` : niveau le
  plus profond, signalé). Parent d'un titre : le dernier titre de niveau
  strictement inférieur ; de toute autre entité : le dernier titre ; à
  défaut, la racine commune `00000000-0000-0000-0000-000000000000`.
- L'arbre est **propre à la plage** : les entrées avant son premier titre
  n'ont pas de rubrique (signalées ; l'alignement ne les apparie jamais).
  La chaîne ne les rattache jamais d'elle-même à la rubrique de la plage
  précédente, souvent non contiguë : le curateur restitue leur titre en tête
  de la table des lignes
  ([guide](guide_annotation_lignes.md#une-plage-qui-commence-au-milieu-dune-rubrique)).
- Le rapport signale aussi les ruptures de l'ordre alphabétique des entrées
  (clé des deux premiers mots, réinitialisée à chaque titre).

## Étape 5 — Empans SUBJ / DESC / ADDR

Un modèle GLiNER-bi découpe chaque ENTRY en `SUBJ` (qui ?), `DESC` (quoi ?)
et `ADDR` (où ?) ; conventions :
[`guide_annotation_ner.md`](guide_annotation_ner.md).

| | `data/ner/gold.ls.json` | `data/ner/train.ls.json` |
|---|---|---|
| Rôle | évaluer (`numrev audit ner`) | entraîner (`numrev train`) |
| Origine | tiré une fois (`numrev gold ner`), relu dans Label Studio | construit par `numrev train-set` depuis les CSV NER |
| Qualité | vérité terrain, découpage `dev` / `test` figé | sortie du modèle, corrigée là où `corrige = oui` |

Les textes du gold sont toujours exclus de l'entraînement (comparaison sur
le texte normalisé, pas sur l'`uid`).

### Inférence : `numrev tag`

`<document>.ner.csv` = le CSV d'entités, avec après `entity` : `tagged_text`
(texte d'origine balisé, `<SUBJ>Dupont</SUBJ>, <ADDR>rue A, 1.</ADDR>`),
`subject_count`, `description_count`, `address_count`, `ner_confidence`
(score du moins sûr des empans) et `ner_suspect`, motifs de relecture
prioritaire (`numrev/ner/suspicion.py`, structurels, sans lexique propre à
un volume) : `aucun empan`, `score bas` (sous `--min-score`, 0,9), `texte non
couvert`, `SUBJ absent en tête`, `signature inhabituelle`. Le modèle
(`--model`, défaut `models/ner.gliner-model`, libellés dans
`<modèle>/ner_config.json`) reçoit le texte sans Markdown ; ses empans sont
reportés sur le texte d'origine. Curation (étape `ner`) : `tagged_text` et
`parent_uuid`, capturés avant le chargement du modèle, réappliqués par
`uuid`.

### Entraînement

```bash
uv run numrev train-set --apply                    # en local : data/ner/train.ls.json, tiré par forme typographique
uv run numrev train data/ner/train.ls.json -o models/<nom>.gliner-model --apply   # machine GPU
```

`train` convertit les empans en caractères en empans de mots, calcule
`max_width`, exclut le gold, valide sur un découpage par page et écrit
`<modèle>/ner_config.json`. Base : `knowledgator/gliner-bi-base-v2.0` ; un
GLiNER-relex fait jeu égal à trois fois la taille
([`experience_gliner_relex.md`](experience_gliner_relex.md)). Le jeu
d'entraînement vaut ce que valent les CSV : le modèle réapprend leurs écarts
de convention.

### Audit : `numrev audit ner`

```bash
uv run numrev audit ner --split dev --model models/ner.gliner-model [--model …] [--predictions nom=fichier]
```

Métrique principale : exactitude par entrée (part des entrées sans
correction à faire, ponctuation de bord ignorée), pondérée par le plan de
sondage du gold, avec intervalle de confiance par bootstrap des pages et
écarts appariés contre le premier système ; aussi F1 par classe, exactitude
par mot, types d'erreurs, couverture et rappel de `ner_suspect`. Sortie :
`reports/ner/`. Sur `dev`, une entrée « courante » pèse ~1,5 point :
comparer aussi les nombres d'erreurs. Ne jamais régler sur `test`.

## Étape 6 — Livrable du volume : `numrev export`

```bash
uv run numrev export annuaires/<volume> --apply
```

Format champ par champ (référence, à tenir à jour avec le code, les schémas
et `viewers/directory.html`) : [`format_export.md`](format_export.md).

- **Couche documentaire** `<volume>.document.json` : pages → blocs → lignes,
  chaque ligne avec son id `<plage>:<cle>`, son origine (OCR ou ajoutée), son
  texte retenu, son texte OCR, sa boîte estimée, ses confiances, sa classe.
  Les lignes supprimées n'existent qu'ici.
- **Couche structurée** `<volume>.structure.json` : titres, entrées et
  éléments hors champ (uuid de l'étape 4, un élément hors champ par ligne
  `OUT OF SCOPE`), chacun avec ses lignes (positions dans son texte, rebâti
  comme à l'étape 4 puis vérifié contre le `.ner.csv`) ; pour une entrée,
  ses empans (identifiant stable, morceaux dans chaque ligne) et, pour un
  groupe à accolade, ses membres. Sur une page partagée par deux plages, la
  partie de l'autre plage est `SUPPRIMÉE` dans chaque copie (convention de
  curation) : rien n'est exporté deux fois.
- **Étape dérivée**, sans correction propre : elle relit `.lines.json`,
  `.lines.csv` et `.ner.csv` et se recalcule à l'identique (date mise à
  part). Elle refuse une plage dont une étape manque ou est périmée, accepte
  un volume partiellement traité (l'en-tête liste les plages).
- **Groupes à accolade** (`numrev/export/braces.py`) : règle déterministe sur
  les signes `{` / `}` ; les cas qu'elle ne résout pas vont dans
  `<volume>.export.report.txt`, à corriger dans la table des lignes ou le CSV
  NER.
- **Provenance** : commit de numrev (et modifications non commitées), date,
  sha256 du PDF et des fichiers lus. Les modèles OCR et NER ne sont pas
  encore enregistrés en amont, donc pas exportés.

`uv run numrev view directory annuaires/<volume>` ouvre le livrable posé sur
son PDF (`src/numrev/viewers/directory.html`, page autonome servie en local,
lecture partielle du PDF par requêtes `Range`) : boîtes des lignes par
classe, entrées et empans de la page, fiche d'une entrée, filtres de
relecture, arbre des rubriques, tableau des entrées (export CSV),
statistiques et rapport d'export. La commande prévient si le livrable est
périmé ; l'export prend quelques secondes, à relancer après une correction.
`-p/--port` fixe le port (la page retient ses réglages par adresse).

## Étapes 7 et 8 — Alignement de deux éditions

Retrouver, entre deux éditions, les ENTRY qui désignent la même personne ou
le même commerce. Deux méthodes, même format de sortie, liens un-à-un :

| | `numrev align nw` | `numrev align dedupe` |
|---|---|---|
| Principe | ordre des entrées : Needleman-Wunsch par rubrique, pair-HMM entre ancres | Dedupe `RecordLink`, poids appris sur des paires étiquetées |
| Étiquetage | aucun | en console, la première fois |
| Patch des entrées | vérifié et réancré, appliqué en mémoire par le viewer et `join` | appliqué à la sortie finale |
| Sorties | `<A>__<B>.nw.csv` | `<A>__<B>.dedupe.csv` (brut), `<A>__<B>.csv` (final) |
| Durée (1807/1808) | ~12 s | quelques minutes |

Les deux méthodes restent à comparer sur des alignements vérifiés.

### Chargement des volumes et rubriques

`numrev/alignment/records.py` lit chaque volume en entier, plage par plage
dans l'ordre des pages (un `.ner.csv` manquant est une erreur). Une ENTRY
est comparée sur sa **rubrique** (titre ancêtre de niveau 2, à défaut 1,
en clé sans accents ni ponctuation), son **SUBJ** et son **texte** (sans
Markdown ; DESC et ADDR, trop variables, n'y entrent que par lui). Il n'y a
pas d'héritage de rubrique d'une plage à l'autre, sauf titre restitué à la
main (clé `debut+n`) de même clé que la rubrique qui précède : les deux
morceaux forment alors une seule rubrique (`continue_sections`).

**Correspondance des rubriques** (`numrev/alignment/sections.py`, partagée
par les deux méthodes, `join` et le viewer) : une rubrique = une suite
contiguë d'ENTRY de même rubrique, identifiée par l'uuid de son titre. Les
rubriques sont alignées par Needleman-Wunsch sur la similarité Jaro-Winkler
de leur clé (seuil 0,8). Le patch `data/alignment/<A>__<B>.sections.csv`
(`left_uuid, right_uuid, left_title, right_title, note`) gagne : une ligne à
deux uuid lie deux rubriques (les composantes connexes forment des groupes
1-1, 1-N, N-M), une ligne à un uuid déclare une rubrique seule. **Les entrées
ne sont appariées qu'entre rubriques appariées** : une rubrique seule perd
toutes ses entrées tant qu'elle n'est pas liée au patch, et tout lien qui
viole cette règle (Dedupe en produit, le patch des entrées aussi) est écarté
(`restrict_to_corresponding`). Les suites sans rubrique (`(sans rubrique :
<document>, à partir de la p. N)`) ne sont jamais appariées.

### Méthode ordonnée : `numrev align nw`

```bash
uv run numrev align nw annuaires/<A> annuaires/<B> --apply [--no-context]
```

Par segment (groupe de rubriques appariées) : Needleman-Wunsch sans
pénalité de trou (seuil τ = 0,75) dont les paires de similarité ≥ 0,9 sont
des **ancres** ; **passe résiduelle** (affectation optimale, ≥ 0,85) pour
les inversions locales ; entre deux ancres, **pair-HMM** gardant les paires
de probabilité a posteriori > 0,5. Similarité `w · JaroWinkler(SUBJ) + (1 −
w) · Indel(texte)` (`--subj-weight`, 0,5 ; texte seul sans SUBJ). Méthode,
formalisation et résultats : [`alignement_ordonne.md`](alignement_ordonne.md)
(pourquoi pas de passe résiduelle plus fine : sa § 6.1). Sources : `nw`,
`nw-contexte` (score = probabilité a posteriori), `nw-residuel`. La console
affiche les transitions estimées et l'accord avec le `.dedupe.csv` s'il
existe.

### Méthode Dedupe : `numrev align dedupe`

```bash
uv run numrev align dedupe annuaires/<A> annuaires/<B> [--label] --apply
uv run numrev align dedupe annuaires/<A> annuaires/<B> --patch-only --apply   # réapplique le patch, sans Dedupe
```

Champs comparés en minuscules : rubrique (par la **clé commune de son
groupe**, aussi substituée à la lecture dans les paires d'entraînement),
SUBJ, texte. L'étiquetage en console (`y` / `n` / `u` / `f`) est enregistré
aussitôt dans `data/alignment/<A>__<B>.training.json` et réutilisé ensuite.
`--threshold` : score minimal (0,5). `prepare_training` prend quelques
minutes sur ~17 000 × 16 000 entrées ; Dedupe 3.0.3 exige `btrees<6`.

### Format des correspondances

`left_file, left_uuid, right_uuid, right_file, score, source`, puis un
instantané pour la relecture : `left_section, right_section,
left_tagged_text, right_tagged_text`. `source` : `dedupe`, `manuel`,
`manuel-incertain`, `nw`, `nw-contexte`, `nw-residuel` ; `score` vide pour
une paire saisie à la main. Une entrée absente du fichier n'a pas de
correspondance.

### Patch des entrées

`data/alignment/<A>__<B>.patch.csv` (`numrev/alignment/patch.py`) :
`left_file, left_uuid, right_uuid, right_file, left_section, right_section,
left_tagged_text, right_tagged_text, note, certitude`. Deux uuid : ces
entrées se correspondent ; un seul : cette entrée est sans correspondance.
Le patch gagne (tout lien automatique qui touche un de ses uuid est
écarté ; une paire validée garde son score). `certitude = incertaine`
(colonne facultative) donne la source `manuel-incertain`. Un uuid disparu
est **réancré** sur l'entrée unique de même texte dans la même rubrique
(réécrit par `align nw` et `align dedupe`, en mémoire seulement pour `join`
et le viewer) ; sinon la ligne est **orpheline** : conservée, non appliquée,
et les commandes paniquent sauf `--force`. Un uuid sur deux lignes, ou une
ligne sans uuid, est une erreur.

### Relecture : `numrev view alignment`

Viewer Streamlit en lecture seule
([`guide_relecture_alignement.md`](guide_relecture_alignement.md)) : on
choisit un `.nw.csv` ou un `.dedupe.csv`, les deux patchs sont appliqués en
mémoire. Une table dans l'ordre naturel (paires et entrées de gauche seules
dans l'ordre de gauche ; une entrée de droite seule après la paire qui
contient l'entrée de droite appariée qui la précède), statut de chaque
ligne, incertitude et motifs, boutons qui copient un uuid ou une ligne de
patch, encart des rubriques, export CSV des lignes affichées au format de
`join`. Pour le tester : `streamlit.testing.v1.AppTest`, ou un script qui
redéfinit `numrev.paths.ALIGNMENTS_DIR`, `ALIGNMENT_DATA_DIR` et
`ANNUAIRES_DIR` (jamais les dossiers réels).

**Motifs de relecture** (`numrev/alignment/review.py`, partagé par le
viewer, `join` et l'audit) : l'alignement ne change pas, on signale après
coup ce qu'un humain peut corriger.

| Motif | Règle |
|---|---|
| `déduite des voisines (p < 0,9)` | paire `nw-contexte` de probabilité a posteriori < 0,9 |
| `homonyme proche` | paire `nw`, `nw-residuel` ou `dedupe` dont une concurrente du segment est à moins de 0,05 (`--margin`) |
| `candidate non appariée` | deux entrées seules, chacune la plus proche de l'autre, similarité dans [τ ; 0,85[ ; jamais un uuid déclaré seul au patch ; ce n'est pas un lien |

Incertitude **ordinale** (pour trier, pas une probabilité) : `faible` sans
motif, `moyenne` avec un motif, `forte` pour une candidate, plusieurs motifs
ou une paire déduite des voisines de probabilité < 0,7. Sur 1807/1808 :
445 lignes à vérifier pour 14 491 paires.

### Étape 8 — Jointure : `numrev join`

```bash
uv run numrev join annuaires/alignments/<A>__<B>.nw.csv [--excel] [--candidates] --apply
```

Pour les utilisateurs des données : les deux volumes relus, le patch des
entrées appliqué en mémoire (`--no-patch`), une ligne par paire ou entrée
seule dans l'ordre naturel, colonnes en français décrites dans
[`donnees.md`](donnees.md). Sortie : `<entrée sans .csv>.join.csv` ;
`--excel` : séparateur `;` et UTF-8 avec BOM.

### Gold des inversions et audit

```bash
uv run numrev gold alignment annuaires/<A> annuaires/<B> --apply   # une fois par paire
uv run numrev audit alignment data/alignment/<A>__<B>.gold-inversions.csv
```

Le gold tire des paires « inversion » (entrées laissées seules par
Needleman-Wunsch qui croisent au moins une paire ordonnée ; meilleure
partenaire croisée de chaque entrée de gauche), stratifiées par déplacement
× similarité, avec leur poids, à étiqueter dans `meme_entree` (`OUI` /
`NON` / `INCERTAIN`) ; il n'est jamais écrasé sans `--force`. L'audit
(`reports/alignment/<A>__<B>.md`) mesure la précision de la passe
résiduelle, le devenir des paires du gold selon la relecture, la charge de
relecture et les rubriques sans correspondance. **Pour une nouvelle paire**
d'annuaires : lier d'abord les rubriques seules, puis tirer un petit gold
(`--per-stratum 4`), l'étiqueter et lancer l'audit avant de se fier aux
seuils.

## Audit du CRF : `numrev audit crf`

```bash
uv run numrev audit crf [a.lines.csv …] [-o reports/mon_audit]
```

Mesure le CRF de l'étape 2 et l'apport de chaque groupe de features contre
les tables de lignes corrigées (`numrev/crf/silver.py`). Les features sont
calculées sur le `.lines.json` qu'a vu l'annotateur, les classes curées y
étant rattachées par `cle` : le texte curé (marqueurs `#`) ferait fuiter
les étiquettes. Ces classes ne diffèrent des prédictions d'origine que sur
~0,2 % des lignes : les scores absolus sont optimistes. Validation croisée
intra-document (blocs de pages contiguës) et inter-volumes, bootstrap par
page, calibration, information mutuelle, ablations jugées contre des
features placebo, groupes candidats, courbe d'apprentissage, analyse
d'erreurs (`reports/crf/rapport.md`). `--no-learning-curve`,
`--no-candidates`, `--no-single-groups` accélèrent ; `--hyperparams` ajoute
une grille c1 × c2. `LEGACY_GROUPS` reproduit l'ancien jeu de features
(`production_v1`).

## Organisation du code

Paquet `src/numrev/`, installé par `uv sync` avec la commande `numrev`, lancé
depuis la racine du dépôt (chemins relatifs).

| Module | Rôle |
|---|---|
| `cli.py` | Table des commandes ; un module n'est importé que pour la commande lancée ; aide en français |
| `command.py` | Conventions : `Writes` et `--apply`, `CommandError`, options partagées, `require_input` / `require_ranges` / `require_volume` |
| `paths.py` | Noms et emplacements de tous les fichiers ; `Pair` pour une paire d'annuaires |
| `curation.py` | Corrections rejouables : clés, empreintes, capture, application, `CurationConflict`, écriture atomique des CSV |
| `document.py` | Le JSON des lignes (pages → `data_blocks` → `lines`) : lecture et parcours |
| `titles.py` | Niveau et texte des titres, racine `ROOT_UUID` |
| `pipeline/` | Une commande par étape : `ocr`, `extract`, `label`, `tabulate`, `assemble`, `tag`, `export`, `align_nw`, `align_dedupe`, `join` |
| `crf/` | `labels`, `features`, `model` (python-crfsuite), `active_learning` (moteur de l'annotateur), `silver` et `evaluation` (audit) |
| `ner/` | `spans` (empans, `tagged_text`, Label Studio, texte sans Markdown), `gliner`, `suspicion`, `corpus`, `shapes`, `metrics`, `html` |
| `export/` | `layers` (les deux couches, fonctions pures), `braces` (groupes à accolade), `schemas/` (JSON Schema) |
| `alignment/` | `records` (volumes, similarité, CSV des liens), `pair` (chargement d'une paire, bilans), `sections` (rubriques et leur patch), `patch` (patch des entrées, réancrage commun aux deux patchs), `nw`, `sequence`, `pair_hmm`, `review`, `export` (ordre naturel, CSV lisible) |
| `viewers/` | `numrev view` : `alignment.py` (Streamlit), `directory.html` et `ocr.html` (pages autonomes servies en local) |
| `devtools/` | `status`, gold (`ner_gold`, `alignment_gold`), NER (`ner_dataset` = `train-set`, `ner_train` = `train`), audits |
| `stats.py`, `reporting.py` | Bootstrap, intervalles, calibration ; mise en forme des rapports |

Chaque module de commande expose `add_arguments(parser)` et `run(args)`
(et `DESCRIPTION`, l'aide en clair). Options : l'essentiel d'abord, les
options d'expert dans le groupe « Réglages fins » (`fine_tuning`) ; mêmes
noms pour les mêmes choses partout (`--model` = un modèle entraîné,
`--force` = passer outre un refus, `--seed`, `--threshold`, `--margin`…).
Une erreur attendue lève `CommandError` (`Erreur : …`, code 1). Tests :
`uv run python -m unittest discover -s tests` ; mise en forme : `uv run
black src tests` puis `uv run ruff check src tests`.
