# Traiter un annuaire, du PDF aux données

Ce guide suit le traitement d'un annuaire de bout en bout, dans l'ordre, en
disant à chaque étape ce que fait la machine et ce que fait l'humain. Les
commandes se lancent depuis la racine du dépôt (installation : voir le
[README](../README.md)). Le détail de chaque commande est dans
[`pipeline.md`](pipeline.md) et dans `uv run numrev <commande> --help`.

## Vocabulaire

| Terme | Sens |
|---|---|
| **OCR** | La lecture automatique du texte des pages scannées, faite ici par le modèle latynna. |
| **volume** | Un annuaire : un dossier `annuaires/<volume>/` qui contient son PDF complet, `<volume>.pdf`. |
| **plage** | Une suite de pages du PDF traitées ensemble, dans un sous-dossier `<première page>-<dernière page>` (pages du PDF, comptées à partir de 1). Chaque plage est un **document** : ses fichiers s'appellent `<volume>.<plage>.<étape>`. |
| **ligne** | Une ligne imprimée, telle que l'OCR l'a lue. |
| **entrée** | Une personne ou une maison de commerce de l'annuaire, sur une ou plusieurs lignes. |
| **titre**, **rubrique** | Les titres forment un arbre (`#` partie, `##` rubrique, `###` sous-rubrique) ; la rubrique d'une entrée est son titre de niveau `##` (à défaut `#`). |
| **empan** | Un morceau du texte d'une entrée : `SUBJ` (qui ?), `DESC` (quoi ?), `ADDR` (où ?). Le modèle qui les repère est le modèle **NER** (reconnaissance d'entités nommées). |
| **uuid** | L'identifiant d'une entrée ou d'un titre (`2aa4…`, 36 caractères) : il sert à les désigner dans les corrections et à relier les fichiers entre eux. |
| **patch** | Le fichier (dans `data/`, suivi par git, qui en garde toutes les versions) où sont recopiées les corrections humaines d'une étape, pour être réappliquées à chaque relance. |
| **simulation** | Le mode par défaut des commandes qui écrivent : elles calculent et vérifient tout, montrent ce qu'elles écriraient, et n'écrivent qu'avec `--apply`. |

À tout moment, `uv run numrev status` montre, pour chaque plage, les étapes
faites, absentes ou périmées, le nombre de lignes corrigées et de lignes
douteuses qui restent à relire, et la commande à lancer ensuite.

## Préparer le volume

1. Créer le dossier `annuaires/<volume>/` (un nom sans point, par exemple
   `1808_AD75-PER292`) et y placer le PDF complet sous le nom
   `<volume>.pdf`.
2. Choisir les plages (ci-dessous). Rien n'est à créer à la main : la
   commande d'OCR crée le dossier de la plage.

### Choisir les plages

Une plage regroupe des pages qui se suivent et qu'on traite ensemble : la
liste principale, une liste annexe… On laisse de côté les pages qui ne sont
pas des listes (couverture, préface, tables), ou on les met dans une plage
à part.

L'arbre des titres est construit **plage par plage**. Les entrées placées
avant le premier titre d'une plage n'ont donc pas de rubrique, même si leur
rubrique a commencé dans la plage précédente, et l'alignement de deux
éditions ne les apparie jamais. Autant que possible, faire commencer chaque
plage par un titre. Sinon, il faudra restituer à la main le titre de leur
rubrique en tête de la table des lignes (étape 3, voir le
[guide d'annotation des lignes](guide_annotation_lignes.md#une-plage-qui-commence-au-milieu-dune-rubrique)).

Les plages se choisissent une fois pour toutes : le nom de la plage entre
dans le nom de tous ses fichiers, de ses patchs et dans l'identifiant de
chaque entrée. Redécouper un volume après coup oblige à refaire ses
corrections.

## Étape 0 · OCR

Sur la machine GPU, lancer le serveur du modèle latynna, puis l'OCR de
chaque plage :

```bash
uv sync --group ocr
uv run latynna serve --detach && uv run latynna serve --wait
uv run numrev ocr annuaires/<volume>/<première>-<dernière> --apply
```

Sans `--apply`, la commande vérifie seulement la plage et le PDF. Un OCR
existant n'est remplacé qu'avec `--force`.

**Contrôle :** `uv run numrev view ocr annuaires/<volume>/<première>-<dernière>`
ouvre la plage dans le navigateur : le texte lu est posé sur le scan, avec
les mots de faible confiance surlignés (Ctrl+C dans le terminal pour fermer).

## Étape 1 · Extraction des lignes

```bash
uv run numrev extract annuaires/<volume> --apply
```

Rien à faire à la main. La commande s'arrête si l'OCR contient du texte
hors de toute ligne, en indiquant la page et le bloc.

## Étape 2 · Annotation des lignes

```bash
uv run numrev label annuaires/<volume>/<plage>
```

L'annotateur propose les lignes une par une (par deux, avec leur voisine) ;
on répond par une touche, selon le
[guide d'annotation des lignes](guide_annotation_lignes.md) :
`1` B-ENTRY, `2` I-ENTRY, `3` SUB-ENTRY, `4` B-TITLE, `5` I-TITLE, `0`
OUT OF SCOPE, `6` incertain, `u` annuler, `p` toute la page en
OUT OF SCOPE, `q` arrêter. Un modèle (CRF) réapprend après chaque réponse,
propose ensuite les lignes dont il est le moins sûr, et classe lui-même
toutes les autres.

- On peut s'arrêter à tout moment (`q`) : la session est enregistrée après
  chaque réponse, et la commande reprend où on s'était arrêté.
- À chaque arrêt, le fichier annoté (`….labeled.json`) est écrit avec les
  classes humaines et celles du modèle : on peut passer à l'étape 3 sans
  avoir tout annoté, et revenir annoter plus tard.
- On s'arrête quand les lignes proposées sont presque toujours bien prédites
  par le modèle : la suite se corrige plus vite dans la table des lignes.
- Si le fichier de lignes a changé depuis la session (OCR refait, nouvelle
  extraction), la session ne se reprend pas : `--reset-session`
  recommence l'annotation.

## Étape 3 · Table des lignes et curation

```bash
uv run numrev tabulate annuaires/<volume> --apply
```

On ouvre `….lines.csv` dans un tableur (LibreOffice Calc, OpenRefine) et on
corrige les classes et le texte, en mettant `corrige = oui` sur chaque ligne
corrigée ou validée, selon le [guide de la curation](guide_curation.md). Par
ordre de priorité :

1. les **niveaux de titres** (`#`, `##`, `###`) : ils décident de l'arbre des
   titres et des rubriques comparées à l'étape 7 ;
2. les **cas signalés par l'étape 4** (lancer `numrev assemble`, lire le
   rapport `….entities.report.txt`, voir ci-dessous) ;
3. les **lignes douteuses** : filtrer `douteuse = oui` (un mot lu avec une
   faible confiance), trier sur `confiance` pour voir les pires d'abord.

On relance `numrev tabulate` après chaque séance : la commande capture les
corrections dans `data/curation/<document>.lines.patch.csv` et les
réapplique.

**Corriger le texte ici, pas plus tard.** Le texte d'une entrée ne se corrige
que dans la table des lignes : une correction de texte faite ensuite
invalide les corrections NER de cette entrée (étape 5), qu'il faut alors
refaire.

## Étape 4 · Entrées et arbre des titres

```bash
uv run numrev assemble annuaires/<volume> --apply
```

La commande recolle les lignes en entrées et en titres, rattache chaque
entrée à son titre et écrit un rapport, `….entities.report.txt` : l'arbre
des titres avec le nombre d'entrées sous chacun, puis les cas à vérifier
(titres sans `#`, suites sans entrée, ruptures de l'ordre alphabétique,
classes incertaines). Il signale aussi, comme la console, les entrées placées
avant le premier titre de la plage : elles n'ont pas de rubrique tant qu'on
n'a pas restitué leur titre en tête de la table des lignes. On corrige la
table des lignes (étape 3) et on relance `tabulate` puis `assemble` jusqu'à
ce que l'arbre soit juste.

## Étape 5 · Empans SUBJ / DESC / ADDR

```bash
uv run numrev tag annuaires/<volume> --apply
```

Le modèle NER découpe chaque entrée en empans. Il tourne sur le processeur,
sans GPU (quelques minutes par volume). Pour relire :

- le viewer du volume (voir l'[étape 6](#étape-6--livrable-du-volume)) montre
  chaque entrée avec ses empans colorés, sur la page du PDF ; le filtre
  « suspectes seulement » donne la file de relecture. Il affiche le livrable :
  lancer `numrev export` avant de l'ouvrir, et le relancer après des
  corrections pour les voir ;
- dans `….ner.csv`, filtrer la colonne `ner_suspect` (non vide) ou trier sur
  `ner_confidence`.

On corrige `tagged_text` (et `parent_uuid` pour rattacher une entrée à une
autre rubrique) avec `corrige = oui`, selon le
[guide d'annotation NER](guide_annotation_ner.md), sans toucher au texte
entre les balises, puis on relance `numrev tag`.

## Étape 6 · Livrable du volume

Une fois toutes les plages d'un volume traitées jusqu'à l'étape 5 :

```bash
uv run numrev export annuaires/<volume> --apply
```

La commande rassemble les plages du volume en deux fichiers, le livrable de
référence de l'annuaire. Le premier, `<volume>.document.json`, décrit le
document : chaque page et chaque ligne, avec sa place sur la page, le texte
lu par l'OCR et le texte corrigé. Le second, `<volume>.structure.json`,
contient la structure : les titres, les entrées et leurs empans, chacun
relié aux lignes dont il vient. Leur format est décrit dans
[`format_export.md`](format_export.md).

**Rien n'est à corriger dans ces fichiers.** Ils se recalculent à partir des
tables des lignes et des CSV NER : pour corriger le livrable, on corrige ces
fichiers comme aux étapes 3 et 5, on relance les étapes suivantes, puis
l'export. La commande refuse de s'exécuter tant qu'une plage n'est pas à
jour (`numrev status` dit quoi relancer). Elle accepte en revanche un volume
dont seule une partie des pages est traitée.

Le rapport `<volume>.export.report.txt` liste les **groupes à accolade** à
vérifier. Une accolade imprimée relie plusieurs personnes à un texte commun
(« Fieffé, Pastoret… } Membres du conseil ») ; l'export donne chaque
personne avec le texte commun dont elle hérite, à condition que chaque
ligne du groupe porte son signe `}` ou `{` et que le texte commun soit sur
la première ligne (voir le
[guide d'annotation des lignes](guide_annotation_lignes.md)). Le rapport
nomme la page et la ligne de chaque cas ; on le corrige dans la table des
lignes.

Pour consulter le livrable :

```bash
uv run numrev view directory annuaires/<volume>
```

Le navigateur s'ouvre sur le volume : à gauche la page du PDF avec ses
lignes, à droite les titres et les entrées de la page, empans en couleur.
Cliquer une entrée la montre sur la page ; cliquer une ligne ouvre la fiche
de son entrée (rubrique, confiances, lignes telles que l'OCR les a lues).
Des filtres organisent la relecture (entrées suspectes, confiance basse,
groupes à accolade…), à côté de l'arbre des rubriques, d'un tableau des
entrées (téléchargeable en CSV), de statistiques et du rapport d'export,
dont chaque cas mène à sa ligne. Après une correction, relancer l'export
(quelques secondes) pour la voir. Ctrl+C dans le terminal arrête la
commande.

## Étape 7 · Alignement de deux éditions

Une fois deux volumes traités jusqu'à l'étape 5 :

```bash
uv run numrev align nw annuaires/<A> annuaires/<B> --apply
uv run numrev view alignment
```

La commande compare les entrées des deux éditions rubrique par rubrique, en
suivant l'ordre des listes ; le viewer sert à relire le résultat. Le
[guide de relecture de l'alignement](guide_relecture_alignement.md#dans-le-viewer)
détaille ces deux temps :

1. **Les rubriques d'abord.** La console de `align nw` et l'encart
   *Rubriques* du viewer listent les rubriques restées sans correspondance :
   leurs entrées ne sont jamais appariées. Lier celles qui ont un équivalent
   dans `data/alignment/<A>__<B>.sections.csv`, puis relancer `align nw`.
2. **Les paires ensuite.** Dans le viewer, filtrer *Incertitude* sur
   moyenne et forte et trancher chaque ligne. Les décisions se collent dans
   `data/alignment/<A>__<B>.patch.csv` grâce aux boutons « copier » du
   viewer ; recharger la page en montre l'effet.

`numrev align dedupe` est une autre méthode, qui apprend à partir de paires
qu'on étiquette en console ; voir [`pipeline.md`](pipeline.md#méthode-dedupe--numrev-align-dedupe).

## Étape 8 · Jointure pour les utilisateurs des données

```bash
uv run numrev join annuaires/alignments/<A>__<B>.nw.csv --excel --apply
```

Le fichier `….nw.join.csv` (une ligne par paire ou entrée seule, patch
appliqué) est le livrable pour les historiens ; ses colonnes sont décrites
dans [`donnees.md`](donnees.md).

## Sauvegarder son travail

Les fichiers de `annuaires/` ne sont pas suivis par git ; les corrections le
sont, dans les patchs de `data/`. Après une séance : relancer la commande de
l'étape (elle met le patch à jour), puis commiter les patchs, à part du code
(`git add data/ && git commit -m "données: …"` ; voir le
[guide de la curation](guide_curation.md#où-sont-gardées-les-corrections-et-comment-les-sauvegarder)).

## Ce qui survit à une relance

| Ce qu'on relance | Corrections des lignes | Corrections NER | Patchs d'alignement |
|---|---|---|---|
| Nouvelle annotation ou nouveau CRF (étapes 2-3) | gardées | gardées, sauf pour une entrée redécoupée (lignes rattachées autrement) | gardés |
| Nouveau modèle NER (étape 5) | — | gardées | gardés |
| Livrable du volume (étape 6) | — | — | — (il ne contient aucune correction propre : il se recalcule) |
| Correction du texte d'une entrée (étape 3) | — | celle de l'entrée est à refaire | gardés |
| **Nouvel OCR** (étape 0) | **perdues** pour toute ligne dont le texte lu change | perdues avec elles | gardées si l'entrée est retrouvée par son texte ; sinon la commande s'arrête et les liste |

Les clés des lignes viennent du texte lu par l'OCR : refaire l'OCR (nouveau
modèle, PDF remplacé) change le texte de nombreuses lignes, donc leurs clés,
et les corrections correspondantes ne s'appliquent plus. La commande le
signale (elle s'arrête et liste les corrections inapplicables) ; `--force`
les abandonne, et l'historique git les garde.
