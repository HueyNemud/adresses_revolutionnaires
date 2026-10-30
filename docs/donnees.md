# Les données produites

Ce document s'adresse aux utilisateurs des données (historiens) : ce que
contiennent les fichiers livrés, comment les lire, et leurs limites. La
façon de les produire est dans le [guide de traitement](guide_traitement.md).

Trois livrables :

| Fichier | Contenu |
|---|---|
| `annuaires/<volume>/<volume>.document.json` et `<volume>.structure.json` | Le **livrable de référence d'un volume** : le document (pages, lignes, leur place sur la page, texte lu et texte corrigé) et sa structure (titres, entrées, sujets, descriptions, adresses, personnes d'un groupe à accolade), chaque information reliée aux lignes dont elle vient. Format détaillé dans [`format_export.md`](format_export.md). |
| `annuaires/<volume>/<plage>/<volume>.<plage>.ner.csv` | Un **annuaire structuré** en tableau : ses entrées, ses titres et le découpage de chaque entrée en sujet, description et adresse. Un fichier par plage de pages ; c'est aussi le fichier de travail où l'on corrige le découpage. |
| `annuaires/alignments/<A>__<B>.<méthode>.join.csv` | La **jointure de deux éditions** : pour chaque entrée, son équivalent dans l'autre édition s'il a été trouvé. |

Les deux JSON sont décrits champ par champ dans
[`format_export.md`](format_export.md), avec des exemples de lecture en
Python. Les autres livrables sont des CSV en UTF-8. Une jointure produite
avec `--excel` utilise le séparateur `;` et s'ouvre directement dans un
tableur réglé en français.

## L'annuaire structuré (`….ner.csv`)

Une ligne par élément de l'annuaire, dans l'ordre du document : une
**entrée** (`ENTRY`, une personne ou une maison de commerce), un **titre**
(`TITLE`) ou une ligne **hors champ** (`OUT OF SCOPE` : titre courant,
numéro de page…).

| Colonne | Contenu |
|---|---|
| `uuid` | Identifiant de l'élément (voir [Identifiants](#identifiants)). |
| `parent_uuid` | `uuid` du titre sous lequel il est rangé ; `00000000-0000-0000-0000-000000000000` pour la racine (titres de premier niveau, éléments placés avant tout titre). |
| `entity` | `ENTRY`, `TITLE` ou `OUT OF SCOPE`. |
| `markdown` | Texte de l'élément, tel que lu par l'OCR et corrigé à la main. L'italique est écrit `*…*`, le gras `**…**` ; un titre commence par `#` (partie), `##` (rubrique) ou `###` (sous-rubrique). Les lignes d'une entrée sont jointes par une espace, ou par un saut de ligne pour une ligne rattachée (seconde personne à la même adresse, complément). |
| `tagged_text` | Pour une entrée, le même texte balisé : `<SUBJ>Dupont</SUBJ>, <DESC>boulanger</DESC>, <ADDR>rue de la Paix, 12</ADDR>.` `SUBJ` : qui (nom, prénoms, état civil, associés) ; `DESC` : quoi (profession, spécialité, fonction, renvoi) ; `ADDR` : où (adresse, section, repères). Conventions : [guide d'annotation NER](guide_annotation_ner.md). Les caractères `<` et `>` du texte sont écrits `&lt;` et `&gt;`. |
| `subject_count`, `description_count`, `address_count` | Nombre d'empans de chaque sorte dans `tagged_text`. |
| `corrige` | `oui` : le balisage ou le rattachement de l'entrée a été relu par un humain ; vide : sortie du modèle. |
| `ner_confidence`, `ner_suspect` | Confiance du modèle (score de son empan le moins sûr) et motifs pour lesquels l'entrée mérite une relecture (`aucun empan`, `score bas`, `texte non couvert`, `SUBJ absent en tête`, `signature inhabituelle`). Ce sont les valeurs du modèle, gardées telles quelles sur une entrée relue. |
| `page_index` | Page(s) du PDF du volume, comptées à partir de 1 ; plusieurs valeurs séparées par des virgules quand l'élément court sur plusieurs lignes. |
| `uid` | Position de chacune de ses lignes : `page.bloc.ligne`. |
| `confiance`, `douteuse` | Lecture de l'OCR : la plus faible confiance d'un mot de l'élément (`0,44`), et `oui` si elle est sous le seuil de doute (0,9). Vides si l'OCR n'a pas donné de scores, et pour une ligne ajoutée à la main. |
| `classe`, `provenance`, `probability`, `timestamp` | Classe de chaque ligne et son origine : `human` (annotée à la main), `model` (prédite par le modèle, avec sa probabilité), `unclassified` (ni l'un ni l'autre) ; vides pour une ligne ajoutée à la main. La classe a pu être corrigée ensuite dans la table des lignes. |
| `cle` | Clés des lignes de l'élément. Une clé qui contient `+` est celle d'une ligne ajoutée à la main, absente de l'OCR : ligne oubliée par l'OCR, ou titre restitué en tête de plage (`debut+1`, voir [Limites connues](#limites-connues)). |
| `chunk_index`, `data_block_index`, `line_index`, `data_block_bbox`, `data_block_label`, `empreinte` | Colonnes techniques (blocs de l'OCR, contrôle des corrections). |

**Retrouver la rubrique d'une entrée.** Remonter les `parent_uuid` jusqu'au
titre qui commence par `##` (à défaut `#`). Le chemin complet des titres
s'obtient de la même façon. En Python :

```python
import pandas as pd

rows = pd.read_csv("1808_AD75-PER292.7-186.ner.csv", dtype=str, keep_default_na=False)
titles = rows[rows.entity == "TITLE"].set_index("uuid")

def path(parent):
    chain = []
    while parent in titles.index:
        chain.insert(0, titles.at[parent, "markdown"])
        parent = titles.at[parent, "parent_uuid"]
    return " > ".join(chain)

entries = rows[rows.entity == "ENTRY"].assign(titres=lambda df: df.parent_uuid.map(path))
```

## La jointure de deux éditions (`….join.csv`)

Une ligne par paire d'entrées qui se correspondent, ou par entrée restée
seule, dans l'ordre des listes de l'édition de gauche.

| Colonne | Contenu |
|---|---|
| `statut` | `apparié` ; `gauche seulement` ou `droite seulement` (aucun équivalent trouvé) ; `candidate` (deux entrées non appariées mais proches, seulement si l'export a été demandé avec les candidates). |
| `certitude` | Pour une paire : `relue` (décidée par un humain), `incertaine` (un humain l'a jugée plausible sans pouvoir trancher) ou `automatique`. |
| `niveau_incertitude` | Pour une paire automatique ou une candidate : `faible`, `moyenne` ou `forte`. C'est un ordre de relecture, pas une probabilité. |
| `motifs_relecture` | Pourquoi l'incertitude n'est pas faible : `déduite des voisines (p < 0,9)`, `homonyme proche`, `candidate non appariée` (séparés par « \| »). |
| `score`, `methode` | Score et méthode de la mise en correspondance. Alignement par l'ordre des listes : `nw` (paire très semblable, à sa place dans les deux listes), `nw-contexte` (paire moins semblable, retenue parce que ses voisines sont appariées ; le score est alors une probabilité), `nw-residuel` (paire très semblable mais déplacée dans la liste). Autres : `dedupe` (méthode Dedupe), `manuel` et `manuel-incertain` (décidée par un relecteur). Le score n'est comparable qu'au sein d'une même méthode. |
| `gauche_…`, `droite_…` | Pour chaque édition : `volume`, `page`, `rubrique` (titre lisible), `texte` (sans mise en forme), `sujet`, `description`, `adresse` (texte des empans ; plusieurs empans d'une sorte séparés par « \| »), `texte_balise`, `uuid`. |

**Lire les entrées seules.** `gauche seulement` ne veut pas dire que la
personne a disparu de l'autre édition. Une entrée reste seule quand :

- son texte a trop changé pour que l'équivalent soit retrouvé (nom réécrit,
  déménagement et changement de désignation à la fois…) ;
- sa rubrique n'a pas d'équivalent reconnu dans l'autre édition : les
  entrées ne sont comparées qu'entre rubriques qui se correspondent ;
- elle n'a pas de rubrique : c'est le cas des entrées placées en début de
  plage, avant le premier titre de la plage, quand leur titre n'a pas été
  restitué (voir [Limites connues](#limites-connues)).

**Choisir les paires à utiliser.** Pour une analyse prudente, garder les
paires `relue` et les paires `automatique` d'incertitude `faible`. Les
paires `incertaine` sont celles qu'un relecteur n'a pas pu trancher : à
utiliser en connaissance de cause.

## Identifiants

L'`uuid` d'un élément est calculé à partir du nom du document (volume et
plage) et du texte lu par l'OCR sur sa première ligne. Il ne change pas
quand on relance la chaîne (corrections, nouveaux modèles), ce qui permet
de suivre une entrée d'une version des données à l'autre et de relier
l'annuaire structuré au livrable JSON du volume (même `uuid` d'entrée) et à
la jointure (`gauche_uuid`, `droite_uuid`). Il change
si l'OCR du volume est refait et que le texte lu de cette ligne change, ou
si les plages du volume sont redécoupées.

## Limites connues

- **OCR** : les erreurs de lecture que la relecture n'a pas corrigées
  restent dans le texte. `douteuse = oui` signale les lignes où un mot a été
  lu avec une faible confiance ; une erreur lue avec assurance n'est pas
  signalée.
- **Empans** : hors des entrées marquées `corrige = oui`, le découpage est
  celui du modèle ; `ner_suspect` signale les entrées les plus douteuses.
- **Accolades** : quand l'annuaire relie plusieurs personnes par une
  accolade à un texte commun, le `….ner.csv` et la jointure en font une
  seule entrée (une ligne par personne, jointes par un saut de ligne), où le
  texte commun n'est écrit qu'une fois. Seul le livrable JSON du volume
  donne chaque personne avec le texte commun dont elle hérite
  ([`format_export.md`](format_export.md#groupes-à-accolade)).
- **Entrées en début de plage** : le volume est traité par plages de pages,
  et l'arbre des titres est propre à chaque plage. Quand une plage commence
  au milieu d'une rubrique, le curateur restitue le titre de cette rubrique
  en tête de la plage : c'est une ligne ajoutée, recopiée du titre imprimé
  plus haut, ou déduite des entrées quand aucune page ne le porte. Rien dans
  son texte ne la distingue d'un titre lu : on la reconnaît dans le
  `….ner.csv` à sa clé (`cle` = `debut+1`) et à sa `confiance` vide ; la
  colonne `rubrique` de la jointure ne fait pas la différence. Sans titre
  restitué, ces entrées n'ont pas de rubrique et ne sont jamais appariées
  dans la jointure.
