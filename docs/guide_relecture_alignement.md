# Guide de relecture de l'alignement entre éditions

Ce guide sert à relire les paires à vérifier dans le viewer d'alignement (`numrev view alignment`, filtre *Incertitude* : moyenne ou forte), et à étiqueter l'échantillon qui sert à évaluer l'alignement (`numrev gold alignment` : des paires d'entrées laissées seules parce que leur ordre diffère d'une édition à l'autre, à trancher dans la colonne `meme_entree`). La question est toujours la même :

> Ces deux entrées désignent-elles **la même entrée de l'annuaire**, d'une édition à l'autre ?

Il s'agit de la même ligne de l'annuaire, avec sa personne ou sa maison de commerce, et pas forcément exactement du même individu. Les conventions ci-dessous valent pour toutes les paires d'annuaires. Les exemples viennent de 1807/1808, à titre d'illustration.

## Dans le viewer

`uv run numrev view alignment` ouvre le viewer dans le navigateur ; on y choisit un alignement (`….nw.csv` ou `….dedupe.csv`, dans `annuaires/alignments/`). Le viewer ne modifie aucun fichier : les décisions s'écrivent à la main dans deux fichiers CSV suivis par git, ouverts dans un tableur ou un éditeur de texte (`<A>` et `<B>` : les noms des dossiers des deux volumes).

| Fichier | Ce qu'on y décide |
|---|---|
| `data/alignment/<A>__<B>.sections.csv` | quelles rubriques se correspondent d'une édition à l'autre |
| `data/alignment/<A>__<B>.patch.csv` | quelles entrées se correspondent, ou n'ont pas d'équivalent |

S'ils n'existent pas encore, les boutons `en-tête` et `en-tête du patch`, en haut de page, copient leur première ligne (les noms des colonnes) : créer le fichier et y coller cette ligne.

### 1. Les rubriques d'abord

Les entrées ne sont comparées qu'entre rubriques qui se correspondent. L'encart **Rubriques** liste les groupes de rubriques appariées, puis les rubriques restées seules, avec la part d'entrées appariées de chaque côté. Une rubrique seule qui a un équivalent dans l'autre édition (renommée, fusionnée, déplacée) se lie dans `….sections.csv`, une ligne par lien, avec les uuid copiés par les boutons de l'encart :

| left_uuid | right_uuid | left_title | right_title | note |
|---|---|---|---|---|
| `3f2…` | `a91…` | Jardiniers-fleuristes, pépiniéristes | Marchands d'arbres | renommée |

Deux rubriques fusionnées dans l'autre édition prennent une ligne chacune, vers la même rubrique. Une ligne avec un seul uuid déclare une rubrique sans équivalent. Les titres et la note ne servent qu'à la relecture.

Une ligne `(sans rubrique : <document>, à partir de la p. N)` regroupe des entrées placées en début de plage, avant le premier titre : elles n'ont pas de rubrique et ne sont jamais appariées. On ne peut pas les lier ici : on restitue leur titre dans la table des lignes (voir le [guide d'annotation des lignes](guide_annotation_lignes.md#une-plage-qui-commence-au-milieu-dune-rubrique)).

Après avoir lié des rubriques, relancer `uv run numrev align nw annuaires/<A> annuaires/<B> --apply` : leurs entrées sont alors comparées.

### 2. Les paires ensuite

Filtrer *Incertitude* sur moyenne et forte. La dernière colonne donne le statut de chaque ligne (`✓ appariée`, `? non appariée · candidate à décider`, `✗ sans correspondance`), puis l'incertitude et son motif (voir ci-dessous) ; le bouton *Légende* les rappelle.

Pour chaque ligne tranchée (voir [Trois réponses](#trois-réponses)), le bouton `copier` met dans le presse-papiers une ligne prête à coller dans `….patch.csv` ; le bouton `incertaine` copie la même ligne avec `certitude=incertaine`. Pour apparier deux entrées que la chaîne n'a pas rapprochées, coller la ligne de l'une et y reporter l'uuid (bouton `uuid`) et le fichier de l'autre.

Recharger la page du viewer montre l'effet des décisions : le viewer et `numrev join` appliquent le patch eux-mêmes, il n'y a rien à relancer. (Avec la méthode Dedupe, `numrev align dedupe … --patch-only --apply` régénère en plus son CSV final.)

## Motifs affichés

| Motif | Ce qu'il signale |
|---|---|
| `déduite des voisines (p < 0,9)` | Paire retenue parce que les entrées voisines sont appariées des deux côtés, mais avec une probabilité modeste : vérifier qu'il ne s'agit pas d'un remplacement (une entrée disparue, une autre apparue au même rang). |
| `homonyme proche` | Une autre entrée est presque aussi proche : vérifier qu'on n'a pas apparié le mauvais homonyme. |
| `candidate non appariée` | Deux entrées restées seules, chacune la plus proche de l'autre, mais pas assez semblables pour être appariées automatiquement : décider si elles se correspondent. |

L'incertitude (faible, moyenne, forte) ne sert qu'à ordonner la relecture : elle n'est pas une probabilité.

## Trois réponses

| Réponse | Gold (`meme_entree`) | Patch des entrées |
|---|---|---|
| Même entrée | `OUI` | ligne de la paire (bouton **copier**) |
| Pas la même entrée | `NON` | rien pour une candidate non appariée ; pour une paire retenue à tort, une ligne par entrée, seule (ou la bonne paire si on la connaît) |
| On ne peut pas trancher | `INCERTAIN` | ligne de la paire avec `certitude=incertaine` (bouton **incertaine**) si la paire est plausible ; sinon rien |

`INCERTAIN` est une réponse à part entière, pas un échec. L'export la transmet aux utilisateurs des données (`certitude = incertaine`), qui décident de s'en servir ou non. Mieux vaut `INCERTAIN` qu'un `OUI` ou un `NON` arbitraire.

## Indices qui vont vers « même entrée »

- **Erreur d'OCR** sur un chiffre ou une lettre, souvent dans un texte dégradé : 65 → 63, 5 → 3, une lettre effacée.
- **Variante de graphie** qui se prononce pareil ou presque : V/W, y/g, n/u, un nom étranger francisé, un accent.
- **Coquille des éditeurs** et non de l'OCR (*BaDin* / *DaBin*, *N/P*). Le contexte doit l'appuyer : ni l'un ni l'autre nom n'existe dans l'autre édition, et la rubrique ne contient pas d'autre candidat.
- **Même adresse, ou rue renommée** entre les deux éditions (rue Neuve-Égalité → rue d'Aboukir, rue de la Loi → rue de Richelieu), ou **précision ajoutée** (quartier ajouté pour distinguer deux rues homonymes).
- **Changement de numéro** dans la même rue, ou **déménagement**, quand le nom n'est porté par personne d'autre dans la rubrique.
- **Désignation de la personne qui change** sans changer l'entrée, par défaut :
  - M.e et Mad. (une femme), une femme devenue veuve ;
  - la femme puis le mari ;
  - le père puis le fils ;
  - un associé ajouté ou retiré (« et comp. »).
- **Sous-rubrique** qui apporte de l'information : même spécialité dans les deux éditions.

## Indices qui vont vers « pas la même entrée »

- **L'une des deux entrées a déjà sa correspondante exacte** dans l'autre édition, par exemple le même nom à la même adresse ailleurs dans la rubrique. C'est le cas le plus fréquent de `NON`.
- **Noms différents à l'écrit comme à l'oral**, sans adresse commune. Un risque de mauvaise recopie est alors peu plausible.
- **Remplacement au même rang** : une entrée disparaît, une autre apparaît à sa place, et rien ne les relie.
- **Adresse incompatible** (hors de Paris dans une édition seulement), sans autre indice.

## Quand répondre `INCERTAIN`

- **Nom courant avec des homonymes** dans la rubrique (Martin, Garnier, Morin…) et adresses différentes : rien ne permet de dire lequel est lequel.
- **Plusieurs entrées concurrentes** proches dans l'autre édition (deux Bourquelot rue des Noyers à des numéros différents, aucun à l'adresse attendue).
- **Graphies assez éloignées** et adresse différente, même si aucune autre entrée ne fait concurrence.
- **Cas qui demandent une source extérieure** : renvoi vers une société, changement de raison sociale.

## Ordre de lecture

On lit toujours les entrées dans **l'ordre du document**, celui des CSV (`uid` = `<page>.<bloc>.<ligne>`). Aucun tri alphabétique n'est imposé. Le déplacement d'une entrée dans la liste n'est pas en soi un indice. Dans une longue rubrique, une entrée dont le nom a été mal recopié peut se retrouver loin de sa place attendue.
