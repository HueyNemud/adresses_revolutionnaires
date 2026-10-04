# Guide de relecture de l'alignement entre éditions

Ce guide sert à relire les paires à vérifier dans `numrev view alignment` (vue **Relecture**, dont la file contient par défaut les lignes à vérifier, pas encore décidées) et à étiqueter un gold d'inversions (`numrev gold alignment`, colonne `meme_entree`). La question est toujours la même :

> Ces deux entrées désignent-elles **la même entrée de l'annuaire**, d'une édition à l'autre ?

Il s'agit de la même ligne de l'annuaire, avec sa personne ou sa maison de commerce, et pas forcément exactement du même individu. Les conventions ci-dessous valent pour toutes les paires d'annuaires. Les exemples viennent de 1807/1808, à titre d'illustration.

## Motifs affichés

| Motif | Ce qu'il signale |
|---|---|
| `déduite des voisines (p < 0,9)` | Paire retenue parce que les entrées voisines sont appariées des deux côtés, mais avec une probabilité modeste : vérifier qu'il ne s'agit pas d'un remplacement (une entrée disparue, une autre apparue au même rang). |
| `homonyme proche` | Une autre entrée est presque aussi proche : vérifier qu'on n'a pas apparié le mauvais homonyme. |
| `candidate non appariée` | Deux entrées restées seules, chacune la plus proche de l'autre, mais pas assez semblables pour être appariées automatiquement : décider si elles se correspondent. |

L'incertitude (faible, moyenne, forte) ne sert qu'à ordonner la relecture : elle n'est pas une probabilité.

## Trois réponses

| Réponse | Gold (`meme_entree`) | Bouton du viewer (raccourci) | Ligne(s) de patch produites |
|---|---|---|---|
| Même entrée | `OUI` | **✓ Même entrée** (`V`) | la paire |
| Pas la même entrée | `NON` | **✗ Pas la même entrée** (`X`) | une ligne par entrée, seule ; si l'une a un autre partenaire, l'apparier ensuite (*Rapprochements possibles* ou vue Documents) |
| On ne peut pas trancher | `INCERTAIN` | **≈ Probablement (incertaine)** (`I`) si la paire est plausible ; sinon **↷ Passer** (`→`) | la paire avec `certitude=incertaine` |

Le patch ne sait pas dire « pas avec celle-là » : *Pas la même entrée*
déclare les deux entrées sans correspondance, ce qui retire aussi la
candidate de la file. Une entrée seule se confirme avec **✓ Confirmer sans
correspondance** (`V`).

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

## Relire dans le viewer

- **Relecture** : une ligne à la fois. Les caractères qui diffèrent entre
  les deux textes sont surlignés. Sous la carte, les **rapprochements
  possibles** donnent les entrées les plus proches de l'autre côté (la
  concurrente d'un `homonyme proche` y est signalée), chacune avec un bouton
  **Apparier**. Le **contexte** montre les deux annuaires autour de la ligne,
  chacun dans son ordre. Après une décision, on passe à la ligne suivante ;
  `←` revient en arrière, `Ctrl+Z` annule la dernière décision. La note
  facultative va dans la colonne `note` du patch.
- **Table** : toutes les lignes filtrées ; **Relire →** ouvre une ligne dans
  la vue Relecture.
- **Documents** : les deux annuaires côte à côte, titres et lignes hors
  sujet compris. Les liens qui se croisent (orange) signalent une inversion.
  Cliquer une entrée de chaque côté propose de les apparier.

Les décisions vont dans un journal gardé par le navigateur (il survit à un
rechargement). En local, **Enregistrer** écrit le patch
`data/alignment/<A>__<B>.patch.csv` ; sur la copie hébergée,
**Télécharger le patch** donne le fichier complet à transmettre, à déposer
tel quel dans `data/alignment/`.

## Ordre de lecture

On lit toujours les entrées dans **l'ordre du document**, celui des CSV (`uid` = `<page>.<bloc>.<ligne>`). Aucun tri alphabétique n'est imposé. Le déplacement d'une entrée dans la liste n'est pas en soi un indice. Dans une longue rubrique, une entrée dont le nom a été mal recopié peut se retrouver loin de sa place attendue.
