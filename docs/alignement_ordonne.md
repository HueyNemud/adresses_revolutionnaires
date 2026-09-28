# Aligner deux éditions en suivant l'ordre : Needleman-Wunsch et pair-HMM

`align_directories_nw.py` aligne les entrées de deux éditions d'un annuaire sans apprentissage supervisé, en exploitant ce que Dedupe ignore : **l'ordre des entrées**. Ce document présente la méthode, ses hypothèses et ses limites. Les chiffres viennent de 1807 → 1808 (17 388 et 16 093 entrées).

## 1. Constat : l'ordre est un signal fort

- **Les rubriques gardent le même ordre** : sur 1807/1808, aucune paire de rubriques appariées n'est inversée. Leurs titres varient (« Liste » / « Listes de non-commerçans », « Sellieres » / « Selliers »), donc on les compare par similarité et non par égalité.
- **Dans une rubrique, l'ordre des entrées est presque stable** : 96 % des liens Dedupe sont dans l'ordre. Les exceptions sont surtout de petites inversions du tri alphabétique (« Gueniot » / « Guéniot »).

**En clair** : pour savoir qui est qui d'une édition à l'autre, il ne suffit pas de comparer les textes. Il faut aussi regarder *où* se trouve chaque entrée.

## 2. Similarité de deux entrées

On reprend les champs de Dedupe (en minuscules) :

$$
\mathrm{sim}(a,b) = w\,\mathrm{JW}(\mathrm{subj}_a,\mathrm{subj}_b) + (1-w)\,\mathrm{Indel}(\mathrm{text}_a,\mathrm{text}_b), \qquad w = 0{,}5
$$

- JW est la similarité de Jaro-Winkler, adaptée au nom (SUBJ). Elle est courte et donne du poids au début de la chaîne.
- Indel est la distance d'édition normalisée, calculée sur le texte complet.
- Si l'une des deux entrées n'a pas de SUBJ, on n'utilise que le texte.

## 3. Première passe : Needleman-Wunsch

On aligne d'abord les suites de rubriques (sur la similarité JW de leurs titres), puis les entrées de chaque paire de rubriques. Needleman-Wunsch cherche l'appariement **croissant** (qui ne croise jamais) maximisant

$$
\sum_{(i,j)\ \text{appariés}} \big(\mathrm{sim}(a_i,b_j) - \tau\big), \qquad \tau = 0{,}75,
$$

sans pénalité de trou. Le calcul se fait par programmation dynamique : $H_{i,j} = \max(H_{i-1,j},\ H_{i,j-1},\ H_{i-1,j-1} + \mathrm{sim}_{ij} - \tau)$.

**En clair** : on parcourt les deux listes en parallèle, comme deux personnes qui pointent la même liste d'invités. Une paire n'est retenue que si elle dépasse le seuil. Sauter une entrée (ajout ou suppression) ne coûte rien.

**Limite** : chaque paire est jugée seule contre un seuil fixe. Pagès (agent de change) est à la rue de l'Échiquier en 1807 et au boulevard Montmartre en 1808 : sim = 0,742 < 0,75, donc pas de paire. Pourtant, ses deux voisins, Mounier et Péan de Saint-Gilles, sont appariés des deux côtés et l'encadrent. Pour Needleman-Wunsch, « un Pagès disparaît, un autre Pagès apparaît au même endroit » ne coûte rien, alors que c'est une coïncidence improbable.

## 4. Le contexte formalisé : un pair-HMM

### Modèle

On voit l'alignement comme le chemin d'un modèle de Markov caché à trois états (Durbin et al., *Biological Sequence Analysis*, ch. 4) :

| état | signification |
|---|---|
| **M** | une entrée de gauche et une de droite se correspondent |
| **X** | entrée seulement à gauche (supprimée) |
| **Y** | entrée seulement à droite (ajoutée) |

Les **transitions** $t_{uv} = P(\text{état suivant } v \mid \text{état } u)$ décrivent la fréquence des ajouts et suppressions. La transition Y → X est interdite, pour qu'un trou mixte n'ait qu'une seule écriture : d'abord les suppressions, puis les ajouts.

Chaque paire M apporte un **rapport de vraisemblance** :

$$
\mathrm{LR}(s) = \frac{p(s \mid \text{même entrée})}{p(s \mid \text{entrées différentes})}, \qquad s = \mathrm{sim}(a_i,b_j)\ \text{discrétisée en 20 classes}.
$$

Le score d'un chemin $\pi$ est $\prod t_{uv} \times \prod_{M} \mathrm{LR}(s_{ij})$. La vraisemblance totale $Z$ est la somme sur tous les chemins.

### Probabilité a posteriori

L'algorithme forward-backward calcule, pour chaque paire candidate,

$$
P(i \leftrightarrow j \mid \text{les deux listes}) = \frac{\sum_{\pi \ni (i,j)} \mathrm{score}(\pi)}{Z}.
$$

On garde les paires dont cette probabilité dépasse 0,5. Chaque entrée a au plus une paire au-dessus de 0,5, et ces paires forment toujours un alignement croissant et un-à-un. Le `score` de ces paires (`nw-contexte`) est cette probabilité, directement interprétable.

### Le cas Pagès, en chiffres

Entre Mounier et Péan, deux explications sont en concurrence. Avec les transitions estimées sur 1807/1808 :

$$
\underbrace{t_{MM}^2 \cdot \mathrm{LR}(0{,}742)}_{\text{Pagès = Pagès}} \quad\text{contre}\quad \underbrace{t_{MX}\, t_{XY}\, t_{YM}}_{\text{un Pagès part, un autre arrive}}
$$

$$
\text{cote a priori} = \frac{0{,}774^2}{0{,}161 \times 0{,}152 \times 0{,}870} \approx 28, \qquad \mathrm{LR}(0{,}742) \approx e^{-2{,}3} \approx 0{,}1
$$

$$
P(\text{Pagès} \leftrightarrow \text{Pagès}) = \frac{28 \times 0{,}1}{1 + 28 \times 0{,}1} \approx 0{,}73.
$$

**En clair** : pris isolément, une similarité de 0,742 penche plutôt vers « deux personnes différentes » (environ 1 contre 10). Mais une suppression et un ajout qui tombent exactement au même rang sont environ 28 fois moins probables qu'une continuité. Au total, la paire l'emporte avec une probabilité de 73 %. Le même Pagès au milieu d'un large trou (plusieurs entrées ajoutées et supprimées autour) n'aurait pas cet avantage, et une paire vraiment dissemblable (sim ≈ 0,4, LR ≈ $e^{-6}$) n'est jamais sauvée par le contexte.

## 5. Estimer les paramètres sans étiquettes

**Émissions : fixes, estimées hors du contexte.**

- $p(s \mid \text{même})$ vient des paires dont le **SUBJ est identique et unique** des deux côtés de la rubrique (≈ 9 000 paires). Elles sont choisies sans regarder ni l'ordre ni la similarité. L'échantillon est biaisé vers le haut, puisqu'un SUBJ identique garantit $s \geq 0{,}5$. Il est donc **prudent** : il sous-estime les vraies paires peu similaires.
- $p(s \mid \text{différentes})$ vient des **voisines des paires sûres**, $(i, j \pm d)$ et $(i \pm d, j)$ pour $d \leq 3$. C'est le bon modèle nul : une paire candidate est comparée à des entrées proches dans l'ordre alphabétique, qui partagent souvent leurs initiales. Des paires prises au hasard se ressembleraient beaucoup moins et gonfleraient artificiellement le rapport de vraisemblance.
- $\mathrm{LR}(s)$ est contraint à **croître** avec $s$ (rapport de vraisemblance monotone). Sans cela, une classe basse et presque vide paraîtrait favorable.

**Transitions : estimées par EM (Baum-Welch).** On remplace $t_{uv}$ par le nombre attendu de transitions $u \to v$ sous le modèle courant, normalisé, et on recommence (4 itérations sur 1807/1808). Résultat :

| de \ vers | M | X | Y |
|---|---|---|---|
| M | 0,774 | 0,161 | 0,065 |
| X | 0,635 | 0,213 | 0,152 |
| Y | 0,870 | — | 0,130 |

**Pourquoi l'EM n'estime pas aussi les émissions.** La première version le faisait, et elle échouait. Le mélange n'est pas identifiable : l'EM expliquait les *remplacements* (Helger → Hermée, Lebrun → Lefebvre au même rang) comme des paires. Cela gonflait $p(s \mid \text{même})$ autour de 0,5, ce qui faisait passer d'autres remplacements, et ainsi de suite. Fixer les émissions sur des échantillons indépendants du contexte coupe cette boucle.

## 6. Calcul en pratique

Un forward-backward complet sur les grandes rubriques représenterait environ 10 millions de cases. On **conditionne sur les paires sûres** : les paires Needleman-Wunsch avec $s \geq 0{,}9$ (11 820 sur ≈ 14 000) servent d'**ancres**, et le HMM ne tourne que dans les fenêtres entre deux ancres consécutives. Chaque fenêtre part de M et revient à M. Cela fait 7 802 cases au total, et le script entier tourne en 12 s.

Les inversions locales sont hors du modèle, puisqu'un HMM d'alignement est croissant par construction. Une **passe résiduelle** (affectation optimale, $s \geq 0{,}85$) les apparie *avant* le HMM, et ces entrées sont retirées des fenêtres. Sinon, le HMM pourrait prendre l'une d'elles pour une paire voisine plausible (Cortet aîné → Cortot au lieu de Cortet aîné → Cortet aîné, 0,90).

**Résumé du pipeline** :

1. rubriques (Needleman-Wunsch, Jaro-Winkler) ;
2. entrées par Needleman-Wunsch, dont on retient les ancres ($s \geq 0{,}9$) ;
3. inversions (passe résiduelle) ;
4. pair-HMM entre les ancres (probabilité a posteriori > 0,5).

## 7. Résultats et limites (1807 → 1808)

| méthode | paires | communes avec Dedupe | seulement Dedupe |
|---|---|---|---|
| Dedupe | 13 582 | — | — |
| Needleman-Wunsch seul (`--no-context`) | 14 355 | 13 114 | 468 |
| NW + pair-HMM | 14 491 | 13 187 | 395 |

Pagès est apparié (probabilité 0,73). Les 171 paires gagnées sont surtout des déménagements (Rey, Morlot, Maradan, Seguin). Deux permutations de noms homonymes sont corrigées (Renard, Marchais).

Limites :

- **Cas ambigus.** Des remplacements plausibles sont encore appariés (Potrel → Prot, 0,57), et le modèle refuse des homonymes fréquents qui ont déménagé (Lambert, Gervais).
- **Appariement des rubriques.** Une rubrique qui change de place dans l'ordre alphabétique (« Jardiniers-fleuristes, marchands d'arbres » → « Marchands d'arbres ») n'est pas alignée automatiquement. Ses entrées le sont quand même si les deux rubriques tombent dans le même « trou » entre deux paires de rubriques, ce qui est le cas ici ; sinon, il faut une ligne dans le patch des rubriques (section 8).
- **Hypothèses.** La probabilité est conditionnée aux ancres, et les émissions reposent sur des échantillons choisis par heuristique.

Seuls les alignements curés à la main permettront de mesurer précision et rappel, de comparer honnêtement les trois méthodes, et de vérifier la calibration des probabilités.

## 8. Correspondance des rubriques et son patch

L'alignement des rubriques (étape 1 du pipeline) est commun à Needleman-Wunsch, à Dedupe et au visualiseur (`lib/section_alignment.py`). Une rubrique est désignée par l'uuid de son titre, et ce qui échappe à l'ordre ou au seuil se corrige dans un patch versionné, `data/alignement/<gauche>__<droite>.sections.csv` :

| ligne | effet |
|---|---|
| `left_uuid` + `right_uuid` | les deux rubriques se correspondent ; un même uuid sur plusieurs lignes forme un **groupe** 1-N, N-1 ou N-M |
| un seul uuid | la rubrique n'a pas de correspondance |

Les rubriques du patch sont retirées de l'alignement automatique.

- **Needleman-Wunsch** aligne les entrées d'un groupe manuel en concaténant celles de ses rubriques, dans l'ordre de chaque annuaire. C'est la fusion des « N » en entrée.
- **Dedupe** compare la rubrique par une **clé canonique** : toutes les rubriques d'un groupe reçoivent la même clé (celle de sa première rubrique de gauche). « Liste » et « Listes de non-commerçans » deviennent donc la même chaîne. La réécriture s'applique aussi aux paires du fichier d'entraînement, à la lecture : aucun réétiquetage n'est nécessaire.
- **Le visualiseur** ne compte plus comme « rubriques non correspondantes » que les paires dont les rubriques ne se correspondent pas. Sur 1807/1808 (NW), on passe de 3 146 paires de clés différentes à 12.
