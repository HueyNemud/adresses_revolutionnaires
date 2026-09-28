"""Pair-HMM d'alignement de deux listes ordonnées (Durbin et al., *Biological
Sequence Analysis*, ch. 4), utilisé par `align_directories_nw.py`.

Trois états cachés : `M` (une entrée de gauche et une de droite se
correspondent), `X` (entrée seulement à gauche), `Y` (entrée seulement à
droite). Un alignement est un chemin dans la grille gauche × droite ; sa
probabilité est le produit des transitions et, pour chaque paire `M`, du
rapport de vraisemblance `p(sim | même entrée) / p(sim | entrées
différentes)` (modèle nul : `X` et `Y` émettent 1). Les similarités sont
discrétisées en `N_BINS` classes.

On ne calcule que dans des **fenêtres** : les suites d'entrées comprises
entre deux paires tenues pour sûres (ancres), une fenêtre commence donc en
`M` et se termine par une transition vers `M`. Forward-backward donne la
probabilité a posteriori de chaque paire sachant les deux listes (et les
ancres) ; les paires de probabilité > 0,5 forment toujours un alignement
monotone et un-à-un.

Les deux distributions d'émission sont estimées **hors du contexte**, par
l'appelant, et restent fixes ; seules les transitions s'estiment sans
étiquettes par EM (Baum-Welch). Laisser l'EM apprendre aussi
`p(sim | même entrée)` ne marche pas : le mélange n'est pas identifiable,
l'EM explique les remplacements (une entrée disparue, une autre apparue au
même rang) comme des paires, ce qui gonfle `p(sim | même)` vers 0,5, ce qui
en fait passer d'autres, etc.
"""

import math
from dataclasses import dataclass

import numpy as np

N_BINS = 20
M, X, Y = 0, 1, 2
STATES = "MXY"
NEG_INF = -math.inf
# Transitions initiales (lignes : état de départ ; Y → X interdit, pour qu'un
# trou mixte n'ait qu'un chemin : d'abord les X, puis les Y).
INITIAL_TRANSITIONS = np.array(
    [
        [0.90, 0.05, 0.05],
        [0.50, 0.30, 0.20],
        [0.70, 0.00, 0.30],
    ]
)
ALLOWED = INITIAL_TRANSITIONS > 0


def bins(similarity: np.ndarray | float) -> np.ndarray:
    """Classe de similarité (0 … N_BINS − 1) de valeurs dans [0 ; 1]."""
    return np.clip((np.asarray(similarity) * N_BINS).astype(int), 0, N_BINS - 1)


def histogram(similarities: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    return np.bincount(bins(similarities).ravel(), weights=None if weights is None else weights.ravel(), minlength=N_BINS).astype(float)


def normalize(counts: np.ndarray, pseudo: float | np.ndarray = 1.0) -> np.ndarray:
    """Distribution lissée (Dirichlet : `pseudo` ajouté à chaque classe)."""
    counts = counts + pseudo
    return counts / counts.sum()


def estimate_same(counts: np.ndarray, different: np.ndarray) -> np.ndarray:
    """`p(sim | même entrée)` à partir de comptes (attendus) par classe.

    Lissage vers `different` (une classe sans données a un rapport de
    vraisemblance de 1, pas davantage), puis rapport de vraisemblance rendu
    croissant en sim (hypothèse de rapport de vraisemblance monotone : une
    similarité plus haute n'est jamais un indice plus faible de
    correspondance) — sinon une classe basse et rare, vide chez les paires
    comme ailleurs, pourrait paraître favorable."""
    same = normalize(counts, N_BINS * different)
    ratio = np.minimum.accumulate((same / different)[::-1])[::-1]
    return normalize(ratio * different, 0.0)


@dataclass
class PairHmm:
    transitions: np.ndarray  # 3 × 3, lignes = état de départ (M, X, Y)
    same: np.ndarray  # p(classe de sim | même entrée)
    different: np.ndarray  # p(classe de sim | entrées différentes)

    def log_transitions(self) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return np.log(self.transitions)

    def log_odds(self, similarity: np.ndarray) -> np.ndarray:
        """Log-rapport de vraisemblance d'une paire, selon sa similarité."""
        return np.log(self.same / self.different)[bins(similarity)]

    def break_even(self) -> float:
        """Plus petite similarité (borne basse de classe) à partir de laquelle
        une paire est plus vraisemblable sous « même entrée »."""
        above = np.nonzero(self.same >= self.different)[0]
        return float(above[0]) / N_BINS if len(above) else 1.0


def logsumexp(*values: float) -> float:
    top = max(values)
    if top == NEG_INF:
        return NEG_INF
    return top + math.log(sum(math.exp(value - top) for value in values))


@dataclass
class WindowResult:
    posterior: np.ndarray  # k × l : P(paire (i, j) | fenêtre)
    log_likelihood: float
    transitions: np.ndarray  # 3 × 3 : nombre attendu de chaque transition


def window_posteriors(similarity: np.ndarray, model: PairHmm) -> WindowResult:
    """Forward-backward (espace log) sur une fenêtre `k × l` de similarités,
    entre deux ancres : départ en M en (0, 0), fin par une transition vers
    M depuis (k, l). `f[s][i, j]` : chemins ayant émis les `i` premières
    entrées de gauche et les `j` premières de droite, finissant en `s`."""
    k, l = similarity.shape
    t = model.log_transitions()
    e = model.log_odds(similarity) if similarity.size else np.zeros((k, l))
    f = np.full((3, k + 1, l + 1), NEG_INF)
    f[M, 0, 0] = 0.0
    for i in range(k + 1):
        for j in range(l + 1):
            if i and j:
                f[M, i, j] = e[i - 1, j - 1] + logsumexp(*(f[s, i - 1, j - 1] + t[s, M] for s in (M, X, Y)))
            if i:
                f[X, i, j] = logsumexp(*(f[s, i - 1, j] + t[s, X] for s in (M, X, Y)))
            if j:
                f[Y, i, j] = logsumexp(*(f[s, i, j - 1] + t[s, Y] for s in (M, X, Y)))

    b = np.full((3, k + 1, l + 1), NEG_INF)
    for i in range(k, -1, -1):
        for j in range(l, -1, -1):
            for s in (M, X, Y):
                terms = []
                if i == k and j == l:
                    terms.append(t[s, M])  # vers l'ancre suivante
                if i < k and j < l:
                    terms.append(t[s, M] + e[i, j] + b[M, i + 1, j + 1])
                if i < k:
                    terms.append(t[s, X] + b[X, i + 1, j])
                if j < l:
                    terms.append(t[s, Y] + b[Y, i, j + 1])
                b[s, i, j] = logsumexp(*terms)

    log_z = b[M, 0, 0]
    counts = np.zeros((3, 3))
    for i in range(k + 1):
        for j in range(l + 1):
            for s in (M, X, Y):
                if f[s, i, j] == NEG_INF:
                    continue
                moves = []
                if i == k and j == l:
                    moves.append((M, t[s, M]))
                if i < k and j < l:
                    moves.append((M, t[s, M] + e[i, j] + b[M, i + 1, j + 1]))
                if i < k:
                    moves.append((X, t[s, X] + b[X, i + 1, j]))
                if j < l:
                    moves.append((Y, t[s, Y] + b[Y, i, j + 1]))
                for target, value in moves:
                    counts[s, target] += math.exp(f[s, i, j] + value - log_z)
    posterior = np.exp(f[M, 1:, 1:] + b[M, 1:, 1:] - log_z)
    return WindowResult(posterior, log_z, counts)


def decode(posterior: np.ndarray) -> list[tuple[int, int]]:
    """Paires de probabilité a posteriori > 0,5 (au plus une par ligne et par
    colonne, puisque chacune somme à 1 au plus)."""
    return [(int(i), int(j)) for i, j in zip(*np.nonzero(posterior > 0.5))]


@dataclass
class Fit:
    model: PairHmm
    log_likelihoods: list[float]  # une par itération d'EM


def fit(
    windows: list[np.ndarray],
    same_counts: np.ndarray,
    different_counts: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 1.0,
) -> Fit:
    """EM (Baum-Welch) sur les transitions, à partir des comptes attendus sur
    toutes les fenêtres ; émissions fixes, tirées des comptes par classe de
    similarité de paires sûrement identiques (`same_counts`) et sûrement
    différentes (`different_counts`). Arrêt quand la log-vraisemblance totale
    gagne moins de `tolerance` (en nats)."""
    different = normalize(different_counts)
    model = PairHmm(INITIAL_TRANSITIONS.copy(), estimate_same(same_counts, different), different)
    history: list[float] = []
    for _ in range(max_iterations):
        transitions = np.zeros((3, 3))
        total = 0.0
        for window in windows:
            result = window_posteriors(window, model)
            transitions += result.transitions
            total += result.log_likelihood
        history.append(total)
        transitions = np.where(ALLOWED, transitions + 1.0, 0.0)  # lissage
        model = PairHmm(transitions / transitions.sum(axis=1, keepdims=True), model.same, different)
        if len(history) > 1 and history[-1] - history[-2] <= tolerance:
            break
    return Fit(model, history)
