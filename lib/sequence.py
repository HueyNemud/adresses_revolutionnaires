"""Alignement de deux suites ordonnées par Needleman-Wunsch, partagé par
l'alignement des rubriques (`lib/section_alignment.py`) et celui des
entrées (`align_directories_nw.py`)."""

import numpy as np


def needleman_wunsch(similarity: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Appariement croissant (i, j) qui maximise Σ (similarité − seuil), sans
    pénalité de trou. Sans pénalité, la récurrence
    H[i, j] = max(H[i-1, j], H[i, j-1], H[i-1, j-1] + s − seuil)
    se calcule ligne par ligne : maximum avec la diagonale, puis maximum
    cumulé le long de la ligne."""
    n, m = similarity.shape
    scores = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        row = scores[i - 1].copy()
        np.maximum(row[1:], scores[i - 1, :-1] + (similarity[i - 1] - threshold), out=row[1:])
        scores[i] = np.maximum.accumulate(row)
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        if scores[i, j] == scores[i - 1, j]:
            i -= 1
        elif scores[i, j] == scores[i, j - 1]:
            j -= 1
        else:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
    return pairs[::-1]
