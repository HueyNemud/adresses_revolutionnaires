"""Entrées suspectes : à relire en priorité après l'inférence.

Deux sources de signal, sans lexique ni règle propres à un volume :

- **la confiance du modèle** : `confidence` = score minimal des empans de
  l'entrée (une entrée n'est pas plus sûre que son empan le moins sûr) ;
  motif `score bas` sous `min_score` ;
- **la structure des empans prédits**, qui contredit le schéma d'une entrée
  d'annuaire (voir `docs/guide_annotation_ner.md`) :
  * `aucun empan` ;
  * `texte non couvert` : un mot (caractère alphanumérique) hors de tout
    empan, alors que toute l'entrée doit être segmentée ;
  * `SUBJ absent en tête` : l'entrée ne commence pas par un SUBJ ;
  * `signature inhabituelle` : suite de classes hors de `USUAL_SIGNATURES`
    (ex. deux SUBJ, typique de deux entrées fusionnées en une ligne).

`audit_ner.py` mesure, sur le gold, la part d'entrées signalées et la part
des erreurs attrapées, motif par motif et pour plusieurs `min_score`.
"""

import re
from collections.abc import Sequence

from lib.ner.spans import Span, signature

USUAL_SIGNATURES = frozenset({"SUBJ,ADDR", "SUBJ,DESC,ADDR", "SUBJ,DESC", "SUBJ"})
DEFAULT_MIN_SCORE = 0.9
REASONS = ("aucun empan", "score bas", "texte non couvert", "SUBJ absent en tête", "signature inhabituelle")
SEPARATOR = " | "
WORD_CHARACTER = re.compile(r"\w")


def confidence(spans: Sequence[Span]) -> float | None:
    """Score minimal des empans ; 0 sans empan, None sans score."""
    if not spans:
        return 0.0
    scores = [span.score for span in spans if span.score is not None]
    return min(scores) if scores else None


def uncovered(text: str, spans: Sequence[Span]) -> bool:
    covered = bytearray(len(text))
    for span in spans:
        covered[span.start : span.end] = b"\x01" * (span.end - span.start)
    return any(not covered[m.start()] for m in WORD_CHARACTER.finditer(text))


def suspicion_reasons(text: str, spans: Sequence[Span], min_score: float = DEFAULT_MIN_SCORE) -> list[str]:
    """Motifs de suspicion, dans l'ordre de `REASONS` (liste vide : rien à
    signaler)."""
    if not spans:
        return ["aucun empan"]
    reasons = []
    score = confidence(spans)
    if score is not None and score < min_score:
        reasons.append("score bas")
    if uncovered(text, spans):
        reasons.append("texte non couvert")
    ordered = signature(spans)
    if not ordered.startswith("SUBJ"):
        reasons.append("SUBJ absent en tête")
    elif ordered not in USUAL_SIGNATURES:
        reasons.append("signature inhabituelle")
    return reasons
