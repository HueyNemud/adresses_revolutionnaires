"""Titres Markdown (`#`, `##`, …) des lignes TITLE : niveau, texte lisible
et racine de l'arbre des titres.

Partagé par `numrev assemble` (arbre des titres), `numrev/alignment/records.py`
(rubrique d'une entrée) et `numrev view directory` (bandeaux de
rubrique).
"""

import re
import uuid

from numrev.ner.spans import normalize_markdown

HEADING_PATTERN = re.compile(r"^\s*(#+)")

# Parent des titres de plus haut niveau et des entrées avant tout titre : racine
# commune à tous les documents (`numrev assemble`).
ROOT_UUID = str(uuid.UUID(int=0))


def title_level(markdown: str) -> int | None:
    """Niveau d'un titre : nombre de `#` en tête, None sans `#`."""
    match = HEADING_PATTERN.match(markdown)
    return len(match.group(1)) if match else None


def title_text(markdown: str) -> str:
    """Titre sans `#` ni emphase, blancs réduits : « ##\\xa0**BAIGNEURS.** » → « BAIGNEURS. »."""
    return " ".join(normalize_markdown(markdown.lstrip().lstrip("#")).text.split())
