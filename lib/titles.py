"""Titres Markdown (`#`, `##`, …) des lignes TITLE : niveau et texte lisible.

Partagé par `build_entity_tree.py` (arbre des titres), `lib/alignment.py`
(rubrique d'une entrée) et `tools/display_directory.py` (bandeaux de
rubrique).
"""

import re

from lib.ner.spans import normalize_markdown

HEADING_PATTERN = re.compile(r"^\s*(#+)")


def title_level(markdown: str) -> int | None:
    """Niveau d'un titre : nombre de `#` en tête, None sans `#`."""
    match = HEADING_PATTERN.match(markdown)
    return len(match.group(1)) if match else None


def title_text(markdown: str) -> str:
    """Titre sans `#` ni emphase, blancs réduits : « ##\\xa0**BAIGNEURS.** » → « BAIGNEURS. »."""
    return " ".join(normalize_markdown(markdown.lstrip().lstrip("#")).text.split())
