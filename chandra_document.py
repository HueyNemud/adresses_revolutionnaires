"""Schéma partagé du JSON produit par extract_chandra_lines.py.

Ce document est une liste de pages, chacune portant une liste de blocs de
données (« data_blocks »), eux-mêmes portant une liste de lignes Markdown
(« lines »). annotate_lines_crf.py et export_lines_csv.py parcourent tous
les deux cette même structure : `iter_line_locations` centralise la
validation et l'itération pour éviter la duplication entre ces scripts.
"""

from typing import Any, Iterator, NamedTuple


class LineLocation(NamedTuple):
    """Une ligne Markdown et son contexte (page, bloc) dans le document."""

    page: dict[str, Any]
    block: dict[str, Any]
    line: dict[str, Any]
    page_pos: int
    block_pos: int
    line_pos: int


def iter_line_locations(document: list[Any]) -> Iterator[LineLocation]:
    """Parcourt page/data_blocks/lines en validant la forme attendue.

    Lève une ValueError explicite dès qu'un niveau ne correspond pas à la
    sortie de extract_chandra_lines.py.
    """
    for page_pos, page in enumerate(document):
        if not isinstance(page, dict):
            raise ValueError("Chaque page doit être un objet JSON.")
        data_blocks = page.get("data_blocks")
        if not isinstance(data_blocks, list):
            raise ValueError(
                "Le JSON ne correspond pas à la sortie de extract_chandra_lines.py : "
                "clé 'data_blocks' manquante ou invalide."
            )
        for block_pos, block in enumerate(data_blocks):
            if not isinstance(block, dict):
                raise ValueError("Chaque bloc de données doit être un objet JSON.")
            lines = block.get("lines")
            if not isinstance(lines, list):
                raise ValueError(
                    "Le JSON ne correspond pas à la sortie de extract_chandra_lines.py : "
                    "clé 'lines' manquante ou invalide."
                )
            for line_pos, line in enumerate(lines):
                if not isinstance(line, dict):
                    raise ValueError("Chaque ligne doit être un objet JSON.")
                yield LineLocation(page, block, line, page_pos, block_pos, line_pos)
