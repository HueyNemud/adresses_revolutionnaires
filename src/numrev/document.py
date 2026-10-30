"""JSON des lignes d'un document, écrit par `numrev extract` et enrichi par
`numrev label` : une liste de pages, chacune avec ses blocs (`data_blocks`),
chacun avec ses lignes (`lines`). Lu, validé et parcouru ici seulement :
c'est là qu'on change ce schéma.
"""

import json
from pathlib import Path
from typing import Any, Iterator, NamedTuple


def parse_pages(text: str) -> list[dict[str, Any]]:
    """Pages d'un JSON de la chaîne (OCR, lignes, lignes annotées) ; ValueError si ce n'est pas une liste d'objets."""
    pages = json.loads(text)
    if not isinstance(pages, list) or not all(isinstance(page, dict) for page in pages):
        raise ValueError("le JSON doit être une liste de pages, chacune un objet JSON.")
    return pages


def load_pages(path: Path) -> list[dict[str, Any]]:
    return parse_pages(Path(path).read_text(encoding="utf-8"))


class LineLocation(NamedTuple):
    """Une ligne et son contexte : page, bloc, et leurs rangs (à partir de 0) dans le document."""

    page: dict[str, Any]
    block: dict[str, Any]
    line: dict[str, Any]
    page_pos: int
    block_pos: int
    line_pos: int


def iter_line_locations(pages: list[Any]) -> Iterator[LineLocation]:
    """Toutes les lignes, dans l'ordre du document ; ValueError dès qu'un niveau n'a pas la forme écrite par `numrev extract`."""
    for page_pos, page in enumerate(pages):
        if not isinstance(page, dict):
            raise ValueError("Chaque page doit être un objet JSON.")
        data_blocks = page.get("data_blocks")
        if not isinstance(data_blocks, list):
            raise ValueError("Le JSON ne correspond pas à la sortie de numrev extract : clé 'data_blocks' manquante ou invalide.")
        for block_pos, block in enumerate(data_blocks):
            if not isinstance(block, dict):
                raise ValueError("Chaque bloc de données doit être un objet JSON.")
            lines = block.get("lines")
            if not isinstance(lines, list):
                raise ValueError("Le JSON ne correspond pas à la sortie de numrev extract : clé 'lines' manquante ou invalide.")
            for line_pos, line in enumerate(lines):
                if not isinstance(line, dict):
                    raise ValueError("Chaque ligne doit être un objet JSON.")
                yield LineLocation(page, block, line, page_pos, block_pos, line_pos)
