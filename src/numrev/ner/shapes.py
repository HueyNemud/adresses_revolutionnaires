"""« Forme » typographique d'une entrée, calculée sans aucune étiquette.

La forme grossière (`coarse_shape`) remplace chaque suite de mots par `w` et
ne garde que ce qui structure une entrée : virgules, points-virgules,
parenthèses, tirets longs, nombres, « et », renvois « Voyez ». Sur les
volumes actuels, deux formes couvrent la moitié des entrées :

    w , w , 9        « Fouteau-Beauregard, R. de Grenelle S. Honoré, 54. »
    w , w , 9 — w    « Bourdois-Delamotte, R. S. Honoré, 87. — Pl. Vendôme. »

et une longue traîne de formes rares porte les cas atypiques. Elle sert de
strate pour l'échantillonnage (tirage ∝ √effectif de la forme, qui plafonne
le cas de base) et de ventilation des résultats de l'audit.
"""

import re

# Nombres, mots (traits d'union et apostrophes internes compris, point
# d'abréviation final inclus), tirets longs, puis tout autre caractère.
TOKEN_PATTERN = re.compile(r"\d+|[^\W\d_]+(?:[-'’.]+[^\W\d_]+)*\.?|[—–]+|\S")
KEPT_PUNCTUATION = {",", ";", ":", "(", ")", "[", "]", "{", "}"}


def token_shapes(text: str) -> list[str]:
    shapes = []
    for match in TOKEN_PATTERN.finditer(text):
        token = match.group()
        if token.isdigit():
            shapes.append("9")
        elif token in KEPT_PUNCTUATION:
            shapes.append(token)
        elif token[0] in "—–":
            shapes.append("—")
        elif token.casefold() in ("et", "&"):
            shapes.append("et")
        elif token.casefold().startswith("voy"):
            shapes.append("V")
        elif token[0].isalpha():
            shapes.append("w")
    return shapes


def coarse_shape(text: str) -> str:
    """Forme grossière : suites de mots fusionnées en un seul `w`."""
    compressed: list[str] = []
    for shape in token_shapes(text):
        if shape == "w" and compressed and compressed[-1] == "w":
            continue
        compressed.append(shape)
    return " ".join(compressed)


def shape_profile(text: str) -> str:
    """Profil très grossier et lisible, pour les tableaux de l'audit :
    parenthèses, tiret de section, « et », renvoi, nombre de virgules."""
    shapes = token_shapes(text)
    parts = []
    if "(" in shapes:
        parts.append("paren")
    if "—" in shapes:
        parts.append("tiret")
    if "et" in shapes:
        parts.append("et")
    if "V" in shapes:
        parts.append("renvoi")
    if "9" not in shapes:
        parts.append("sans-num")
    commas = shapes.count(",") + shapes.count(";")
    parts.append(f"{min(commas, 4)}{'+' if commas >= 4 else ''}virg")
    return "·".join(parts)
