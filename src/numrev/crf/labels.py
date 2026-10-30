"""Classes de lignes prédites par le CRF."""

from enum import StrEnum


class AnnotationLabel(StrEnum):
    """Classes de lignes (annotateur, sessions, table des lignes) ; `UNK` = incertain, à trancher à la curation."""

    BENTRY = "B-ENTRY"
    IENTRY = "I-ENTRY"
    SUBENTRY = "SUB-ENTRY"
    BTITLE = "B-TITLE"
    ITITLE = "I-TITLE"
    UNK = "¯\\_(ツ)_/¯"
    OOS = "OUT OF SCOPE"


CLASSES: tuple[str, ...] = tuple(label.value for label in AnnotationLabel)
