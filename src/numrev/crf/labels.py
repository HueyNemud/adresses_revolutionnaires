"""Classes de lignes prédites par le CRF."""

from enum import StrEnum


class AnnotationLabel(StrEnum):
    """Labels supported by the sequence classifier and persisted sessions."""

    BENTRY = "B-ENTRY"
    IENTRY = "I-ENTRY"
    SUBENTRY = "SUB-ENTRY"
    BTITLE = "B-TITLE"
    ITITLE = "I-TITLE"
    UNK = "¯\\_(ツ)_/¯"
    OOS = "OUT OF SCOPE"


CLASSES: tuple[str, ...] = tuple(label.value for label in AnnotationLabel)
