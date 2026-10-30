"""Groupes à accolade : les membres d'une entrée groupée et le texte commun
dont chacun hérite.

Une accolade imprimée relie plusieurs lignes à un texte commun, écrit une
seule fois. Selon la convention de curation (`docs/guide_annotation_lignes.md`),
le groupe forme une seule entrée : sa première ligne est `B-ENTRY`, les
suivantes `SUB-ENTRY`. Chaque ligne porte son signe, et le texte commun
n'est écrit que sur la première ligne de la suite :

    Fieffé. } *Membres du conseil…*      `}` : le texte à droite est commun
    Pastoret. }                           à toute la suite de `}`
    Commissaires. { MM. Touzet.           `{` : le texte à gauche est commun
    { Lalande, rue Mandar.                à toute la suite de `{`

Un **segment** est la partie de l'entrée qui vient d'une ligne `B-ENTRY` ou
`SUB-ENTRY`, ses lignes `I-ENTRY` comprises : c'est un membre du groupe. Dans
un segment, le texte qui précède `{` et celui qui suit `}` sont communs, le
texte entre les deux est propre au membre. Un texte commun non vide ouvre une
nouvelle suite (« Rive gauche » puis « Rive droite » sous la même accolade
`{`), un signe sans texte commun prolonge la suite en cours, et un segment
sans le signe la termine. Les deux sortes de suites s'emboîtent : un membre
hérite du texte commun de la suite `{` et de la suite `}` qui le contiennent
(arrondissement `{` quartiers `}` juge de paix).

Les cas que la règle ne résout pas sont rendus sous forme de messages, et
l'export les signale dans son rapport : une ligne sans signe dans un groupe,
un signe sans texte commun pour ouvrir sa suite, des signes en double ou dans
l'ordre `}` … `{`, un empan à cheval sur un signe. Ils se corrigent dans la
table des lignes (signes, texte commun) ou dans le CSV NER (empans).
"""

from dataclasses import dataclass, field

OPEN = "{"
CLOSE = "}"


@dataclass(frozen=True)
class Zone:
    """Portion `[start, end)` du texte de l'entrée."""

    start: int
    end: int

    def contains(self, start: int, end: int) -> bool:
        return self.start <= start and end <= self.end


@dataclass
class Member:
    """Un membre : le segment d'où il vient, son texte propre et les textes communs dont il hérite."""

    segment: int  # rang du segment dans l'entrée, à partir de 0
    own: Zone
    inherited: list[Zone] = field(default_factory=list)


@dataclass
class Grouping:
    """Résultat de la règle sur une entrée groupée."""

    members: list[Member]
    problems: list[tuple[int, str]]  # (rang du segment, message)
    commons: list[Zone]  # textes communs, chacun une fois


def has_signs(text: str) -> bool:
    return OPEN in text or CLOSE in text


def _trimmed(text: str, start: int, end: int) -> Zone | None:
    """Zone sans blancs de bord ; None si elle ne contient que des blancs."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return Zone(start, end) if end > start else None


def group_members(text: str, segments: list[Zone]) -> Grouping | None:
    """Membres d'une entrée dont les `segments` (dans l'ordre) découpent `text` ;
    None si l'entrée n'est pas un groupe à accolade (un seul segment, ou aucun signe)."""
    if len(segments) < 2 or not has_signs(text):
        return None
    members: list[Member] = []
    problems: list[tuple[int, str]] = []
    commons: list[Zone] = []
    open_common: Zone | None = None  # texte commun de la suite `{` en cours
    close_common: Zone | None = None  # texte commun de la suite `}` en cours
    open_run = close_run = False  # une suite est en cours, même sans texte commun (signalée une fois)
    for index, segment in enumerate(segments):
        piece = text[segment.start : segment.end]
        opens = [segment.start + i for i, char in enumerate(piece) if char == OPEN]
        closes = [segment.start + i for i, char in enumerate(piece) if char == CLOSE]
        if len(opens) > 1 or len(closes) > 1 or (opens and closes and closes[0] < opens[0]):
            problems.append((index, f"signes d'accolade inattendus ({len(opens)} « {OPEN} », {len(closes)} « {CLOSE} »)"))
            members.append(Member(index, _trimmed(text, segment.start, segment.end) or segment))
            open_common = close_common = None
            open_run = close_run = False
            continue
        own_start, own_end = segment.start, segment.end
        if opens:
            left = _trimmed(text, segment.start, opens[0])
            if left is not None:
                open_common = left
                commons.append(left)
            elif not open_run:
                problems.append((index, f"« {OPEN} » sans texte commun à sa gauche"))
            open_run = True
            own_start = opens[0] + 1
        else:
            open_common, open_run = None, False
        if closes:
            right = _trimmed(text, closes[0] + 1, segment.end)
            if right is not None:
                close_common = right
                commons.append(right)
            elif not close_run:
                problems.append((index, f"« {CLOSE} » sans texte commun à sa droite"))
            close_run = True
            own_end = closes[0]
        else:
            close_common, close_run = None, False
        if not opens and not closes:
            problems.append((index, "ligne sans accolade dans un groupe"))
        own = _trimmed(text, own_start, own_end)
        if own is None:
            problems.append((index, "membre sans texte propre"))
            own = Zone(own_start, own_start)
        members.append(Member(index, own, [zone for zone in (open_common, close_common) if zone is not None]))
    return Grouping(members, problems, commons)
