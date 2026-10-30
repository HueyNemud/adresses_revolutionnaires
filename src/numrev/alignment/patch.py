"""Patch des entrées d'un alignement : les décisions humaines,
`data/alignment/<gauche>__<droite>.patch.csv` (versionné, édité à la main).

Une ligne est une **paire** (`left_uuid` et `right_uuid` : ces deux entrées
se correspondent) ou un **uuid seul** (cette entrée n'a pas de
correspondance). Le patch gagne : tout lien automatique qui touche un de ses
uuid est écarté, puis ses paires sont ajoutées (source `manuel`, ou
`manuel-incertain` si `certitude = incertaine`). Colonne `certitude`
facultative.

Rubrique et texte balisé sont recopiés pour la relecture et le
**réancrage** : un uuid disparu (entrée re-segmentée en amont) est
remplacé par celui de l'entrée unique de même texte dans la même rubrique ;
faute de quoi la ligne est **orpheline** (jamais appliquée ; les commandes
qui produisent un alignement paniquent, sauf `--force`).

Ce module porte aussi ce que partagent les deux patchs d'alignement
(entrées, et rubriques : `numrev/alignment/sections.py`) : `PatchLine`,
`Resolution`, `resolve_lines`.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Generic, TypeVar

from numrev.alignment.records import SOURCE_MANUAL, SOURCE_MANUAL_UNCERTAIN, Link, Record, clean_text, clean_title, display_text
from numrev.curation import read_dataclasses, write_csv
from numrev.ner.spans import normalize_markdown, parse_tagged_text

SIDES = ("left", "right")
UNCERTAIN = "incertaine"


class PatchLine:
    """Ligne d'un patch d'alignement (dataclass à champs `left_uuid` et `right_uuid`)."""

    def uuids(self) -> list[tuple[str, str]]:
        """(côté, uuid) renseignés."""
        return [(side, getattr(self, f"{side}_uuid")) for side in SIDES if getattr(self, f"{side}_uuid")]

    @property
    def is_pair(self) -> bool:
        return bool(self.left_uuid and self.right_uuid)


Line = TypeVar("Line", bound=PatchLine)


@dataclass
class Resolution(Generic[Line]):
    """Lignes d'un patch confrontées aux volumes : applicables (uuid à jour), réancrées, orphelines."""

    entries: list[Line] = field(default_factory=list)
    reanchored: list[tuple[Line, Line]] = field(default_factory=list)  # (avant, après)
    orphans: list[Line] = field(default_factory=list)

    def updated(self, lines: list[Line]) -> list[Line]:
        """Le patch à réécrire : lignes réancrées remplacées, orphelines gardées (à corriger à la main), ordre conservé."""
        after = {id(before): new for before, new in self.reanchored}
        return [after.get(id(line), line) for line in lines]


Target = TypeVar("Target")


def resolve_lines(
    lines: list[Line],
    locate: Callable[[str, str, Line], tuple[Target, bool] | None],
    refresh: Callable[[Line, str, Target], Line],
) -> Resolution[Line]:
    """Confronte chaque ligne aux volumes. `locate(côté, uuid, ligne)` rend
    (cible, réancrée) ou None (introuvable) ; `refresh(ligne, côté, cible)`
    remet à jour l'uuid et l'instantané de ce côté."""
    resolution: Resolution[Line] = Resolution()
    for line in lines:
        updated, moved = line, False
        for side, uuid in line.uuids():
            found = locate(side, uuid, line)
            if found is None:
                resolution.orphans.append(line)
                break
            target, reanchored = found
            moved |= reanchored
            updated = refresh(updated, side, target)
        else:
            if moved:
                resolution.reanchored.append((line, updated))
            resolution.entries.append(updated)
    return resolution


@dataclass(frozen=True)
class PatchEntry(PatchLine):
    left_file: str = ""
    left_uuid: str = ""  # vide : pas de décision sur ce côté
    right_uuid: str = ""
    right_file: str = ""
    left_section: str = ""
    right_section: str = ""
    left_tagged_text: str = ""
    right_tagged_text: str = ""
    note: str = ""
    certitude: str = ""  # vide : décision sûre ; UNCERTAIN : relecture sans conclusion ferme

    @property
    def is_uncertain(self) -> bool:
        return self.certitude.lower() == UNCERTAIN


PATCH_FIELDS = [f.name for f in fields(PatchEntry)]


@dataclass
class PatchStats:
    overridden: int = 0  # liens automatiques écartés par le patch
    manual_pairs: int = 0
    unmatched_left: int = 0  # uuid de gauche déclarés sans correspondance
    unmatched_right: int = 0


def read_patch(path: Path) -> list[PatchEntry]:
    """Lignes du patch, dans l'ordre du fichier (vide s'il n'existe pas)."""
    return read_dataclasses(path, PatchEntry)


def write_patch(path: Path, entries: list[PatchEntry]) -> None:
    write_csv(path, PATCH_FIELDS, (asdict(entry) for entry in entries))


def validate(entries: list[PatchEntry]) -> None:
    """ValueError si une ligne n'a aucun uuid, si sa `certitude` n'est ni vide
    ni `incertaine`, ou si un uuid figure sur plusieurs lignes (numéros de
    ligne du CSV, en-tête = 1)."""
    problems = [f"ligne {number} : aucun uuid" for number, entry in enumerate(entries, start=2) if not entry.uuids()]
    problems += [
        f"ligne {number} : certitude « {entry.certitude} » (attendu : vide ou « {UNCERTAIN} »)"
        for number, entry in enumerate(entries, start=2)
        if entry.certitude and not entry.is_uncertain
    ]
    seen: dict[tuple[str, str], list[int]] = defaultdict(list)
    for number, entry in enumerate(entries, start=2):
        for key in entry.uuids():
            seen[key].append(number)
    problems += [
        f"{side} {uuid} présent aux lignes {', '.join(map(str, numbers))}" for (side, uuid), numbers in seen.items() if len(numbers) > 1
    ]
    if problems:
        raise ValueError("Patch incohérent :\n- " + "\n- ".join(problems))


def side_snapshot(side: str, record: Record) -> dict[str, str]:
    """Colonnes d'un côté d'une ligne de patch : fichier, uuid, rubrique, texte balisé."""
    return {
        f"{side}_file": record.document,
        f"{side}_uuid": record.uuid,
        f"{side}_section": record.section_title,
        f"{side}_tagged_text": display_text(record),
    }


def entry_from_records(left: Record | None, right: Record | None, note: str = "", certitude: str = "") -> PatchEntry:
    """Ligne de patch pour une paire, ou une entrée sans correspondance (l'autre à None)."""
    values = {"note": note, "certitude": certitude}
    for side, record in (("left", left), ("right", right)):
        if record is not None:
            values |= side_snapshot(side, record)
    return PatchEntry(**values)


def anchor_key(section: str, tagged_text: str) -> tuple[str, str]:
    """(rubrique, texte) d'une ligne de patch, normalisés comme `Record.section` et `Record.text`."""
    try:
        text, _ = parse_tagged_text(tagged_text)
    except ValueError:
        text = tagged_text
    return clean_title(section), clean_text(normalize_markdown(text).text)


def resolve(entries: list[PatchEntry], left: dict[str, Record], right: dict[str, Record]) -> Resolution[PatchEntry]:
    """Uuid du patch confrontés aux volumes ; un uuid disparu est réancré par
    (rubrique, texte), jamais sur une entrée déjà présente dans le patch."""
    records = {"left": left, "right": right}
    used = {(side, uuid) for entry in entries for side, uuid in entry.uuids() if uuid in records[side]}
    indexes: dict[str, dict[tuple[str, str], list[Record]]] = {}

    def locate(side: str, uuid: str, entry: PatchEntry) -> tuple[Record, bool] | None:
        if uuid in records[side]:
            return records[side][uuid], False
        if side not in indexes:
            indexes[side] = defaultdict(list)
            for record in records[side].values():
                indexes[side][(record.section, record.text)].append(record)
        key = anchor_key(getattr(entry, f"{side}_section"), getattr(entry, f"{side}_tagged_text"))
        candidates = [record for record in indexes[side].get(key, []) if (side, record.uuid) not in used]
        if len(candidates) != 1:
            return None
        used.add((side, candidates[0].uuid))
        return candidates[0], True

    return resolve_lines(entries, locate, lambda entry, side, record: replace(entry, **side_snapshot(side, record)))


def load_patch(path: Path, left: dict[str, Record], right: dict[str, Record]) -> tuple[list[PatchEntry], Resolution[PatchEntry]]:
    """Lit, valide (ValueError) et résout le patch."""
    entries = read_patch(path)
    validate(entries)
    return entries, resolve(entries, left, right)


def declared_unmatched(resolution: Resolution[PatchEntry]) -> set[str]:
    """Uuid déclarés sans correspondance par le patch (jamais proposés comme candidates)."""
    return {uuid for entry in resolution.entries if not entry.is_pair for _, uuid in entry.uuids()}


def orphan_descriptions(resolution: Resolution[PatchEntry]) -> list[str]:
    """Lignes orphelines, lisibles : côté, uuid, rubrique et texte."""
    return [
        " ↔ ".join(
            f"{side} {getattr(entry, f'{side}_uuid')} · {getattr(entry, f'{side}_section')} · {getattr(entry, f'{side}_tagged_text')}"
            for side, _ in entry.uuids()
        )
        for entry in resolution.orphans
    ]


def apply_patch(links: list[Link], entries: list[PatchEntry]) -> tuple[list[Link], PatchStats]:
    """Liens automatiques + patch (voir la docstring du module). Une paire
    validée telle que la machine l'avait trouvée garde son score."""
    patched_left = {entry.left_uuid for entry in entries if entry.left_uuid}
    patched_right = {entry.right_uuid for entry in entries if entry.right_uuid}
    scores = {(link.left_uuid, link.right_uuid): link.score for link in links}
    manual = [
        Link(
            entry.left_uuid,
            entry.right_uuid,
            scores.get((entry.left_uuid, entry.right_uuid)),
            SOURCE_MANUAL_UNCERTAIN if entry.is_uncertain else SOURCE_MANUAL,
        )
        for entry in entries
        if entry.is_pair
    ]
    kept = [link for link in links if link.left_uuid not in patched_left and link.right_uuid not in patched_right]
    validated = sum(1 for link in manual if (link.left_uuid, link.right_uuid) in scores)
    stats = PatchStats(
        overridden=len(links) - len(kept) - validated,  # une paire validée n'est pas écartée
        manual_pairs=len(manual),
        unmatched_left=sum(1 for entry in entries if entry.left_uuid and not entry.right_uuid),
        unmatched_right=sum(1 for entry in entries if entry.right_uuid and not entry.left_uuid),
    )
    return kept + manual, stats
