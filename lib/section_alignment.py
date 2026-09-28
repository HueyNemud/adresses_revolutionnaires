"""Correspondance des rubriques de deux annuaires, partagée par
`align_directories.py` (Dedupe), `align_directories_nw.py` et
`tools/display_alignment.py`.

Une **rubrique** est la suite contiguë des ENTRY de même rubrique
(`Record.section`, `lib/alignment.py`) ; elle est identifiée par l'uuid de
son TITLE (`Record.section_uuid`).

L'ordre des rubriques est stable d'une édition à l'autre ; elles sont donc
alignées par Needleman-Wunsch sur la similarité Jaro-Winkler de leur clé
(« liste » / « listes de non-commerçans »). Ce qui échappe à l'ordre ou au
seuil se corrige dans un **patch** versionné et édité à la main,
`data/alignement/<gauche>__<droite>.sections.csv` :

- une ligne à deux uuid lie deux rubriques ; un même uuid peut figurer dans
  plusieurs lignes : les composantes connexes forment des **groupes** 1-1,
  1-N, N-1 ou N-M (« Jardiniers-fleuristes, pépiniéristes, marchands
  d'arbres » ↔ « Marchands d'arbres », hors ordre alphabétique) ;
- une ligne à un seul uuid déclare une rubrique sans correspondance.

Le patch gagne : ses rubriques sont retirées de l'alignement automatique.
Les titres recopiés servent à la relecture et au **réancrage** : si un uuid
disparaît (re-segmentation amont), on cherche la rubrique de même clé,
unique dans l'annuaire ; faute de quoi la ligne est orpheline (signalée,
non appliquée).
"""

import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cdist

from lib.alignment import SOURCE_MANUAL, Record, clean_title
from lib.sequence import needleman_wunsch

SOURCE_AUTO = "auto"
DEFAULT_THRESHOLD = 0.8  # similarité Jaro-Winkler minimale de deux clés alignées
SECTION_PATCH_SUFFIX = ".sections.csv"
SIDES = ("left", "right")


@dataclass
class Section:
    uuid: str  # uuid du TITLE, vide pour les entrées sans rubrique
    key: str  # clé comparée (`clean_title`)
    title: str  # titre lisible
    records: list[Record]


def sections(records: list[Record]) -> list[Section]:
    """Rubriques d'un annuaire, dans l'ordre : suites contiguës d'entrées de
    même rubrique."""
    result: list[Section] = []
    for record in sorted(records, key=lambda record: record.order):
        if not result or (result[-1].uuid, result[-1].key) != (record.section_uuid, record.section):
            result.append(Section(record.section_uuid, record.section, record.section_title, []))
        result[-1].records.append(record)
    return result


# ----------------------------------------------------------------------
# Patch
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SectionPatchEntry:
    left_uuid: str = ""
    right_uuid: str = ""
    left_title: str = ""
    right_title: str = ""
    note: str = ""

    def uuids(self) -> list[tuple[str, str]]:
        """(côté, uuid) renseignés."""
        return [(side, getattr(self, f"{side}_uuid")) for side in SIDES if getattr(self, f"{side}_uuid")]

    @property
    def is_pair(self) -> bool:
        return bool(self.left_uuid and self.right_uuid)


SECTION_PATCH_FIELDS = [f.name for f in fields(SectionPatchEntry)]


def read_section_patch(path: Path) -> list[SectionPatchEntry]:
    """Lignes du patch, dans l'ordre du fichier (vide s'il n'existe pas)."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            SectionPatchEntry(**{name: (row.get(name) or "").strip() for name in SECTION_PATCH_FIELDS})
            for row in csv.DictReader(handle)
        ]


def write_section_patch(path: Path, entries: list[SectionPatchEntry]) -> None:
    """Écriture atomique (fichier temporaire puis renommage)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SECTION_PATCH_FIELDS)
        writer.writeheader()
        for entry in entries:
            writer.writerow({name: getattr(entry, name) for name in SECTION_PATCH_FIELDS})
    os.replace(temporary, path)


def validate_section_patch(entries: list[SectionPatchEntry]) -> None:
    """Lève ValueError si une ligne n'a aucun uuid, si une paire est en double
    ou si une rubrique est à la fois appariée et déclarée sans correspondance
    (numéros de ligne du CSV, en-tête = 1)."""
    problems = [f"ligne {number} : aucun uuid" for number, entry in enumerate(entries, start=2) if not entry.uuids()]
    pairs: dict[tuple[str, str], list[int]] = defaultdict(list)
    paired: dict[tuple[str, str], list[int]] = defaultdict(list)
    alone: dict[tuple[str, str], list[int]] = defaultdict(list)
    for number, entry in enumerate(entries, start=2):
        if entry.is_pair:
            pairs[(entry.left_uuid, entry.right_uuid)].append(number)
            for key in entry.uuids():
                paired[key].append(number)
        elif entry.uuids():
            alone[entry.uuids()[0]].append(number)
    problems += [f"paire {left} ↔ {right} aux lignes {', '.join(map(str, numbers))}" for (left, right), numbers in pairs.items() if len(numbers) > 1]
    problems += [f"{side} {uuid} sans correspondance aux lignes {', '.join(map(str, numbers))}" for (side, uuid), numbers in alone.items() if len(numbers) > 1]
    problems += [
        f"{side} {uuid} apparié (lignes {', '.join(map(str, paired[(side, uuid)]))}) et sans correspondance (ligne {numbers[0]})"
        for (side, uuid), numbers in alone.items()
        if (side, uuid) in paired
    ]
    if problems:
        raise ValueError("Patch des rubriques incohérent :\n- " + "\n- ".join(problems))


def section_patch_line(left: Section | None, right: Section | None, note: str = "") -> SectionPatchEntry:
    """Ligne de patch pour deux rubriques, ou une seule (l'autre à None)."""
    values = {"note": note}
    for side, section in (("left", left), ("right", right)):
        if section is not None:
            values |= {f"{side}_uuid": section.uuid, f"{side}_title": section.title}
    return SectionPatchEntry(**values)


@dataclass
class SectionResolution:
    entries: list[SectionPatchEntry]  # lignes applicables
    reanchored: list[tuple[SectionPatchEntry, SectionPatchEntry]] = field(default_factory=list)  # (avant, après)
    orphans: list[SectionPatchEntry] = field(default_factory=list)


def resolve_section_patch(entries: list[SectionPatchEntry], left: list[Section], right: list[Section]) -> SectionResolution:
    """Vérifie que les uuid du patch existent ; réancre par la clé du titre
    (unique dans l'annuaire) ceux qui ont disparu ; met à jour les titres."""
    available = {"left": left, "right": right}
    by_uuid = {side: {section.uuid: section for section in available[side] if section.uuid} for side in SIDES}
    by_key: dict[str, dict[str, list[Section]]] = {side: defaultdict(list) for side in SIDES}
    for side in SIDES:
        for section in available[side]:
            by_key[side][section.key].append(section)
    resolution = SectionResolution(entries=[])
    for entry in entries:
        updated, moved, orphan = entry, False, False
        for side, uuid in entry.uuids():
            section = by_uuid[side].get(uuid)
            if section is None:
                candidates = by_key[side].get(clean_title(getattr(entry, f"{side}_title")), [])
                if len(candidates) != 1 or not candidates[0].uuid:
                    orphan = True
                    break
                section, moved = candidates[0], True
            updated = replace(updated, **{f"{side}_uuid": section.uuid, f"{side}_title": section.title})
        if orphan:
            resolution.orphans.append(entry)
            continue
        if moved:
            resolution.reanchored.append((entry, updated))
        resolution.entries.append(updated)
    return resolution


def updated_section_patch(entries: list[SectionPatchEntry], resolution: SectionResolution) -> list[SectionPatchEntry]:
    """Le patch à réécrire : lignes réancrées remplacées, orphelines gardées."""
    after = {id(before): new for before, new in resolution.reanchored}
    return [after.get(id(entry), entry) for entry in entries]


# ----------------------------------------------------------------------
# Alignement
# ----------------------------------------------------------------------
@dataclass
class SectionGroup:
    left: list[Section]
    right: list[Section]
    source: str  # SOURCE_AUTO ou SOURCE_MANUAL


@dataclass
class SectionAlignment:
    groups: list[SectionGroup]  # groupes manuels puis automatiques
    auto_left: list[Section]  # suites alignées automatiquement (sans le patch)
    auto_right: list[Section]
    auto_pairs: list[tuple[int, int]]  # indices dans auto_left / auto_right
    unmatched_left: list[Section]  # sans groupe (déclarées au patch ou non alignées)
    unmatched_right: list[Section]
    declared: set[tuple[str, str]]  # (côté, uuid) déclarés sans correspondance au patch
    resolution: SectionResolution


def manual_groups(entries: list[SectionPatchEntry], left: list[Section], right: list[Section]) -> list[SectionGroup]:
    """Composantes connexes des paires du patch, rubriques dans l'ordre de
    l'annuaire, groupes dans l'ordre de leur première rubrique de gauche."""
    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for entry in entries:
        if entry.is_pair:
            parent[find(("left", entry.left_uuid))] = find(("right", entry.right_uuid))
    components: dict[tuple[str, str], SectionGroup] = {}
    for side, side_sections in (("left", left), ("right", right)):
        for section in side_sections:
            if section.uuid and (side, section.uuid) in parent:
                group = components.setdefault(find((side, section.uuid)), SectionGroup([], [], SOURCE_MANUAL))
                getattr(group, side).append(section)
    position = {id(section): index for index, section in enumerate(left)}
    return sorted(components.values(), key=lambda group: position[id(group.left[0])])


def align_sections(
    left_records: list[Record],
    right_records: list[Record],
    patch: list[SectionPatchEntry] = (),
    threshold: float = DEFAULT_THRESHOLD,
) -> SectionAlignment:
    left, right = sections(left_records), sections(right_records)
    resolution = resolve_section_patch(list(patch), left, right)
    manual = manual_groups(resolution.entries, left, right)
    declared = {key for entry in resolution.entries if not entry.is_pair for key in entry.uuids()}
    taken = {(side, section.uuid) for group in manual for side in SIDES for section in getattr(group, side)} | declared
    auto_left = [section for section in left if ("left", section.uuid) not in taken]
    auto_right = [section for section in right if ("right", section.uuid) not in taken]
    auto_pairs = []
    if auto_left and auto_right:
        similarity = cdist(
            [section.key for section in auto_left],
            [section.key for section in auto_right],
            scorer=JaroWinkler.normalized_similarity,
            workers=-1,
        )
        auto_pairs = needleman_wunsch(similarity, threshold)
    groups = manual + [SectionGroup([auto_left[i]], [auto_right[j]], SOURCE_AUTO) for i, j in auto_pairs]
    grouped = {id(section) for group in groups for side in SIDES for section in getattr(group, side)}
    return SectionAlignment(
        groups=groups,
        auto_left=auto_left,
        auto_right=auto_right,
        auto_pairs=auto_pairs,
        unmatched_left=[section for section in left if id(section) not in grouped],
        unmatched_right=[section for section in right if id(section) not in grouped],
        declared=declared,
        resolution=resolution,
    )


def load_section_alignment(
    left_records: list[Record], right_records: list[Record], patch_path: Path, threshold: float = DEFAULT_THRESHOLD, rewrite: bool = True
) -> SectionAlignment:
    """Lit, valide et applique le patch (ValueError s'il est incohérent) ; le
    réécrit si des lignes ont été réancrées (sauf `rewrite=False`)."""
    entries = read_section_patch(patch_path)
    validate_section_patch(entries)
    alignment = align_sections(left_records, right_records, entries, threshold)
    if rewrite and alignment.resolution.reanchored:
        write_section_patch(patch_path, updated_section_patch(entries, alignment.resolution))
    return alignment


def corresponding(alignment: SectionAlignment) -> set[tuple[str, str]]:
    """(uuid de gauche, uuid de droite) des rubriques qui se correspondent."""
    return {(left.uuid, right.uuid) for group in alignment.groups for left in group.left for right in group.right}


def canonical_keys(alignment: SectionAlignment) -> tuple[dict[str, str], dict[str, str]]:
    """Tables clé → clé canonique, pour chaque côté : la clé de la première
    rubrique de gauche du groupe (idempotent). Une rubrique sans groupe
    garde sa clé."""
    left_keys: dict[str, str] = {}
    right_keys: dict[str, str] = {}
    for group in alignment.groups:
        canonical = group.left[0].key
        left_keys |= {section.key: canonical for section in group.left}
        right_keys |= {section.key: canonical for section in group.right}
    return left_keys, right_keys


def canonical_training(data: dict, left_keys: dict[str, str], right_keys: dict[str, str]) -> dict:
    """Paires d'entraînement Dedupe (`match` / `distinct`, chacune
    (gauche, droite)) avec la rubrique réécrite en clé canonique."""

    def rewrite(record: dict, keys: dict[str, str]) -> dict:
        section = record.get("section")
        return record | {"section": keys.get(section, section)} if section else record

    result = {}
    for label, pairs in data.items():
        rewritten = []
        for pair in pairs:
            wrapped = isinstance(pair, dict) and "__value__" in pair
            left, right = pair["__value__"] if wrapped else pair
            values = [rewrite(left, left_keys), rewrite(right, right_keys)]
            rewritten.append(pair | {"__value__": values} if wrapped else values)
        result[label] = rewritten
    return result


def canonical_training_text(text: str, left_keys: dict[str, str], right_keys: dict[str, str]) -> str:
    return json.dumps(canonical_training(json.loads(text), left_keys, right_keys), ensure_ascii=False)
