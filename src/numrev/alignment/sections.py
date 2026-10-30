"""Correspondance des rubriques de deux annuaires (`numrev align`, `numrev join`,
`numrev view alignment`) : les entrées ne sont jamais appariées qu'entre
rubriques qui se correspondent.

Une **rubrique** est une suite contiguë d'ENTRY de même rubrique
(`Record.section`), identifiée par l'uuid de son TITLE. L'ordre des
rubriques étant stable d'une édition à l'autre, elles sont alignées par
Needleman-Wunsch sur la similarité Jaro-Winkler de leur clé. Le **patch**
`data/alignment/<gauche>__<droite>.sections.csv` (édité à la main) gagne :
une ligne à deux uuid lie deux rubriques (les composantes connexes forment
des groupes 1-1, 1-N ou N-M), une ligne à un uuid déclare une rubrique sans
correspondance. Un uuid disparu est réancré sur la rubrique de même clé de
titre, unique, sinon la ligne est orpheline (`numrev/alignment/patch.py`).

Une suite d'entrées **sans rubrique** (avant le premier titre d'une plage)
n'a pas d'uuid : jamais appariée, ni inscriptible au patch ; on restitue son
titre dans la table des lignes (`docs/guide_annotation_lignes.md`).
"""

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cdist

from numrev.alignment.patch import SIDES, PatchLine, Resolution, resolve_lines
from numrev.alignment.records import SOURCE_MANUAL, Link, Record, clean_title
from numrev.alignment.sequence import needleman_wunsch
from numrev.command import Writes
from numrev.curation import read_dataclasses, write_csv
from numrev.paths import NER

SOURCE_AUTO = "auto"
DEFAULT_THRESHOLD = 0.8  # similarité Jaro-Winkler minimale de deux clés alignées


@dataclass
class Section:
    uuid: str  # uuid du TITLE, vide pour les entrées sans rubrique
    key: str  # clé comparée (`clean_title`)
    title: str  # titre lisible (pour une suite sans rubrique : où elle commence)
    records: list[Record]


def untitled_label(record: Record) -> str:
    """Libellé d'une suite d'entrées sans rubrique, d'après sa première entrée."""
    return f"(sans rubrique : {record.document.removesuffix(NER)}, à partir de la p. {record.page})"


def split_sections(records: list[Record]) -> list[Section]:
    """Rubriques d'un annuaire, dans l'ordre : suites contiguës d'entrées de
    même rubrique."""
    result: list[Section] = []
    for record in sorted(records, key=lambda record: record.order):
        if not result or (result[-1].uuid, result[-1].key) != (record.section_uuid, record.section):
            title = record.section_title if record.section_uuid else untitled_label(record)
            result.append(Section(record.section_uuid, record.section, title, []))
        result[-1].records.append(record)
    return result


# ----------------------------------------------------------------------
# Patch
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SectionPatchEntry(PatchLine):
    left_uuid: str = ""
    right_uuid: str = ""
    left_title: str = ""
    right_title: str = ""
    note: str = ""


SECTION_PATCH_FIELDS = [f.name for f in fields(SectionPatchEntry)]


def read_section_patch(path: Path) -> list[SectionPatchEntry]:
    """Lignes du patch, dans l'ordre du fichier (vide s'il n'existe pas)."""
    return read_dataclasses(path, SectionPatchEntry)


def write_section_patch(path: Path, entries: list[SectionPatchEntry]) -> None:
    """Écriture atomique (numrev/curation.py)."""
    write_csv(path, SECTION_PATCH_FIELDS, (asdict(entry) for entry in entries))


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
    problems += [
        f"paire {left} ↔ {right} aux lignes {', '.join(map(str, numbers))}" for (left, right), numbers in pairs.items() if len(numbers) > 1
    ]
    problems += [
        f"{side} {uuid} sans correspondance aux lignes {', '.join(map(str, numbers))}"
        for (side, uuid), numbers in alone.items()
        if len(numbers) > 1
    ]
    problems += [
        f"{side} {uuid} apparié (lignes {', '.join(map(str, paired[(side, uuid)]))}) et sans correspondance (ligne {numbers[0]})"
        for (side, uuid), numbers in alone.items()
        if (side, uuid) in paired
    ]
    if problems:
        raise ValueError("Patch des rubriques incohérent :\n- " + "\n- ".join(problems))


def resolve_section_patch(entries: list[SectionPatchEntry], left: list[Section], right: list[Section]) -> Resolution[SectionPatchEntry]:
    """Uuid du patch confrontés aux rubriques ; un uuid disparu est réancré
    sur la rubrique de même clé de titre, si elle est unique ; les titres
    recopiés sont mis à jour."""
    by_uuid: dict[str, dict[str, Section]] = {}
    by_key: dict[str, dict[str, list[Section]]] = {}
    for side, side_sections in zip(SIDES, (left, right)):
        by_uuid[side] = {section.uuid: section for section in side_sections if section.uuid}
        by_key[side] = defaultdict(list)
        for section in side_sections:
            by_key[side][section.key].append(section)

    def locate(side: str, uuid: str, entry: SectionPatchEntry) -> tuple[Section, bool] | None:
        if uuid in by_uuid[side]:
            return by_uuid[side][uuid], False
        candidates = by_key[side].get(clean_title(getattr(entry, f"{side}_title")), [])
        return (candidates[0], True) if len(candidates) == 1 and candidates[0].uuid else None

    def refresh(entry: SectionPatchEntry, side: str, section: Section) -> SectionPatchEntry:
        return replace(entry, **{f"{side}_uuid": section.uuid, f"{side}_title": section.title})

    return resolve_lines(entries, locate, refresh)


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
    groups: list[SectionGroup]  # groupes du patch puis paires automatiques
    unmatched_left: list[Section]  # sans groupe (déclarées au patch ou non alignées)
    unmatched_right: list[Section]
    declared: set[tuple[str, str]]  # (côté, uuid) déclarés sans correspondance au patch
    resolution: Resolution[SectionPatchEntry]


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
    """Groupes du patch, puis paires automatiques (Needleman-Wunsch, similarité ≥ `threshold`) des autres rubriques titrées."""
    left, right = split_sections(left_records), split_sections(right_records)
    resolution = resolve_section_patch(list(patch), left, right)
    manual = manual_groups(resolution.entries, left, right)
    declared = {key for entry in resolution.entries if not entry.is_pair for key in entry.uuids()}
    taken = {(side, section.uuid) for group in manual for side in SIDES for section in getattr(group, side)} | declared
    # Une suite sans rubrique (sans uuid) n'est jamais alignée automatiquement : voir la docstring du module.
    auto_left = [section for section in left if section.uuid and ("left", section.uuid) not in taken]
    auto_right = [section for section in right if section.uuid and ("right", section.uuid) not in taken]
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
        unmatched_left=[section for section in left if id(section) not in grouped],
        unmatched_right=[section for section in right if id(section) not in grouped],
        declared=declared,
        resolution=resolution,
    )


def load_section_alignment(
    left_records: list[Record],
    right_records: list[Record],
    patch_path: Path,
    threshold: float = DEFAULT_THRESHOLD,
    writes: Writes | None = None,
) -> SectionAlignment:
    """Lit, valide et applique le patch (ValueError s'il est incohérent). Si
    des lignes ont été réancrées, le patch est réécrit via `writes`
    (numrev/command.py) ; sans `writes` (lecteurs : viewer, export, audit), jamais."""
    entries = read_section_patch(patch_path)
    validate_section_patch(entries)
    alignment = align_sections(left_records, right_records, entries, threshold)
    if writes is not None and alignment.resolution.reanchored:
        writes.add(patch_path, lambda: write_section_patch(patch_path, alignment.resolution.updated(entries)))
    return alignment


def section_orphans(alignment: SectionAlignment) -> list[str]:
    """Lignes orphelines du patch des rubriques, lisibles (numrev/curation.py :
    elles font paniquer les scripts, sauf --force)."""
    return [
        " ↔ ".join(filter(None, (f"{entry.left_uuid} {entry.left_title}".strip(), f"{entry.right_uuid} {entry.right_title}".strip())))
        for entry in alignment.resolution.orphans
    ]


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


def segment_entries(alignment: SectionAlignment) -> list[tuple[list[Record], list[Record]]]:
    """Segments à aligner : un par groupe de rubriques appariées (entrées de
    ses rubriques concaténées, dans l'ordre de chaque annuaire). Une rubrique
    sans correspondance n'est dans aucun segment : ses entrées ne sont jamais
    appariées tant qu'elle n'est pas liée dans le patch des rubriques."""
    return [
        (
            [record for section in group.left for record in section.records],
            [record for section in group.right for record in section.records],
        )
        for group in alignment.groups
    ]


def restrict_to_corresponding(
    links: list[Link], left: dict[str, Record], right: dict[str, Record], alignment: SectionAlignment
) -> tuple[list[Link], list[Link]]:
    """(liens gardés, liens écartés) : un lien entre deux entrées dont les
    rubriques ne se correspondent pas (`corresponding`) est écarté, quelle
    que soit sa source — Dedupe, ou patch des entrées (corriger alors le
    patch des rubriques). Les liens vers des entrées inconnues sont gardés
    (signalés ailleurs)."""
    matching = corresponding(alignment)
    kept, dropped = [], []
    for link in links:
        left_record, right_record = left.get(link.left_uuid), right.get(link.right_uuid)
        known = left_record is not None and right_record is not None
        (dropped if known and (left_record.section_uuid, right_record.section_uuid) not in matching else kept).append(link)
    return kept, dropped
